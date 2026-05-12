//! MILBlob v2 weight file writer.
//!
//! Format reference: `apple/coremltools/mlmodel/src/MILBlob/Blob/StorageFormat.hpp`
//! at the commit pinned in `proto/MIL_PROTO_VERSION.md`.
//!
//! Layout:
//! ```text
//! offset 0:    storage_header   (64 bytes: count u32, version u32 = 2,
//!                                 reserved [u64; 7])
//! offset 64:   blob_metadata_0  (64 bytes: sentinel 0xDEADBEEF u32,
//!                                 dtype u32, size_in_bytes u64,
//!                                 data_offset u64, padding_bits u64,
//!                                 reserved [u64; 4])
//! offset 128:  data_0           (size_in_bytes_0 bytes; pad to 64-byte
//!                                 alignment after)
//! offset N:    blob_metadata_1 ...
//! ```
//!
//! `BlobFileValue.offset` in MIL points to a `blob_metadata` struct, not the
//! data — verified empirically in the Phase 1 spike.

const BLOB_HEADER_SIZE: u64 = 64;
const BLOB_METADATA_SIZE: u64 = 64;
const BLOB_ALIGNMENT: u64 = 64;
const BLOB_SENTINEL: u32 = 0xDEAD_BEEF;
const BLOB_VERSION: u32 = 2;

/// MILBlob v2 dtype enum, from `apple/coremltools/mlmodel/src/MILBlob/Blob/BlobDataType.hpp`.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum BlobDataType {
    Float16 = 1,
    Float32 = 2,
    UInt8 = 3,
    Int8 = 4,
    BFloat16 = 5,
    Int16 = 6,
    UInt16 = 7,
    Int4 = 8,
    UInt1 = 9,
    UInt2 = 10,
    UInt4 = 11,
    UInt3 = 12,
    UInt6 = 13,
    Int32 = 14,
    UInt32 = 15,
    Float8E4M3FN = 16,
    Float8E5M2 = 17,
}

/// Builds a MILBlob v2 file incrementally. Returns the metadata-offset for each
/// added blob — pass that offset into `BlobFileValue.offset` in the MIL program.
///
/// Example:
/// ```ignore
/// let mut b = BlobBuilder::new();
/// let conv_weight_offset = b.add(BlobDataType::Float16, &fp16_weights);
/// let bias_offset       = b.add(BlobDataType::Float16, &fp16_bias);
/// let bytes = b.finish();
/// ```
pub struct BlobBuilder {
    out: Vec<u8>,
    count: u32,
}

impl Default for BlobBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BlobBuilder {
    pub fn new() -> Self {
        // Pre-allocate the header; finalized in `finish()`.
        Self { out: vec![0u8; BLOB_HEADER_SIZE as usize], count: 0 }
    }

    /// Add a blob and return the offset of its metadata struct (the value to put
    /// in `BlobFileValue.offset` in MIL).
    pub fn add(&mut self, dtype: BlobDataType, data: &[u8]) -> u64 {
        // Align current write position so this metadata starts on a 64-byte boundary.
        // (The data immediately following — exactly 64 bytes later — will then also
        // be 64-aligned.)
        self.pad_to_alignment();
        let metadata_offset = self.out.len() as u64;
        let data_offset = metadata_offset + BLOB_METADATA_SIZE;

        let mut meta = [0u8; 64];
        meta[0..4].copy_from_slice(&BLOB_SENTINEL.to_le_bytes());
        meta[4..8].copy_from_slice(&(dtype as u32).to_le_bytes());
        meta[8..16].copy_from_slice(&(data.len() as u64).to_le_bytes());
        meta[16..24].copy_from_slice(&data_offset.to_le_bytes());
        // padding_size_in_bits + reserved_1..4 stay zero
        self.out.extend_from_slice(&meta);
        self.out.extend_from_slice(data);

        self.count += 1;
        metadata_offset
    }

    fn pad_to_alignment(&mut self) {
        let n = self.out.len() as u64;
        let pad = (BLOB_ALIGNMENT - (n % BLOB_ALIGNMENT)) % BLOB_ALIGNMENT;
        for _ in 0..pad as usize {
            self.out.push(0);
        }
    }

    /// Finalize and return the MILBlob v2 byte stream.
    pub fn finish(mut self) -> Vec<u8> {
        self.out[0..4].copy_from_slice(&self.count.to_le_bytes());
        self.out[4..8].copy_from_slice(&BLOB_VERSION.to_le_bytes());
        // Reserved fields (8..64) stay zero.
        self.out
    }
}

/// Convenience wrapper for the single-blob case.
pub fn build_single_blob(dtype: BlobDataType, data: &[u8]) -> Vec<u8> {
    let mut b = BlobBuilder::new();
    b.add(dtype, data);
    b.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin the byte layout against the format spec. If this test breaks, either
    /// the format has changed (re-vendor protos + check StorageFormat.hpp) or
    /// the writer has regressed.
    ///
    /// The expected layout matches what coremltools emitted for the Phase 1
    /// spike's `SpikeConv_A.mlpackage/Data/com.apple.CoreML/weights/weight.bin`.
    #[test]
    fn single_blob_v2_layout_byte_exact() {
        // 8 * 4 * 3 * 3 FP16 weights = 288 elements * 2 bytes = 576 bytes.
        let data: Vec<u8> = (0..576u32).map(|i| (i & 0xff) as u8).collect();
        let bytes = build_single_blob(BlobDataType::Float16, &data);

        // Total: 64 (storage_header) + 64 (blob_metadata) + 576 (data) = 704
        assert_eq!(bytes.len(), 704, "total blob size");

        // storage_header
        assert_eq!(&bytes[0..4], &1u32.to_le_bytes(), "count = 1");
        assert_eq!(&bytes[4..8], &2u32.to_le_bytes(), "version = 2");
        for (i, &b) in bytes.iter().enumerate().take(64).skip(8) {
            assert_eq!(b, 0, "header reserved byte {i} should be zero");
        }

        // blob_metadata at offset 64
        assert_eq!(&bytes[64..68], &0xDEAD_BEEFu32.to_le_bytes(), "sentinel = 0xDEADBEEF");
        assert_eq!(
            &bytes[68..72],
            &(BlobDataType::Float16 as u32).to_le_bytes(),
            "dtype = Float16"
        );
        assert_eq!(&bytes[72..80], &576u64.to_le_bytes(), "size_in_bytes");
        assert_eq!(&bytes[80..88], &128u64.to_le_bytes(), "data_offset = 128");
        for (i, &b) in bytes.iter().enumerate().take(128).skip(88) {
            assert_eq!(b, 0, "metadata reserved byte {i} should be zero");
        }

        // Data at offset 128
        assert_eq!(&bytes[128..704], &data[..]);
    }

    #[test]
    fn multi_blob_offsets_are_64_aligned() {
        let mut b = BlobBuilder::new();
        // First blob: 100 bytes — leaves writer at offset 64+64+100 = 228, not 64-aligned.
        let off1 = b.add(BlobDataType::Float16, &[1u8; 100]);
        // Second blob's metadata MUST start at the next 64-aligned offset (256).
        let off2 = b.add(BlobDataType::Float32, &[2u8; 200]);
        let bytes = b.finish();

        assert_eq!(off1, 64, "first blob_metadata at offset 64");
        assert_eq!(off2, 256, "second blob_metadata at next 64-aligned offset");

        // Sentinels at both offsets
        assert_eq!(&bytes[off1 as usize..off1 as usize + 4], &0xDEAD_BEEFu32.to_le_bytes());
        assert_eq!(&bytes[off2 as usize..off2 as usize + 4], &0xDEAD_BEEFu32.to_le_bytes());

        // count = 2 in the storage_header
        assert_eq!(&bytes[0..4], &2u32.to_le_bytes());
    }
}
