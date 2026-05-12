//! Conversion between tract `Tensor` (always contiguous) and `MLMultiArray`
//! (may have padded strides for ANE alignment).
//!
//! Critical: **always read `MLMultiArray.strides()` on the output side.** Core
//! ML returns buffers with non-contiguous strides — observed in the Phase 1
//! spike: a logical `[1, 8, 8, 8]` FP16 tensor came back with strides
//! `[2048, 256, 32, 1]` (W padded to 32 for ANE). Naive
//! `slice::from_raw_parts(dataPointer, count)` returns garbage past the first
//! row. See `feedback_mlmultiarray_strides.md` in the project memory.
//!
//! TODO(phase 2): migrate from `dataPointer` (deprecated) to
//! `getBytesWithHandler` / `getMutableBytesWithHandler` per ORT bug-lessons §7
//! (#17726). Requires the `block2` async-callback dance; out of scope for the
//! Phase 1 single-Conv MVP.
#![allow(deprecated)]

use anyhow::{Result, anyhow, bail, ensure};
use objc2::AllocAnyThread;
use objc2::rc::Retained;
use objc2_core_ml::{MLMultiArray, MLMultiArrayDataType};
use objc2_foundation::{NSArray, NSNumber};

use tract_core::internal::*;

/// Map a tract `DatumType` to the equivalent `MLMultiArrayDataType`.
///
/// Phase 1 covers FP32 + FP16 + I32 + I8 — extend as Phase 2 needs more.
pub fn datum_type_to_ml(dt: DatumType) -> Result<MLMultiArrayDataType> {
    Ok(match dt {
        DatumType::F32 => MLMultiArrayDataType::Float32,
        DatumType::F16 => MLMultiArrayDataType::Float16,
        DatumType::I32 => MLMultiArrayDataType::Int32,
        DatumType::I8 => MLMultiArrayDataType::Int8,
        other => bail!("DatumType {other:?} has no MLMultiArrayDataType equivalent"),
    })
}

/// Map an `MLMultiArrayDataType` back to a tract `DatumType`.
pub fn ml_to_datum_type(dt: MLMultiArrayDataType) -> Result<DatumType> {
    Ok(match dt {
        MLMultiArrayDataType::Float32 => DatumType::F32,
        MLMultiArrayDataType::Float16 => DatumType::F16,
        MLMultiArrayDataType::Int32 => DatumType::I32,
        MLMultiArrayDataType::Int8 => DatumType::I8,
        other => bail!("MLMultiArrayDataType({:?}) has no DatumType equivalent", other.0),
    })
}

fn elem_size(dt: MLMultiArrayDataType) -> Result<usize> {
    Ok(match dt {
        MLMultiArrayDataType::Float32 => 4,
        MLMultiArrayDataType::Float16 => 2,
        MLMultiArrayDataType::Int32 => 4,
        MLMultiArrayDataType::Int8 => 1,
        other => bail!("element size for MLMultiArrayDataType({:?}) unknown", other.0),
    })
}

/// Convert a contiguous tract `Tensor` into a fresh `MLMultiArray` of matching
/// shape and dtype.
///
/// The MLMultiArray is allocated via `initWithShape:dataType:error:` (Core ML
/// picks the strides — typically contiguous for a fresh user allocation). If
/// Core ML returns non-contiguous strides we walk them; the contiguous-fast-path
/// covers the common case with one `memcpy`.
pub fn tensor_to_mlmultiarray(t: &Tensor) -> Result<Retained<MLMultiArray>> {
    let dt = datum_type_to_ml(t.datum_type())?;
    let elem_bytes = elem_size(dt)?;
    let shape: Vec<isize> = t.shape().iter().map(|&s| s as isize).collect();

    let shape_nums: Vec<Retained<NSNumber>> =
        shape.iter().copied().map(NSNumber::new_isize).collect();
    let shape_refs: Vec<&NSNumber> = shape_nums.iter().map(|n| &**n).collect();
    let shape_array: Retained<NSArray<NSNumber>> = NSArray::from_slice(&shape_refs);

    let alloc = MLMultiArray::alloc();
    let arr = unsafe {
        MLMultiArray::initWithShape_dataType_error(alloc, &shape_array, dt)
            .map_err(|e| anyhow!("MLMultiArray init failed: {e:?}"))?
    };

    let n_elems = unsafe { arr.count() } as usize;
    ensure!(n_elems == t.len(), "MLMultiArray count mismatch: ml={n_elems}, tract={}", t.len());

    let strides = read_strides(&arr);
    let dst = unsafe { arr.dataPointer() }.as_ptr() as *mut u8;
    let src = t.as_bytes().as_ptr();

    if is_contiguous(&shape, &strides) {
        unsafe { std::ptr::copy_nonoverlapping(src, dst, n_elems * elem_bytes) };
    } else {
        // Defensive fallback: walk strides. Should not hit for fresh inputs;
        // if we do, that's a Core ML quirk worth logging.
        log::warn!(
            "tensor_to_mlmultiarray: non-contiguous input strides {strides:?} for shape {shape:?} \
             (should be rare for fresh allocations); falling back to stride-walk copy"
        );
        unsafe { copy_strided(src, dst, &shape, &strides, elem_bytes) };
    }

    Ok(arr)
}

/// Copy an `MLMultiArray` into a fresh contiguous tract `Tensor`.
///
/// Walks the source strides — Core ML routinely returns padded layouts on the
/// output side (see module docs).
pub fn mlmultiarray_to_tensor(arr: &MLMultiArray) -> Result<Tensor> {
    let dt_ml = unsafe { arr.dataType() };
    let dt = ml_to_datum_type(dt_ml)?;
    let elem_bytes = elem_size(dt_ml)?;

    let shape_nsa = unsafe { arr.shape() };
    let shape: Vec<usize> =
        (0..shape_nsa.count()).map(|i| shape_nsa.objectAtIndex(i).as_isize() as usize).collect();
    let n_elems: usize = shape.iter().product();
    let strides: Vec<isize> = shape.iter().map(|&s| s as isize).collect();
    let src_strides = read_strides(arr);
    ensure!(
        src_strides.len() == shape.len(),
        "rank mismatch: shape rank {} vs strides rank {}",
        shape.len(),
        src_strides.len()
    );

    let mut out = unsafe { Tensor::uninitialized_dt(dt, &shape)? };
    let dst = out.as_bytes_mut().as_mut_ptr();
    let src = unsafe { arr.dataPointer() }.as_ptr() as *const u8;

    if is_contiguous_isize(&strides, &src_strides) {
        unsafe { std::ptr::copy_nonoverlapping(src, dst, n_elems * elem_bytes) };
    } else {
        // Walk source strides; destination is contiguous (row-major shape).
        unsafe { copy_strided_in(src, dst, &shape, &src_strides, elem_bytes) };
    }

    Ok(out)
}

// ---------------- internals ----------------

fn read_strides(arr: &MLMultiArray) -> Vec<isize> {
    let strides_nsa = unsafe { arr.strides() };
    (0..strides_nsa.count()).map(|i| strides_nsa.objectAtIndex(i).as_isize()).collect()
}

/// True iff `strides` matches the contiguous row-major layout of `shape`
/// (in **elements**, the unit MLMultiArray uses).
fn is_contiguous(shape: &[isize], strides: &[isize]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }
    let mut expected: isize = 1;
    for i in (0..shape.len()).rev() {
        if strides[i] != expected {
            return false;
        }
        expected *= shape[i];
    }
    true
}

fn is_contiguous_isize(shape_isize: &[isize], strides: &[isize]) -> bool {
    is_contiguous(shape_isize, strides)
}

/// Stride-walk copy: tract → MLMultiArray. Source is contiguous; destination
/// uses `dst_strides` (in elements).
unsafe fn copy_strided(
    src: *const u8,
    dst: *mut u8,
    shape: &[isize],
    dst_strides: &[isize],
    elem_bytes: usize,
) {
    let mut indices = vec![0isize; shape.len()];
    let total: usize = shape.iter().map(|&s| s as usize).product();
    let dim = shape.len();

    for linear in 0..total {
        let mut off: isize = 0;
        for d in 0..dim {
            off += indices[d] * dst_strides[d];
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.add(linear * elem_bytes),
                dst.add(off as usize * elem_bytes),
                elem_bytes,
            );
        }

        // odometer
        for d in (0..dim).rev() {
            indices[d] += 1;
            if indices[d] < shape[d] {
                break;
            }
            indices[d] = 0;
        }
    }
}

/// Stride-walk copy: MLMultiArray → tract. Destination is contiguous; source
/// uses `src_strides` (in elements).
unsafe fn copy_strided_in(
    src: *const u8,
    dst: *mut u8,
    shape: &[usize],
    src_strides: &[isize],
    elem_bytes: usize,
) {
    let mut indices = vec![0usize; shape.len()];
    let total: usize = shape.iter().product();
    let dim = shape.len();

    for linear in 0..total {
        let mut off: isize = 0;
        for d in 0..dim {
            off += indices[d] as isize * src_strides[d];
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.add(off as usize * elem_bytes),
                dst.add(linear * elem_bytes),
                elem_bytes,
            );
        }

        for d in (0..dim).rev() {
            indices[d] += 1;
            if indices[d] < shape[d] {
                break;
            }
            indices[d] = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn fp32_roundtrip_contiguous() {
        // Build a tract Tensor → MLMultiArray → tract Tensor, verify equal.
        let data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.5 - 3.0).collect();
        let t1 = Tensor::from_shape::<f32>(&[2, 3, 4], &data).unwrap();
        let arr = tensor_to_mlmultiarray(&t1).expect("to_ml");
        let t2 = mlmultiarray_to_tensor(&arr).expect("from_ml");
        assert_eq!(t1.shape(), t2.shape());
        assert_eq!(t1.datum_type(), t2.datum_type());
        assert_eq!(t1.as_bytes(), t2.as_bytes());
    }

    #[test]
    fn fp16_roundtrip_contiguous() {
        let data: Vec<f16> = (0..16).map(|i| f16::from_f32((i as f32) * 0.25 - 2.0)).collect();
        let t1 = Tensor::from_shape::<f16>(&[2, 8], &data).unwrap();
        let arr = tensor_to_mlmultiarray(&t1).expect("to_ml");
        let t2 = mlmultiarray_to_tensor(&arr).expect("from_ml");
        assert_eq!(t1.shape(), t2.shape());
        assert_eq!(t1.datum_type(), t2.datum_type());
        assert_eq!(t1.as_bytes(), t2.as_bytes());
    }

    #[test]
    fn dtype_mapping_round_trips() {
        for dt in [DatumType::F32, DatumType::F16, DatumType::I32, DatumType::I8] {
            let ml = datum_type_to_ml(dt).unwrap();
            let back = ml_to_datum_type(ml).unwrap();
            assert_eq!(dt, back, "round-trip failed for {dt:?}");
        }
    }

    #[test]
    fn contiguous_predicate_basic() {
        // Logical row-major strides for [1, 8, 8, 8] should be [512, 64, 8, 1].
        assert!(is_contiguous(&[1, 8, 8, 8], &[512, 64, 8, 1]));
        // The Phase 1 spike's observed ANE-padded strides are NOT contiguous.
        assert!(!is_contiguous(&[1, 8, 8, 8], &[2048, 256, 32, 1]));
        // Single-dim cases.
        assert!(is_contiguous(&[5], &[1]));
        assert!(!is_contiguous(&[5], &[2]));
    }
}
