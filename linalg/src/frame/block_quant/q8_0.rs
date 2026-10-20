use super::opaque::impl_opaque_block_quant;
use super::opaque::load_f16;
use super::opaque::store_f16;
use super::*;

/// ggml `block_q8_0`: 32 values as an f16 scale followed by 32 `i8`, 34 bytes.
///
/// The quantizer is llama.cpp `quantize_row_q8_0_ref`.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Q8_0F;

/// The ggml Q8_0 block format.
pub const Q8_0: Q8_0F = Q8_0F;

fn quant_q8_0(block: &[f32], quant: &mut [u8]) {
    let amax = block.iter().fold(0f32, |acc, v| acc.max(v.abs()));
    let d = amax / 127.0;
    let id = if d == 0.0 { 0.0 } else { 1.0 / d };
    store_f16(&mut quant[..2], d);
    for (q, &v) in quant[2..].iter_mut().zip(block) {
        *q = (v * id).round() as i8 as u8;
    }
}

fn dequant_q8_0(quant: &[u8], block: &mut [f32]) {
    let d = load_f16(quant);
    for (dst, &q) in block.iter_mut().zip(&quant[2..]) {
        *dst = (q as i8) as f32 * d;
    }
}

impl_opaque_block_quant!(Q8_0F, "Q8_0", 32, 34, quant_q8_0, dequant_q8_0);

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(len: usize, zeros: usize) -> Vec<f32> {
        (0..len)
            .map(|i| if i < zeros { 0.0 } else { (((i * 17 + 3) % 41) as f32 - 20.0) / 10.0 })
            .collect()
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn q8_0_roundtrip() -> TractResult<()> {
        let data = ramp(64, 0);
        let back = Q8_0.dequant_f32(&Q8_0.quant_f32(&data)?)?;
        let back = back.try_as_plain_ram()?;
        let err = data.iter().zip(back.as_slice::<f32>()?).map(|(a, b)| (a - b).abs());
        assert!(err.fold(0f32, f32::max) < 0.01);
        Ok(())
    }

    /// Bytes and values from llama.cpp `quantize_row_q8_0_ref` / `dequantize_row_q8_0`.
    #[test]
    fn q8_0_matches_ggml_ref() {
        let mut q = [0u8; 34];
        let mut back = [0f32; 32];
        Q8_0.quant_block_f32(&ramp(32, 0), &mut q);
        assert_eq!(hex(&q), "082494006cd440a7137fe753ba268efa66cd39a10d79e04cb42087f35fc7339a0672");
        Q8_0.dequant_block_f32(&q, &mut back);
        assert_eq!(back[0].to_bits(), 0xbfd9_b000);
        assert_eq!(back[8].to_bits(), 0xbec9_9000);
        assert_eq!(back[31].to_bits(), 0x3fe5_c800);
        Q8_0.quant_block_f32(&ramp(32, 8), &mut q);
        assert_eq!(hex(&q), "a9230000000000000000e557b62888f96bcb3c9c0d7fdf50b02181f364c435950778");
        Q8_0.dequant_block_f32(&q, &mut back);
        assert_eq!(back[0].to_bits(), 0);
        assert_eq!(back[8].to_bits(), 0xbece_d300);
        assert_eq!(back[31].to_bits(), 0x3fe5_ce00);
        Q8_0.quant_block_f32(&[0f32; 32], &mut q);
        assert!(q.iter().all(|b| *b == 0));
    }
}
