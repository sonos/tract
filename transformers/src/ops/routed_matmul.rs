//! CPU routed packing and packed block-quant matmul execution.
//!
//! The low-level MMM interface requires unsafe stores and kernel calls. Using
//! ndarray matmul here would require unpacking quantized expert weights and
//! would lose the fused scaled accumulation into the destination token row.
//! Indirect row packing avoids a separate gathered activation tensor; kernel
//! scratch is reused between synchronous calls. Packing operands share ownership
//! of their input tensors and validate all row offsets before execution.

use std::fmt;
use std::hash::{Hash, Hasher};

use tract_nnef::internal::*;
use tract_nnef::tract_core::tract_linalg::mmm::{MMMInputFormat, MMMInputValue, PackedExoticFact};
use tract_nnef::tract_core::tract_linalg::pack::{PackedFormat, PackingWriter};

/// Share f32 inputs without copying their storage; convert other input dtypes once.
pub(super) fn f32_input(input: &TValue) -> TractResult<Arc<Tensor>> {
    if input.datum_type() == DatumType::F32 {
        Ok(input.clone().into_arc_tensor())
    } else {
        Ok(Arc::new(input.cast_to::<f32>()?.into_owned()))
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(super) enum RoutedRowOffsets {
    Single(usize),
    Regular { start: usize, len: usize, stride: usize },
    Explicit(Vec<usize>),
}

impl RoutedRowOffsets {
    fn len(&self) -> usize {
        match self {
            Self::Single(_) => 1,
            Self::Regular { len, .. } => *len,
            Self::Explicit(offsets) => offsets.len(),
        }
    }

    fn get(&self, ix: usize) -> usize {
        match self {
            Self::Single(offset) => *offset,
            Self::Regular { start, stride, .. } => start + stride * ix,
            Self::Explicit(offsets) => offsets[ix],
        }
    }
}

/// Element offsets only: reusable route metadata never retains an input tensor.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(super) struct RoutedInputRows {
    row_offsets: RoutedRowOffsets,
    k_stride: usize,
}

impl RoutedInputRows {
    pub(super) fn single(row_offset: usize, k_stride: usize) -> Self {
        Self { row_offsets: RoutedRowOffsets::Single(row_offset), k_stride }
    }

    pub(super) fn regular(start: usize, len: usize, stride: usize, k_stride: usize) -> Self {
        Self { row_offsets: RoutedRowOffsets::Regular { start, len, stride }, k_stride }
    }

    pub(super) fn explicit(row_offsets: Vec<usize>, k_stride: usize) -> Self {
        match row_offsets.as_slice() {
            [offset] => Self::single(*offset, k_stride),
            [first, second, ..] => {
                if let Some(stride) = second.checked_sub(*first)
                    && row_offsets.windows(2).all(|w| w[1].checked_sub(w[0]) == Some(stride))
                {
                    return Self::regular(*first, row_offsets.len(), stride, k_stride);
                }
                Self { row_offsets: RoutedRowOffsets::Explicit(row_offsets), k_stride }
            }
            _ => Self { row_offsets: RoutedRowOffsets::Explicit(row_offsets), k_stride },
        }
    }

    pub(super) fn len(&self) -> usize {
        self.row_offsets.len()
    }

    fn validate(&self, len: usize, k: usize) -> TractResult<()> {
        let width = if k == 0 {
            0
        } else {
            self.k_stride
                .checked_mul(k - 1)
                .and_then(|n| n.checked_add(1))
                .context("routed row width overflow")?
        };
        let check = |offset: usize| -> TractResult<()> {
            let end = offset.checked_add(width).context("routed row end overflow")?;
            ensure!(end <= len, "routed row ends at {end}, beyond input length {len}");
            Ok(())
        };
        match &self.row_offsets {
            RoutedRowOffsets::Single(offset) => check(*offset)?,
            RoutedRowOffsets::Regular { start, len, stride } if *len > 0 => {
                let last = start
                    .checked_add(stride.checked_mul(len - 1).context("routed row stride overflow")?)
                    .context("routed row offset overflow")?;
                check(last)?;
            }
            RoutedRowOffsets::Regular { .. } => (),
            RoutedRowOffsets::Explicit(offsets) => {
                for &offset in offsets {
                    check(offset)?;
                }
            }
        }
        Ok(())
    }
}

/// A live packing operand owns its source. Cloning it shares storage, while
/// validated element offsets permit safe slice reads without gathered copies.
#[derive(Clone, Debug)]
pub(super) struct RoutedRowsInput {
    source: Arc<Tensor>,
    rows: RoutedInputRows,
    k: usize,
    format: PackedFormat,
    fact: PackedExoticFact,
}

impl RoutedRowsInput {
    pub(super) fn new(
        source: Arc<Tensor>,
        rows: RoutedInputRows,
        k: usize,
        format: PackedFormat,
    ) -> TractResult<Self> {
        rows.validate(source.try_as_plain_ram()?.as_slice::<f32>()?.len(), k)?;
        let fact =
            PackedExoticFact { format: Box::new(format.clone()), mn: rows.len().to_dim(), k };
        Ok(Self { source, rows, k, format, fact })
    }

    fn read(&self, data: &[f32], mn: usize, k: usize) -> f32 {
        data[self.rows.row_offsets.get(mn) + self.rows.k_stride * k]
    }
}

impl PartialEq for RoutedRowsInput {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.source, &other.source)
            && self.rows == other.rows
            && self.k == other.k
            && self.format == other.format
    }
}

impl Eq for RoutedRowsInput {}

impl Hash for RoutedRowsInput {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.source).hash(state);
        self.rows.hash(state);
        self.k.hash(state);
        self.format.hash(state);
    }
}

impl fmt::Display for RoutedRowsInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RoutedRowsInput(mn={}, k={}, {})", self.mn(), self.k, self.format)
    }
}

impl MMMInputValue for RoutedRowsInput {
    fn format(&self) -> &dyn MMMInputFormat {
        &self.format
    }

    fn scratch_panel_buffer_layout(&self) -> Option<std::alloc::Layout> {
        Some(self.format.single_panel_layout(self.k, f32::datum_type().size_of()))
    }

    fn panel_bytes(&self, i: usize, buffer: Option<*mut u8>) -> TractResult<*const u8> {
        let buffer = buffer.context("RoutedRowsInput requires a scratch panel buffer")?;
        let r = self.format.r();
        let mn_start = i.checked_mul(r).context("routed panel offset overflow")?;
        let panel_end = mn_start.checked_add(r).context("routed panel end overflow")?;
        let mn_end = panel_end.min(self.rows.len());
        ensure!(mn_start < self.rows.len(), "panel {i} starts past routed rows");

        let plain = self.source.try_as_plain_ram()?;
        let data = plain.as_slice::<f32>()?;
        unsafe {
            std::ptr::write_bytes(buffer, 0, self.format.single_panel_layout(self.k, 4).size());
            let mut writer = self.format.write_with_k_outer(buffer as *mut f32, self.k, r);
            for k in 0..self.k {
                for mn in mn_start..panel_end {
                    writer.write(if mn < mn_end { self.read(data, mn, k) } else { 0.0 });
                }
            }
        }
        Ok(buffer)
    }

    fn mn(&self) -> usize {
        self.rows.len()
    }

    fn k(&self) -> usize {
        self.k
    }

    fn exotic_fact(&self) -> &dyn ExoticFact {
        &self.fact
    }

    fn extract_at_mn_f16(&self, _mn: usize, _slice: &mut [f16]) -> TractResult<()> {
        bail!("RoutedRowsInput only supports f32 extraction")
    }

    fn extract_at_mn_f32(&self, mn: usize, slice: &mut [f32]) -> TractResult<()> {
        ensure!(slice.len() == self.k);
        ensure!(mn < self.mn());
        let plain = self.source.try_as_plain_ram()?;
        let data = plain.as_slice::<f32>()?;
        for (k, slot) in slice.iter_mut().enumerate() {
            *slot = self.read(data, mn, k);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routed_operand_owns_shared_storage() -> TractResult<()> {
        fn send_sync<T: Send + Sync>() {}
        send_sync::<RoutedInputRows>();
        send_sync::<RoutedRowsInput>();
        let source =
            Arc::new(Tensor::from_shape(&[3, 4], &(0..12).map(|n| n as f32).collect::<Vec<_>>())?);
        let weak = Arc::downgrade(&source);
        let operand = RoutedRowsInput::new(
            source.clone(),
            RoutedInputRows::explicit(vec![8, 0, 4], 2),
            2,
            PackedFormat::new(DatumType::F32, 2, 4),
        )?;
        let clone = operand.clone();
        assert!(Arc::ptr_eq(&source, &clone.source));
        assert_eq!(Arc::strong_count(&source), 3);
        assert_eq!(operand, clone);
        drop(source);
        drop(operand);
        let mut values = [0f32; 2];
        clone.extract_at_mn_f32(0, &mut values)?;
        assert_eq!(values, [8., 10.]);
        clone.extract_at_mn_f32(1, &mut values)?;
        assert_eq!(values, [0., 2.]);
        let mut panel = vec![0f32; clone.scratch_panel_buffer_layout().unwrap().size() / 4];
        clone.panel_bytes(0, Some(panel.as_mut_ptr().cast()))?;
        assert_eq!(&panel[..4], &[8., 0., 10., 2.]);
        clone.panel_bytes(1, Some(panel.as_mut_ptr().cast()))?;
        assert_eq!(&panel[..4], &[4., 0., 6., 0.]);
        assert!(clone.extract_at_mn_f32(3, &mut values).is_err());
        assert!(clone.panel_bytes(usize::MAX, Some(panel.as_mut_ptr().cast())).is_err());
        drop(clone);
        assert!(weak.upgrade().is_none());
        Ok(())
    }

    #[test]
    fn routed_operand_rejects_invalid_offsets() -> TractResult<()> {
        let source = Arc::new(Tensor::zero::<f32>(&[8])?);
        let format = PackedFormat::new(DatumType::F32, 2, 4);
        for (rows, k) in [
            (RoutedInputRows::single(8, 1), 1),
            (RoutedInputRows::single(0, usize::MAX), 2),
            (RoutedInputRows::regular(usize::MAX, 2, 1, 1), 1),
            (RoutedInputRows::regular(0, 3, usize::MAX, 1), 1),
            (RoutedInputRows::explicit(vec![0, 7], 1), 2),
        ] {
            assert!(RoutedRowsInput::new(source.clone(), rows, k, format.clone()).is_err());
            assert_eq!(Arc::strong_count(&source), 1);
        }
        let wrong_type = Arc::new(Tensor::zero::<i32>(&[8])?);
        assert!(
            RoutedRowsInput::new(wrong_type, RoutedInputRows::single(0, 1), 1, format.clone())
                .is_err()
        );
        let empty = RoutedRowsInput::new(source, RoutedInputRows::single(8, 1), 0, format)?;
        empty.extract_at_mn_f32(0, &mut [])?;
        Ok(())
    }

    #[test]
    fn f32_inputs_are_shared_and_f16_inputs_are_converted() -> TractResult<()> {
        let source = Arc::new(tensor1(&[1f32, 2.]));
        let shared = f32_input(&source.clone().into_tvalue())?;
        assert!(Arc::ptr_eq(&source, &shared));
        let half = source.cast_to::<f16>()?.into_owned().into_tvalue();
        let converted = f32_input(&half)?;
        assert_eq!(converted.datum_type(), DatumType::F32);
        converted.close_enough(&source, Approximation::Exact)?;
        Ok(())
    }
}
