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
use tract_nnef::tract_core::tract_linalg::BinOp;
use tract_nnef::tract_core::tract_linalg::block_quant::{
    BlockQuantStorage, PackedBlockQuantFormat,
};
use tract_nnef::tract_core::tract_linalg::mmm::{
    AsInputValue, EagerPackedInput, FusedSpec, MMMInputFormat, MMMInputValue, MatMatMul,
    PackedExoticFact, ScratchSpace,
};
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

#[derive(Clone, Debug)]
struct PreparedPackedMatMul {
    mmm: Box<dyn MatMatMul>,
    packing: usize,
    left_format: Box<dyn MMMInputFormat>,
    right_format: PackedFormat,
    k_dim: usize,
    out_dim: usize,
}

// CPU implementation of the reusable routed-matmul contract: one prepared
// packed-left matrix per group, plus an indirect list of runtime input rows.
#[derive(Clone, Debug)]
pub(super) struct PreparedRoutedMatMul {
    kernel: PreparedPackedMatMul,
    left_by_group: Vec<Box<dyn MMMInputValue>>,
}

#[derive(Clone, Debug)]
pub(super) struct RoutedMatMulGroup {
    pub(super) group: usize,
    pub(super) rows: RoutedInputRows,
    pub(super) output_row_offset: usize,
}

#[derive(Default)]
pub(super) struct PreparedRoutedMatMulState {
    scratch: Option<Box<dyn ScratchSpace>>,
}

impl Clone for PreparedRoutedMatMulState {
    fn clone(&self) -> Self {
        PreparedRoutedMatMulState::default()
    }
}

impl fmt::Debug for PreparedRoutedMatMulState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedRoutedMatMulState")
            .field("has_scratch", &self.scratch.is_some())
            .finish()
    }
}

#[allow(clippy::type_complexity)]
fn select_block_quant_left_mmm(
    weight: &Tensor,
    out_dim: usize,
    k_dim: usize,
) -> TractResult<PreparedPackedMatMul> {
    let bqs = weight.try_storage_as::<BlockQuantStorage>()?;
    let mut best: Option<(
        (bool, isize),
        bool,
        usize,
        usize,
        Box<dyn MatMatMul>,
        usize,
        PackedBlockQuantFormat,
        PackedFormat,
    )> = None;

    for mmm in tract_nnef::tract_core::tract_linalg::MmmDispatch::native().runnable() {
        if !mmm.runnable()
            || mmm.internal_type() != f32::datum_type()
            || !mmm.stores().contains(&f32::datum_type())
        {
            continue;
        }
        for (packing, (a, b)) in mmm.packings().iter().enumerate() {
            let Some(a_format) = a.downcast_ref::<PackedBlockQuantFormat>() else {
                continue;
            };
            if !a_format.bq.dyn_eq(bqs.format()) {
                continue;
            }
            let Some(b_format) = b.downcast_ref::<PackedFormat>() else {
                continue;
            };
            if !b_format.precursor().as_dt().is_some_and(|dt| dt == f32::datum_type()) {
                continue;
            }
            if !k_dim.is_multiple_of(a_format.k_alignment())
                || !k_dim.is_multiple_of(b_format.k_alignment())
            {
                continue;
            }

            // A kernel written for this architecture always beats a portable one,
            // whatever their preference; among kernels of the same kind, higher
            // preference wins. Mirrors `retain_best`'s own ranking key.
            let score = (mmm.arch().is_some(), mmm.preference());
            let candidate = (
                score,
                mmm.nr() == 1,
                mmm.mr().min(out_dim),
                mmm.nr(),
                mmm.clone(),
                packing,
                a_format.clone(),
                b_format.clone(),
            );
            if best.as_ref().is_none_or(|current| {
                (&candidate.0, &candidate.1, &candidate.2, &candidate.3)
                    > (&current.0, &current.1, &current.2, &current.3)
            }) {
                best = Some(candidate);
            }
        }
    }

    let Some((_score, _is_gemv, _mr, _nr, mmm, packing, a_format, b_format)) = best else {
        bail!("no runnable f32 block-quant MMM found for weight shape [{out_dim}, {k_dim}]");
    };

    Ok(PreparedPackedMatMul {
        mmm,
        packing,
        left_format: Box::new(a_format),
        right_format: b_format,
        k_dim,
        out_dim,
    })
}

fn pack_left_rows(t: &Tensor, format: &dyn MMMInputFormat) -> TractResult<Box<dyn MMMInputValue>> {
    ensure!(t.rank() == 2, "packed-left tensor must be rank 2, got {:?}", t.shape());
    format.prepare_one(t, 1, 0)
}

fn pack_right_rows(
    kernel: &PreparedPackedMatMul,
    source: &Arc<Tensor>,
    rows: RoutedInputRows,
) -> TractResult<Box<dyn MMMInputValue>> {
    let input =
        RoutedRowsInput::new(source.clone(), rows, kernel.k_dim, kernel.right_format.clone())?;
    let panel_layout =
        kernel.right_format.single_panel_layout(kernel.k_dim, f32::datum_type().size_of());
    let panel_count = input.mn().divceil(kernel.right_format.r());
    let panel_bytes = panel_layout.size();
    let mut packed = unsafe {
        Blob::for_layout(std::alloc::Layout::from_size_align(
            panel_bytes * panel_count,
            panel_layout.align(),
        )?)
    };
    for panel in 0..panel_count {
        let buffer = unsafe { packed.as_mut_ptr().add(panel * panel_bytes) };
        input.panel_bytes(panel, Some(buffer))?;
    }
    Ok(Box::new(EagerPackedInput {
        fact: PackedExoticFact {
            format: Box::new(kernel.right_format.clone()),
            mn: input.mn().to_dim(),
            k: kernel.k_dim,
        },
        packed: packed.into(),
        panel_bytes,
        mn: input.mn(),
    }))
}

fn ensure_scratch<'a>(
    kernel: &PreparedPackedMatMul,
    state: &'a mut PreparedRoutedMatMulState,
) -> TractResult<&'a mut Box<dyn ScratchSpace>> {
    if state.scratch.as_ref().is_none_or(|s| !unsafe { kernel.mmm.can_use_scratch_space(&**s) }) {
        state.scratch = Some(unsafe { kernel.mmm.allocate_scratch_space() });
    }
    state.scratch.as_mut().context("prepared MMM scratch was not allocated")
}

pub(super) fn build_block_quant_routed_matmul(
    group_weights: Vec<Tensor>,
) -> TractResult<PreparedRoutedMatMul> {
    let sample =
        group_weights.first().context("prepared routed matmul needs at least one group")?;
    ensure!(sample.rank() == 2, "prepared routed matmul weights must be rank 2");
    let out_dim = sample.shape()[0];
    let k_dim = sample.shape()[1];
    let kernel = select_block_quant_left_mmm(sample, out_dim, k_dim)?;

    let mut left_by_group = Vec::with_capacity(group_weights.len());
    for (group, weight) in group_weights.iter().enumerate() {
        ensure!(
            weight.shape() == sample.shape(),
            "prepared routed matmul group {group} shape {:?} does not match sample {:?}",
            weight.shape(),
            sample.shape()
        );
        left_by_group.push(pack_left_rows(weight, &*kernel.left_format)?);
    }

    Ok(PreparedRoutedMatMul { kernel, left_by_group })
}

pub(super) fn pack_prepared_routed_matmul_rhs(
    plan: &PreparedRoutedMatMul,
    source: &Arc<Tensor>,
    rows: RoutedInputRows,
) -> TractResult<Box<dyn MMMInputValue>> {
    ensure!(rows.len() > 0, "prepared routed matmul cannot pack an empty RHS route list");
    pack_right_rows(&plan.kernel, source, rows)
}

fn run_prepared_routed_matmul_with_rhs(
    plan: &PreparedRoutedMatMul,
    group: usize,
    rhs: &dyn MMMInputValue,
    route_count: usize,
    output: &mut Tensor,
    output_row_offset: usize,
    state: &mut PreparedRoutedMatMulState,
) -> TractResult<()> {
    ensure!(
        group < plan.left_by_group.len(),
        "prepared routed matmul group {group} out of range for {} groups",
        plan.left_by_group.len()
    );
    let out_dim = plan.kernel.out_dim;
    if route_count == 0 {
        return Ok(());
    }
    ensure!(
        output.len() >= (output_row_offset + route_count) * out_dim,
        "routed matmul output scratch has {} values, needs {}",
        output.len(),
        (output_row_offset + route_count) * out_dim
    );
    ensure!(
        rhs.mn() >= route_count && rhs.k() == plan.kernel.k_dim,
        "routed matmul RHS has shape [{}, {}], needs at least [{route_count}, {}]",
        rhs.mn(),
        rhs.k(),
        plan.kernel.k_dim
    );

    let scratch = ensure_scratch(&plan.kernel, state)?;
    let shape = [route_count, out_dim];
    let strides = [out_dim as isize, 1];
    let output_offset_bytes = (output_row_offset * out_dim * f32::datum_type().size_of()) as isize;
    let view = unsafe { TensorView::from_bytes(output, output_offset_bytes, &shape, &strides) };
    let store = unsafe {
        plan.kernel
            .mmm
            .c_from_data_and_strides(f32::datum_type().size_of(), 1, out_dim as isize)
            .wrap(&view)
    };
    let uops = tvec![
        FusedSpec::AddMatMul {
            a: AsInputValue::Borrowed(&*plan.left_by_group[group]),
            b: AsInputValue::Borrowed(rhs),
            packing: plan.kernel.packing,
        },
        FusedSpec::Store(store),
    ];
    unsafe { plan.kernel.mmm.run_with_scratch_space(out_dim, route_count, scratch.as_mut(), &uops) }
}

pub(super) fn run_prepared_routed_matmul(
    plan: &PreparedRoutedMatMul,
    group: usize,
    source: &Arc<Tensor>,
    rows: RoutedInputRows,
    output: &mut Tensor,
    state: &mut PreparedRoutedMatMulState,
) -> TractResult<()> {
    let route_count = rows.len();
    if route_count == 0 {
        return Ok(());
    }
    let input = RoutedRowsInput::new(
        source.clone(),
        rows,
        plan.kernel.k_dim,
        plan.kernel.right_format.clone(),
    )?;
    run_prepared_routed_matmul_with_rhs(plan, group, &input, route_count, output, 0, state)
}

pub(super) fn run_prepared_routed_matmul_many(
    plan: &PreparedRoutedMatMul,
    source: &Arc<Tensor>,
    groups: &[RoutedMatMulGroup],
    output: &mut Tensor,
    state: &mut PreparedRoutedMatMulState,
) -> TractResult<()> {
    for group in groups {
        let route_count = group.rows.len();
        if route_count == 0 {
            continue;
        }
        let input = RoutedRowsInput::new(
            source.clone(),
            group.rows.clone(),
            plan.kernel.k_dim,
            plan.kernel.right_format.clone(),
        )?;
        run_prepared_routed_matmul_with_rhs(
            plan,
            group.group,
            &input,
            route_count,
            output,
            group.output_row_offset,
            state,
        )?;
    }
    Ok(())
}

pub(super) fn run_prepared_routed_matmul_many_same_rhs(
    plan: &PreparedRoutedMatMul,
    groups: &[(usize, usize)],
    rhs: &dyn MMMInputValue,
    route_count: usize,
    output: &mut Tensor,
    state: &mut PreparedRoutedMatMulState,
) -> TractResult<()> {
    for &(group, output_row_offset) in groups {
        run_prepared_routed_matmul_with_rhs(
            plan,
            group,
            rhs,
            route_count,
            output,
            output_row_offset,
            state,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_prepared_routed_matmul_accumulate_one(
    plan: &PreparedRoutedMatMul,
    group: usize,
    source: &Arc<Tensor>,
    rows: RoutedInputRows,
    output: &mut Tensor,
    output_row: usize,
    scale: &Tensor,
    state: &mut PreparedRoutedMatMulState,
) -> TractResult<()> {
    ensure!(
        group < plan.left_by_group.len(),
        "prepared routed matmul group {group} out of range for {} groups",
        plan.left_by_group.len()
    );
    ensure!(rows.len() == 1, "accumulating routed matmul expects exactly one RHS row");
    ensure!(scale.rank() == 0 && scale.datum_type() == f32::datum_type());
    let out_dim = plan.kernel.out_dim;
    ensure!(
        output.len() >= (output_row + 1) * out_dim,
        "routed matmul output has {} values, needs {}",
        output.len(),
        (output_row + 1) * out_dim
    );

    let scratch = ensure_scratch(&plan.kernel, state)?;
    let input = RoutedRowsInput::new(
        source.clone(),
        rows,
        plan.kernel.k_dim,
        plan.kernel.right_format.clone(),
    )?;
    let shape = [1, out_dim];
    let strides = [out_dim as isize, 1];
    let output_offset_bytes = (output_row * out_dim * f32::datum_type().size_of()) as isize;
    let view = unsafe { TensorView::from_bytes(output, output_offset_bytes, &shape, &strides) };
    let store = unsafe {
        plan.kernel
            .mmm
            .c_from_data_and_strides(f32::datum_type().size_of(), 1, out_dim as isize)
            .wrap(&view)
    };
    let uops = tvec![
        FusedSpec::AddMatMul {
            a: AsInputValue::Borrowed(&*plan.left_by_group[group]),
            b: AsInputValue::Borrowed(&input),
            packing: plan.kernel.packing,
        },
        FusedSpec::BinScalar(scale, BinOp::Mul),
        FusedSpec::AddUnicast(store),
        FusedSpec::Store(store),
    ];
    unsafe { plan.kernel.mmm.run_with_scratch_space(out_dim, 1, scratch.as_mut(), &uops) }
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
