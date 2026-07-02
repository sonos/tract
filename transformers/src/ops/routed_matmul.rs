use std::fmt;
use std::hash::{Hash, Hasher};

use tract_nnef::internal::*;
use tract_nnef::tract_core::tract_linalg::block_quant::{
    BlockQuantStorage, PackedBlockQuantFormat,
};
use tract_nnef::tract_core::tract_linalg::mmm::{
    AsInputValue, FusedSpec, MMMInputFormat, MMMInputValue, MatMatMul, PackedExoticFact,
    ScratchSpace,
};
use tract_nnef::tract_core::tract_linalg::pack::{PackedFormat, PackingWriter};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct RoutedRowsInput {
    base: usize,
    row_byte_offsets: Vec<isize>,
    k: usize,
    k_stride_bytes: isize,
    format: PackedFormat,
    fact: PackedExoticFact,
}

impl RoutedRowsInput {
    pub(super) fn new(
        base: *const f32,
        row_byte_offsets: Vec<isize>,
        k: usize,
        k_stride_bytes: isize,
        format: PackedFormat,
    ) -> Self {
        let fact = PackedExoticFact {
            format: Box::new(format.clone()),
            mn: row_byte_offsets.len().to_dim(),
            k,
        };
        RoutedRowsInput { base: base as usize, row_byte_offsets, k, k_stride_bytes, format, fact }
    }

    fn read(&self, mn: usize, k: usize) -> f32 {
        unsafe {
            let ptr = (self.base as *const u8)
                .offset(self.row_byte_offsets[mn] + self.k_stride_bytes * k as isize);
            *(ptr as *const f32)
        }
    }
}

impl Hash for RoutedRowsInput {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.base.hash(state);
        self.row_byte_offsets.hash(state);
        self.k.hash(state);
        self.k_stride_bytes.hash(state);
        self.format.hash(state);
    }
}

impl fmt::Display for RoutedRowsInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RoutedRowsInput(mn={}, k={}, {})", self.mn(), self.k, self.format)
    }
}

unsafe impl Send for RoutedRowsInput {}
unsafe impl Sync for RoutedRowsInput {}

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
        let mn_start = i * r;
        let mn_end = (mn_start + r).min(self.row_byte_offsets.len());
        ensure!(mn_start < self.row_byte_offsets.len(), "panel {i} starts past routed rows");

        unsafe {
            std::ptr::write_bytes(buffer, 0, self.format.single_panel_layout(self.k, 4).size());
            let mut writer = self.format.write_with_k_outer(buffer as *mut f32, self.k, r);
            for k in 0..self.k {
                for mn in mn_start..mn_start + r {
                    writer.write(if mn < mn_end { self.read(mn, k) } else { 0.0 });
                }
            }
        }
        Ok(buffer)
    }

    fn mn(&self) -> usize {
        self.row_byte_offsets.len()
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
        for (k, slot) in slice.iter_mut().enumerate() {
            *slot = self.read(mn, k);
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
pub(super) struct RoutedInputRows {
    pub(super) base: *const f32,
    pub(super) row_byte_offsets: Vec<isize>,
    pub(super) k_stride_bytes: isize,
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

fn select_block_quant_left_mmm(
    weight: &Tensor,
    out_dim: usize,
    k_dim: usize,
) -> TractResult<PreparedPackedMatMul> {
    let bqs = weight.try_storage_as::<BlockQuantStorage>()?;
    let mut best: Option<(
        isize,
        bool,
        usize,
        usize,
        Box<dyn MatMatMul>,
        usize,
        PackedBlockQuantFormat,
        PackedFormat,
    )> = None;

    for mmm in tract_nnef::tract_core::tract_linalg::ops().mmm_impls() {
        if !mmm.is_supported_here()
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
            if k_dim % a_format.k_alignment() != 0 || k_dim % b_format.k_alignment() != 0 {
                continue;
            }

            let score = -(mmm.quality().cost() as isize * 1000) + mmm.dynamic_boost();
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

pub(super) fn run_prepared_routed_matmul(
    plan: &PreparedRoutedMatMul,
    group: usize,
    rows: RoutedInputRows,
    output: &mut Tensor,
    state: &mut PreparedRoutedMatMulState,
) -> TractResult<()> {
    ensure!(
        group < plan.left_by_group.len(),
        "prepared routed matmul group {group} out of range for {} groups",
        plan.left_by_group.len()
    );
    let route_count = rows.row_byte_offsets.len();
    let out_dim = plan.kernel.out_dim;
    if route_count == 0 {
        return Ok(());
    }
    ensure!(
        output.len() >= route_count * out_dim,
        "routed matmul output scratch has {} values, needs {}",
        output.len(),
        route_count * out_dim
    );

    if state
        .scratch
        .as_ref()
        .is_none_or(|s| !unsafe { plan.kernel.mmm.can_use_scratch_space(&**s) })
    {
        state.scratch = Some(unsafe { plan.kernel.mmm.allocate_scratch_space() });
    }
    let scratch = state.scratch.as_mut().context("prepared MMM scratch was not allocated")?;

    let input = RoutedRowsInput::new(
        rows.base,
        rows.row_byte_offsets,
        plan.kernel.k_dim,
        rows.k_stride_bytes,
        plan.kernel.right_format.clone(),
    );
    let shape = [route_count, out_dim];
    let strides = [out_dim as isize, 1];
    let view = unsafe { TensorView::from_bytes(output, 0, &shape, &strides) };
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
        FusedSpec::Store(store),
    ];
    unsafe { plan.kernel.mmm.run_with_scratch_space(out_dim, route_count, scratch.as_mut(), &uops) }
}
