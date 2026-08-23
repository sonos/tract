use downcast_rs::{Downcast, impl_downcast};
use dyn_clone::DynClone;
use dyn_eq::DynEq;
use dyn_hash::DynHash;
use std::alloc::Layout;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::sync::Arc;
use tract_data::internal::*;

use crate::WeightType;

pub trait MMMInputFormat:
    Downcast + Debug + DynHash + dyn_eq::DynEq + DynClone + Send + Sync + Display
{
    fn prepare_tensor(&self, t: &Tensor, k_axis: usize, mn_axis: usize) -> TractResult<Tensor>;
    /// Pack one matmul out of a possibly strided view — one batch of a batched input, which a
    /// `&Tensor` cannot name. A format whose input is not a strided array of numbers (block
    /// quant, lazy im2col) says so here rather than being silently unreachable.
    fn prepare_one_view(
        &self,
        t: &TensorView,
        k_axis: usize,
        mn_axis: usize,
    ) -> TractResult<Box<dyn MMMInputValue>>;
    fn prepare_one(
        &self,
        t: &Tensor,
        k_axis: usize,
        mn_axis: usize,
    ) -> TractResult<Box<dyn MMMInputValue>> {
        self.prepare_one_view(&t.view(), k_axis, mn_axis)
    }
    fn precursor(&self) -> WeightType;
    /// Round `tensor` through the numeric precision this format stores its data
    /// in, so a reference matmul can reproduce the kernel's pack-time precision
    /// loss. Default: identity, for formats that store inputs losslessly.
    fn simulate_precision_loss(&self, tensor: Tensor) -> TractResult<Tensor> {
        Ok(tensor)
    }
    fn r(&self) -> usize;
    fn k_alignment(&self) -> usize;
    fn merge_with<'o, 'a: 'o, 'b: 'o>(
        &'a self,
        other: &'b dyn MMMInputFormat,
    ) -> Option<&'o dyn MMMInputFormat> {
        if self.dyn_eq(other) { Some(other) } else { None }
    }
    fn mem_size(&self, k: TDim, mn: TDim) -> TDim;
    fn extract_at_mn_f16(
        &self,
        data: &EagerPackedInput,
        mn: usize,
        slice: &mut [f16],
    ) -> TractResult<()>;
    fn extract_at_mn_f32(
        &self,
        data: &EagerPackedInput,
        mn: usize,
        slice: &mut [f32],
    ) -> TractResult<()>;
}

dyn_clone::clone_trait_object!(MMMInputFormat);
impl_downcast!(MMMInputFormat);
dyn_hash::hash_trait_object!(MMMInputFormat);
dyn_eq::eq_trait_object!(MMMInputFormat);

pub trait MMMInputValue:
    DynClone + Debug + DynHash + dyn_eq::DynEq + Send + Sync + Display + Downcast
{
    fn format(&self) -> &dyn MMMInputFormat;
    fn scratch_panel_buffer_layout(&self) -> Option<Layout>;
    fn panel_bytes(&self, i: usize, buffer: Option<*mut u8>) -> TractResult<*const u8>;
    fn panels_count(&self) -> usize {
        self.mn().divceil(self.format().r())
    }
    fn mn(&self) -> usize;
    fn k(&self) -> usize;
    fn exotic_fact(&self) -> &dyn ExoticFact;

    fn extract_at_mn_f16(&self, mn: usize, slice: &mut [f16]) -> TractResult<()>;
    fn extract_at_mn_f32(&self, mn: usize, slice: &mut [f32]) -> TractResult<()>;
}
dyn_clone::clone_trait_object!(MMMInputValue);
impl_downcast!(MMMInputValue);
dyn_hash::hash_trait_object!(MMMInputValue);
dyn_eq::eq_trait_object!(MMMInputValue);

#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Clone, Hash, Debug)]
pub struct PackedExoticFact {
    pub format: Box<dyn MMMInputFormat>,
    pub mn: TDim,
    pub k: usize,
}

impl Display for PackedExoticFact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Eager {} tensor (mn={} k={})", self.format, self.mn, self.k)
    }
}

impl ExoticFact for PackedExoticFact {
    fn buffer_sizes(&self) -> TVec<TDim> {
        tvec!(self.format.mem_size(self.k.to_dim(), self.mn.clone()))
    }
}

impl PartialEq for PackedExoticFact {
    fn eq(&self, other: &Self) -> bool {
        self.format == other.format && self.mn == other.mn && self.k == other.k
    }
}
impl Eq for PackedExoticFact {}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct EagerPackedInput {
    pub fact: PackedExoticFact,
    pub packed: Arc<Blob>,
    pub panel_bytes: usize,
    pub mn: usize,
}

impl MMMInputValue for EagerPackedInput {
    fn scratch_panel_buffer_layout(&self) -> Option<Layout> {
        None
    }
    fn panel_bytes(&self, i: usize, _buffer: Option<*mut u8>) -> TractResult<*const u8> {
        unsafe { Ok(self.packed.as_ptr().add(i * self.panel_bytes)) }
    }
    fn k(&self) -> usize {
        self.fact.k
    }
    fn mn(&self) -> usize {
        self.mn
    }
    fn format(&self) -> &dyn MMMInputFormat {
        &*self.fact.format
    }
    fn exotic_fact(&self) -> &dyn ExoticFact {
        &self.fact
    }
    fn extract_at_mn_f16(&self, mn: usize, slice: &mut [f16]) -> TractResult<()> {
        ensure!(slice.len() == self.k());
        ensure!(mn < self.mn());
        self.fact.format.extract_at_mn_f16(self, mn, slice)
    }
    fn extract_at_mn_f32(&self, mn: usize, slice: &mut [f32]) -> TractResult<()> {
        ensure!(slice.len() == self.k());
        ensure!(mn < self.mn());
        self.fact.format.extract_at_mn_f32(self, mn, slice)
    }
}

impl Display for EagerPackedInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (&self.fact as &dyn Display).fmt(f)
    }
}

impl Debug for EagerPackedInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as Display>::fmt(self, f)
    }
}

#[cfg(test)]
pub mod view_check {
    use super::*;

    /// Packing one batch of a batched tensor through a strided view must give the same bytes as
    /// packing that batch alone: what a dynamic packing op relies on, and what an unimplemented
    /// [`MMMInputFormat::prepare_one_view`] used to break.
    pub fn view_matches_tensor(
        format: &dyn MMMInputFormat,
        k: usize,
        mn: usize,
    ) -> TractResult<()> {
        let WeightType::Plain(dt) = format.precursor() else {
            bail!("{format} does not pack a plain tensor")
        };
        let mut parent = Tensor::zero_dt(dt, &[2, k, mn])?;
        parent.as_bytes_mut().iter_mut().enumerate().for_each(|(ix, b)| *b = (ix % 101) as u8 + 1);
        let batch = parent.slice(0, 1, 2)?.into_shape(&[k, mn])?;
        let from_tensor = format.prepare_one(&batch, 0, 1)?;
        let offset = parent.strides()[0] * dt.size_of() as isize;
        let view =
            unsafe { TensorView::from_bytes(&parent, offset, parent.shape(), parent.strides()) };
        let from_view = format.prepare_one_view(&view, 1, 2)?;
        let bytes = |v: &dyn MMMInputValue| -> TractResult<Arc<Blob>> {
            Ok(v.downcast_ref::<EagerPackedInput>()
                .with_context(|| format!("{format} did not pack to an EagerPackedInput"))?
                .packed
                .clone())
        };
        ensure!(
            *bytes(&*from_tensor)? == *bytes(&*from_view)?,
            "{format}: packing k={k} mn={mn} from a view differs from packing it from a tensor"
        );
        Ok(())
    }
}
