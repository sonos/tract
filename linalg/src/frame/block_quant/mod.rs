use downcast_rs::{impl_downcast, Downcast};
use dyn_clone::{clone_box, DynClone};
use dyn_hash::DynHash;
use num_traits::Zero;
use tract_data::internal::*;
use tract_data::itertools::Itertools;

use std::alloc::Layout;
use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::sync::Arc;

mod helpers;
mod q4_0;
mod value;

pub use helpers::{NibbleReader, NibbleWriter};
pub use q4_0::Q4_0;
pub use value::{BlockQuantFact, BlockQuantValue, PackedBlockQuantFact};

use crate::mmm::{EagerPackedInput, MMMInputFormat};
use crate::pack::PackedFormat;

use crate::WeightType;

use super::mmm::MMMInputValue;

pub trait BlockQuant: Debug + Display + Send + Sync + DynClone + DynHash + Downcast {
    fn same_as(&self, other: &dyn BlockQuant) -> bool;

    fn block_len(&self) -> usize;

    fn block_bytes(&self) -> usize;

    fn dequant_block_f32(&self, quant: &[u8], block: &mut [f32]);
    fn dequant_block_f16(&self, quant: &[u8], block: &mut [f16]);
    fn quant_block_f16(&self, block: &[f16], quant: &mut [u8]);
    fn quant_block_f32(&self, block: &[f32], quant: &mut [u8]);

    fn quant_f16(&self, input: &[f16]) -> TractResult<Blob> {
        unsafe {
            let blocks = input.len() / self.block_len();
            let mut quant = Blob::for_layout(
                Layout::from_size_align(blocks * self.block_bytes(), 128).unwrap(),
            );
            for b in 0..blocks {
                let block = &input[b * self.block_len()..][..self.block_len()];
                let qblock = &mut quant[b * self.block_bytes()..][..self.block_bytes()];
                self.quant_block_f16(block, qblock);
            }
            Ok(quant)
        }
    }

    fn quant_f32(&self, input: &[f32]) -> TractResult<Blob> {
        unsafe {
            let blocks = input.len() / self.block_len();
            let mut quant = Blob::for_layout(
                Layout::from_size_align(blocks * self.block_bytes(), 128).unwrap(),
            );
            for b in 0..blocks {
                let block = &input[b * self.block_len()..][..self.block_len()];
                let qblock = &mut quant[b * self.block_bytes()..][..self.block_bytes()];
                self.quant_block_f32(block, qblock);
            }
            Ok(quant)
        }
    }

    fn dequant_f32(&self, input: &[u8]) -> TractResult<Tensor> {
        unsafe {
            let blocks = input.len() / self.block_bytes();
            let mut tensor = Tensor::uninitialized::<f32>(&[blocks * self.block_len()])?;
            let slice = tensor.as_slice_mut::<f32>()?;
            for b in 0..blocks {
                let block = &mut slice[b * self.block_len()..][..self.block_len()];
                let qblock = &input[b * self.block_bytes()..][..self.block_bytes()];
                self.dequant_block_f32(qblock, block);
            }
            Ok(tensor)
        }
    }

    fn dequant_f16(&self, input: &[u8]) -> TractResult<Tensor> {
        unsafe {
            let blocks = input.len() / self.block_bytes();
            let mut tensor = Tensor::uninitialized::<f16>(&[blocks * self.block_len()])?;
            let slice = tensor.as_slice_mut::<f16>()?;
            for b in 0..blocks {
                let block = &mut slice[b * self.block_len()..][..self.block_len()];
                let qblock = &input[b * self.block_bytes()..][..self.block_bytes()];
                self.dequant_block_f16(qblock, block);
            }
            Ok(tensor)
        }
    }

    fn extract_at_offset_f16(&self, input: &[u8], offset: usize) -> f16 {
        let len = self.block_len();
        let block_id = offset / len;
        let mut block = vec![f16::zero(); self.block_len()];
        self.dequant_block_f16(
            &input[block_id * self.block_bytes()..][..self.block_bytes()],
            &mut block,
        );
        block[offset % len]
    }

    fn extract_at_offset_f32(&self, input: &[u8], offset: usize) -> f32 {
        let len = self.block_len();
        let block_id = offset / len;
        let mut block = vec![f32::zero(); self.block_len()];
        self.dequant_block_f32(
            &input[block_id * self.block_bytes()..][..self.block_bytes()],
            &mut block,
        );
        block[offset % len]
    }

    fn simulate_precision_loss(
        &self,
        mut tensor: Tensor,
        block_axis: usize,
    ) -> TractResult<Tensor> {
        ensure!(block_axis == tensor.rank() - 1);
        ensure!(tensor.shape()[block_axis] % self.block_len() == 0);
        let mut scratch = vec![0u8; self.block_bytes()];
        if tensor.datum_type() == f32::datum_type() {
            for block in tensor.as_slice_mut::<f32>()?.chunks_mut(self.block_len()) {
                self.quant_block_f32(block, &mut scratch);
                self.dequant_block_f32(&scratch, block);
            }
            Ok(tensor)
        } else if tensor.datum_type() == f16::datum_type() {
            for block in tensor.as_slice_mut::<f16>()?.chunks_mut(self.block_len()) {
                self.quant_block_f16(block, &mut scratch);
                self.dequant_block_f16(&scratch, block);
            }
            Ok(tensor)
        } else {
            todo!()
        }
    }

    fn pack(
        &self,
        input: &[u8],
        k: usize,
        r: usize,
        zip: usize,
        scales_at_end: bool,
    ) -> TractResult<EagerPackedInput>;

    unsafe fn extract_packed_panel(
        &self,
        value: &EagerPackedInput,
        target: &PackedFormat,
        panel: usize,
        scratch: *mut u8,
    ) -> TractResult<()>;

    fn extract_at_mn_f16(
        &self,
        value: &EagerPackedInput,
        mn: usize,
        target: &mut [f16],
    ) -> TractResult<()>;

    fn extract_at_mn_f32(
        &self,
        value: &EagerPackedInput,
        mn: usize,
        target: &mut [f32],
    ) -> TractResult<()>;
}

dyn_clone::clone_trait_object!(BlockQuant);
dyn_hash::hash_trait_object!(BlockQuant);
impl_downcast!(BlockQuant);

#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Clone, Hash)]
pub struct PackedBlockQuantFormat {
    pub bq: Box<dyn BlockQuant>,
    pub r: usize,
    pub zip: usize,
    pub scales_at_end: bool,
}

impl PartialEq for PackedBlockQuantFormat {
    fn eq(&self, other: &Self) -> bool {
        self.bq.same_as(&*other.bq)
            && self.r == other.r
            && self.zip == other.zip
            && self.scales_at_end == other.scales_at_end
    }
}

impl Eq for PackedBlockQuantFormat {}

impl Display for PackedBlockQuantFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Packed{}[{}]", &*self.bq, self.r)?;
        if self.zip != 0 {
            write!(f, "Z{}", self.zip)?;
        }
        if self.scales_at_end {
            write!(f, "Se")?;
        }
        Ok(())
    }
}

impl Debug for PackedBlockQuantFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as Display>::fmt(self, f)
    }
}

impl PackedBlockQuantFormat {
    pub fn new(bq: &dyn BlockQuant, r: usize, zip: usize, scales_at_end: bool) -> Self {
        PackedBlockQuantFormat { bq: clone_box(bq), r, zip, scales_at_end }
    }

    pub fn simulate_precision_loss(
        &self,
        tensor: Tensor,
        block_axis: usize,
    ) -> TractResult<Tensor> {
        self.bq.simulate_precision_loss(tensor, block_axis)
    }

    pub fn pack(&self, input: &[u8], k: usize) -> TractResult<EagerPackedInput> {
        self.bq.pack(input, k, self.r, self.zip, self.scales_at_end)
    }
}

impl MMMInputFormat for PackedBlockQuantFormat {
    fn prepare_tensor(&self, t: &Tensor, _k_axis: usize, _mn_axis: usize) -> TractResult<Tensor> {
        let packed = t
            .as_slice::<Opaque>()?
            .iter()
            .map(|o| {
                let bqv = o.downcast_ref::<BlockQuantValue>().unwrap();
                let packed = self.pack(&bqv.value, bqv.fact.k())?;
                Ok(Opaque(Arc::new(Box::new(packed) as Box<dyn MMMInputValue>)))
            })
            .collect::<TractResult<Vec<Opaque>>>()?;
        tensor1(&packed).into_shape(t.shape())
    }

    fn prepare_one(
        &self,
        t: &Tensor,
        k_axis: usize,
        mn_axis: usize,
    ) -> TractResult<Box<dyn MMMInputValue>> {
        // this code path is essentially there for test scenarios
        let t = if t.datum_type().is_number() {
            let k = t.shape()[k_axis];
            let m = t.shape()[mn_axis];
            assert!(k % self.bq.block_len() == 0);
            let t: Cow<Tensor> = if k_axis == 1 && mn_axis == 0 {
                Cow::Borrowed(t)
            } else {
                Cow::Owned(t.clone().move_axis(1, 0)?)
            };
            let quant = if t.datum_type() == f32::datum_type() {
                self.bq.quant_f32(t.as_slice()?)?
            } else if t.datum_type() == f16::datum_type() {
                self.bq.quant_f16(t.as_slice()?)?
            } else {
                todo!()
            };
            Cow::Owned(tensor0(Opaque(Arc::new(BlockQuantValue {
                value: Arc::new(quant),
                fact: BlockQuantFact::new(self.bq.clone(), tvec!(m, k)),
            }))))
        } else {
            Cow::Borrowed(t)
        };
        ensure!(mn_axis == 0);
        ensure!(k_axis == 1);
        let bqv = t.to_scalar::<Opaque>()?.downcast_ref::<BlockQuantValue>().unwrap();
        let packed = self.pack(&bqv.value, bqv.fact.k())?;
        Ok(Box::new(packed))
    }

    fn precursor(&self) -> WeightType {
        WeightType::BlockQuant(self.bq.clone())
    }

    fn k_alignment(&self) -> usize {
        self.bq.block_len()
    }

    fn r(&self) -> usize {
        self.r
    }

    fn mem_size(&self, k: TDim, mn: TDim) -> TDim {
        k * mn * self.bq.block_bytes() / self.bq.block_len()
    }

    fn same_as(&self, other: &dyn MMMInputFormat) -> bool {
        other.downcast_ref::<Self>().is_some_and(|other| self == other)
    }

    fn extract_at_mn_f16(
        &self,
        data: &EagerPackedInput,
        mn: usize,
        slice: &mut [f16],
    ) -> TractResult<()> {
        self.bq.extract_at_mn_f16(data, mn, slice)
    }

    fn extract_at_mn_f32(
        &self,
        data: &EagerPackedInput,
        mn: usize,
        slice: &mut [f32],
    ) -> TractResult<()> {
        self.bq.extract_at_mn_f32(data, mn, slice)
    }
}
