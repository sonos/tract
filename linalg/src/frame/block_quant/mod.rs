use downcast_rs::{impl_downcast, Downcast};
use dyn_clone::DynClone;
use dyn_hash::DynHash;
use tract_data::internal::*;
use tract_data::itertools::Itertools;

use std::alloc::Layout;
use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::Deref;

mod helpers;
mod q4_0;

pub use helpers::{NibbleReader, NibbleWriter};
pub use q4_0::Q4_0;

use crate::mmm::{EagerPackedInput, MMMInputFormat};

use super::PackedFormat;

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

    fn pack(
        &self,
        input: &[u8],
        k: usize,
        r: usize,
        zip: usize,
        scales_at_end: bool,
    ) -> TractResult<EagerPackedInput>;

    unsafe fn extract_panel(
        &self,
        value: &EagerPackedInput,
        target: &PackedFormat,
        panel: usize,
        scratch: *mut u8,
    ) -> TractResult<()>;
}

dyn_clone::clone_trait_object!(BlockQuant);
dyn_hash::hash_trait_object!(BlockQuant);
impl_downcast!(BlockQuant);

#[derive(Clone, Hash)]
pub enum StaticBlockQuant {
    Owned(Box<dyn BlockQuant>),
    Borrow(&'static dyn BlockQuant),
}

impl Deref for StaticBlockQuant {
    type Target = dyn BlockQuant;
    fn deref(&self) -> &dyn BlockQuant {
        match self {
            StaticBlockQuant::Owned(o) => &**o,
            StaticBlockQuant::Borrow(o) => *o,
        }
    }
}

#[derive(Clone, Hash)]
pub struct PackedBlockQuantFormat {
    pub bq: StaticBlockQuant,
    pub r: usize,
    pub zip: usize,
    pub scales_at_end: bool,
}

impl Display for PackedBlockQuantFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Packed{} (r={})", &*self.bq, self.r)
    }
}

impl Debug for PackedBlockQuantFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as Display>::fmt(self, f)
    }
}

impl PackedBlockQuantFormat {
    pub const fn new(
        bq: &'static dyn BlockQuant,
        r: usize,
        zip: usize,
        scales_at_end: bool,
    ) -> Self {
        PackedBlockQuantFormat { bq: StaticBlockQuant::Borrow(bq), r, zip, scales_at_end }
    }

    #[cfg(test)]
    pub fn simulate_precision_loss(
        &self,
        mut tensor: Tensor,
        block_axis: usize,
    ) -> TractResult<Tensor> {
        ensure!(block_axis == tensor.rank() - 1);
        ensure!(tensor.shape()[block_axis] % self.bq.block_len() == 0);
        let mut scratch = vec![0u8; self.bq.block_bytes()];
        if tensor.datum_type() == f32::datum_type() {
            for block in tensor.as_slice_mut::<f32>()?.chunks_mut(self.bq.block_len()) {
                self.bq.quant_block_f32(block, &mut scratch);
                self.bq.dequant_block_f32(&scratch, block);
            }
            Ok(tensor)
        } else if tensor.datum_type() == f16::datum_type() {
            for block in tensor.as_slice_mut::<f16>()?.chunks_mut(self.bq.block_len()) {
                self.bq.quant_block_f16(block, &mut scratch);
                self.bq.dequant_block_f16(&scratch, block);
            }
            Ok(tensor)
        } else {
            todo!()
        }
    }

    pub fn pack(&self, input: &[u8], k: usize) -> TractResult<EagerPackedInput> {
        self.bq.pack(input, k, self.r, self.zip, self.scales_at_end)
    }
}

impl MMMInputFormat for PackedBlockQuantFormat {
    fn can_prepare_types(&self) -> Vec<DatumType> {
        vec![]
    }

    fn prepare_tensor(
        &self,
        t: &Tensor,
        k_axis: usize,
        mn_axis: usize,
    ) -> TractResult<Box<dyn crate::mmm::MMMInputValue>> {
        let k = t.shape()[k_axis];
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
        Ok(Box::new(self.pack(&quant, k)?))
    }

    fn k_alignment(&self) -> usize {
        self.bq.block_len()
    }

    fn r(&self) -> usize {
        self.r
    }
}
