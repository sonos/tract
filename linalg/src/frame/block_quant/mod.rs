use downcast_rs::{impl_downcast, Downcast};
use dyn_clone::DynClone;
use dyn_hash::DynHash;
use tract_data::internal::*;
use tract_data::itertools::Itertools;

use std::alloc::Layout;
use std::fmt::{Debug, Display};
use std::hash::Hash;

mod helpers;
mod repack;
mod q4_0;

pub use helpers::{NibbleReader, NibbleWriter};
pub use q4_0::Q4_0;

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

    fn pack(&self, input: &[u8], k: usize, r: usize) -> TractResult<Blob>;
    fn panel_f32(&self, packed: &Blob, k: usize, r: usize, panel: usize, scratch: &mut [f32]);
    fn panel_f16(&self, packed: &Blob, k: usize, r: usize, panel: usize, scratch: &mut [f16]);
}

dyn_clone::clone_trait_object!(BlockQuant);
dyn_hash::hash_trait_object!(BlockQuant);
impl_downcast!(BlockQuant);

