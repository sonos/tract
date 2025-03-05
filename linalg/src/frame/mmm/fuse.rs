use std::fmt::Debug;
use std::ops::Deref;

use crate::pack::PackedFormat;
use crate::BinOp;

use super::{MMMInputValue, OutputStore, OutputStoreKer};
use tract_data::internal::*;

#[repr(usize)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RoundingPolicy {
    Native,
    Zero,
    Away,
    MinusInf,
    PlusInf,
    Even,
    Odd,
}

#[derive(Clone, Debug)]
pub enum AsInputValue<'t> {
    Owned(Box<dyn MMMInputValue>),
    Borrowed(&'t dyn MMMInputValue),
}

impl Deref for AsInputValue<'_> {
    type Target = dyn MMMInputValue;
    fn deref(&self) -> &Self::Target {
        match self {
            AsInputValue::Owned(b) => &**b,
            AsInputValue::Borrowed(r) => *r,
        }
    }
}

#[derive(Clone, Debug)]
pub enum FusedSpec<'t> {
    BinScalar(&'t Tensor, BinOp),
    BinPerRow(TensorView<'t>, BinOp),
    BinPerCol(TensorView<'t>, BinOp),
    AddRowColProducts(&'t Tensor, &'t Tensor),
    AddUnicast(OutputStore),
    LeakyRelu(&'t Tensor),
    QScale(isize, RoundingPolicy, i32),
    RoundingShiftRight(usize, RoundingPolicy),
    ShiftLeft(usize),
    Store(OutputStore),
    AddMatMul { a: AsInputValue<'t>, b: AsInputValue<'t>, packing: usize },
}

impl FusedSpec<'_> {
    pub fn prefer_col_outer(&self) -> Option<bool> {
        if let FusedSpec::AddMatMul { a, b, .. } = self {
            let a_is_eager = a.format().is::<PackedFormat>();
            let b_is_eager = b.format().is::<PackedFormat>();
            if a_is_eager == b_is_eager {
                None
            } else {
                Some(a_is_eager)
            }
        } else {
            None
        }
    }
}

// Careful here, the jump_to comments are used by the build script.
#[repr(C, usize)]
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
#[rustfmt::skip]
pub enum FusedKerSpec<TI: Copy> {
    Done,                                       // jump_to:done
    Clear,                                      // jump_to:clear
                                                //
    LoadTile(*const TI, *const TI),             // jump_to:load_tile

    ScalarMin(TI),                              // jump_to:scalar_min
    ScalarMax(TI),                              // jump_to:scalar_max
    ScalarAdd(TI),                              // jump_to:scalar_add
    ScalarMul(TI),                              // jump_to:scalar_mul
    ScalarSub(TI),                              // jump_to:scalar_sub
    ScalarSubF(TI),                             // jump_to:scalar_sub_flipped

    LeakyRelu(TI),                              // jump_to:leaky_relu

    PerRowMin(*const TI),                       // jump_to:per_row_min
    PerRowMax(*const TI),                       // jump_to:per_row_max
    PerRowAdd(*const TI),                       // jump_to:per_row_add
    PerRowMul(*const TI),                       // jump_to:per_row_mul
    PerRowSub(*const TI),                       // jump_to:per_row_sub
    PerRowSubF(*const TI),                      // jump_to:per_row_sub_flipped

    PerColMin(*const TI),                       // jump_to:per_col_min
    PerColMax(*const TI),                       // jump_to:per_col_max
    PerColAdd(*const TI),                       // jump_to:per_col_add
    PerColMul(*const TI),                       // jump_to:per_col_mul
    PerColSub(*const TI),                       // jump_to:per_col_sub
    PerColSubF(*const TI),                      // jump_to:per_col_sub_flipped

    QScale(isize, RoundingPolicy, i32),         // jump_to:q_scale
    RoundingShiftRight(usize, RoundingPolicy),  // jump_to:q_shr
    ShiftLeft(usize),                           // jump_to:q_shl
    AddUnicast(OutputStoreKer),                 // jump_to:add_unicast
    AddRowColProducts(*const TI, *const TI),    // jump_to:add_row_col_products
    Store(OutputStoreKer),                      // jump_to:store

    // jump_to:add_mat_mul
    AddMatMul { k: usize, pa: *const u8, pb: *const u8, packing: usize },
}

unsafe impl<TI: Copy> Send for FusedKerSpec<TI> {}
unsafe impl<TI: Copy> Sync for FusedKerSpec<TI> {}

#[cfg(test)]
#[test]
fn check_non_linear_enum_size() {
    assert_eq!(std::mem::size_of::<RoundingPolicy>(), std::mem::size_of::<usize>());
    assert_eq!(
        std::mem::size_of::<FusedKerSpec<f32>>(),
        std::mem::size_of::<usize>() + std::mem::size_of::<OutputStoreKer>()
    );
    assert_eq!(std::mem::size_of::<FusedKerSpec<f32>>(), 5 * std::mem::size_of::<usize>());
}
