pub mod de_block_quant;
pub mod optimized;
pub mod pack;
pub mod quant;

use crate::internal::*;

pub fn output_type(input: DatumType) -> DatumType {
    if input.is_float() {
        input
    } else {
        i32::datum_type()
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum ModePicker {
    Single,
    VecVsMat,
}

impl ModePicker {
    #[inline]
    pub fn pick(&self, n: usize) -> TractResult<usize> {
        match self {
            ModePicker::Single => Ok(0),
            ModePicker::VecVsMat => Ok((n > 1) as usize),
        }
    }
}
