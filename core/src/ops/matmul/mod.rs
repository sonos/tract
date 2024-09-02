pub mod de_block_quant;
pub mod optimized;
pub mod mir_quant;
pub mod pack;

use crate::internal::*;

pub fn output_type(input: DatumType) -> DatumType {
    if input.is_float() {
        input
    } else {
        i32::datum_type()
    }
}
