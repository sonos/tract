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
