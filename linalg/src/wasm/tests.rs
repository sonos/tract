use proptest::prelude::*;

use crate::{
    frame::mmm::kernel::MatMatMulKer,
    generic::GenericMmm4x4,
    mmm::{FusedKerSpec, OutputStoreKer, RoundingPolicy},
    wasm::WasmMmm4x4,
};

fn f32_4() -> impl Strategy<Value = [f32; 4]> {
    [any::<f32>(); 4]
}

fn f32_4x4() -> impl Strategy<Value = [[f32; 4]; 4]> {
    [[any::<f32>(); 4]; 4]
}

fn remove_nan(mut a: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    for i in 0..4 {
        for j in 0..4 {
            if a[i][j].is_nan() {
                a[i][j] = 12345678.9;
            }
        }
    }
    a
}

fn run(
    mut initial: [[f32; 4]; 4],
    elem: FusedKerSpec<f32>,
) -> ((isize, [[f32; 4]; 4]), (isize, [[f32; 4]; 4])) {
    let mut output = [[0.0; 4]; 4];

    let spec = [
        FusedKerSpec::Clear,
        FusedKerSpec::AddUnicast(OutputStoreKer {
            ptr: initial.as_mut_ptr() as *mut u8,
            row_byte_stride: 16,
            col_byte_stride: 4,
            item_size: 4,
        }),
        elem,
        FusedKerSpec::Store(OutputStoreKer {
            ptr: output.as_mut_ptr() as *mut u8,
            row_byte_stride: 16,
            col_byte_stride: 4,
            item_size: 4,
        }),
        FusedKerSpec::Done,
    ];

    let r1 = WasmMmm4x4::kernel(&spec);
    let actual = remove_nan(output.clone());

    let r2 = GenericMmm4x4::<f32, f32, f32>::kernel(&spec);
    let expected = remove_nan(output.clone());

    ((r1, actual), (r2, expected))
}

proptest! {
    #[test]
    fn test_clear(initial in f32_4x4()) {
        let (actual, expected) = run(
            initial,
            FusedKerSpec::Clear
        );
        prop_assert_eq!(actual, expected);
    }
}

proptest! {
    #[test]
    fn test_scalar_min(initial in f32_4x4(), m in any::<f32>()) {
        let (actual, expected) = run(
            initial,
            FusedKerSpec::ScalarMin(m)
        );
        prop_assert_eq!(actual, expected);
    }
}

proptest! {
    #[test]
    fn test_scalar_max(initial in f32_4x4(), m in any::<f32>()) {
        let (actual, expected) = run(
            initial,
            FusedKerSpec::ScalarMax(m)
        );
        prop_assert_eq!(actual, expected);
    }
}

proptest! {
    #[test]
    fn test_scalar_add(initial in f32_4x4(), m in any::<f32>()) {
        let (actual, expected) = run(
            initial,
            FusedKerSpec::ScalarAdd(m)
        );
        prop_assert_eq!(actual, expected);
    }
}

proptest! {
    #[test]
    fn test_scalar_mul(initial in f32_4x4(), m in any::<f32>()) {
        let (actual, expected) = run(
            initial,
            FusedKerSpec::ScalarMul(m)
        );
        prop_assert_eq!(actual, expected);
    }
}

proptest! {
    #[test]
    fn test_scalar_sub(initial in f32_4x4(), m in any::<f32>()) {
        let (actual, expected) = run(
            initial,
            FusedKerSpec::ScalarSub(m)
        );
        prop_assert_eq!(actual, expected);
    }
}

proptest! {
    #[test]
    fn test_scalar_subf(initial in f32_4x4(), m in any::<f32>()) {
        let (actual, expected) = run(
            initial,
            FusedKerSpec::ScalarSubF(m)
        );
        prop_assert_eq!(actual, expected);
    }
}

proptest! {
    #[test]
    fn test_leaky_relu(initial in f32_4x4(), m in any::<f32>()) {
        let (actual, expected) = run(
            initial,
            FusedKerSpec::LeakyRelu(m)
        );
        prop_assert_eq!(actual, expected);
    }
}

proptest! {
    #[test]
    fn test_per_row_min(initial in f32_4x4(), row in f32_4()) {
        let (actual, expected) = run(
            initial,
            FusedKerSpec::PerRowMin(row.as_ptr())
        );
        prop_assert_eq!(actual, expected);
    }
}

proptest! {
    #[test]
    fn test_per_row_max(initial in f32_4x4(), row in f32_4()) {
        let (actual, expected) = run(
            initial,
            FusedKerSpec::PerRowMax(row.as_ptr())
        );
        prop_assert_eq!(actual, expected);
    }
}

proptest! {
    #[test]
    fn test_per_row_add(initial in f32_4x4(), row in f32_4()) {
        let (actual, expected) = run(
            initial,
            FusedKerSpec::PerRowAdd(row.as_ptr())
        );
        prop_assert_eq!(actual, expected);
    }
}

proptest! {
    #[test]
    fn test_per_row_mul(initial in f32_4x4(), row in f32_4()) {
        let (actual, expected) = run(
            initial,
            FusedKerSpec::PerRowMul(row.as_ptr())
        );
        prop_assert_eq!(actual, expected);
    }
}

proptest! {
    #[test]
    fn test_per_row_sub(initial in f32_4x4(), row in f32_4()) {
        let (actual, expected) = run(
            initial,
            FusedKerSpec::PerRowSub(row.as_ptr())
        );
        prop_assert_eq!(actual, expected);
    }
}

proptest! {
    #[test]
    fn test_per_row_subf(initial in f32_4x4(), row in f32_4()) {
        let (actual, expected) = run(
            initial,
            FusedKerSpec::PerRowSubF(row.as_ptr())
        );
        prop_assert_eq!(actual, expected);
    }
}

proptest! {
    #[test]
    fn test_per_col_min(initial in f32_4x4(), col in f32_4()) {
        let (actual, expected) = run(
            initial,
            FusedKerSpec::PerColMin(col.as_ptr())
        );
        prop_assert_eq!(actual, expected);
    }
}

proptest! {
    #[test]
    fn test_per_col_max(initial in f32_4x4(), col in f32_4()) {
        let (actual, expected) = run(
            initial,
            FusedKerSpec::PerColMax(col.as_ptr())
        );
        prop_assert_eq!(actual, expected);
    }
}

proptest! {
    #[test]
    fn test_per_col_add(initial in f32_4x4(), col in f32_4()) {
        let (actual, expected) = run(
            initial,
            FusedKerSpec::PerColAdd(col.as_ptr())
        );
        prop_assert_eq!(actual, expected);
    }
}

proptest! {
    #[test]
    fn test_per_col_mul(initial in f32_4x4(), col in f32_4()) {
        let (actual, expected) = run(
            initial,
            FusedKerSpec::PerColMul(col.as_ptr())
        );
        prop_assert_eq!(actual, expected);
    }
}

proptest! {
    #[test]
    fn test_per_col_sub(initial in f32_4x4(), col in f32_4()) {
        let (actual, expected) = run(
            initial,
            FusedKerSpec::PerColSub(col.as_ptr())
        );
        prop_assert_eq!(actual, expected);
    }
}

proptest! {
    #[test]
    fn test_per_col_subf(initial in f32_4x4(), col in f32_4()) {
        let (actual, expected) = run(
            initial,
            FusedKerSpec::PerColSubF(col.as_ptr())
        );
        prop_assert_eq!(actual, expected);
    }
}

proptest! {
    #[test]
    fn test_qscale(initial in f32_4x4(), shift in -100..100, mult in -1000..1000) {
        let (actual, expected) = run(
            initial,
            FusedKerSpec::QScale(shift as isize, RoundingPolicy::Native, mult)
        );
        prop_assert_eq!(actual, expected);
    }
}

proptest! {
    #[test]
    fn test_rounding_shift_right(initial in f32_4x4(), shift in 0..100) {
        let (actual, expected) = run(
            initial,
            FusedKerSpec::RoundingShiftRight(shift as usize, RoundingPolicy::Native)
        );
        prop_assert_eq!(actual, expected);
    }
}

proptest! {
    #[test]
    fn test_shift_left(initial in f32_4x4(), shift in 0..100) {
        let (actual, expected) = run(
            initial,
            FusedKerSpec::ShiftLeft(shift as usize)
        );
        prop_assert_eq!(actual, expected);
    }
}

proptest! {
    #[test]
    fn test_add_row_col_products(initial in f32_4x4(), rows in f32_4(), cols in f32_4()) {
        let (actual, expected) = run(
            initial,
            FusedKerSpec::AddRowColProducts(rows.as_ptr(), cols.as_ptr())
        );
        prop_assert_eq!(actual, expected);
    }
}

proptest! {
    #[test]
    fn test_add_mat_mul(initial in f32_4x4(), pa in f32_4x4(), pb in f32_4x4()) {
        let (actual, expected) = run(
            initial,
            FusedKerSpec::AddMatMul {
                k: 4,
                pa: pa.as_ptr() as *const u8,
                pb: pb.as_ptr() as *const u8,
                cpu_variant: 0,
            },
        );
        prop_assert_eq!(actual, expected);
    }
}
