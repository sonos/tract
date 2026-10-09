use crate::mmm::input_store::MMMInputFormat;
use crate::mmm::kernel::MatMatMulKer;
use crate::mmm::{AsInputValue, FusedKerSpec, FusedSpec, MatMatMul, OutputStoreKer};
use crate::strided_panel::{StridedKMajor, StridedPanelInput};
use std::sync::Arc;
use tract_data::internal::*;

/// Test battery for kernels registering a [`StridedKMajor`] input packing:
/// `mmm_strided_tests!(&*kernel_static, mod_name: packing_index)` inside the
/// kernel's `mod tests`. Verifies a `[k, m]`-contiguous f32 operand reaches
/// the kernel as a view with the expected panel descriptor, and that the
/// kernel predicates its loads on `valid` rather than reading the padded tail.
#[macro_export]
macro_rules! mmm_strided_tests {
    ($ker:expr, $packing_id:ident : $packing:expr) => {
        mod $packing_id {
            use super::*;

            #[test]
            fn k_major() -> TractResult<()> {
                $crate::frame::mmm::tests::strided::k_major($ker, $packing)
            }

            #[test]
            fn predicated_tail() -> TractResult<()> {
                $crate::frame::mmm::tests::strided::poison_tail($ker, $packing)
            }
        }
    };
}

fn strided_format(ker: &impl MatMatMulKer, packing: usize) -> StridedKMajor {
    *ker.packings()[packing]
        .0
        .downcast_ref::<StridedKMajor>()
        .expect("packing is not StridedKMajor")
}

/// `m` spans the cases a descriptor-driven kernel must get right: a full
/// panel pair, a predicated last panel, and rows shorter than `r`.
pub fn k_major(ker: &impl MatMatMulKer<Acc = f32>, packing: usize) -> TractResult<()> {
    if !ker.runnable() {
        return Ok(());
    }
    let format = strided_format(ker, packing);
    let r = format.r();
    let shapes = [
        (3usize, r + 8, 32usize),
        (5, 2 * r, 32),
        (1, r / 2, 32),
        (7, r + 1, 48),
        (4, r / 2 + 4, 32),
    ];
    for (k, m, n) in shapes {
        let a_data: Vec<f32> = (0..k * m).map(|i| (i as f32) * 0.01 - 0.4).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02 + 0.1).collect();
        let a = Arc::new(Tensor::from_shape(&[k, m], &a_data)?);
        let start = a.as_bytes().as_ptr() as usize;
        let end = start + a.as_bytes().len();
        let pa = format.panel_from_arc(a, 0, 0, 1)?;
        let view = pa.downcast_ref::<StridedPanelInput>().unwrap();
        assert!(view.is_view(), "k={k} m={m} packed instead of viewing");
        let panel0 = pa.panel_bytes(0, None)? as *const usize;
        let (ptr, stride, valid) = unsafe { (*panel0, *panel0.add(1), *panel0.add(2)) };
        assert!(ptr >= start && ptr < end, "panel pointer is outside the tensor");
        assert_eq!(stride, m * std::mem::size_of::<f32>(), "k byte stride");
        assert_eq!(valid, m.min(r));
        if m > r {
            let panel1 = pa.panel_bytes(1, None)? as *const usize;
            assert_eq!(unsafe { *panel1.add(2) }, m - r);
        }
        let b = Tensor::from_shape(&[k, n], &b_data)?;
        let pb = ker.packings()[packing].1.prepare_one(&b, 0, 1)?;
        let mut found = Tensor::zero::<f32>(&[m, n])?;
        unsafe {
            let c = ker.c_view(Some(0), Some(1)).wrap(&found.view_mut());
            ker.run(
                m,
                n,
                &[
                    FusedSpec::AddMatMul {
                        a: AsInputValue::Borrowed(&*pa),
                        b: AsInputValue::Borrowed(&*pb),
                        packing,
                    },
                    FusedSpec::Store(c),
                ],
            )?;
        }
        let mut expected = vec![0f32; m * n];
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = 0f32;
                for ki in 0..k {
                    acc += a_data[ki * m + mi] * b_data[ki * n + ni];
                }
                expected[mi * n + ni] = acc;
            }
        }
        let expected = Tensor::from_shape(&[m, n], &expected)?;
        found
            .close_enough(&expected, Approximation::Approximate)
            .unwrap_or_else(|e| panic!("strided A {k}x{m}x{n}: {e}"));
    }
    Ok(())
}

/// `valid < r` must not consume the poison sitting one vector past the row.
pub fn poison_tail(ker: &impl MatMatMulKer<Acc = f32>, packing: usize) -> TractResult<()> {
    if !ker.runnable() {
        return Ok(());
    }
    let r = strided_format(ker, packing).r();
    let nr = ker.nr();
    for valid in [16usize, 20, 8] {
        let mut row = vec![0f32; r];
        for (i, v) in row.iter_mut().enumerate().take(valid) {
            *v = (i as f32) * 0.05 - 0.2;
        }
        for (i, v) in row.iter_mut().enumerate().skip(valid) {
            *v = 1000.0 + i as f32;
        }
        let a = Tensor::from_shape(&[r], &row)?;
        let desc = [a.as_bytes().as_ptr() as usize, r * std::mem::size_of::<f32>(), valid];
        let b_data: Vec<f32> = (0..nr).map(|i| (i as f32) * 0.03 - 0.5).collect();
        let b = Tensor::from_shape(&[1, nr], &b_data)?;
        let pb = ker.packings()[packing].1.prepare_one(&b, 0, 1)?;
        let mut found = vec![f32::NAN; r * nr];
        let c = OutputStoreKer {
            ptr: found.as_mut_ptr() as _,
            row_byte_stride: (std::mem::size_of::<f32>() * nr) as isize,
            col_byte_stride: std::mem::size_of::<f32>() as isize,
            item_size: std::mem::size_of::<f32>(),
        };
        let ops = [
            FusedKerSpec::<f32>::Clear,
            FusedKerSpec::AddMatMul {
                k: 1,
                pa: desc.as_ptr() as *const u8,
                pb: pb.panel_bytes(0, None)?,
                packing,
            },
            FusedKerSpec::Store(c),
            FusedKerSpec::Done,
        ];
        let err = ker.kernel(&ops);
        assert_eq!(err, 0);
        for m in 0..r {
            for n in 0..nr {
                let got = found[m * nr + n];
                let want = if m < valid { row[m] * b_data[n] } else { 0.0 };
                assert!(
                    (got - want).abs() < 1e-4,
                    "valid {valid} C[{m},{n}] got {got} want {want}"
                );
            }
        }
    }
    Ok(())
}
