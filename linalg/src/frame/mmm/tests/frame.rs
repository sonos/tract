use crate::frame::mmm::*;
use crate::{BinOp, LADatum};
use num_traits::AsPrimitive;
use std::ops::Neg;
use tests::display_error;
use tract_data::internal::*;

#[macro_export]
macro_rules! mmm_frame_tests {
    ($ker:expr, $ta:ty, $tb:ty, $tc:ty, $ti:ty) => {
        mod frame {
            use tract_data::internal::*;
            #[allow(unused_imports)]
            use $crate::frame::mmm::tests::frame::*;

            #[test]
            fn row_mul_2_1_3() -> TractResult<()> {
                unsafe { row_mul::<_, $ta, $tb, $tc, $ti>($ker, 2, 3)? }
                Ok(())
            }

            #[test]
            fn row_add_2_1_3() -> TractResult<()> {
                unsafe { row_add::<_, $ta, $tb, $tc, $ti>($ker, 2, 3)? }
                Ok(())
            }

            #[test]
            fn col_mul_2_1_3() -> TractResult<()> {
                unsafe { col_mul::<_, $ta, $tb, $tc, $ti>($ker, 2, 3)? }
                Ok(())
            }

            #[test]
            fn col_add_2_1_3() -> TractResult<()> {
                unsafe { col_add::<_, $ta, $tb, $tc, $ti>($ker, 2, 3)? }
                Ok(())
            }

            #[test]
            fn max_2_1_3() -> TractResult<()> {
                unsafe { max::<_, $ta, $tb, $tc, $ti>($ker, 2, 3)? }
                Ok(())
            }

            #[test]
            fn min_2_1_3() -> TractResult<()> {
                unsafe { min::<_, $ta, $tb, $tc, $ti>($ker, 2, 3)? }
                Ok(())
            }

            #[test]
            fn add_d_2_1_3() -> TractResult<()> {
                unsafe { add_d::<_, $ta, $tb, $tc, $ti>($ker, 2, 3)? }
                Ok(())
            }

            #[test]
            fn add_d_big() -> TractResult<()> {
                unsafe { add_d::<_, $ta, $tb, $tc, $ti>($ker, 197, 1)? }
                Ok(())
            }
        }
    };
}

pub unsafe fn fused_ops<
    K: MatMatMulKer<Acc = TI> + 'static,
    TA,
    TB,
    TC,
    TI,
    F: Fn(usize, usize) -> TC,
>(
    ker: &K,
    m: usize,
    n: usize,
    spec: &[FusedSpec],
    expect: F,
) -> TractResult<()>
where
    TA: LADatum + AsPrimitive<TI> + 'static,
    TB: LADatum + AsPrimitive<TI> + 'static,
    TC: LADatum + AsPrimitive<TI> + 'static,
    TI: LADatum + AsPrimitive<TC> + 'static,
    i32: AsPrimitive<TI>,
    usize: AsPrimitive<TI>,
{
    if !ker.is_supported_here() {
        return Ok(());
    };
    crate::setup_test_logger();

    let mut found = Tensor::zero::<TC>(&[m, n])?;
    let c_store = ker
        .c_from_data_and_strides(TC::datum_type().size_of(), n as isize, 1)
        .wrap(&found.view_mut());
    let mut spec: TVec<FusedSpec> = spec.into();
    spec.push(FusedSpec::Store(c_store));

    ker.run(m, n, &spec)?;
    let expected =
        tract_ndarray::prelude::Array2::from_shape_fn((m, n), |(r, c)| expect(r, c)).into_tensor();
    let err = found.close_enough(&expected, true);
    if err.is_err() {
        display_error(found.as_slice::<TC>()?, expected.as_slice::<TC>()?, m, n);
    }
    err
}

pub unsafe fn row_add<K: MatMatMulKer<Acc = TI> + 'static, TA, TB, TC, TI>(
    ker: &K,
    m: usize,
    n: usize,
) -> TractResult<()>
where
    TA: LADatum + AsPrimitive<TI> + 'static,
    TB: LADatum + AsPrimitive<TI> + 'static,
    TC: LADatum + AsPrimitive<TI> + 'static,
    TI: LADatum + AsPrimitive<TC> + 'static + Neg<Output = TI>,
    i32: AsPrimitive<TI>,
    usize: AsPrimitive<TI>,
{
    let bias = (0..m).map(|i| i.as_()).collect::<Vec<TI>>();
    fused_ops::<K, TA, TB, TC, TI, _>(
        ker,
        m,
        n,
        &[FusedSpec::BinPerRow(tensor1(&bias).view(), BinOp::Add)],
        |r, _| bias[r].as_(),
    )
}

pub unsafe fn row_mul<K: MatMatMulKer<Acc = TI> + 'static, TA, TB, TC, TI>(
    ker: &K,
    m: usize,
    n: usize,
) -> TractResult<()>
where
    TA: LADatum + AsPrimitive<TI> + 'static,
    TB: LADatum + AsPrimitive<TI> + 'static,
    TC: LADatum + AsPrimitive<TI> + 'static,
    TI: LADatum + AsPrimitive<TC> + 'static + Neg<Output = TI>,
    i32: AsPrimitive<TI>,
    usize: AsPrimitive<TI>,
{
    let bias = (0..m).map(|i| i.as_()).collect::<Vec<TI>>();
    fused_ops::<K, TA, TB, TC, TI, _>(
        ker,
        m,
        n,
        &[
            FusedSpec::BinScalar(&tensor0(1i32.as_()), BinOp::Add),
            FusedSpec::BinPerRow(tensor1(&bias).view(), BinOp::Mul),
        ],
        |r, _| bias[r].as_(),
    )
}

pub unsafe fn col_add<K: MatMatMulKer<Acc = TI> + 'static, TA, TB, TC, TI>(
    ker: &K,
    m: usize,
    n: usize,
) -> TractResult<()>
where
    TA: LADatum + AsPrimitive<TI> + 'static,
    TB: LADatum + AsPrimitive<TI> + 'static,
    TC: LADatum + AsPrimitive<TI> + 'static,
    TI: LADatum + AsPrimitive<TC> + 'static + Neg<Output = TI>,
    i32: AsPrimitive<TI>,
    usize: AsPrimitive<TI>,
{
    let bias = (0..n).map(|i| i.as_()).collect::<Vec<TI>>();
    fused_ops::<K, TA, TB, TC, TI, _>(
        ker,
        m,
        n,
        &[FusedSpec::BinPerCol(tensor1(&bias).view(), BinOp::Add)],
        |_, c| bias[c].as_(),
    )
}

pub unsafe fn col_mul<K: MatMatMulKer<Acc = TI> + 'static, TA, TB, TC, TI>(
    ker: &K,
    m: usize,
    n: usize,
) -> TractResult<()>
where
    TA: LADatum + AsPrimitive<TI> + 'static,
    TB: LADatum + AsPrimitive<TI> + 'static,
    TC: LADatum + AsPrimitive<TI> + 'static,
    TI: LADatum + AsPrimitive<TC> + 'static + Neg<Output = TI>,
    i32: AsPrimitive<TI>,
    usize: AsPrimitive<TI>,
{
    let bias = (0..n).map(|i| i.as_()).collect::<Vec<TI>>();
    fused_ops::<K, TA, TB, TC, TI, _>(
        ker,
        m,
        n,
        &[
            FusedSpec::BinScalar(&tensor0(1i32.as_()), BinOp::Add),
            FusedSpec::BinPerCol(tensor1(&bias).view(), BinOp::Mul),
        ],
        |_, c| bias[c].as_(),
    )
}

pub unsafe fn add_d<K: MatMatMulKer<Acc = TI> + 'static, TA, TB, TC, TI>(
    ker: &K,
    m: usize,
    n: usize,
) -> TractResult<()>
where
    TA: LADatum + AsPrimitive<TI> + 'static,
    TB: LADatum + AsPrimitive<TI> + 'static,
    TC: LADatum + AsPrimitive<TI> + 'static,
    TI: LADatum + AsPrimitive<TC> + 'static + Neg<Output = TI>,
    i32: AsPrimitive<TI>,
    usize: AsPrimitive<TI>,
{
    let d = (0..m * n).map(|i| i.as_()).collect::<Vec<TI>>();
    let d = tensor1(&d).into_shape(&[m, n])?;
    let store_spec =
        OutputStoreSpec::View { m_axis: Some(0), n_axis: Some(1), mr: ker.mr(), nr: ker.nr() };
    let view_d = d.to_array_view::<TI>()?.into_dimensionality()?;
    fused_ops::<K, TA, TB, TC, TI, _>(
        ker,
        m,
        n,
        &[FusedSpec::AddUnicast(store_spec.wrap(&d.view()))],
        |r, c| view_d[(r, c)].as_(),
    )
}

pub unsafe fn max<K: MatMatMulKer<Acc = TI>, TA, TB, TC, TI>(
    ker: &K,
    m: usize,
    n: usize,
) -> TractResult<()>
where
    TA: LADatum + AsPrimitive<TI> + 'static,
    TB: LADatum + AsPrimitive<TI> + 'static,
    TC: LADatum + AsPrimitive<TI> + 'static,
    TI: LADatum + AsPrimitive<TC> + 'static + Neg<Output = TI>,
    i32: AsPrimitive<TI>,
    usize: AsPrimitive<TI>,
{
    let five: TI = 5.as_();
    fused_ops::<K, TA, TB, TC, TI, _>(
        ker,
        m,
        n,
        &[FusedSpec::BinScalar(&tensor0(five), BinOp::Max)],
        |_, _| five.as_(),
    )
}

pub unsafe fn min<K: MatMatMulKer<Acc = TI>, TA, TB, TC, TI>(
    ker: &K,
    m: usize,
    n: usize,
) -> TractResult<()>
where
    TA: LADatum + AsPrimitive<TI> + 'static,
    TB: LADatum + AsPrimitive<TI> + 'static,
    TC: LADatum + AsPrimitive<TI> + 'static,
    TI: LADatum + AsPrimitive<TC> + 'static + Neg<Output = TI>,
    i32: AsPrimitive<TI>,
    usize: AsPrimitive<TI>,
{
    let five: TI = 5.as_();
    fused_ops::<K, TA, TB, TC, TI, _>(
        ker,
        m,
        n,
        &[FusedSpec::BinScalar(&tensor0(five), BinOp::Min)],
        |_, _| TC::zero(),
    )
}
