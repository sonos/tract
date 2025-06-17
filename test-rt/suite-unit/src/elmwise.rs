use std::ops::Div;

use infra::{Test, TestResult, TestSuite};
use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ndarray::ArrayD;
use tract_core::num_traits::{AsPrimitive, FromPrimitive, Num, ToPrimitive};
use tract_core::ops::element_wise::ElementWiseOp;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub enum ElWiseOps {
    Neg,
    Abs,
    Sqr,
    Sqrt,
    Rsqrt,
    Recip,
    Ceil,
    Floor,
    Round,
    RoundHalfToEven,
    Exp,
    Sigmoid,
    Sin,
    Sinh,
    Asin,
    Asinh,
    Cos,
    Cosh,
    Acos,
    Acosh,
    Tan,
    Tanh,
    Atan,
    Atanh,
    // Erf, Erf is unstable in rust
    Ln,
}

pub const ALL_OPS: [ElWiseOps; 25] = [
    ElWiseOps::Neg,
    ElWiseOps::Abs,
    ElWiseOps::Sqr,
    ElWiseOps::Sqrt,
    ElWiseOps::Rsqrt,
    ElWiseOps::Recip,
    ElWiseOps::Ceil,
    ElWiseOps::Floor,
    ElWiseOps::Round,
    ElWiseOps::RoundHalfToEven,
    ElWiseOps::Exp,
    ElWiseOps::Sigmoid,
    ElWiseOps::Sin,
    ElWiseOps::Sinh,
    ElWiseOps::Asin,
    ElWiseOps::Asinh,
    ElWiseOps::Cos,
    ElWiseOps::Cosh,
    ElWiseOps::Acos,
    ElWiseOps::Acosh,
    ElWiseOps::Tan,
    ElWiseOps::Tanh,
    ElWiseOps::Atan,
    ElWiseOps::Atanh,
    //ElWiseOps::Erf,
    ElWiseOps::Ln,
];

pub trait SupportedElement:
    Datum
    + Num
    + Copy
    + FromPrimitive
    + ToPrimitive
    + 'static
    + Div<Output = Self>
    + PartialOrd
    + AsPrimitive<usize>
    + AsPrimitive<f32>
    + AsPrimitive<f64>
{
}

impl<T> SupportedElement for T where
    T: Datum
        + Num
        + Copy
        + FromPrimitive
        + ToPrimitive
        + 'static
        + Div<Output = Self>
        + PartialOrd
        + AsPrimitive<usize>
        + AsPrimitive<f32>
        + AsPrimitive<f64>
{
}

#[derive(Debug, Clone)]
pub struct ElWiseOpProblem<T>
where
    T: SupportedElement,
{
    pub op: ElWiseOps,
    pub input: ArrayD<T>,
}

impl<T> Arbitrary for ElWiseOpProblem<T>
where
    T: SupportedElement,
{
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        let shape_strategy = prop::collection::vec(1usize..=5, 0..=4);

        shape_strategy
            .prop_flat_map(|shape| {
                let len = shape.iter().product::<usize>();
                let input = vec(
                    (2u8..=10u8).prop_map(|i| T::from_u8(i).unwrap() / T::from_u8(2).unwrap()),
                    len..=len,
                )
                .prop_map(move |vec| ArrayD::from_shape_vec(shape.to_vec(), vec).unwrap())
                .boxed();

                let mut ops = ALL_OPS.to_vec();

                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u8>()
                    || std::any::TypeId::of::<T>() == std::any::TypeId::of::<u16>()
                    || std::any::TypeId::of::<T>() == std::any::TypeId::of::<u32>()
                    || std::any::TypeId::of::<T>() == std::any::TypeId::of::<u64>()
                    || std::any::TypeId::of::<T>() == std::any::TypeId::of::<i8>()
                    || std::any::TypeId::of::<T>() == std::any::TypeId::of::<i16>()
                    || std::any::TypeId::of::<T>() == std::any::TypeId::of::<i32>()
                    || std::any::TypeId::of::<T>() == std::any::TypeId::of::<i64>()
                {
                    ops.retain(|op| {
                        matches!(op, ElWiseOps::Ceil | ElWiseOps::Floor | ElWiseOps::Round)
                    });
                }

                let op_strategy = prop::sample::select(ops);
                (input, op_strategy)
            })
            .prop_map(|(input, op)| ElWiseOpProblem { input, op })
            .boxed()
    }
}

pub fn to_tract_op(op: &ElWiseOps) -> Box<dyn TypedOp> {
    let el_mini_op: Box<dyn ElementWiseMiniOp> = match op {
        ElWiseOps::Neg => Box::new(tract_core::ops::math::Neg {}),
        ElWiseOps::Abs => Box::new(tract_core::ops::math::Abs {}),
        ElWiseOps::Sqr => Box::new(tract_core::ops::math::Square {}),
        ElWiseOps::Sqrt => Box::new(tract_core::ops::math::Sqrt {}),
        ElWiseOps::Rsqrt => Box::new(tract_core::ops::math::Rsqrt {}),
        ElWiseOps::Recip => Box::new(tract_core::ops::math::Recip {}),
        ElWiseOps::Ceil => Box::new(tract_core::ops::math::Ceil {}),
        ElWiseOps::Floor => Box::new(tract_core::ops::math::Floor {}),
        ElWiseOps::Round => Box::new(tract_core::ops::math::Round {}),
        ElWiseOps::RoundHalfToEven => Box::new(tract_core::ops::math::RoundHalfToEven {}),
        ElWiseOps::Exp => Box::new(tract_core::ops::math::Exp {}),
        ElWiseOps::Sigmoid => Box::new(tract_core::ops::nn::Sigmoid {}),
        ElWiseOps::Sin => Box::new(tract_core::ops::math::Sin {}),
        ElWiseOps::Sinh => Box::new(tract_core::ops::math::Sinh {}),
        ElWiseOps::Asin => Box::new(tract_core::ops::math::Asin {}),
        ElWiseOps::Asinh => Box::new(tract_core::ops::math::Asinh {}),
        ElWiseOps::Cos => Box::new(tract_core::ops::math::Cos {}),
        ElWiseOps::Cosh => Box::new(tract_core::ops::math::Cosh {}),
        ElWiseOps::Acos => Box::new(tract_core::ops::math::Acos {}),
        ElWiseOps::Acosh => Box::new(tract_core::ops::math::Acosh {}),
        ElWiseOps::Tan => Box::new(tract_core::ops::math::Tan {}),
        ElWiseOps::Tanh => Box::new(tract_core::ops::math::Tanh {}),
        ElWiseOps::Atan => Box::new(tract_core::ops::math::Atan {}),
        ElWiseOps::Atanh => Box::new(tract_core::ops::math::Atanh {}),
        //ElWiseOps::Erf => Box::new(tract_core::ops::math::Erf {}),
        ElWiseOps::Ln => Box::new(tract_core::ops::math::Ln {}),
    };
    Box::new(ElementWiseOp(el_mini_op, None))
}

fn eval_reference<FI: Datum, FO: Datum>(
    a: &Tensor,
    func: impl Fn(&mut FO, &FI),
) -> TractResult<Tensor> {
    let mut out = unsafe { Tensor::uninitialized_dt(FO::datum_type(), a.shape())? };
    let a_view = a.to_array_view::<FI>()?;
    let mut c = out.to_array_view_mut::<FO>()?;
    tract_core::ndarray::Zip::from(&mut c).and_broadcast(a_view).for_each(func);
    Ok(out)
}

impl<T> ElWiseOpProblem<T>
where
    T: SupportedElement,
{
    fn f32_elmwise<F>(a: &T, op: F) -> T
    where
        T: SupportedElement,
        F: Fn(f32) -> f32,
    {
        let a_f32 = a.to_f32().unwrap();
        T::from_f32(op(a_f32)).unwrap()
    }

    pub fn reference(&self) -> TractResult<Tensor> {
        let inp = self.input.clone().into_tensor();

        let res = match self.op {
            ElWiseOps::Neg => eval_reference(&inp, |c, a| *c = Self::f32_elmwise(a, |x| -x))?,
            ElWiseOps::Abs => eval_reference(&inp, |c, a| *c = Self::f32_elmwise(a, f32::abs))?,
            ElWiseOps::Sqr => eval_reference(&inp, |c, a| *c = Self::f32_elmwise(a, |x| x * x))?,
            ElWiseOps::Sqrt => eval_reference(&inp, |c, a| *c = Self::f32_elmwise(a, f32::sqrt))?,
            ElWiseOps::Rsqrt => {
                eval_reference(&inp, |c, a| *c = Self::f32_elmwise(a, |x| 1.0 / x.sqrt()))?
            }
            ElWiseOps::Recip => {
                eval_reference(&inp, |c, a| *c = Self::f32_elmwise(a, |x| 1.0 / x))?
            }
            ElWiseOps::Ceil => eval_reference(&inp, |c, a| *c = Self::f32_elmwise(a, f32::ceil))?,
            ElWiseOps::Floor => eval_reference(&inp, |c, a| *c = Self::f32_elmwise(a, f32::floor))?,
            ElWiseOps::Round => eval_reference(&inp, |c, a| *c = Self::f32_elmwise(a, f32::round))?,
            ElWiseOps::RoundHalfToEven => {
                eval_reference(&inp, |c, a| *c = Self::f32_elmwise(a, f32::round_ties_even))?
            }
            ElWiseOps::Exp => eval_reference(&inp, |c, a| *c = Self::f32_elmwise(a, f32::exp))?,
            ElWiseOps::Sigmoid => {
                eval_reference(&inp, |c, a| *c = Self::f32_elmwise(a, |x| 1. / (1. + (-x).exp())))?
            }
            ElWiseOps::Sin => eval_reference(&inp, |c, a| *c = Self::f32_elmwise(a, f32::sin))?,
            ElWiseOps::Sinh => eval_reference(&inp, |c, a| *c = Self::f32_elmwise(a, f32::sinh))?,
            ElWiseOps::Asin => eval_reference(&inp, |c, a| *c = Self::f32_elmwise(a, f32::asin))?,
            ElWiseOps::Asinh => eval_reference(&inp, |c, a| *c = Self::f32_elmwise(a, f32::asinh))?,
            ElWiseOps::Cos => eval_reference(&inp, |c, a| *c = Self::f32_elmwise(a, f32::cos))?,
            ElWiseOps::Cosh => eval_reference(&inp, |c, a| *c = Self::f32_elmwise(a, f32::cosh))?,
            ElWiseOps::Acos => eval_reference(&inp, |c, a| *c = Self::f32_elmwise(a, f32::acos))?,
            ElWiseOps::Acosh => eval_reference(&inp, |c, a| *c = Self::f32_elmwise(a, f32::acosh))?,
            ElWiseOps::Tan => eval_reference(&inp, |c, a| *c = Self::f32_elmwise(a, f32::tan))?,
            ElWiseOps::Tanh => eval_reference(&inp, |c, a| *c = Self::f32_elmwise(a, f32::tanh))?,
            ElWiseOps::Atan => eval_reference(&inp, |c, a| *c = Self::f32_elmwise(a, f32::atan))?,
            ElWiseOps::Atanh => eval_reference(&inp, |c, a| *c = Self::f32_elmwise(a, f32::atanh))?,
            ElWiseOps::Ln => eval_reference(&inp, |c, a| *c = Self::f32_elmwise(a, f32::ln))?,
        };
        Ok(res)
    }

    fn tract(&self) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();

        let input = model
            .add_source("input", TypedFact::shape_and_dt_of(&self.input.clone().into_tensor()))?;

        let output = model.wire_node("bin_op", to_tract_op(&self.op), &[input])?;
        model.set_output_outlets(&output)?;

        model = model.into_decluttered()?;
        Ok(model)
    }
}

impl<T> Test for ElWiseOpProblem<T>
where
    T: SupportedElement,
{
    fn run_with_approx(
        &self,
        _suite: &str,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> TestResult {
        let reference = self.reference()?;
        let mut model = self.tract()?;

        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));

        let mut output = runtime.prepare(model)?.run(tvec![self.input.clone().into_tvalue()])?;
        let output = output.remove(0).into_tensor();
        output.close_enough(&reference, approx)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();

    suite.add_arbitrary::<ElWiseOpProblem<f32>>("proptest_f32", ());
    suite.add_arbitrary::<ElWiseOpProblem<f16>>("proptest_f16", ());
    suite.add_arbitrary::<ElWiseOpProblem<u8>>("proptest_u8", ());
    suite.add_arbitrary::<ElWiseOpProblem<u32>>("proptest_u32", ());
    suite.add_arbitrary::<ElWiseOpProblem<i16>>("proptest_i16", ());
    suite.add_arbitrary::<ElWiseOpProblem<i64>>("proptest_i64", ());

    Ok(suite)
}
