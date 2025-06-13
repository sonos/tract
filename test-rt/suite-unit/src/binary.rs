use std::ops::Div;

use super::*;
use infra::{Test, TestResult};
use tract_core::num_traits::{AsPrimitive, FromPrimitive, Num, ToPrimitive};
use proptest::collection::vec;
use tract_core::ndarray::ArrayD;
use tract_core::ops::binary::TypedBinOp;
use tract_core::ops::logic::Comp;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub enum BinOps {
    Mul,
    Add,
    Div,
    Sub,
    Pow,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Equals,
    NotEquals,
}

pub const ALL_OPS: [BinOps; 11] = [
    BinOps::Mul,
    BinOps::Add,
    BinOps::Div,
    BinOps::Sub,
    BinOps::Pow,
    BinOps::Less,
    BinOps::LessEqual,
    BinOps::Greater,
    BinOps::GreaterEqual,
    BinOps::Equals,
    BinOps::NotEquals,
];

fn to_tract_op(op: BinOps) -> Box<dyn TypedOp>{
    match op {
        BinOps::Mul => Box::new(TypedBinOp(Box::new(tract_core::ops::math::Mul), None)),
        BinOps::Add => Box::new(TypedBinOp(Box::new(tract_core::ops::math::Add), None)),
        BinOps::Div => Box::new(TypedBinOp(Box::new(tract_core::ops::math::Div), None)),
        BinOps::Sub => Box::new(TypedBinOp(Box::new(tract_core::ops::math::Sub), None)),
        BinOps::Pow => Box::new(TypedBinOp(Box::new(tract_core::ops::math::Pow), None)),
        BinOps::Less => Box::new(Comp::LT),
        BinOps::LessEqual => Box::new(Comp::LTE),
        BinOps::Greater => Box::new(Comp::GT),
        BinOps::GreaterEqual => Box::new(Comp::GTE),
        BinOps::Equals => Box::new(Comp::Eq),
        BinOps::NotEquals => Box::new(Comp::NE),
    }
}

pub trait SupportedElement:
Datum + Num + Copy + FromPrimitive + ToPrimitive + 'static + Div<Output = Self> + PartialOrd + AsPrimitive<usize> + AsPrimitive<f32> + AsPrimitive<f64>
{
}

impl<T> SupportedElement for T where
    T: Datum + Num + Copy + FromPrimitive + ToPrimitive + 'static + Div<Output = Self> + PartialOrd + AsPrimitive<usize> + AsPrimitive<f32> + AsPrimitive<f64>
{
}

#[derive(Debug, Clone)]
pub struct BinaryOpProblem<T>
where
    T: SupportedElement
{   
    pub op: BinOps,
    pub lhs: ArrayD<T>,
    pub rhs: ArrayD<T>
}

impl<T> Arbitrary for BinaryOpProblem<T>
where
    T: SupportedElement,
{
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        let lhs_shape_strat = prop::collection::vec(1usize..=5, 0..=4);
        lhs_shape_strat
            .prop_flat_map(|lhs_shape| {
                let rank = lhs_shape.len();
                let rhs_shape_strat = prop::collection::vec(1usize..=5, rank..=rank);
                (Just(lhs_shape), rhs_shape_strat)
            }
            ) 
            .prop_flat_map(|(mut lhs_shape, rhs_shape)| {
                for idx in 0..lhs_shape.len() {
                    if (lhs_shape[idx] != rhs_shape[idx]) && lhs_shape[idx] != 1 && rhs_shape[idx] != 1 {
                        lhs_shape[idx] = rhs_shape[idx]
                    }
                }

                let lhs_len = lhs_shape.iter().product::<usize>();
                let rhs_len = rhs_shape.iter().product::<usize>();
                let lhs = vec((2u8..=10u8).prop_map(|i| T::from_u8(i).unwrap() / T::from_u8(2).unwrap()), lhs_len..=lhs_len)
                    .prop_map(move |vec| ArrayD::from_shape_vec(lhs_shape.to_vec(), vec).unwrap())
                    .boxed();
                let rhs = vec((2u8..=10u8).prop_map(|i| T::from_u8(i).unwrap() / T::from_u8(2).unwrap()), rhs_len..=rhs_len)
                    .prop_map(move |vec| ArrayD::from_shape_vec(rhs_shape.to_vec(), vec).unwrap())
                    .boxed();

                let mut  ops = ALL_OPS.to_vec();

                // Avoid Sub overfow for uints and Unsupported type for Pow
                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u8>()
                || std::any::TypeId::of::<T>() == std::any::TypeId::of::<u16>()
                || std::any::TypeId::of::<T>() == std::any::TypeId::of::<u32>()
                || std::any::TypeId::of::<T>() == std::any::TypeId::of::<u64>() {
                    ops.retain(|op| !matches!(op, BinOps::Sub | BinOps::Pow));
                }

                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i8>()
                || std::any::TypeId::of::<T>() == std::any::TypeId::of::<i16>()
                {
                    ops.retain(|op| !matches!(op, BinOps::Pow));
                }
                let op_strategy = prop::sample::select(ops);

                (lhs, rhs, op_strategy)
            })
            .prop_map(|(lhs, rhs, op)| BinaryOpProblem {
                lhs,
                rhs,
                op,
            })
            .boxed()
    }
}

fn eval_reference<FI: Datum, FO: Datum>(
a: &Tensor,
b: &Tensor,
cab: impl Fn(&mut FO, &FI, &FI),
) -> TractResult<Tensor> {
    let out_shape = tract_core::broadcast::multi_broadcast(&[a.shape(), b.shape()])?;
    let mut out = unsafe { Tensor::uninitialized_dt(FO::datum_type(), &out_shape)? };
    let a_view = a.to_array_view::<FI>()?;
    let b_view = b.to_array_view::<FI>()?;
    let mut c = out.to_array_view_mut::<FO>()?;
    tract_core::ndarray::Zip::from(&mut c)
        .and_broadcast(a_view)
        .and_broadcast(b_view)
        .for_each(cab);
    Ok(out)
}

impl<T> BinaryOpProblem<T>
where
    T: SupportedElement
{
    pub fn reference(&self) -> TractResult<Tensor> {
        let lhs = self.lhs.clone().into_tensor();
        let rhs = self.rhs.clone().into_tensor();

        let res = match self.op {
            BinOps::Add => eval_reference(&lhs, &rhs, |c: &mut T, a: &T, b: &T| *c = *a + *b)?,
            BinOps::Sub => eval_reference(&lhs, &rhs, |c: &mut T, a: &T, b: &T| *c = *a - *b)?,
            BinOps::Mul => eval_reference(&lhs, &rhs, |c: &mut T, a: &T, b: &T| *c = *a * *b)?,
            BinOps::Div => eval_reference(&lhs, &rhs, |c: &mut T, a: &T, b: &T| *c = *a / *b)?,
            BinOps::Pow => eval_reference(&lhs, &rhs, |c: &mut T, a: &T, b: &T| *c = T::from_f32(a.to_f32().unwrap().powf(b.to_f32().unwrap())).unwrap())?,
            BinOps::Less => eval_reference(&lhs, &rhs, |c: &mut bool, a: &T, b: &T| *c = *a < *b)?,
            BinOps::LessEqual => eval_reference(&lhs, &rhs, |c: &mut bool, a: &T, b: &T| *c = *a <= *b)?,
            BinOps::Greater => eval_reference(&lhs, &rhs, |c: &mut bool, a: &T, b: &T| *c = *a > *b)?,
            BinOps::GreaterEqual => eval_reference(&lhs, &rhs, |c: &mut bool, a: &T, b: &T| *c = *a >= *b)?,
            BinOps::Equals => eval_reference(&lhs, &rhs, |c: &mut bool, a: &T, b: &T| *c = *a == *b)?,
            BinOps::NotEquals => eval_reference(&lhs, &rhs, |c: &mut bool, a: &T, b: &T| *c = *a != *b)?,
        };
        Ok(res)
    }

    fn tract(&self) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();

        let lhs = model
            .add_source("lhs", TypedFact::shape_and_dt_of(&self.lhs.clone().into_tensor()))?;
        let rhs = model
            .add_source("rhs", TypedFact::shape_and_dt_of(&self.rhs.clone().into_tensor()))?;

        let output = model.wire_node(
            "bin_op",
            to_tract_op(self.op),
            &[lhs, rhs],
        )?;
        model.set_output_outlets(&output)?;

        model = model.into_decluttered()?;
        Ok(model)
    }
}

impl<T> Test for BinaryOpProblem<T>
where
    T: SupportedElement
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

        let mut output = runtime
                    .prepare(model)?
                    .run(tvec![self.lhs.clone().into_tvalue(), self.rhs.clone().into_tvalue()])?;
        let output = output.remove(0).into_tensor();
        output.close_enough(&reference, approx)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();

    suite.add_arbitrary::<BinaryOpProblem<f32>>("proptest_f32", ());
    suite.add_arbitrary::<BinaryOpProblem<f16>>("proptest_f16", ());
    suite.add_arbitrary::<BinaryOpProblem<u8>>("proptest_u8", ());
    suite.add_arbitrary::<BinaryOpProblem<u32>>("proptest_u32", ());
    suite.add_arbitrary::<BinaryOpProblem<i16>>("proptest_i16", ());
    suite.add_arbitrary::<BinaryOpProblem<i64>>("proptest_i64", ());

    Ok(suite)
}