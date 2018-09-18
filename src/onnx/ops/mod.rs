use std::fmt;

use ndarray::prelude::*;
use ops::prelude::*;
use analyser::rules::prelude::*;

trait ElementUnary {
    fn name() -> &'static str;
    fn f32(v: &f32) -> f32;
    fn f64(v: &f64) -> f64;
}

#[derive(Clone,new)]
struct ElementUnaryOp<U>(PhantomData<U>)
where U:ElementUnary+Send+Sync+fmt::Debug+Clone+'static;

impl<U> Op for ElementUnaryOp<U>
where U:ElementUnary+Send+Sync+fmt::Debug+Clone+'static
{
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Value>,) -> Result<TVec<Value>> {
        let a = args_1!(inputs);
        let it = match a.datum_type() {
            DatumType::F32 => a.into_array()?.map(U::f32).into(),
            _ => bail!("unsupported type {:?} for {}", a.datum_type(), U::name())
        };
        Ok(tvec![it])
    }

    /// Evaluates one step of the operation on the given input tensors.
    fn step(
        &self,
        mut inputs: TVec<StepValue>,
        _buffer: &mut Box<OpBuffer>,
    ) -> Result<Option<TVec<Value>>> {
        let a = args_1!(inputs);
        match a.into_value() {
            None => Ok(None),
            Some(tv) => Ok(Some(self.eval(tvec![tv])?)),
        }
    }
}

impl<U:ElementUnary> InferenceRulesOp for ElementUnaryOp<U>
where U:ElementUnary+Send+Sync+fmt::Debug+Clone+'static
{
    fn rules<'r, 'p: 'r, 's: 'r>(
    &'s self,
    solver: &mut Solver<'r>,
    inputs: &'p TensorsProxy,
    outputs: &'p TensorsProxy
) {
    }
}

impl<U:ElementUnary> fmt::Debug for ElementUnaryOp<U>
where U:ElementUnary+Send+Sync+fmt::Debug+Clone+'static {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{}", U::name())
    }
}

#[derive(Debug,Clone)]
struct AbsUnary;
impl ElementUnary for AbsUnary {
    fn name() -> &'static str { "Abs" }
    fn f32(v:&f32) -> f32 { v.abs() }
    fn f64(v:&f64) -> f64 { v.abs() }
}

pub fn build(op_type: &str) -> Box<Op> {
    match op_type {
        "Abs" => Box::new(ElementUnaryOp::<AbsUnary>::new()),
        _ => Box::new(::ops::unimpl::UnimplementedOp(op_type.to_string(), "".to_string()))
    }
}
