use tract_hir::internal::*;
use tract_ndarray::Dimension;

use crate::model::ParsingContext;
use crate::pb::NodeProto;

#[derive(Debug, Clone, Hash)]
pub struct NonZero(Symbol);

pub fn non_zero(
    ctx: &ParsingContext,
    _node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    // symbol table is shared between all templates and models
    let x = ctx.template.symbols.new_with_prefix("x");
    Ok((Box::new(NonZero(x)) as _, vec!()))
}

impl NonZero {
    unsafe fn eval_t<T: Datum + tract_num_traits::Zero>(input: &Tensor) -> TractResult<Tensor> {
        let count = input.as_slice_unchecked::<T>().iter().filter(|d| !d.is_zero()).count();
        let view = input.to_array_view_unchecked::<T>();
        let mut output = Tensor::uninitialized::<i64>(&[input.rank(), count])?;
        let mut view_mut: tract_ndarray::ArrayViewMut2<i64> =
            output.to_array_view_mut_unchecked::<i64>().into_dimensionality().unwrap();
        for (i, (coords, _)) in
            view.indexed_iter().filter(|(_, value)| !value.is_zero()).enumerate()
        {
            view_mut
                .index_axis_mut(tract_ndarray::Axis(1), i)
                .assign(&coords.as_array_view().map(|d| *d as i64));
        }
        Ok(output)
    }
}

impl Op for NonZero {
    fn name(&self) -> Cow<str> {
        "NonZero".into()
    }

    op_as_typed_op!();
}

impl EvalOp for NonZero {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        unsafe {
            let input = args_1!(inputs);
            let output = if input.datum_type() == bool::datum_type() {
                Self::eval_t::<u8>(&input)?
            } else {
                dispatch_numbers!(Self::eval_t(input.datum_type())(&input))?
            };
            Ok(tvec!(output.into_tvalue()))
        }
    }
}

impl InferenceRulesOp for NonZero {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> TractResult<()> {
        check_input_arity(inputs, 1)?;
        check_output_arity(outputs, 1)?;
        s.equals(&outputs[0].datum_type, i64::datum_type())?;
        s.equals(&outputs[0].rank, 2)?;
        s.equals(&outputs[0].shape[0], inputs[0].rank.bex().to_dim())?;
        Ok(())
    }

    as_op!();
    to_typed!();
}

impl TypedOp for NonZero {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(i64::fact(dims![inputs[0].rank(), self.0])))
    }

    as_op!();
}
