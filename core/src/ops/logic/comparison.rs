use crate::broadcast::multi_broadcast;
use crate::internal::*;
use crate::ndarray::Zip;

#[derive(Clone, Copy, Debug, Hash)]
pub enum Comp {
    Eq,
    NE,
    LT,
    GT,
    GTE,
    LTE,
}

use tract_data::TooEarly;
use Comp::*;

impl Op for Comp {
    fn name(&self) -> Cow<str> {
        match *self {
            Eq => "==",
            NE => "!=",
            LT => "<",
            GT => ">",
            LTE => "<=",
            GTE => ">=",
        }
        .into()
    }

    op_as_typed_op!();
}

impl Comp {
    fn eval<T: Datum + PartialOrd>(&self, a: &Tensor, b: &Tensor) -> TractResult<Tensor> {
        let a = a.to_array_view::<T>()?;
        let b = b.to_array_view::<T>()?;
        let shape = multi_broadcast(&[a.shape(), b.shape()])?;
        let mut c = unsafe { Tensor::uninitialized::<bool>(&shape)? };
        let mut view = c.to_array_view_mut::<bool>()?;
        let zipped = Zip::from(&mut view).and_broadcast(&a).and_broadcast(&b);
        match *self {
            Eq => zipped.for_each(|c, a, b| *c = a == b),
            NE => zipped.for_each(|c, a, b| *c = a != b),
            LT => zipped.for_each(|c, a, b| *c = a < b),
            GT => zipped.for_each(|c, a, b| *c = a > b),
            LTE => zipped.for_each(|c, a, b| *c = a <= b),
            GTE => zipped.for_each(|c, a, b| *c = a >= b),
        }
        Ok(c)
    }
}

impl EvalOp for Comp {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        session: &SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        if inputs[0].datum_type() == TDim::datum_type() {
            let mut a = inputs[0].clone().into_tensor();
            let mut b = inputs[1].clone().into_tensor();
            for a in a.as_slice_mut::<TDim>()? {
                *a = a.eval(&session.resolved_symbols);
            }
            for b in b.as_slice_mut::<TDim>()? {
                *b = b.eval(&session.resolved_symbols);
            }
            if let (Ok(a), Ok(b)) = (a.cast_to::<i64>(), b.cast_to::<i64>()) {
                return Ok(tvec!(self.eval::<i64>(&a, &b)?.into_tvalue()));
            }
            let a = inputs[0].to_array_view::<TDim>()?;
            let b = inputs[0].to_array_view::<TDim>()?;
            let shape = multi_broadcast(&[a.shape(), b.shape()])?;
            let mut c = unsafe { Tensor::uninitialized::<bool>(&shape)? };
            let mut view = c.to_array_view_mut::<bool>()?;
            let a = a.broadcast(&*shape).unwrap();
            let b = b.broadcast(&*shape).unwrap();
            for ixs in tract_ndarray::indices(&*shape) {
                let (a, b) = (&a[&ixs], &b[&ixs]);
                let diff = a.clone() - b;
                view[&ixs] = match *self {
                    Eq => a == b,
                    NE => a != b,
                    GTE => {
                        if diff.prove_positive_or_zero() {
                            true
                        } else if diff.prove_strict_negative() {
                            false
                        } else {
                            bail!(TooEarly::UndeterminedSymbol(diff));
                        }
                    }
                    GT => {
                        if diff.prove_strict_positive() {
                            true
                        } else if diff.prove_negative_or_zero() {
                            false
                        } else {
                            bail!(TooEarly::UndeterminedSymbol(diff));
                        }
                    }
                    LTE => {
                        if diff.prove_negative_or_zero() {
                            true
                        } else if diff.prove_strict_positive() {
                            false
                        } else {
                            bail!(TooEarly::UndeterminedSymbol(diff));
                        }
                    }
                    LT => {
                        if diff.prove_strict_negative() {
                            true
                        } else if diff.prove_negative_or_zero() {
                            false
                        } else {
                            bail!(TooEarly::UndeterminedSymbol(diff));
                        }
                    }
                };
            }
            Ok(tvec!(c.into_tvalue()))
        } else {
            let t = dispatch_numbers!(Self::eval(inputs[0].datum_type())(
                self, &inputs[0], &inputs[1]
            ))?;
            Ok(tvec!(t.into_tvalue()))
        }
    }
}

impl TypedOp for Comp {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let shape = multi_broadcast(&[&inputs[0].shape, &inputs[1].shape])?;
        Ok(tvec!(bool::datum_type().fact(shape)))
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        if let AxisOp::Rm(rm) = change {
            let (inputs, outputs) = model.node_facts(node.id)?;
            if !inputs[0].shape[*rm].is_one()
                || !inputs[0].shape[*rm].is_one()
                || !outputs[0].shape[*rm].is_one()
            {
                return Ok(None);
            }
        }
        Ok(Some(AxisChangeConsequence::new(model, node, None, change)))
    }

    fn slice(
        &self,
        patch: &mut TypedModelPatch,
        _model: &TypedModel,
        _node: &TypedNode,
        prefix: &str,
        inputs: &[OutletId],
        _output_axis: usize,
        _start: &TDim,
        _end: &TDim,
    ) -> TractResult<Option<TVec<OutletId>>> {
        Ok(Some(patch.wire_node(prefix, *self, inputs)?))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        AxesMapping::natural(inputs, outputs)
    }

    as_op!();
}
