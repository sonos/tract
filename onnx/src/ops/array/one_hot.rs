use tract_hir::internal::*;
use tract_ndarray::Dimension;

use crate::model::ParsingContext;
use crate::pb::NodeProto;

pub fn one_hot(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(-1);
    Ok((expand(OneHot::new(axis)), vec![]))
}

#[derive(Debug, PartialEq, Clone, new, Hash)]
struct OneHot {
    axis: i64,
}

tract_linalg::impl_dyn_hash!(OneHot);

impl OneHot {
    pub fn split_values_t<T: Datum>(values: &Tensor) -> TractResult<(Tensor, Tensor)> {
        let slice = values.as_slice::<T>()?;
        let v_off = tensor0(slice[0].clone());
        let v_on = tensor0(slice[1].clone());
        Ok((v_off, v_on))
    }
}

impl Expansion for OneHot {
    fn name(&self) -> Cow<str> {
        "OneHot".into()
    }

    op_onnx!();

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let dim = model.outlet_fact(inputs[1])?;
        let values = model.outlet_fact(inputs[2])?;
        if let (Some(dim), Some(values)) = (&dim.konst, &values.konst) {
            let rank = model.outlet_fact(inputs[0])?.rank();
            let axis = if self.axis < 0 { self.axis + rank as i64 + 1 } else { self.axis } as usize;
            let dim = dim.cast_to::<i64>()?;
            let dim = dim.as_slice::<i64>()?[0];
            if dim < 0 {
                bail!("Expected positive dimension, got {}", dim)
            }
            let (off, on) = dispatch_datum!(Self::split_values_t(values.datum_type())(&values))?;
            let op = MirOneHot { axis, dim: dim as usize, off, on };
            model.wire_node(prefix, op, &[inputs[0]])
        } else {
            bail!("Expected dim and value to be determined, got {:?} and {:?}", dim, values)
        }
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 3)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[2].datum_type, &outputs[0].datum_type)?;
        s.equals(inputs[0].rank.bex() + 1, &outputs[0].rank)?;
        s.equals(&inputs[2].rank, 1)?;
        s.equals(&inputs[2].shape[0], 2.to_dim())?;
        s.given(&inputs[0].rank, move |s, irank| {
            let axis = if self.axis < 0 { self.axis + irank + 1 } else { self.axis } as usize;
            for ix in 0..axis {
                s.equals(&inputs[0].shape[ix], &outputs[0].shape[ix])?;
            }
            for ix in axis + 1..irank as usize + 1 {
                s.equals(&inputs[0].shape[ix - 1], &outputs[0].shape[ix])?;
            }
            s.given(&inputs[1].value, move |s, value| {
                let dim = value.cast_to_scalar::<i64>()?;
                s.equals(&outputs[0].shape[axis], dim.to_dim())
            })
        })
    }
}

#[derive(Debug, PartialEq, Clone, new, Hash)]
struct MirOneHot {
    axis: usize,
    dim: usize,
    off: Tensor,
    on: Tensor,
}

tract_linalg::impl_dyn_hash!(MirOneHot);

impl Op for MirOneHot {
    fn name(&self) -> Cow<str> {
        "MirOnehot".into()
    }

    op_onnx!();
    op_as_typed_op!();
}

impl TypedOp for MirOneHot {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut shape = inputs[0].shape.to_tvec();
        shape.insert(self.axis, self.dim.to_dim());
        Ok(tvec!(TypedFact::dt_shape(self.off.datum_type(), &*shape)?))
    }

    as_op!();
}

impl EvalOp for MirOneHot {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let output = dispatch_datum!(Self::eval_t(self.off.datum_type())(self, &input))?;
        Ok(tvec!(output.into_arc_tensor()))
    }
}

impl MirOneHot {
    fn eval_t<T: Datum + Clone>(&self, input: &Tensor) -> TractResult<Tensor> {
        let off = self.off.to_scalar::<T>()?;
        let on = self.on.to_scalar::<T>()?;
        let mut shape: TVec<usize> = input.shape().into();
        shape.insert(self.axis, self.dim);
        let mut array = tract_ndarray::ArrayD::<T>::from_elem(&*shape, off.to_owned());
        let input = input.cast_to::<i32>()?;
        let input = input.to_array_view::<i32>()?;
        dbg!(&input);
        for icoord in tract_ndarray::indices_of(&input) {
            let mut ocoord:Vec<usize> = icoord.slice().into();
            let coord = input[&icoord];
            let coord = if coord < 0 { coord + self.dim as i32 } else { coord } as usize;
            ocoord.insert(self.axis, coord);
            array[&*ocoord] = on.clone();
        }
        Ok(array.into_tensor())
    }
}
