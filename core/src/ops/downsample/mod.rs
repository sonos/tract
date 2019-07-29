use crate::internal::*;
use crate::ops;
use ndarray::prelude::*;

mod array;
mod scan;

#[derive(Debug, Clone, new, Default, PartialEq)]
pub struct Downsample {
    axis: usize,
    stride: usize,
    modulo: usize,
}

impl Downsample {
    fn eval_t<T: Datum>(&self, input: &Tensor) -> TractResult<Arc<Tensor>> {
        let input = input.to_array_view::<T>()?;
        let sampled = if self.modulo < input.shape()[self.axis] {
            input
                .slice_axis(
                    Axis(self.axis),
                    ndarray::Slice::new(self.modulo as isize, None, self.stride as isize),
                )
                .to_owned()
                .into_arc_tensor()
        } else {
            let mut shape = input.shape().to_vec();
            shape[self.axis] = 0;
            unsafe { Tensor::uninitialized::<T>(&shape)?.into_arc_tensor() }
        };
        Ok(sampled)
    }

    fn transform_fact(&self, input_fact: &TypedTensorInfo) -> TractResult<TypedTensorInfo> {
        let mut downed = input_fact.clone();
        let down_len = (input_fact.shape.dim(self.axis) - self.modulo).div_ceil(self.stride.into());
        downed.shape.set_dim(self.axis, down_len.clone())?;
        Ok(downed)
    }
}


impl Op for Downsample {
    fn name(&self) -> Cow<str> {
        "Downsample".into()
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if self.stride == 1 {
            return Ok(Some(TypedModelPatch::shunt_one_op(model, node)?));
        }
        pull_downsample_up(model, node)
    }

    impl_op_same_as!();
}

impl StatelessOp for Downsample {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        Ok(tvec!(dispatch_datum!(Self::eval_t(input.datum_type())(self, &*input))?))
    }
}

impl InferenceRulesOp for Downsample {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.given(&inputs[0].rank, move |s, r| {
            for i in 0..(r as usize) {
                if i == self.axis {
                    s.given(&inputs[0].shape[i], move |s, d| {
                        s.equals(
                            &outputs[0].shape[i],
                            (d - self.modulo).div_ceil(self.stride.to_dim()),
                        )
                    })?
                } else {
                    s.equals(&inputs[0].shape[i], &outputs[0].shape[i])?
                }
            }
            Ok(())
        })
    }

    inference_op_as_op!();
}

fn pull_downsample_up(
    model: &TypedModel,
    down_node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    let down_op = down_node.op_as::<Downsample>().unwrap();
    if let Some(prec) = model.single_prec(down_node.id)? {
        let invariants = prec.op().translation_invariants(model, prec)?;
        println!("Considering moving {:?} over {:?} of invariants {:?}", down_node, prec, invariants);
        if invariants
            .iter()
            .find(|inv| inv.axis == down_op.axis && down_op.stride % inv.period == 0)
            .is_some()
        {
            println!("Doing it because invariants");
            let mut patch = TypedModelPatch::default();
            patch.tap_model(model, prec.inputs[0])?;
            let input_outlet = prec.inputs[0].clone();
            let input_fact = model.outlet_fact(input_outlet).unwrap();
            let downed = down_op.transform_fact(&input_fact)?;
            patch.chain(&*down_node.name, down_op.clone(), tvec!(downed))?;
            let other = patch.chain(
                &*prec.name,
                objekt::clone_box(prec.op()),
                tvec!(down_node.outputs[0].fact.clone()),
            )?;
            patch.shunt_outside(OutletId::new(down_node.id, 0), OutletId::new(other, 0))?;
            return Ok(Some(patch));
        }
        if let Some(crop_op) = prec.op_as::<ops::array::Crop>() {
            return array::pull_downsample_over_crop(model, prec, crop_op, down_node, down_op);
        } else if let Some(other_op) = prec.op_as::<ops::array::RmDims>() {
            return array::pull_downsample_over_rmdims(model, prec, other_op, down_node, down_op);
        } else if let Some(other_op) = prec.op_as::<ops::array::AddDims>() {
            return array::pull_downsample_over_adddims(model, prec, other_op, down_node, down_op);
        } else if let Some(other_op) = prec.op_as::<ops::scan::Typed>() {
            return scan::pull_downsample_over_scan(model, prec, other_op, down_node, down_op);
        }
    }
    Ok(None)
}

