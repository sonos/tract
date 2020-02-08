use tract_core::infer::*;
use tract_core::internal::*;

use crate::model::ParsingContext;
use crate::tfpb::tensorflow::NodeDef;

pub fn build(_ctx: &ParsingContext, _pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    Ok(Box::new(ExpandDims))
}

#[derive(Debug, Clone)]
pub struct ExpandDims;

impl ExpandDims {
    fn eval_t<T: Datum>(
        &self,
        data: Arc<Tensor>,
        shape: &[usize],
    ) -> TractResult<TVec<Arc<Tensor>>> {
        let data = data.into_tensor().into_array::<T>()?;
        Ok(tvec![Tensor::from(data.into_shape(&*shape)?).into()])
    }
}

impl Op for ExpandDims {
    fn name(&self) -> Cow<str> {
        "tf.ExpandDims".into()
    }

    not_a_typed_op!();
}

impl StatelessOp for ExpandDims {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (data, dims) = args_2!(inputs);
        let dims = dims.to_array_view::<i32>()?;
        let mut shape: TVec<usize> = data.shape().into();
        for d in dims.iter() {
            let d = if *d >= 0 { *d } else { *d + 1 + data.shape().len() as i32 } as usize;
            shape.insert(d, 1);
        }
        dispatch_datum!(Self::eval_t(data.datum_type())(self, data, &*shape))
    }
}

impl InferenceRulesOp for ExpandDims {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        let data = &inputs[0];
        let dims = &inputs[1];
        let output = &outputs[0];

        check_input_arity(&inputs, 2)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&dims.datum_type, DatumType::I32)?;
        s.equals(&data.datum_type, &output.datum_type)?;
        s.equals(data.rank.bex() + 1, &output.rank)?;
        s.given_2(&dims.value, &data.rank, move |s, index, rank| {
            let mut index = *(index.to_scalar::<i32>()?);
            if index < 0 {
                index += rank + 1
            }
            let index = index as usize;

            for i in 0..index {
                s.equals(&output.shape[i], &data.shape[i])?;
            }

            s.equals(output.shape[index].bex(), 1i32.to_dim().bex())?;

            s.given(&data.rank, move |s, rank| {
                for i in index..(rank as usize) {
                    s.equals(&output.shape[i + 1], &data.shape[i])?;
                }
                Ok(())
            })
        })
    }

    as_op!();

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        if let Some(ref axes) = target.outlet_fact(mapping[&node.inputs[1]])?.konst {
            let mut axes = axes
                .cast_to::<i64>()?
                .as_slice::<i64>()?
                .iter()
                .map(|&axis| {
                    Ok(if axis < 0 {
                        axis + target.outlet_fact(mapping[&node.inputs[0]])?.shape.rank() as i64
                    } else {
                        axis
                    })
                })
                .collect::<TractResult<Vec<_>>>()?;
            axes.sort();
            let mut wire = mapping[&node.inputs[0]];
            for axis in axes.iter().rev() {
                wire = target.wire_node(
                    format!("{}-axis-{}", node.name, axis),
                    AxisOp::Add(*axis as _),
                    &[wire],
                )?[0];
            }
            Ok(tvec!(wire))
        } else {
            bail!("Need axes to be const")
        }
    }
}
