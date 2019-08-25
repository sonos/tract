use tract_core::internal::*;

use crate::model::ParsingContext;
use crate::tfpb::node_def::NodeDef;

#[derive(Debug, Clone, new)]
pub struct Max {
    t: DatumType,
    t_idx: DatumType,
    keep_dims: bool,
}

pub fn max(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let t = pb.get_attr_datum_type("T")?;
    let t_idx = pb.get_attr_datum_type("Tidx")?;
    let keep_dims = pb.get_attr_bool("keep_dims")?;
    Ok(Box::new(Max::new(t, t_idx, keep_dims)))
}

impl Max {}

impl Op for Max {
    fn name(&self) -> Cow<str> {
        "tf.Max".into()
    }
}

impl StatelessOp for Max {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (input, axes) = args_2!(inputs);
        let axes: TVec<usize> = axes
            .cast_to::<i32>()?
            .as_slice::<i32>()?
            .iter()
            .map(|&ax| if ax >= 0 { ax as usize } else { ax as usize + input.shape().len() })
            .collect();
        dispatch_numbers!(self::eval_t(self.t)(input, &*axes, self.keep_dims))
    }
}

impl InferenceRulesOp for Max {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 2)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&inputs[1].rank, 1)?;
        if self.keep_dims {
            s.equals(&inputs[0].rank, &outputs[0].rank)?;
        } else {
            s.equals(
                inputs[0].rank.bex().to_dim(),
                inputs[1].shape[0].bex() + outputs[0].rank.bex().to_dim(),
            )?;
        }
        s.given_3(
            &inputs[0].rank,
            &outputs[0].rank,
            &inputs[1].value,
            move |s, irank, orank, axes| {
                let axes: TVec<usize> = axes
                    .cast_to::<i32>()?
                    .as_slice::<i32>()?
                    .iter()
                    .map(|&ax| if ax > 0 { ax } else { ax + irank } as usize)
                    .collect();
                let mut od = 0;
                for id in 0..(irank as usize) {
                    if axes.contains(&id) {
                        if self.keep_dims {
                            s.equals(&outputs[0].shape[od], 1.to_dim())?;
                            od += 1;
                        }
                    } else {
                        if od < orank as usize {
                            s.equals(&outputs[0].shape[od], &inputs[0].shape[id])?;
                        }
                    }
                }
                Ok(())
            },
        )?;
        Ok(())
    }

    fn to_typed(
        &self,
        source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        if let Some(ref axes) = target.outlet_fact(mapping[&node.inputs[1]])?.konst {
            let axes: TVec<usize> = axes
                .cast_to::<i32>()?
                .as_slice::<i32>()?
                .iter()
                .map(|&ax| {
                    Ok(if ax >= 0 {
                        ax as usize
                    } else {
                        ax as usize + target.outlet_fact(mapping[&node.inputs[0]])?.shape.rank()
                    })
                })
                .collect::<TractResult<_>>()?;
            let op = TypedMax::new(self.t, self.t_idx, self.keep_dims, axes);
            tract_core::ops::trivial_inference_op_to_typed(
                Box::new(op),
                source,
                node,
                target,
                mapping,
            )
        } else {
            bail!("Nees axes to be const")
        }
    }

    inference_op_as_op!();
}

#[derive(Debug, Clone, new)]
pub struct TypedMax {
    t: DatumType,
    t_idx: DatumType,
    keep_dims: bool,
    axes: TVec<usize>,
}

impl Op for TypedMax {
    fn name(&self) -> Cow<str> {
        "tf.TypedMax".into()
    }
}

impl StatelessOp for TypedMax {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        dispatch_numbers!(self::eval_t(self.t)(input, &*self.axes, self.keep_dims))
    }
}

impl TypedOp for TypedMax {
    typed_op_as_op!();

    fn output_facts(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<TypedTensorInfo>> {
        let output_shape: TVec<TDim> = inputs[0]
            .shape
            .iter()
            .enumerate()
            .filter_map(|(ax, d)| {
                if self.axes.contains(&ax) {
                    if self.keep_dims {
                        Some(1.to_dim())
                    } else {
                        None
                    }
                } else {
                    Some(d)
                }
            })
            .collect();
        Ok(tvec!(TypedTensorInfo::dt_shape(self.t, &*output_shape)?))
    }
}

fn eval_t<T>(input: Arc<Tensor>, axes: &[usize], keep_dims: bool) -> TractResult<TVec<Arc<Tensor>>>
where
    T: Copy + Datum + PartialOrd + num_traits::Bounded,
{
    use ndarray::*;
    let input = input.to_array_view::<T>()?;
    let full_output_shape: TVec<usize> = input
        .shape()
        .iter()
        .enumerate()
        .map(|(ax, &d)| if axes.contains(&ax) { 1 } else { d })
        .collect();
    let mut result = Array::from_shape_fn(&*full_output_shape, |coords| {
        let slice_spec: Vec<SliceOrIndex> = coords
            .slice()
            .iter()
            .enumerate()
            .map(|(ax, &d)| if axes.contains(&ax) { (..).into() } else { d.into() })
            .collect();
        let slice_info = SliceInfo::<_, IxDyn>::new(&slice_spec).unwrap();
        let slice = input.slice(slice_info.as_ref());
        slice.iter().fold(T::min_value(), |a, &b| if a < b { b } else { a })
    });
    if !keep_dims {
        for ax in (0..full_output_shape.len()).rev() {
            if axes.contains(&ax) {
                result = result.index_axis_move(Axis(ax), 0);
            }
        }
    }
    Ok(tvec!(result.into_arc_tensor()))
}
