use crate::infer::*;
use tract_core::ops::nn::*;

use tract_core::ops::nn::Reduce as TypedReduce;
pub use tract_core::ops::nn::Reducer as Reducer;

#[derive(Clone, Debug, new)]
pub struct Reduce {
    axes: Option<Vec<i64>>,
    keep_dims: bool,
    reducer: Reducer,
}

impl Reduce {
    pub fn must_reduce(&self, ax: usize, rank: usize) -> bool {
        let resolved_axes: Option<Vec<usize>> = match &self.axes {
            None => None,
            Some(original_axes) => {
                let mut ans: Vec<usize> = vec![];
                for or_ax in original_axes.iter() {
                    ans.push(Self::resolve_axis(*or_ax, rank).unwrap());
                }
                Some(ans)
            }
        };

        resolved_axes.as_ref().map(|axes| axes.contains(&ax)).unwrap_or(true)
    }

    fn output_shape(&self, shape: &[TDim]) -> TVec<TDim> {
        shape
            .iter()
            .enumerate()
            .filter_map(|(ix, d)| {
                if self.must_reduce(ix, shape.len()) {
                    if self.keep_dims {
                        Some(1.to_dim())
                    } else {
                        None
                    }
                } else {
                    Some(d.clone())
                }
            })
            .collect()
    }

    fn resolve_axis(axis: i64, rank: usize) -> TractResult<usize> {
        if 0 <= axis && axis as usize <= rank - 1 {
            Ok(axis as usize)
        } else if -(rank as i64) <= axis && axis < 0 {
            Ok((axis + rank as i64) as usize)
        } else {
            bail!("Illegal combination of values for rank and axis: {} and {}", rank, axis)
        }
    }

    fn resolve_axes(&self, input_rank: usize) -> TractResult<TVec<usize>> {
        let mut axes: TVec<usize> = match self.axes.as_ref() {
            None => Ok((0..input_rank).collect()),
            Some(axis) => axis.iter().map(|&a| Self::resolve_axis(a, input_rank)).collect(),
        }?;
        axes.sort();
        Ok(axes)
    }
}

impl Op for Reduce {
    fn name(&self) -> Cow<str> {
        format!("Reduce<{:?}>", self.reducer).into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axes: {:?} keep_dims: {}", self.axes, self.keep_dims)])
    }
    not_a_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for Reduce {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let axes = self.resolve_axes(inputs[0].shape().len())?;
        let mut result = self.reducer.reduce(&*axes, args_1!(inputs))?;
        if !self.keep_dims {
            let mut final_shape: TVec<usize> = result.shape().into();
            for &ax in axes.iter().rev() {
                final_shape.remove(ax);
            }
            result = unsafe { result.into_shape(&*final_shape)? };
        }
        Ok(tvec!(result.into_arc_tensor()))
    }
}

impl InferenceRulesOp for Reduce {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        if self.keep_dims {
            s.equals(&inputs[0].rank, &outputs[0].rank)?;
        } else if let Some(axes) = self.axes.as_ref() {
            s.equals(inputs[0].rank.bex() - axes.len() as i32, &outputs[0].rank)?;
        } else {
            s.equals(&outputs[0].rank, 0)?;
        }
        s.given(&inputs[0].shape, move |s, shape| {
            let out_shape = self.output_shape(&*shape);
            s.equals(&outputs[0].shape, out_shape)
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
        let input = target.outlet_fact(mapping[&node.inputs[0]])?;
        let mut axes = self.resolve_axes(input.shape.rank())?;
        let mut wire = target.wire_node(
            &*node.name,
            TypedReduce::new(axes.clone(), self.reducer.clone()),
            [mapping[&node.inputs[0]]].as_ref(),
        )?;
        if !self.keep_dims {
            axes.sort();
            for axis in axes.into_iter().rev() {
                wire = target.wire_node(
                    format!("{}-dispose-dims-{}", node.name, axis),
                    AxisOp::Rm(axis),
                    &wire,
                )?;
            }
        }
        Ok(wire)
    }
}

