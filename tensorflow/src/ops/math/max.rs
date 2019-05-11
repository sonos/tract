use tract_core::internal::*;

#[derive(Debug, Clone, new)]
pub struct Max {
    t: DatumType,
    t_idx: DatumType,
    keep_dims: bool,
}

pub fn max(pb: &crate::tfpb::node_def::NodeDef) -> TractResult<Box<Op>> {
    let t = pb.get_attr_datum_type("T")?;
    let t_idx = pb.get_attr_datum_type("Tidx")?;
    let keep_dims = pb.get_attr_bool("keep_dims")?;
    Ok(Box::new(Max::new(t, t_idx, keep_dims)))
}

impl Max {
    fn eval_t<T>(
        &self,
        input: Arc<Tensor>,
        full_output_shape: TVec<usize>,
        axes: TVec<usize>,
    ) -> TractResult<TVec<Arc<Tensor>>>
    where
        T: Copy + Datum + PartialOrd + num_traits::Bounded,
    {
        use ndarray::*;
        let input = input.to_array_view::<T>()?;
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
        if !self.keep_dims {
            for ax in (0..full_output_shape.len()).rev() {
                if axes.contains(&ax) {
                    result = result.index_axis_move(Axis(ax), 0);
                }
            }
        }
        Ok(tvec!(result.into_arc_tensor()))
    }
}

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
        let full_output_shape: TVec<usize> = input
            .shape()
            .iter()
            .enumerate()
            .map(|(ax, &d)| if axes.contains(&ax) { 1 } else { d })
            .collect();
        dispatch_numbers!(Self::eval_t(self.t)(self, input, full_output_shape, axes))
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
}
