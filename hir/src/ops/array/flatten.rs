use crate::infer::*;
use crate::internal::*;

#[derive(Debug, Clone, new, Default, Hash)]
pub struct Flatten {
    pub axis: i64,
}

impl Flatten {
    pub fn compute_shape<D: DimLike>(&self, shape: &[D]) -> TractResult<[D; 2]> {
        if shape.iter().filter(|d| d.to_usize().is_err()).count() > 1 {
            bail!("Can not compute a shape with square of symbols")
        }
        let axis = if self.axis >= 0 { self.axis } else { self.axis + shape.len() as i64 } as usize;
        Ok([shape[..axis].iter().cloned().product::<D>(), shape[axis..].iter().cloned().product()])
    }
}

impl Expansion for Flatten {
    fn name(&self) -> Cow<str> {
        "Flatten".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.given(&inputs[0].shape, move |s, shape| {
            let [shape_0, shape_1] = self.compute_shape(&shape)?;
            s.equals(&outputs[0].shape, ShapeFactoid::from(vec![shape_0, shape_1]))
        })
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let input_shape = model.outlet_fact(inputs[0])?.shape.to_tvec();
        let output_shape = self.compute_shape(&input_shape)?;
        let mut wire = tvec!(inputs[0]);
        for (ix, op) in
            tract_core::ops::change_axes::to_axis_ops_with_tf_rules(&input_shape, &output_shape)?
                .into_iter()
                .enumerate()
        {
            wire = model.wire_node(format!("{prefix}.{ix}"), op, &wire)?;
        }
        Ok(wire)
    }
}
