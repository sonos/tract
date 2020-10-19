use crate::infer::*;
use crate::internal::*;

#[derive(Debug, Clone, new, Hash)]
pub struct PermuteAxes {
    pub axes: Option<TVec<usize>>,
}

tract_data::impl_dyn_hash!(PermuteAxes);

impl PermuteAxes {
    fn compute_shape<D: DimLike>(&self, input: &[D]) -> TractResult<TVec<D>> {
        if let Some(ref axes) = self.axes {
            if input.len() != axes.len() {
                bail!(
                    "Op expects tensor of rank {}, input is actually of rank {}.",
                    axes.len(),
                    input.len()
                );
            }
            let mut new_shape = tvec![D::zero(); input.len()];
            for (ix, &d) in axes.iter().enumerate() {
                new_shape[ix] = input[d].clone();
            }
            Ok(new_shape)
        } else {
            let mut new_shape: TVec<D> = input.iter().cloned().collect();
            new_shape.reverse();
            Ok(new_shape)
        }
    }
}

impl Expansion for PermuteAxes {
    fn name(&self) -> Cow<str> {
        "PermuteAxes".into()
    }

    op_hir!();

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("{:?}", self.axes)])
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&outputs[0].rank, &inputs[0].rank)?;
        s.given(&inputs[0].shape, move |s, shape| {
            let output_shape = self.compute_shape(&shape)?;
            s.equals(&outputs[0].shape, output_shape)
        })?;
        if let Some(axes) = &self.axes {
            s.equals(&outputs[0].rank, axes.len() as i64)?;
        }
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let fact = target.outlet_fact(inputs[0])?;
        let axes = if let Some(axes) = &self.axes {
            if fact.rank() != axes.len() {
                bail!(
                    "Op expects tensor of rank {}, input is actually of rank {}.",
                    axes.len(),
                    fact.rank()
                );
            }
            axes.clone()
        } else {
            (0..fact.rank()).rev().collect()
        };
        let mut wire: TVec<OutletId> = inputs.into();
        for (ix, op) in perm_to_ops(&axes).into_iter().enumerate() {
            wire = target.wire_node(
                format!("{}.{}-{}", prefix, op.name(), ix),
                op,
                &wire,
            )?;
        }
        Ok(wire)
    }
}
