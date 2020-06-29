use crate::infer::*;
use crate::internal::*;

#[derive(Debug, Clone, new, Hash)]
pub struct RmDims {
    pub axes: Vec<usize>,
}

tract_linalg::impl_dyn_hash!(RmDims);

impl RmDims {
    fn compute_shape<D: DimLike>(&self, input: &[D]) -> TVec<D> {
        input
            .iter()
            .enumerate()
            .filter(|(ix, _d)| !self.axes.contains(ix))
            .map(|(_ix, d)| d.clone())
            .collect()
    }
}

impl Expansion for RmDims {
    fn name(&self) -> Cow<str> {
        "RmDims".into()
    }

    op_hir!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&outputs[0].rank, (&inputs[0].rank).bex() - self.axes.len() as i32)?;
        for axis in &self.axes {
            s.equals(&inputs[0].shape[*axis], 1.to_dim())?;
        }
        s.given(&inputs[0].shape, move |s, shape| {
            let output_shape = self.compute_shape(&shape);
            s.equals(&outputs[0].shape, output_shape)
        })
    }

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let mut wire = inputs[0];
        let mut axes = self.axes.clone();
        axes.sort();
        for axis in axes.into_iter().rev() {
            wire =
                target.wire_node(format!("{}.axis-{}", prefix, axis), AxisOp::Rm(axis), &[wire])?
                    [0];
        }
        Ok(tvec!(wire))
    }
}
