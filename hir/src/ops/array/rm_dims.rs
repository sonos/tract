use crate::infer::*;
use crate::internal::*;
use tract_itertools::Itertools;

#[derive(Debug, Clone, new, Hash)]
pub struct RmDims {
    pub axes: Vec<isize>,
}

tract_data::impl_dyn_hash!(RmDims);

impl RmDims {
    fn compute_shape<D: DimLike>(&self, input: &[D]) -> TVec<D> {
        let axes = self
            .axes
            .iter()
            .map(|&a| if a < 0 { a + input.len() as isize } else { a } as usize)
            .collect::<Vec<_>>();
        input
            .iter()
            .enumerate()
            .filter(|(ix, _d)| !axes.contains(ix))
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
        s.equals(&outputs[0].rank, (&inputs[0].rank).bex() - self.axes.len() as i64)?;
        s.given(&inputs[0].rank, move |s, rank| {
            for axis in &self.axes {
                let axis = if *axis < 0 { axis + rank as isize } else { *axis } as usize;
                s.equals(&inputs[0].shape[axis], 1.to_dim())?;
            }
            Ok(())
        })?;
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
        let rank = target.outlet_fact(inputs[0])?.rank();
        let axes = self
            .axes
            .iter()
            .map(|&a| if a < 0 { a + rank as isize } else { a } as usize)
            .sorted()
            .rev();
        for axis in axes {
            wire =
                target.wire_node(format!("{}.axis-{}", prefix, axis), AxisOp::Rm(axis), &[wire])?
                    [0];
        }
        Ok(tvec!(wire))
    }
}
