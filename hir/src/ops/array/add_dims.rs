use crate::infer::*;
use crate::internal::*;
use tract_itertools::Itertools;

#[derive(Debug, Clone, new, Hash)]
pub struct AddDims {
    pub axes: Vec<isize>,
}

tract_data::impl_dyn_hash!(AddDims);

impl AddDims {
    fn compute_shape<D: DimLike>(&self, input: &[D]) -> TVec<D> {
        let rank = input.len() as isize;
        let mut shape: TVec<D> = input.iter().cloned().collect();
        let axes = self
            .axes
            .iter()
            .map(|&axis| if axis < 0 { axis + rank } else { axis } as usize)
            .sorted();
        for axis in axes {
            shape.insert(axis, D::one())
        }
        shape
    }
}

impl Expansion for AddDims {
    fn name(&self) -> Cow<str> {
        "AddDims".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("Axes: {:?}", self.axes)])
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
        s.equals(&outputs[0].rank, (&inputs[0].rank).bex() + self.axes.len() as i64)?;
        s.given(&inputs[0].shape, move |s, shape| {
            let output_shape = self.compute_shape(&shape);
            s.equals(&outputs[0].shape, output_shape)
        })
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let rank = model.outlet_fact(inputs[0])?.rank() as isize;
        let mut wire: TVec<OutletId> = inputs.into();
        let axes = self
            .axes
            .iter()
            .map(|&axis| if axis < 0 { axis + rank } else { axis } as usize)
            .sorted();
        for axis in axes {
            wire =
                model.wire_node(format!("{}.axis-{}", prefix, axis), AxisOp::Add(axis), &wire)?;
        }
        Ok(wire)
    }
}
