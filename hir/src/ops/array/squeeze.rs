use crate::infer::*;
use crate::internal::*;

use super::RmDims;

#[derive(Debug, Clone, new, Default, Hash)]
pub struct Squeeze {
    axes: Option<Vec<isize>>,
}

tract_data::impl_dyn_hash!(Squeeze);

impl Squeeze {
    fn compute_shape<D: DimLike>(&self, input: &[D]) -> TractResult<TVec<D>> {
        if let Some(ref axes) = self.axes {
            let axes = axes
                .iter()
                .map(|&a| if a < 0 { a + input.len() as isize } else { a } as usize)
                .collect::<Vec<_>>();
            let mut shape: TVec<D> = input.iter().cloned().collect();
            for &axis in axes.iter().rev() {
                if shape.remove(axis) != D::one() {
                    bail!(
                        "Attempt to squeeze an axis which dimension is not one {:?}, {:?}",
                        self,
                        input
                    );
                }
            }
            Ok(shape)
        } else {
            Ok(input.into_iter().filter(|&d| d != &D::one()).cloned().collect())
        }
    }
}

impl Expansion for Squeeze {
    fn name(&self) -> Cow<str> {
        "Squeeze".into()
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
        if let Some(ref axes) = self.axes {
            s.equals(&outputs[0].rank, (&inputs[0].rank).bex() - axes.len() as i64)?;
        }
        s.given(&inputs[0].shape, move |s, shape| {
            let output_shape = self.compute_shape(&shape)?;
            s.equals(&outputs[0].shape, output_shape)
        })
    }

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let input = inputs[0];
        let axes = if let Some(axes) = &self.axes {
            axes.clone()
        } else {
            let input_fact = target.outlet_fact(input)?;
            input_fact
                .shape
                .iter()
                .enumerate()
                .filter(|(_ix, d)| d == &1.to_dim())
                .map(|(ix, _d)| ix as isize)
                .collect()
        };
        RmDims::new(axes).wire(prefix, target, inputs)
    }
}
