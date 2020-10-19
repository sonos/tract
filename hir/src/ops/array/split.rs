use crate::infer::*;
use crate::internal::*;

#[derive(Debug, Clone, new, Default, Hash)]
pub struct Split {
    axis: isize,
    outputs: usize,
    split: Option<Vec<usize>>,
}

tract_data::impl_dyn_hash!(Split);

impl Split {
    fn split_dims<D: DimLike>(&self, input: &D) -> TractResult<TVec<D>> {
        if let Some(ref split) = self.split.as_ref() {
            Ok(split.iter().map(|&d| D::from(d)).collect())
        } else {
            Ok(tvec!(input.clone()/self.outputs;self. outputs))
        }
    }
}

impl Expansion for Split {
    fn name(&self) -> Cow<str> {
        "Split".into()
    }

    op_hir!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, self.outputs)?;
        (0..self.outputs).try_for_each(|i| {
            s.equals(&inputs[0].datum_type, &outputs[i].datum_type)?;
            s.equals(&inputs[0].rank, &outputs[i].rank)
        })?;
        s.given(&inputs[0].shape, move |s, shape| {
            let axis =
                if self.axis < 0 { self.axis + shape.len() as isize } else { self.axis } as usize;
            let dims = self.split_dims(&shape[axis])?;
            for i in 0..self.outputs {
                let mut shape = shape.clone();
                shape[axis] = dims[i].clone();
                s.equals(&outputs[i].shape, shape)?;
            }
            Ok(())
        })?;
        Ok(())
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(self.outputs)
    }

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let input = target.outlet_fact(inputs[0])?.clone();
        let mut outputs = tvec!();
        let mut current = 0.to_dim();
        let axis =
            if self.axis < 0 { self.axis + input.rank() as isize } else { self.axis } as usize;
        for len in self.split_dims(&input.shape[axis])? {
            let end = current.clone() + len;
            outputs.push(
                target.wire_node(
                    format!("{}.axis_{}_{}..{}", prefix, axis, current, end),
                    crate::ops::array::Slice::new(axis, current, end.clone()),
                    inputs,
                )?[0],
            );
            current = end;
        }
        Ok(outputs)
    }
}
