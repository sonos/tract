use crate::internal::*;
use tract_ndarray::prelude::*;

#[derive(Debug, Clone, Hash, PartialEq)]
pub struct QSumB {
    pub r: usize,
    pub n: usize,
    pub k: usize,
}

impl_dyn_hash!(QSumB);

impl Op for QSumB {
    fn name(&self) -> Cow<str> {
        "QSumB".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("r:{}, n:{}, k:{}", self.r, self.n, self.k)])
    }

    op_core_lir!();
    impl_op_same_as!();
    op_as_typed_op!();
}

impl EvalOp for QSumB {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let mut shape: TVec<usize> = input.shape().into();
        shape[input.rank() - 1] = self.n;
        let mut output = ArrayD::zeros(&*shape);
        for prefix in tract_ndarray::indices(&shape[0..output.ndim() - 1]) {
            let mut panel = input.to_array_view::<i8>()?;
            for (ix, d) in prefix.slice().iter().enumerate() {
                panel.index_axis_inplace(Axis(ix), *d);
                output.index_axis_inplace(Axis(ix), *d);
                let panel = panel.as_slice().unwrap();
                for p in 0..(self.n.div_ceil(self.r)) {
                    let mut vec = vec![0i32; self.r];
                    for k in 0..self.k {
                        for r in 0..self.r {
                            vec[r] += panel[k * self.r + r] as i32;
                        }
                    }
                    let r_slice = &mut output.as_slice_mut().unwrap()[p * self.r..];
                    let len = r_slice.len().min(self.r);
                    r_slice.copy_from_slice(&vec[..len]);
                }
            }
        }
        Ok(tvec!(output.into_arc_tensor()))
    }
}

impl TypedOp for QSumB {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut shape: ShapeFact = inputs[0].shape.clone();
        shape.set(shape.rank() - 1, self.n.to_dim());
        Ok(tvec!(TypedFact::shape::<i32, _>(shape)))
    }
}
