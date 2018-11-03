use ndarray::prelude::*;
use tract_core::ops::prelude::*;

pub fn build(pb: &::tfpb::node_def::NodeDef) -> TractResult<Box<Op>> {
    let n = pb.get_attr_int("N")?;
    let t = pb.get_attr_datum_type("T")?;
    let tidx = pb.get_attr_datum_type("Tidx")?;
    Ok(boxed_new!(ConcatV2(t)(n, tidx)))
}

#[derive(Debug, Clone, new)]
pub struct ConcatV2<T: Datum> {
    n: usize,
    tidx: DatumType,
    t: PhantomData<T>,
}

impl<T: Datum> StatelessOp for ConcatV2<T> {
    fn eval(&self, mut inputs: TVec<Value>) -> TractResult<TVec<Value>> {
        let axis: i32 = inputs
            .pop()
            .and_then(|t| t.as_i32())
            .ok_or("Expected a i32 scalar")?;
        let mats: TractResult<Vec<ArrayViewD<T>>> =
            inputs.iter().map(|mat| mat.to_array_view()).collect();
        let result = ::ndarray::stack(Axis(axis as usize), &*mats?)?;
        Ok(tvec![result.into()])
    }
}

impl<T: Datum> Op for ConcatV2<T> {
    fn name(&self) -> &str {
        "tf.ConvatV2"
    }
}

impl<T: Datum> InferenceRulesOp for ConcatV2<T> {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        let n = self.n;
        s.equals(&inputs.len, n as i32 + 1)?;
        s.equals(&outputs.len, 1)?;
        s.equals_all((0..self.n).map(|i| (&inputs[i].datum_type).bex()).collect())?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&inputs[n].datum_type, DatumType::I32)?;
        s.equals_all((0..self.n).map(|i| (&inputs[i].rank).bex()).collect())?;
        s.equals(&inputs[n].rank, 0)?;
        s.equals(&outputs[0].rank, &inputs[0].rank)?;
        s.given(&inputs[n].value, move |s, axis: Tensor| {
            let axis = axis.as_i32().unwrap() as usize; // checked
            trace!("axis for Concatv2: {}", axis);
            for d in 0..axis {
                s.equals_all((0..n).map(|i| (&inputs[i].shape[d]).bex()).collect())?;
            }
            for d in 0..axis {
                s.equals(&inputs[0].shape[d], &outputs[0].shape[d])?;
            }
            s.given(&inputs[0].rank, move |s, rank| {
                trace!("Given rank {}", rank);
                for d in (axis + 1)..(rank as usize) {
                    s.equals(&inputs[0].shape[d], &outputs[0].shape[d])?;
                }
                for d in (axis + 1)..(rank as usize) {
                    s.equals_all((0..n).map(|i| (&inputs[i].shape[d]).bex()).collect())?;
                }
                Ok(())
            })?;

            let mut concat_dim = -1 * outputs[0].shape[axis].bex();
            for i in 0..n {
                concat_dim = concat_dim + inputs[i].shape[axis].bex();
            }
            s.equals_zero(concat_dim)
        })
    }
}
