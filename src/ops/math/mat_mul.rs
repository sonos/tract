use analyser::rules::prelude::*;
use ndarray::*;
use ops::prelude::*;

#[derive(Debug, Clone, new, Default)]
pub struct MatMul {}

impl MatMul {
    fn eval_t<T: Datum + LinalgScalar>(&self, a: Value, b: Value) -> TfdResult<Value> {
        let a = a.to_array_view::<T>()?;
        let b = b.to_array_view::<T>()?;
        let mut ashape = a.shape().to_vec();
        let mut bshape = b.shape().to_vec();
        if ashape.len() < 2 {
            ashape.insert(0, 1);
        }
        if bshape.len() < 2 {
            bshape.push(1);
        }
        let cshape_prefix = ::broadcast::multi_broadcast(&[
            &ashape[..(ashape.len() - 2)],
            &bshape[..(bshape.len() - 2)],
        ]).ok_or("Could not broadcast")?;
        let mut cshape: Vec<usize> = cshape_prefix.clone();
        cshape.push(ashape[ashape.len() - 2]);
        cshape.push(bshape[bshape.len() - 1]);
        let mut c = unsafe { Array::uninitialized(&*cshape) };
        for prefix in indices(&cshape_prefix[..]).into_iter() {
            let mut a_slice: Vec<SliceOrIndex> = prefix
                .slice()
                .iter()
                .map(|&ix| SliceOrIndex::Index(ix as _))
                .collect();
            let mut b_slice: Vec<SliceOrIndex> = prefix
                .slice()
                .iter()
                .map(|&ix| SliceOrIndex::Index(ix as _))
                .collect();
            let mut c_slice: Vec<SliceOrIndex> = prefix
                .slice()
                .iter()
                .map(|&ix| SliceOrIndex::Index(ix as _))
                .collect();
            a_slice.push(SliceOrIndex::Slice {
                start: 0,
                end: None,
                step: 1,
            });
            a_slice.push(SliceOrIndex::Slice {
                start: 0,
                end: None,
                step: 1,
            });
            b_slice.push(SliceOrIndex::Slice {
                start: 0,
                end: None,
                step: 1,
            });
            b_slice.push(SliceOrIndex::Slice {
                start: 0,
                end: None,
                step: 1,
            });
            c_slice.push(SliceOrIndex::Slice {
                start: 0,
                end: None,
                step: 1,
            });
            c_slice.push(SliceOrIndex::Slice {
                start: 0,
                end: None,
                step: 1,
            });
            linalg::general_mat_mul(
                T::one(),
                &a.slice(SliceInfo::new(a_slice)?.as_ref()),
                &b.slice(SliceInfo::new(b_slice)?.as_ref()),
                T::zero(),
                &mut c.slice_mut(SliceInfo::new(c_slice)?.as_ref()),
            );
        }
        Ok(c.into())
    }
}

impl Op for MatMul {
    fn name(&self) -> &str {
        "MatMul"
    }

    fn eval(&self, mut inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        let (a, b) = args_2!(inputs);
        let c = dispatch_floatlike!(Self::eval_t(a.datum_type())(self, a, b))?;
        Ok(tvec!(c))
    }
}

impl InferenceRulesOp for MatMul {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 2)?;
        s.equals(&outputs.len, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[1].datum_type, &outputs[0].datum_type)?;
        s.given_2(
            &inputs[0].shape,
            &inputs[1].shape,
            move |s, mut ashape, mut bshape| {
                if ashape.len() < 2 {
                    ashape.insert(0, 1.to_dim());
                }
                if bshape.len() < 2 {
                    bshape.push(1.to_dim());
                }
                let mut cshape = ::broadcast::multi_broadcast(&[
                    &ashape[..(ashape.len() - 2)],
                    &bshape[..(bshape.len() - 2)],
                ]).ok_or("Could not broadcast")?;
                cshape.push(ashape[ashape.len() - 2]);
                cshape.push(bshape[bshape.len() - 1]);
                s.equals(&outputs[0].shape, cshape)
            },
        )?;
        Ok(())
    }
}
