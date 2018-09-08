use super::local_patch::*;
use analyser::rules::prelude::*;
use ndarray::prelude::*;
use ops::prelude::*;

pub trait Pooler: Send + Sync + ::std::clone::Clone + ::std::fmt::Debug + 'static {
    type State;
    fn state() -> Self::State;
    fn ingest(state: &mut Self::State, v: f32);
    fn digest(state: &mut Self::State) -> f32;
}

#[derive(Debug, Clone)]
pub struct Pool<P: Pooler>(LocalPatch, (usize, usize), PhantomData<P>);

pub fn pool<P: Pooler>(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
    let ksize: Vec<usize> = pb.get_attr_list_int("ksize")?;
    Ok(Box::new(Pool::<P>(
        LocalPatch::build(pb)?,
        (ksize[1], ksize[2]),
        PhantomData,
    )))
}

impl<P: Pooler + ::std::fmt::Debug> Op for Pool<P> {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Value>) -> Result<TVec<Value>> {
        let m_input = args_1!(inputs);
        let data = m_input
            .into_tensor()
            .take_f32s()
            .ok_or("Expected a f32 matrix")?;
        let data = into_4d(data)?;
        let images = BatchImageWrapper(data.view());

        let out_h = self
            .0
            .adjusted_rows(images.h().into(), (self.1).0)
            .to_integer()? as usize;
        let out_w = self
            .0
            .adjusted_cols(images.w().into(), (self.1).1)
            .to_integer()? as usize;

        let padded = self
            .0
            .pad(data.view(), self.1, ::std::f32::NAN, true, true)?;
        let data = padded.as_ref().map(|a| a.view()).unwrap_or(data.view());
        let out_shape = (images.count(), out_h, out_w, images.d());

        let transformed = Array4::from_shape_fn(out_shape, |(b, h, w, d)| {
            let mut state = P::state();
            for y in (h * self.0.v_stride)..(h * self.0.v_stride) + (self.1).0 {
                for x in (w * self.0.h_stride)..(w * self.0.h_stride) + (self.1).1 {
                    let v = data[(b, y, x, d)];
                    if !v.is_nan() {
                        P::ingest(&mut state, v);
                    }
                }
            }
            P::digest(&mut state)
        });

        Ok(tvec![Tensor::from(transformed.into_dyn()).into()])
    }
}

impl<P: Pooler + ::std::fmt::Debug> InferenceRulesOp for Pool<P> {
    /// Registers the inference rules of the operator.
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) {
        solver
            .equals(&inputs.len, 1)
            .equals(&outputs.len, 1)
            .equals(&inputs[0].datum_type, DatumType::F32)
            .equals(&outputs[0].datum_type, DatumType::F32)
            .equals(&inputs[0].rank, 4)
            .equals(&outputs[0].rank, 4)
            .equals(&inputs[0].shape[0], &outputs[0].shape[0])
            .given(&inputs[0].shape[1], move |solver, h| {
                solver.given(&inputs[0].shape[2], move |solver, w| {
                    let oh = self.0.adjusted_rows(h, (self.1).0);
                    let ow = self.0.adjusted_cols(w, (self.1).1);
                    solver
                        .equals(&outputs[0].shape[1], oh)
                        .equals(&outputs[0].shape[2], ow);
                });
            });
    }
}

#[derive(Debug, Clone)]
pub struct MaxPooler;
impl Pooler for MaxPooler {
    type State = f32;
    fn state() -> f32 {
        ::std::f32::NEG_INFINITY
    }
    fn ingest(state: &mut Self::State, v: f32) {
        if v > *state {
            *state = v
        }
    }
    fn digest(state: &mut Self::State) -> f32 {
        *state
    }
}

#[derive(Debug, Clone)]
pub struct AvgPooler;
impl Pooler for AvgPooler {
    type State = (f32, usize);
    fn state() -> (f32, usize) {
        (0.0, 0)
    }
    fn ingest(state: &mut Self::State, v: f32) {
        state.0 += v;
        state.1 += 1;
    }
    fn digest(state: &mut Self::State) -> f32 {
        state.0 / state.1 as f32
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use super::*;
    use Tensor;

    #[test]
    fn test_maxpool_1() {
        let pool = Pool::<MaxPooler>(LocalPatch::same(1, 1), (2, 1), PhantomData);
        let data = Tensor::f32s(&[1, 1, 1, 1], &[-1.0]).unwrap();
        let exp: Tensor = Tensor::f32s(&[1, 1, 1, 1], &[-1.0]).unwrap();
        let found = pool.eval(tvec![data.into()]).unwrap();

        assert!(
            exp.close_enough(&found[0], false),
            "expected: {:?} found: {:?}",
            exp,
            found[0]
        )
    }

    #[test]
    fn test_maxpool_2() {
        let pool = Pool::<MaxPooler>(LocalPatch::same(3, 3), (3, 3), PhantomData);
        let data = Tensor::f32s(&[1, 2, 4, 1], &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let exp: Tensor = Tensor::f32s(&[1, 1, 2, 1], &[1.0, 0.0]).unwrap();
        let found = pool.eval(tvec![data.into()]).unwrap();

        assert!(
            exp.close_enough(&found[0], true),
            "expected: {:?} found: {:?}",
            exp,
            found[0]
        )
    }

    #[test]
    fn test_avgpool_1() {
        let pool = Pool::<AvgPooler>(LocalPatch::same(1, 1), (1, 2), PhantomData);
        let data = Tensor::f32s(&[1, 1, 2, 1], &[0.0, 0.0]).unwrap();
        let exp: Tensor = Tensor::f32s(&[1, 1, 2, 1], &[0.0, 0.0]).unwrap();
        let found = pool.eval(tvec![data.into()]).unwrap();

        assert!(
            exp.close_enough(&found[0], true),
            "expected: {:?} found: {:?}",
            exp,
            found[0]
        )
    }

}
