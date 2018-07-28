use std::collections::HashMap;

use super::local_patch::*;
use ops::prelude::*;
use analyser::interface::*;
use ndarray::prelude::*;

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
    fn eval(&self, mut inputs: Vec<TensorView>) -> Result<Vec<TensorView>> {
        let m_input = args_1!(inputs);
        let data = m_input
            .into_tensor()
            .take_f32s()
            .ok_or("Expected a f32 matrix")?;
        let data = into_4d(data)?;
        let images = BatchImageWrapper(data.view());

        let (out_h, out_w) = self.0.adjusted_dim(images.h(), images.w(), self.1);

        let padded = self.0.pad(data.view(), self.1, ::std::f32::NAN, true, true)?;
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

        Ok(vec![Tensor::from(transformed.into_dyn()).into()])
    }

    /// Returns the attributes of the operation and their values.
    fn get_attributes(&self) -> HashMap<&'static str, Attr> {
        let mut attributes = hashmap!{
            "ksize" => Attr::UsizeVec(vec![(self.1).0, (self.1).1]),
        };

        attributes.extend(self.0.get_attributes());
        attributes
    }

}

impl<P: Pooler + ::std::fmt::Debug> InferenceRulesOp for Pool<P> {
    /// Registers the inference rules of the operator.
    fn rules<'r, 'p: 'r, 's: 'r>(&'s self, solver: &mut Solver<'r>, inputs: &'p TensorsProxy, outputs: &'p TensorsProxy) {
        solver
            .equals(&inputs.len, 1)
            .equals(&outputs.len, 1)
            .equals(&inputs[0].datatype, DataType::DT_FLOAT)
            .equals(&outputs[0].datatype, DataType::DT_FLOAT)
            .equals(&inputs[0].rank, 4)
            .equals(&outputs[0].rank, 4)
            .equals(&inputs[0].shape[0], &outputs[0].shape[0])
            .given(&inputs[0].shape[1], move |solver, h| {
                solver.given(&inputs[0].shape[2], move |solver, w| {
                    let (oh, ow) = self.0.adjusted_dim(h, w, self.1);
                    solver
                        .equals(&outputs[0].shape[1], oh as isize)
                        .equals(&outputs[0].shape[2], ow as isize);
                });
            })
            ;
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
        let found = pool.eval(vec![data.into()]).unwrap();

        assert!(
            exp.close_enough(&found[0]),
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
        let found = pool.eval(vec![data.into()]).unwrap();

        assert!(
            exp.close_enough(&found[0]),
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
        let found = pool.eval(vec![data.into()]).unwrap();

        assert!(
            exp.close_enough(&found[0]),
            "expected: {:?} found: {:?}",
            exp,
            found[0]
        )
    }

}
