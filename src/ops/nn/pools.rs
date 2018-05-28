use analyser::ATensor;
use {Matrix, Result};
use super::{Input, Op};
use ndarray::prelude::*;
use std::marker::PhantomData;
use super::local_patch::*;

pub trait Pooler: Send + Sync + ::std::fmt::Debug + 'static {
    type State;
    fn state() -> Self::State;
    fn ingest(state: &mut Self::State, v: f32);
    fn digest(state: &mut Self::State) -> f32;
}

#[derive(Debug)]
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
    fn eval(&self, mut inputs: Vec<Input>) -> Result<Vec<Input>> {
        let m_input = args_1!(inputs);
        let data = m_input
            .into_matrix()
            .take_f32s()
            .ok_or("Expected a f32 matrix")?;
        let data = into_4d(data)?;
        let images = BatchImageWrapper(data.view());

        let (out_h, out_w) = self.0.adjusted_dim(images.h(), images.w(), self.1);

        let padded = self.0.pad(data.view(), self.1, ::std::f32::NAN)?;
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

        Ok(vec![Matrix::from(transformed.into_dyn()).into()])
    }

    /// Infers properties about the output tensors from the input tensors.
    fn infer_forward(&self, inputs: Vec<&ATensor>) -> Result<Vec<ATensor>> {
        if inputs.len() != 2 {
            bail!("Pool operations only supports one input.");
        }

        try_infer_forward_concrete!(self, &inputs);

        // If we don't know the actual value, we can still compute the shape.
        let shape = match inputs[0].shape.concretize()?.as_slice() {
            // TODO(liautaud): Take the data_format parameter into account.
            [batch, in_height, in_width, in_channels] => {
                let (height, width) = self.0.adjusted_dim(*in_height, *in_width, self.1);
                ashape![(*batch), height, width, (*in_channels)]
            },

            _ => bail!("The input dimensions are invalid.")
        };

        let output = ATensor {
            datatype: inputs[0].datatype.clone(),
            shape,
            value: avalue!(_),
        };

        Ok(vec![output])
    }

    /// Infers properties about the input tensors from the output tensors.
    fn infer_backward(&self, outputs: Vec<&ATensor>) -> Result<Vec<ATensor>> {
        if outputs.len() != 1 {
            bail!("Pool operations only supports one output.");
        }

        let shape = match outputs[0].shape.concretize()?.as_slice() {
            // TODO(liautaud): Take the data_format parameter into account.
            [batch, _, _, out_channels] =>
                ashape![(*batch), _, _, (*out_channels)],
            _ => bail!("The output dimensions are invalid.")
        };

        let input = ATensor {
            datatype: outputs[0].datatype.clone(),
            shape,
            value: avalue!(_)
        };

        Ok(vec![input])
    }
}

#[derive(Debug)]
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

#[derive(Debug)]
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
    use Matrix;
    use super::*;

    #[test]
    fn test_maxpool_1() {
        let pool = Pool::<MaxPooler>(LocalPatch::same(1, 1), (2, 1), PhantomData);
        let data = Matrix::f32s(&[1, 1, 1, 1], &[-1.0]).unwrap();
        let exp: Matrix = Matrix::f32s(&[1, 1, 1, 1], &[-1.0]).unwrap();
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
        let data = Matrix::f32s(&[1, 2, 4, 1], &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let exp: Matrix = Matrix::f32s(&[1, 1, 2, 1], &[1.0, 0.0]).unwrap();
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
        let data = Matrix::f32s(&[1, 1, 2, 1], &[0.0, 0.0]).unwrap();
        let exp: Matrix = Matrix::f32s(&[1, 1, 2, 1], &[0.0, 0.0]).unwrap();
        let found = pool.eval(vec![data.into()]).unwrap();

        assert!(
            exp.close_enough(&found[0]),
            "expected: {:?} found: {:?}",
            exp,
            found[0]
        )
    }

}
