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
    let ksize = pb.get_attr().get("ksize").unwrap().get_list().get_i();
    Ok(Box::new(Pool::<P>(
        LocalPatch::build(pb)?,
        (ksize[1] as usize, ksize[2] as usize),
        PhantomData,
    )))
}

impl<P: Pooler + ::std::fmt::Debug> Op for Pool<P> {
    fn eval(&self, mut inputs: Vec<Input>) -> Result<Vec<Input>> {
        let m_input = args_1!(inputs);
        let data = m_input
            .into_matrix()
            .take_f32s()
            .ok_or("Expected a f32 matrix")?;
        let data = into_4d(data)?;
        let images = BatchImageWrapper(data.view());

        let (out_h, out_w) = self.0.adjusted_dim(images.h(), images.w(), self.1);

        let h_stride = self.0.strides[1];
        let w_stride = self.0.strides[2];
        let padded = self.0.pad(data.view(), self.1, ::std::f32::NAN)?;
        let data = padded.as_ref().map(|a| a.view()).unwrap_or(data.view());
        let out_shape = (images.count(), out_h, out_w, images.d());

        let transformed = Array4::from_shape_fn(out_shape, |(b, h, w, d)| {
            let mut state = P::state();
            for y in (h * h_stride)..(h * h_stride) + (self.1).0 {
                for x in (w * w_stride)..(w * w_stride) + (self.1).1 {
                    let v = data[(b, y, x, d)];
                    P::ingest(&mut state, v);
                }
            }
            P::digest(&mut state)
        });

        Ok(vec![Matrix::from(transformed.into_dyn()).into()])
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
        let pool = Pool::<MaxPooler>(
            LocalPatch {
                padding: Padding::Same,
                strides: vec![1, 1, 1, 1],
                _data_format: DataFormat::NHWC,
            },
            (2, 1),
            PhantomData,
        );
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
        let pool = Pool::<MaxPooler>(
            LocalPatch {
                padding: Padding::Same,
                strides: vec![1, 3, 3, 1],
                _data_format: DataFormat::NHWC,
            },
            (3, 3),
            PhantomData,
        );
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

}

#[cfg(all(test, feature = "tensorflow"))]
mod proptests {
    #![allow(non_snake_case)]
    use proptest::prelude::*;
    use ndarray::prelude::*;
    use protobuf::core::Message;
    use tfpb;
    use tfpb::types::DataType::DT_FLOAT;
    use ops::proptests::*;

    use Matrix;

    fn img_and_pool(
        ih: usize,
        iw: usize,
        ic: usize,
        kh: usize,
        kw: usize,
    ) -> BoxedStrategy<(Matrix, usize, usize)> {
        (1..ih, 1..iw, 1..ic, 1..kh, 1..kw)
            .prop_flat_map(|(ih, iw, ic, kh, kw)| {
                let i_size = iw * ih * ic;
                (
                    Just((1, ih, iw, ic)),
                    Just(kh),
                    Just(kw),
                    ::proptest::collection::vec(-255f32..255f32, i_size..i_size + 1),
                )
            })
            .prop_map(|(img_shape, kh, kw, img)| {
                (
                    Array::from_vec(img).into_shape(img_shape).unwrap().into(),
                    kw,
                    kh,
                )
            })
            .boxed()
    }

    fn maxpool_pb(
        v_stride: usize,
        h_stride: usize,
        kw: usize,
        kh: usize,
        valid: bool,
    ) -> ::Result<Vec<u8>> {
        let pool = tfpb::node()
            .name("pool")
            .op("MaxPool")
            .input("data")
            .attr("T", DT_FLOAT)
            .attr("strides", vec![1, v_stride as i64, h_stride as i64, 1])
            .attr("ksize", vec![1, kw as i64, kh as i64, 1])
            .attr("padding", if valid { "VALID" } else { "SAME" });

        let graph = tfpb::graph().node(placeholder_f32("data")).node(pool);

        Ok(graph.write_to_bytes()?)
    }

    proptest! {
        #[test]
        fn maxpool((ref i, kh, kw) in img_and_pool(32, 32, 5, 16, 16),
                           valid in ::proptest::bool::ANY,
                           stride in 1usize..4) {
            prop_assume!(stride <= kh);
            prop_assume!(stride <= kw);
            if valid {
                prop_assume!(i.shape()[1] >= kh);
                prop_assume!(i.shape()[2] >= kw);
            }
            let model = maxpool_pb(stride, stride, kh, kw, valid).unwrap();
            let mut tf = ::tf::for_slice(&model)?;
            let tfd = ::Model::for_reader(&*model)?;
            let data = tfd.node_id_by_name("data").unwrap();
            let pool = tfd.node_id_by_name("pool").unwrap();
            let mut tfds = tfd.state();
            let expected = tf.run(vec!(("data", i.clone())), "pool")?;
            tfds.set_value(data, i.clone())?;
            tfd.plan_for_one(pool).unwrap().run(&mut tfds).unwrap();
            let found = tfds.take(pool)?;
            prop_assert!(expected[0].close_enough(&found[0]), "expected: {:?} found: {:?}", expected, found)
        }
    }
}
