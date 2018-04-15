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
                    if !v.is_nan() {
                        P::ingest(&mut state, v);
                    }
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
        println!("new state");
        (0.0, 0)
    }
    fn ingest(state: &mut Self::State, v: f32) {
        state.0 += v;
        state.1 += 1;
        println!("ingested: {} -> {:?}", v, state);
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

    #[test]
    fn test_avgpool_1() {
        let pool = Pool::<AvgPooler>(
            LocalPatch {
                padding: Padding::Same,
                strides: vec![1, 1, 1, 1],
                _data_format: DataFormat::NHWC,
            },
            (1, 2),
            PhantomData,
        );
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
    ) -> BoxedStrategy<(Matrix, (usize, usize), String, usize)> {
        (1..ih, 1..iw, 1..ic)
            .prop_flat_map(move |(ih, iw, ic)| {
                (
                    Just((ih, iw, ic)),
                    (1..kh.min(ih + 1).max(2), 1..kw.min(iw + 1).max(2)),
                )
            })
            .prop_flat_map(|((ih, iw, ic), k)| {
                let i_size = iw * ih * ic;
                (
                    Just((1, ih, iw, ic)),
                    Just(k),
                    ::proptest::collection::vec(-255f32..255f32, i_size..i_size + 1),
                    prop_oneof!("VALID", "SAME"),
                    1..(k.0.min(k.1).max(2)),
                )
            })
            .prop_map(|(img_shape, k, img, padding, stride)| {
                (
                    Array::from_vec(img).into_shape(img_shape).unwrap().into(),
                    k,
                    padding,
                    stride,
                )
            })
            .boxed()
    }

    proptest! {
        #[test]
        fn maxpool((ref i, k, ref padding, stride) in img_and_pool(32, 32, 5, 16, 16)) {
            let graph = tfpb::graph()
                .node(placeholder_f32("data"))
                .node(tfpb::node()
                    .name("pool")
                    .op("MaxPool")
                    .input("data")
                    .attr("T", DT_FLOAT)
                    .attr("strides", vec![1, stride as i64, stride as i64, 1])
                    .attr("ksize", vec![1, k.0 as i64, k.1 as i64, 1])
                    .attr("padding", &**padding))
                .write_to_bytes()?;

            compare(&graph, vec!(("data", i.clone())), "pool")?;
        }
    }

    proptest! {
        #[test]
        fn avgpool((ref i, k, ref padding, stride) in img_and_pool(32, 32, 5, 16, 16)) {
            let graph = tfpb::graph()
                .node(placeholder_f32("data"))
                .node(tfpb::node()
                    .name("pool")
                    .op("AvgPool")
                    .input("data")
                    .attr("T", DT_FLOAT)
                    .attr("strides", vec![1, stride as i64, stride as i64, 1])
                    .attr("ksize", vec![1, k.0 as i64, k.1 as i64, 1])
                    .attr("padding", &**padding))
                .write_to_bytes()?;

            compare(&graph, vec!(("data", i.clone())), "pool")?;
        }
    }
}
