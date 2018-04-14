use {Matrix, Result};
use super::{Input, Op};
use ndarray::prelude::*;
use super::local_patch::*;

#[derive(Debug)]
pub struct Conv2D(LocalPatch);

impl Conv2D {
    pub fn build(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        Self::for_patch(LocalPatch::build(pb)?)
    }

    pub fn for_patch(patch: LocalPatch) -> Result<Box<Op>> {
        Ok(Box::new(Conv2D(patch)))
    }
}

impl Op for Conv2D {
    fn eval(&self, mut inputs: Vec<Input>) -> Result<Vec<Input>> {
        let (m_data, m_filter) = args_2!(inputs);
        let data = m_data
            .into_matrix()
            .take_f32s()
            .ok_or("Expected a f32 matrix")?;
        let filter = m_filter.as_f32s().ok_or("Expected a f32 matrix")?;
        let data = into_4d(data)?;
        let images = BatchImageWrapper(data.view());

        let filter_rows = filter.shape()[0];
        let filter_cols = filter.shape()[1];
        let out_depth = filter.shape()[3];

        let (out_height, out_width) =
            self.0
                .adjusted_dim(images.h(), images.w(), (filter_rows, filter_cols));

        let filter = filter
            .view()
            .into_shape((filter_rows * filter_cols * images.d(), out_depth))?;

        let mut transformed: Vec<f32> = Vec::with_capacity(out_height * out_width * out_depth);
        for image in data.outer_iter() {
            let patches = self.0.mk_patches(image, (filter_rows, filter_cols))?;
            transformed.extend(patches.dot(&filter).into_iter());
        }
        let transformed: Matrix = Array::from_vec(transformed)
            .into_shape((1, out_height, out_width, out_depth))?
            .into_dyn()
            .into();
        Ok(vec![transformed.into()])
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use Matrix;
    use super::*;

    fn mk(sizes: &[usize]) -> Matrix {
        ::ndarray::Array::range(1f32, sizes.iter().product::<usize>() as f32 + 1.0, 1.0)
            .into_shape(sizes)
            .unwrap()
            .into()
    }

    fn verify(input: &[usize], filter: &[usize], stride: usize, padding: Padding, expect: &[f32]) {
        let strides = vec![1, stride, stride, 1];
        let result = Conv2D(LocalPatch {
            padding: padding,
            strides: strides,
            _data_format: DataFormat::NHWC,
        }).eval(vec![mk(input).into(), mk(filter).into()])
            .unwrap()
            .remove(0);
        assert_eq!(expect.len(), result.shape().iter().product::<usize>());
        let found = result
            .into_matrix()
            .take_f32s()
            .unwrap()
            .into_shape(expect.len())
            .unwrap();
        assert_eq!(expect, found.as_slice().unwrap());
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn testConv2D1x1Filter() {
        verify(&[1,2,3,3], &[1, 1, 3, 3], 1, Padding::Valid, &[
        30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0, 138.0, 171.0,
        204.0, 174.0, 216.0, 258.0, 210.0, 261.0, 312.0 ]);
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn testConv2D1x2Filter() {
        verify(&[1, 2, 3, 3], &[1, 2, 3, 3] , 1, Padding::Valid, &[
        231.0, 252.0, 273.0, 384.0, 423.0, 462.0, 690.0, 765.0, 840.0, 843.0,
        936.0, 1029.0
    ])}

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn testConv2D2x1Filter() {
        verify(&[1, 2, 3, 3], &[2, 1, 3, 3] , 1, Padding::Valid,
          &[465.0, 504.0, 543.0, 618.0, 675.0, 732.0, 771.0, 846.0, 921.0]);
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn testConv2D2x2Filter() {
        verify(&[1, 2, 3, 3], &[2, 2, 3, 3] , 1, Padding::Valid,
               &[ 2271.0, 2367.0, 2463.0, 2901.0, 3033.0, 3165.0 ])
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn testConv2D2x2FilterStride2() {
        verify(&[1, 2, 3, 3], &[2, 2, 3, 3] , 2, Padding::Valid,
               &[2271.0, 2367.0, 2463.0])
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn testConv2D2x2FilterStride2Same() {
        verify(&[1, 2, 3, 3], &[2, 2, 3, 3] , 2, Padding::Same,
               &[2271.0, 2367.0, 2463.0, 1230.0, 1305.0, 1380.0]);
    }

    #[test]
    fn test_conv_1() {
        let conv = Conv2D(LocalPatch {
            padding: Padding::Same,
            strides: vec![1, 1, 1, 1],
            _data_format: DataFormat::NHWC,
        });
        // NHWC
        let data: Matrix = Matrix::f32s(&[1, 1, 1, 1], &[1f32]).unwrap();
        // HWIO
        let filter = Matrix::f32s(&[3, 1, 1, 1], &[0.0, 1.0, 0.0]).unwrap();
        let exp: Matrix = Matrix::f32s(&[1, 1, 1, 1], &[1.0]).unwrap();

        let result = conv.eval(vec![data.into(), filter.into()])
            .unwrap()
            .remove(0);
        assert_eq!(exp, result.into_matrix());
    }

    #[test]
    fn test_conv_2() {
        let conv = Conv2D(LocalPatch {
            padding: Padding::Same,
            strides: vec![1, 1, 1, 1],
            _data_format: DataFormat::NHWC,
        });
        let data =
            Matrix::f32s(&[1, 2, 2, 1], &[142.3088, 48.891083, 208.3187, -11.274994]).unwrap();
        let filter: Matrix = Matrix::f32s(
            &[2, 2, 1, 1],
            &[160.72833, 107.84076, 247.50552, -38.738464],
        ).unwrap();
        let exp: Matrix =
            Matrix::f32s(&[1, 2, 2, 1], &[80142.31, 5067.5586, 32266.81, -1812.2109]).unwrap();

        assert!(exp.close_enough(&conv.eval(vec![data.into(), filter.into()]).unwrap()[0],))
    }
}

#[cfg(all(test, feature = "tensorflow"))]
pub mod proptests {
    #![allow(non_snake_case)]
    use proptest::prelude::*;
    use ndarray::prelude::*;
    use protobuf::core::Message;
    use tfpb;
    use tfpb::types::DataType::DT_FLOAT;
    use ops::proptests::*;

    use Matrix;

    fn convolution_pb(v_stride: usize, h_stride: usize, valid: bool) -> ::Result<Vec<u8>> {
        let conv = tfpb::node()
            .name("conv")
            .op("Conv2D")
            .input("data")
            .input("kernel")
            .attr("strides", vec![1, v_stride as i64, h_stride as i64, 1])
            .attr("padding", if valid { "VALID" } else { "SAME" })
            .attr("T", DT_FLOAT);

        let graph = tfpb::graph()
            .node(placeholder_f32("data"))
            .node(placeholder_f32("kernel"))
            .node(conv);

        Ok(graph.write_to_bytes()?)
    }

    fn img_and_ker(
        ih: usize,
        iw: usize,
        ic: usize,
        kh: usize,
        kw: usize,
        kc: usize,
    ) -> BoxedStrategy<(Matrix, Matrix)> {
        (1..ih, 1..iw, 1..ic, 1..kh, 1..kw, 1..kc)
            .prop_flat_map(|(ih, iw, ic, kh, kw, kc)| {
                let i_size = iw * ih * ic;
                let k_size = kw * kh * kc * ic;
                (
                    Just((1, ih, iw, ic)),
                    Just((kh, kw, ic, kc)),
                    ::proptest::collection::vec(-255f32..255f32, i_size..i_size + 1),
                    ::proptest::collection::vec(-255f32..255f32, k_size..k_size + 1),
                )
            })
            .prop_map(|(img_shape, ker_shape, img, ker)| {
                (
                    Array::from_vec(img).into_shape(img_shape).unwrap().into(),
                    Array::from_vec(ker).into_shape(ker_shape).unwrap().into(),
                )
            })
            .boxed()
    }

    proptest! {
        #[test]
        fn conv((ref i, ref k) in img_and_ker(32, 32, 5, 16, 16, 8),
                           valid in ::proptest::bool::ANY,
                           stride in 1usize..4) {
            prop_assume!(stride <= k.shape()[0]);
            prop_assume!(stride <= k.shape()[1]);
            if valid {
                prop_assume!(i.shape()[1] >= k.shape()[0]);
                prop_assume!(i.shape()[2] >= k.shape()[1]);
            }
            let model = convolution_pb(stride, stride, valid).unwrap();
            let mut tf = ::tf::for_slice(&model)?;
            let tfd = ::Model::for_reader(&*model)?;
            let data = tfd.node_id_by_name("data").unwrap();
            let kernel = tfd.node_id_by_name("kernel").unwrap();
            let conv = tfd.node_id_by_name("conv").unwrap();
            let mut tfds = tfd.state();
            let expected = tf.run(vec!(("data", i.clone()), ("kernel", k.clone())), "conv")?;
            tfds.set_value(data, i.clone())?;
            tfds.set_value(kernel, k.clone())?;
            tfd.plan_for_one(conv).unwrap().run(&mut tfds).unwrap();
            let found = tfds.take(conv)?;
            prop_assert!(expected[0].close_enough(&found[0]))
        }
    }

}
