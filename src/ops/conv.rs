use {Matrix, Result};
use super::Op;

#[derive(Debug)]
pub enum DataFormat {
    NHWC,
}

#[derive(Debug, PartialEq)]
pub enum Padding {
    Valid,
    Same,
}

#[derive(Debug)]
pub struct Conv2D {
    pub _data_format: DataFormat,
    pub padding: Padding,
    pub strides: Vec<usize>,
}

impl Conv2D {
    pub fn build(pb: &::tfpb::node_def::NodeDef) -> Result<Conv2D> {
        if let Some(data_format) = pb.get_attr().get("data_format") {
            if data_format.get_s() == b"NCHW" {
                Err("NCHW data_format not implemented")?
            }
        }
        let strides = pb.get_attr()
            .get("strides")
            .ok_or("expect strides in Conv2D args")?
            .get_list()
            .get_i()
            .iter()
            .map(|a| *a as usize)
            .collect();
        let padding = pb.get_attr().get("padding").ok_or(
            "expect padding in Conv2D args",
        )?;
        let padding = match padding.get_s() {
            b"VALID" => Padding::Valid,
            b"SAME" => Padding::Same,
            _ => Err("Only VALID padding supported for now on Conv2D")?,
        };
        Ok(Conv2D {
            _data_format: DataFormat::NHWC,
            padding,
            strides,
        })
    }
}

impl Op for Conv2D {
    fn eval(&self, mut inputs: Vec<Matrix>) -> Result<Vec<Matrix>> {
        // [ filter_rows, filter_cols, in_depth, out_depth]
        let filter = inputs.remove(1).take_f32s().ok_or(
            "Expect input #1 to be f32",
        )?;
        // [ batch, in_rows, in_cols, in_depth ]
        let data = inputs.remove(0).take_f32s().ok_or(
            "Expect input #0 to be f32",
        )?;
        //        println!("kernel is {:?}", filter.shape());

        if self.strides.len() != 4 || self.strides[0] != 1 && self.strides[3] != 1 ||
            self.strides[1] != self.strides[2]
        {
            Err(format!(
                "strides must be of the form [1, s, s, 1], found {:?}",
                self.strides
            ))?
        }
        if data.shape().len() != 4 || filter.shape().len() != 4 {
            Err(format!(
                "data and filter must be of dimension 4. data is {:?}, filter is {:?}",
                data.shape(),
                filter.shape()
            ))?
        }
        if data.shape()[3] != filter.shape()[2] {
            Err(format!(
                "data fourth dim (in_depth) must match filter third (data is {:?}, filter is {:?})",
                data.shape(),
                filter.shape()
            ))?
        }

        let stride = self.strides[1];
        let batches = data.shape()[0];
        let in_rows = data.shape()[1];
        let in_cols = data.shape()[2];
        let in_depth = data.shape()[3];
        let filter_rows = filter.shape()[0];
        let filter_cols = filter.shape()[1];
        let out_depth = filter.shape()[3];

        let mut data = data.into_shape((batches, in_rows, in_cols, in_depth))?;
        let filter = filter.into_shape(
            (filter_rows, filter_cols, in_depth, out_depth),
        )?;

        let (out_height, out_width) = match self.padding {
            Padding::Same => (
                (in_rows as f32 / stride as f32).ceil() as usize,
                (in_cols as f32 / stride as f32).ceil() as usize,
            ),
            Padding::Valid => (
                ((in_rows - filter_rows + 1) as f32 / stride as f32).ceil() as usize,
                ((in_cols - filter_cols + 1) as f32 / stride as f32).ceil() as usize,
            ),
        };
        let out_shape = (data.shape()[0], out_height, out_width, out_depth);
        //        println!("data.shape:{:?} out_shape:{:?} stride:{}", data.shape(), out_shape, stride);
        //        println!("{:?}", data);
        //        println!("{:?}", filter);
        let patches_size = (
            (out_height * out_width) as usize,
            filter_rows * filter_cols * in_depth,
        );
        unsafe {
            let mut results = vec![];
            let mut patches = ::ndarray::Array2::<f32>::uninitialized(patches_size);
            //            println!("{:?}", patches);
            let filters_mat = filter.into_shape((patches_size.1, out_depth))?;
            if self.padding == Padding::Same {
                //                println!("data:{:?}", data.shape());
                let right_padding = stride * (in_cols / stride) + filter_cols - in_cols;
                let bottom_padding = stride * (in_rows / stride) + filter_rows - in_rows;
                //                println!("padding at right:{} bottom:{}", right_padding, bottom_padding);
                let right_padding =
                    ::ndarray::Array4::<f32>::zeros((batches, in_rows, right_padding, in_depth));
                data = ::ndarray::stack(::ndarray::Axis(2), &[data.view(), right_padding.view()])?;
                let bottom_padding = ::ndarray::Array4::<f32>::zeros(
                    (batches, bottom_padding, data.shape()[2], in_depth),
                );
                data = ::ndarray::stack(::ndarray::Axis(1), &[data.view(), bottom_padding.view()])?;
                //                println!("padded data:{:?} patches:{:?}", data.shape(), patches.shape());
            }
            for b in 0..batches {
                //                println!("writting patches for id {}", b);
                for i_x in 0..out_width {
                    for i_y in 0..out_height {
                        //                        println!("getting row {}", i_y * out_width + i_x);
                        let mut patch_row = patches.row_mut(i_y * out_width + i_x);
                        for f_x in 0..filter_cols {
                            for f_y in 0..filter_rows {
                                //                                println!("i_x:{} i_y:{} f_x:{} f_y:{}", i_x, i_y, f_x, f_y);
                                //                                println!("writting loc: {:?}", (b, i_y * stride + f_y, i_x * stride + f_x));
                                for d in 0..in_depth {
                                    let loc = &mut patch_row[f_y * in_depth * filter_cols +
                                                                 f_x * in_depth +
                                                                 d];
                                    *loc = data[(b, i_y * stride + f_y, i_x * stride + f_x, d)];
                                }
                            }
                        }
                    }
                }
                //                println!("doing product");
                results.push(patches.dot(&filters_mat));
            }
            //            println!("building results");
            let views: Vec<_> = results.iter().map(|m| m.view()).collect();
            let result = ::ndarray::stack(::ndarray::Axis(0), &*views)?
                .into_shape(out_shape)?
                .into_dyn();
            return Ok(vec![result.into()]);
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use proptest::prelude::*;
    use ndarray::prelude::*;

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
        let result = Conv2D {
            padding: padding,
            strides: strides,
            _data_format: DataFormat::NHWC,
        }.eval(vec![mk(input), mk(filter)])
            .unwrap()
            .remove(0);
        assert_eq!(expect.len(), result.shape().iter().product::<usize>());
        let found = result
            .take_f32s()
            .unwrap()
            .into_shape((expect.len()))
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
    fn test_image_1() {
        let conv = Conv2D {
            padding: Padding::Same,
            strides: vec![1, 1, 1, 1],
            _data_format: DataFormat::NHWC,
        };
        // NHWC
        let data: Matrix = ::ndarray::arr3(
            &[
                [[1.0, 3.0], [0.0, 2.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ],
        ).into_shape((1, 4, 3, 2))
            .unwrap()
            .into_dyn()
            .into();

        let filter = ::ndarray::arr3(&[[[1.0], [0.0]], [[0.0], [0.0]]])
            .into_shape((1, 2, 2, 1))
            .unwrap()
            .into_dyn()
            .into();

        let exp: Matrix = ::ndarray::arr3(
            &[
                [[1.0], [0.0], [0.0]],
                [[0.0], [0.0], [0.0]],
                [[0.0], [0.0], [0.0]],
                [[0.0], [0.0], [0.0]],
            ],
        ).into_shape((1, 4, 3, 1))
            .unwrap()
            .into_dyn()
            .into();
        assert_eq!(vec![exp], conv.eval(vec![data.clone(), filter]).unwrap());

        let filter = ::ndarray::arr3(&[[[0.0], [1.0]], [[5.0], [0.0]]])
            .into_shape((1, 2, 2, 1))
            .unwrap()
            .into_dyn()
            .into();

        let exp: Matrix = ::ndarray::arr3(
            &[
                [[3.0], [2.0], [0.0]],
                [[0.0], [0.0], [0.0]],
                [[0.0], [0.0], [0.0]],
                [[0.0], [0.0], [0.0]],
            ],
        ).into_shape((1, 4, 3, 1))
            .unwrap()
            .into_dyn()
            .into();
        assert_eq!(vec![exp], conv.eval(vec![data.clone(), filter]).unwrap());

    }

    #[test]
    fn test_image_2() {
        let conv = Conv2D {
            padding: Padding::Same,
            strides: vec![1, 1, 1, 1],
            _data_format: DataFormat::NHWC,
        };
        // NHWC
        let data: Matrix = ::ndarray::arr1(&[1f32])
            .into_shape((1, 1, 1, 1))
            .unwrap()
            .into_dyn()
            .into();

        let filter = ::ndarray::arr1(&[0.0, 1.0, 0.0])
            .into_shape((3, 1, 1, 1))
            .unwrap()
            .into_dyn()
            .into();

        let exp: Matrix = ::ndarray::arr1(&[1.0])
            .into_shape((1, 1, 1, 1))
            .unwrap()
            .into_dyn()
            .into();
        assert_eq!(vec![exp], conv.eval(vec![data.clone(), filter]).unwrap());
    }

    fn convolution_pb(v_stride: usize, h_stride: usize) -> ::Result<Vec<u8>> {
        use protobuf::core::Message;
        use tfpb;

        let mut graph = tfpb::graph::GraphDef::new();
        let mut dt_float = tfpb::attr_value::AttrValue::new();
        dt_float.set_field_type(tfpb::types::DataType::DT_FLOAT);

        let mut data = tfpb::node_def::NodeDef::new();
        data.set_name("data".into());
        data.set_op("Placeholder".into());
        data.mut_attr().insert(
            "dtype".to_string(),
            dt_float.clone(),
        );
        graph.mut_node().push(data);

        let mut kernel = tfpb::node_def::NodeDef::new();
        kernel.set_name("kernel".into());
        kernel.set_op("Placeholder".into());
        kernel.mut_attr().insert(
            "dtype".to_string(),
            dt_float.clone(),
        );
        graph.mut_node().push(kernel);

        let mut conv = tfpb::node_def::NodeDef::new();
        conv.set_name("conv".into());
        conv.set_op("Conv2D".into());
        conv.mut_input().push("data".into());
        conv.mut_input().push("kernel".into());
        let mut strides_list = tfpb::attr_value::AttrValue_ListValue::new();
        strides_list.set_i(vec![1, v_stride as i64, h_stride as i64, 1]);
        let mut strides = tfpb::attr_value::AttrValue::new();
        strides.set_list(strides_list);
        conv.mut_attr().insert("strides".to_string(), strides);
        let mut same = tfpb::attr_value::AttrValue::new();
        same.set_s("SAME".as_bytes().to_vec());
        conv.mut_attr().insert("padding".to_string(), same);
        conv.mut_attr().insert("T".to_string(), dt_float.clone());
        graph.mut_node().push(conv);

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
                    Just(ih),
                    Just(iw),
                    Just(ic),
                    Just(kh),
                    Just(kw),
                    Just(kc),
                    ::proptest::collection::vec(-255f32..255f32, i_size..i_size + 1),
                    ::proptest::collection::vec(-255f32..255f32, k_size..k_size + 1),
                )
            })
            .prop_map(|(ih, iw, ic, kh, kw, kc, img, ker)| {
                (
                    Matrix::F32(
                        Array::from_vec(img)
                            .into_shape((1, ih, iw, ic))
                            .unwrap()
                            .into_dyn(),
                    ),
                    Matrix::F32(
                        Array::from_vec(ker)
                            .into_shape((kh, kw, ic, kc))
                            .unwrap()
                            .into_dyn(),
                    ),
                )
            })
            .boxed()
    }

    proptest! {
        #[test]
        fn test_image_conv((ref i, ref k) in img_and_ker(32, 32, 5, 16, 16, 8)) {
        //    println!("i:{:?} k:{:?}", i.shape(), k.shape());
            let model = convolution_pb(1,1).unwrap();
            let mut tf = ::tf::for_slice(&model)?;
            let mut tfd = ::GraphAnalyser::from_reader(&*model)?;
            let expected = tf.run(vec!(("data", i.clone()), ("kernel", k.clone())), "conv")?;
            tfd.set_value("data", i.clone())?;
            tfd.set_value("kernel", k.clone())?;
            let found = tfd.take("conv")?;
            prop_assert_eq!(expected, found)
        }
    }
}
