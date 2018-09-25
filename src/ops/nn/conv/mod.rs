use analyser::rules::prelude::*;
use ndarray::prelude::*;
use ops::prelude::*;

use insideout::InsideOut;

#[derive(Debug, Clone, new, Default)]
pub struct Conv {
    data_is_nhwc: bool,   // default is nchw (onnx)
    kernel_is_hwio: bool, // default is oihw (onnx)
    dilations: Option<Vec<usize>>,
    kernel_shape: Option<Vec<usize>>,
    pads: Option<Vec<usize>>,
    strides: Option<Vec<usize>>,
}

impl Conv {
    fn spatial_input_dim(&self) -> usize {
        if self.data_is_nhwc {
            1
        } else {
            2
        }
    }

    fn spatial_kernel_dim(&self) -> usize {
        if self.kernel_is_hwio {
            0
        } else {
            2
        }
    }

    fn output_shape(&self, ishape: &[TDim], kshape: &[TDim]) -> Vec<TDim> {
        let ko = if self.kernel_is_hwio {
            kshape[3] // hwio
        } else {
            kshape[0] // oihw
        };
        let mut v = vec![ishape[0]]; // nchw
        if !self.data_is_nhwc {
            v.push(ko); // nchw
        }
        let spatial_rank = ishape.len() - 2;
        v.extend((0..spatial_rank).map(move |ix| {
            compute_output_spatial_dim(
                ishape[ix + self.spatial_input_dim()],
                self.dilations.as_ref().map(|ds| ds[ix]).unwrap_or(1),
                kshape[ix + self.spatial_kernel_dim()].to_integer().unwrap() as usize,
                self.pads.as_ref().map(|p| p[ix]).unwrap_or(0),
                self.pads
                    .as_ref()
                    .map(|p| p[ix + kshape.len() - 2])
                    .unwrap_or(0),
                self.strides.as_ref().map(|s| s[ix]).unwrap_or(1),
            )
        }));
        if self.data_is_nhwc {
            v.push(ko); // nhwc
        }
        v
    }
}

impl Op for Conv {
    fn name(&self) -> &str {
        "Conv"
    }

    fn eval(&self, mut inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        let (input, kernel, bias) = if inputs.len() == 2 {
            let (input, kernel) = args_2!(inputs);
            (input, kernel, None)
        } else {
            let (input, kernel, bias) = args_3!(inputs);
            (input, kernel, Some(bias))
        };
        let input = input.to_array_view()?;
        let kernel = kernel.to_array_view::<f32>()?;
        let bias = bias.as_ref().map(|b| b.to_array_view::<f32>()).inside_out()?;

        let spatial_rank = input.ndim() - 2;

        let ax_img_c = if self.data_is_nhwc {
            Axis(input.ndim() - 1)
        } else {
            Axis(1)
        };
        let (ax_ker_in, ax_ker_out) = if self.kernel_is_hwio {
            (Axis(kernel.ndim() - 2), Axis(kernel.ndim() - 1))
        } else {
            (Axis(1), Axis(0)) // oihw
        };

        let ishape: Vec<_> = input.shape().iter().map(|&i| TDim::from(i)).collect();
        let kshape: Vec<_> = kernel.shape().iter().map(|&i| TDim::from(i)).collect();
        let shape: Vec<_> = self
            .output_shape(&ishape, &kshape)
            .into_iter()
            .map(|d| d.to_integer().unwrap() as usize)
            .collect();
        let input_channels = input.shape()[ax_img_c.index()];

        let kernel_spatial_shape =
            &kernel.shape()[self.spatial_kernel_dim()..self.spatial_kernel_dim() + spatial_rank];

        let ones = vec![1; spatial_rank];
        let dilations = self.dilations.as_ref().unwrap_or(&ones);
        let strides = self.strides.as_ref().unwrap_or(&ones);
        let kernel_field = field(kernel_spatial_shape, &ones, None)?;
        let image_field = field(
            kernel_spatial_shape,
            &dilations,
            self.pads.as_ref().map(|p| p.as_slice()),
        )?;

        let output = ArrayD::from_shape_fn(shape, |coords| -> f32 {
            let (n, ker_out, space): (usize, usize, &[usize]) = if self.data_is_nhwc {
                (
                    coords[0],
                    coords[spatial_rank + 1],
                    &coords.slice()[1..spatial_rank + 1],
                )
            } else {
                (coords[0], coords[1], &coords.slice()[2..spatial_rank + 2]) // nchw
            };
            let space: ArrayView1<usize> = ArrayView1::from(space);
            let mut result = bias.as_ref().map(|b| b[ker_out]).unwrap_or(0.0);
            for (kerf, imgf) in kernel_field.outer_iter().zip(image_field.outer_iter()) {
                let i_coords: Vec<usize> = izip!(space.iter(), imgf.iter(), strides.iter())
                    .map(|(x, i, s)| (x * s).wrapping_add(*i))
                    .collect();
                for input_c in 0..input_channels {
                    let i_value: f32 = *input
                        .subview(ax_img_c, input_c)
                        .subview(Axis(0), n) // careful, need to start with higher ranking
                        .get(&*i_coords)
                        .unwrap_or(&0.0);
                    let k_value = kernel
                        .subview(ax_ker_out, ker_out)
                        .subview(ax_ker_in, input_c) // higher ranking again
                        [kerf.as_slice().unwrap()];
                    result += i_value * k_value;
                }
            }
            result
        });
        Ok(tvec!(output.into()))
    }
}

impl InferenceRulesOp for Conv {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) {
        if let Some(kshape) = &self.kernel_shape {
            for (ix, dim) in kshape.iter().enumerate() {
                solver.equals(
                    &inputs[1].shape[ix + self.spatial_kernel_dim()],
                    TDim::from(*dim as i64),
                );
            }
        }
        solver.equals(&outputs.len, 1).equals_all(wrap![
            &outputs[0].datum_type,
            &inputs[0].datum_type,
            &inputs[1].datum_type
        ]);
        solver.given(&inputs.len, move |solver, len| {
            if len == 3 {
                solver
                    .equals(&inputs[2].rank, 1)
                    .equals(&outputs[0].datum_type, &inputs[2].datum_type);
                solver.given(&inputs[1].rank, move |solver, krank| {
                    let filter_o = if self.kernel_is_hwio {
                        &inputs[1].shape[krank as usize - 1]
                    } else {
                        &inputs[1].shape[0] // oihw
                    };
                    solver.equals(&inputs[2].shape[0], filter_o);
                });
            }
        });
        solver.given_2(
            &inputs[0].rank,
            &inputs[1].rank,
            move |solver, irank, krank| {
                let input_c = if self.data_is_nhwc {
                    &inputs[0].shape[irank as usize - 1]
                } else {
                    &inputs[0].shape[1]
                };
                let filter_i = if self.kernel_is_hwio {
                    &inputs[1].shape[krank as usize - 2]
                } else {
                    &inputs[1].shape[1]
                };
                solver.equals(input_c, filter_i);
            },
        );
        solver.given_2(
            &inputs[0].shape,
            &inputs[1].shape,
            move |solver, ishape, kshape| {
                solver.equals(&outputs[0].shape, self.output_shape(&ishape, &kshape));
            },
        );
    }
}

/*
struct DeterminedConvN<T: Datum> {
    dilations: Vec<usize>,
    pads: Vec<usize>,
    strides: Vec<usize>,
    kernel: ArrayD<T>,
    bias: Option<ArrayD<T>>,
}

impl<T: Datum> DeterminedConvN<T> {
    fn eval(&self, input: &Tensor) -> TfdResult<Tensor> {
        unimplemented!();
    }
}
*/

fn compute_output_spatial_dim(
    input: TDim,
    dilation: usize,
    kernel: usize,
    pad_before: usize,
    pad_after: usize,
    stride: usize,
) -> TDim {
    let field = (kernel - 1) * dilation + 1;
    let out = (input + pad_before + pad_after - field) / stride + 1;
    trace!(
        "input:{:?} dilation:{} kernel:{} pads:{},{}, stride:{} -> field:{} out:{:?}",
        input,
        dilation,
        kernel,
        pad_before,
        pad_after,
        stride,
        field,
        out
    );
    out
}

fn field(shape: &[usize], dilations: &[usize], pads: Option<&[usize]>) -> TfdResult<Array2<usize>> {
    let square = ArrayD::from_shape_fn(shape, |id| id.slice().to_vec());
    let len = square.len();
    let points: Array1<Vec<usize>> = square.into_shape((len,))?;
    Ok(Array2::from_shape_fn(
        (points.len(), shape.len()),
        |(pt, axis)| {
            (points[pt][axis] * dilations[axis]).wrapping_sub(pads.map(|p| p[axis]).unwrap_or(0))
        },
    ))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_output_spatial_dim() {
        // onnx test_basic_conv_without_padding
        assert_eq!(
            compute_output_spatial_dim(5.into(), 1, 3, 0, 0, 1),
            3.into()
        );

        // onnx test_conv_with_strides_no_padding
        assert_eq!(
            compute_output_spatial_dim(7.into(), 1, 3, 0, 0, 2),
            3.into()
        );
        assert_eq!(
            compute_output_spatial_dim(5.into(), 1, 3, 0, 0, 2),
            2.into()
        );

        // onnx test_conv_with_strides_padding
        assert_eq!(
            compute_output_spatial_dim(7.into(), 1, 3, 1, 1, 2),
            4.into()
        );
        assert_eq!(
            compute_output_spatial_dim(5.into(), 1, 3, 1, 1, 2),
            3.into()
        );

        // onnx test_conv_with_strides_and_asymmetric_padding
        assert_eq!(
            compute_output_spatial_dim(7.into(), 1, 3, 1, 1, 2),
            4.into()
        );
        assert_eq!(
            compute_output_spatial_dim(5.into(), 1, 3, 0, 0, 2),
            2.into()
        );
    }

    #[test]
    fn test_kernel_field() {
        assert_eq!(field(&[3], &[1], None).unwrap(), arr2(&[[0], [1], [2]]));
        assert_eq!(field(&[3], &[2], None).unwrap(), arr2(&[[0], [2], [4]]));
        assert_eq!(
            field(&[2, 2], &[1, 1], None).unwrap(),
            arr2(&[[0, 0], [0, 1], [1, 0], [1, 1]])
        );
        assert_eq!(
            field(&[2, 2], &[2, 1], None).unwrap(),
            arr2(&[[0, 0], [0, 1], [2, 0], [2, 1]])
        );
    }

    #[test]
    fn test_infer_with_known_kshape() {
        let mut op = Conv::default();
        op.strides = Some(vec![2, 2]);
        op.kernel_shape = Some(vec![3, 3]);
        let facts =
            op.infer_facts(
                tvec!(
                    TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 7, 5)),
                    TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 3, 3)),
                ),
                tvec!(TensorFact::default()),
            ).unwrap();
        assert_eq!(
            facts.1,
            tvec!(TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 3, 2)))
        );
    }

    #[test]
    fn test_infer_channels() {
        let mut op = Conv::default();
        let facts =
            op.infer_facts(
                tvec!(
                    TensorFact::dt_shape(DatumType::F32, shapefact!(1, 2, 1,1)),
                    TensorFact::dt_shape(DatumType::F32, shapefact!(3, 2, 1, 1)),
                ),
                tvec!(TensorFact::default()),
            ).unwrap();
        assert_eq!(
            facts.1,
            tvec!(TensorFact::dt_shape(DatumType::F32, shapefact!(1, 3, 1, 1)))
        );
    }

    #[test]
    fn test_infer_onxx_strides_no_padding() {
        let mut op = Conv::default();
        op.strides = Some(vec![2, 2]);
        let facts =
            op.infer_facts(
                tvec!(
                    TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 7, 5)),
                    TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 3, 3)),
                ),
                tvec!(TensorFact::default()),
            ).unwrap();
        assert_eq!(
            facts.1,
            tvec!(TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 3, 2)))
        );
    }
}
