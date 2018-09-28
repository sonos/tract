use analyser::rules::prelude::*;
use ndarray::prelude::*;
use ops::prelude::*;

use dim::DimLike;

use super::patches::{PaddingSpec, Patch};

use insideout::InsideOut;

#[derive(Debug, Clone, new, Default)]
pub struct Conv {
    data_is_nhwc: bool,   // default is nchw (onnx)
    kernel_is_hwio: bool, // default is oihw (onnx)
    dilations: Option<Vec<usize>>,
    kernel_shape: Option<Vec<usize>>,
    padding: PaddingSpec,
    strides: Option<Vec<usize>>,
}

impl Conv {
    fn spatial_kernel_dim(&self) -> usize {
        if self.kernel_is_hwio {
            0
        } else {
            2
        }
    }

    fn patch<D: DimLike>(&self, input_full_shape: &[D], kernel_full_shape: &[D]) -> Patch<D> {
        let spatial_rank = input_full_shape.len() - 2;
        let dilations = self.dilations.clone().unwrap_or(vec![1; spatial_rank]);
        let strides = self.strides.clone().unwrap_or(vec![1; spatial_rank]);
        let kernel_spatial_shape = &kernel_full_shape[self.spatial_kernel_dim()..][..spatial_rank];
        assert_eq!(spatial_rank, kernel_spatial_shape.len());
        assert_eq!(spatial_rank, dilations.len());
        assert_eq!(spatial_rank, strides.len());
        let patch = Patch::new(
            self.data_is_nhwc,
            dilations,
            kernel_spatial_shape.to_vec(),
            &self.padding,
            strides,
            input_full_shape.to_vec(),
        );
        patch
    }

    fn output_shape<D: DimLike>(&self, ishape: &[D], kshape: &[D]) -> Vec<D> {
        let patch = self.patch(ishape, kshape);
        let ko = if self.kernel_is_hwio {
            *kshape.last().unwrap() // hwio
        } else {
            kshape[0] // oihw
        };
        patch.output_full_shape(ko)
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
        let input = input.to_array_view::<f32>()?;
        let kernel = kernel.to_array_view::<f32>()?;
        let bias = bias
            .as_ref()
            .map(|b| b.to_array_view::<f32>())
            .inside_out()?;

        let (ax_ker_in, ax_ker_out) = if self.kernel_is_hwio {
            (kernel.ndim() - 2, kernel.ndim() - 1)
        } else {
            (1, 0) // oihw
        };

        let mut patch = self.patch(input.shape(), kernel.shape());

        let shape: Vec<usize> = patch.output_full_shape(kernel.shape()[ax_ker_out]);
        let input_channels = input.shape()[patch.axis_data_channel()];
        let output_channels = shape[patch.axis_data_channel()];

        let k = kernel.len() / output_channels;
        let m = output_channels;
        let n = shape.iter().skip(2).product();
        let kernel = kernel.into_shape((m, k))?;
        let mut output = unsafe { ArrayD::<f32>::uninitialized(&*shape) };
        for i in 0..input.shape()[0] {
            let mut mega_matrix = unsafe { Array2::<f32>::uninitialized((k, n)) };
            for ((mut coords, _uninit), mut col) in output
                .slice_axis(Axis(0), (i..(i + 1)).into())
                .slice_axis(Axis(1), (0..1).into())
                .indexed_iter()
                .zip(mega_matrix.axis_iter_mut(Axis(1)))
            {
                let mut col = col.iter_mut();
                for ci in 0..input_channels {
                    coords[1] = ci;
                    for v in patch.patch_data_iter(&input, &coords.slice()) {
                        *col.next().unwrap() = v.unwrap_or(0.0);
                    }
                }
            }

            let mut output_subview = output.slice_axis_mut(Axis(0), (i..(i + 1)).into());
            let mut output_panel = output_subview.into_shape((m, n))?;
            ::ndarray::linalg::general_mat_mul(1.0, &kernel, &mega_matrix, 0.0, &mut output_panel);
        }

        if let Some(bias) = bias {
            let mut bias_shape:Vec<usize> = ::std::iter::repeat(1).take(shape.len()).collect();
            bias_shape[patch.axis_data_channel()] = output_channels;
            let bias:ArrayViewD<f32> = bias.view().into_shape(&*bias_shape)?;
            output += &bias;
        }
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
            solver.equals(&inputs[1].rank, kshape.len() as i64 + 2);
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
                solver.equals(&outputs[0].shape, self.output_shape(&*ishape, &*kshape));
            },
        );
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_infer_with_known_kshape() {
        let mut op = Conv::default();
        op.strides = Some(vec![2, 2]);
        op.kernel_shape = Some(vec![3, 3]);
        let facts = op
            .infer_facts(
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
        let op = Conv::default();
        let facts = op
            .infer_facts(
                tvec!(
                    TensorFact::dt_shape(DatumType::F32, shapefact!(1, 2, 1, 1)),
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
        let facts = op
            .infer_facts(
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
