use analyser::rules::prelude::*;
use ndarray::prelude::*;
use ops::prelude::*;

use dim::DimLike;

use super::patches::Patch;

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
    fn spatial_kernel_dim(&self) -> usize {
        if self.kernel_is_hwio {
            0
        } else {
            2
        }
    }

    fn patch<D:DimLike>(&self, data_full_shape:&[D], kernel_full_shape: &[D]) -> Patch<D> {
        let spatial_rank = data_full_shape.len() -2;
        Patch::new(self.data_is_nhwc,
                   self.dilations.clone().unwrap_or(vec!(1; spatial_rank)),
                   kernel_full_shape.iter().skip(self.spatial_kernel_dim()).take(spatial_rank).cloned().collect(),
                   self.pads.as_ref().map(|p| p[..spatial_rank].iter().map(|i| D::from(*i)).collect()).unwrap_or(vec!(D::from(0); spatial_rank)),
                   self.pads.as_ref().map(|p| p[spatial_rank..].iter().map(|i| D::from(*i)).collect()).unwrap_or(vec!(D::from(0); spatial_rank)),
                   self.strides.clone().unwrap_or(vec!(1; spatial_rank)),
                   data_full_shape.to_vec())
    }

    fn output_shape<D:DimLike>(&self, ishape: &[D], kshape: &[D]) -> Vec<D> {
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
        let input = input.to_array_view()?;
        let kernel = kernel.to_array_view::<f32>()?;
        let bias = bias.as_ref().map(|b| b.to_array_view::<f32>()).inside_out()?;

        let (ax_ker_in, ax_ker_out) = if self.kernel_is_hwio {
            (kernel.ndim() - 2, kernel.ndim() - 1)
        } else {
            (1, 0) // oihw
        };

        let patch = self.patch(input.shape(), kernel.shape());

        let shape: Vec<usize> = patch.output_full_shape(kernel.shape()[ax_ker_out]);
        let input_channels = input.shape()[patch.axis_data_channel()];

        let data_field = patch.mk_data_field();
        let kernel_field = patch.mk_kernel_field();

        let output = ArrayD::from_shape_fn(shape, |coords| -> f32 {
            let output = patch.split_data_coords(coords.slice());
            let space: ArrayView1<usize> = ArrayView1::from(output.space);
            let mut result = bias.as_ref().map(|b| b[output.chan]).unwrap_or(0.0);
            for (kerf, imgf) in kernel_field.outer_iter().zip(data_field.outer_iter()) {
                let i_coords: Vec<usize> = izip!(space.iter(), imgf.iter(), patch.strides.iter())
                    .map(|(x, i, s)| (x * s).wrapping_add(*i))
                    .collect();
                for input_c in 0..input_channels {
                    let i_value: f32 = *input
                        .subview(Axis(patch.axis_data_channel()), input_c)
                        .subview(Axis(patch.axis_data_batch()), output.n) // careful, need to start with higher ranking
                        .get(&*i_coords)
                        .unwrap_or(&0.0);
                    let k_value = kernel
                        .subview(Axis(ax_ker_out), output.chan)
                        .subview(Axis(ax_ker_in), input_c) // higher ranking again
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
