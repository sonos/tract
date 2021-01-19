use crate::internal::*;
use crate::ops::cnn::conv::KernelFormat;
use crate::ops::cnn::*;
use crate::ops::nn::*;
use proptest::collection::vec;
use proptest::prelude::*;
use tract_ndarray::prelude::*;

#[derive(Debug)]
struct DeconvProblem {
    input: Array4<f32>,
    kernel: Array4<f32>,
}

fn tensor(shape: [usize; 4]) -> BoxedStrategy<Array4<f32>> {
    let len = shape.iter().product::<usize>();
    vec(any::<i8>().prop_map(|i| i as f32), len..=len)
        .prop_map(move |vec| Array4::from_shape_vec(shape.clone(), vec).unwrap())
        .boxed()
}

impl Arbitrary for DeconvProblem {
    type Strategy = BoxedStrategy<DeconvProblem>;
    type Parameters = ();
    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        (1usize..3, 1usize..4, 1usize..4, 1usize..4, 1usize..4, 1usize..8, 1usize..8)
            .prop_flat_map(|(n, ci, co, hk, wk, hi, wi)| {
                (tensor([n, ci, hi, wi]), tensor([ci, co, hk, wk]))
            })
            .prop_map(|(input, kernel)| DeconvProblem { input, kernel })
            .boxed()
    }
}

impl DeconvProblem {
    fn tract(&self) -> Array4<f32> {
        let op = DeconvUnary::new(PaddingSpec::Valid, self.kernel.clone().into_arc_tensor());
        let mut outputs = op.eval(tvec!(self.input.clone().into_arc_tensor())).unwrap();
        outputs.remove(0).into_tensor().into_array().unwrap().into_dimensionality().unwrap()
    }
    fn reference(&self) -> Array4<f32> {
        let output_shape = [
            self.input.shape()[0],
            self.kernel.shape()[1],
            self.input.shape()[2] + self.kernel.shape()[2] - 1,
            self.input.shape()[3] + self.kernel.shape()[3] - 1,
        ];
        let mut output = Array4::zeros(output_shape);
        for n in 0..self.input.shape()[0] {
            for co in 0..output.shape()[1] {
                for ci in 0..self.input.shape()[1] {
                    for hi in 0..self.input.shape()[2] {
                        for wi in 0..self.input.shape()[3] {
                            for hk in 0..self.kernel.shape()[2] {
                                for wk in 0..self.kernel.shape()[3] {
                                    output[(n, co, hi + hk, wi + wk)] +=
                                        self.input[(n, ci, hi, wi)] * self.kernel[(ci, co, hk, wk)];
                                }
                            }
                        }
                    }
                }
            }
        }
        output
    }
}

proptest::proptest! {
    #[test]
    fn prop(pb in any::<DeconvProblem>()) {
        prop_assert_eq!(pb.tract(), pb.reference());
    }
}

#[test]
fn test_trivial_0() {
    let pb = DeconvProblem { input: arr4(&[[[[0.0]]]]), kernel: arr4(&[[[[0.0]]]]) };
    assert_eq!(pb.tract(), pb.reference());
}
