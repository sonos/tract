use num_traits::Zero;
use std::ops::{Add, AddAssign, Mul};

use crate::internal::*;
use ndarray::prelude::*;

use crate::ops::cnn::conv::KernelFormat;
use crate::ops::cnn::Patch;
use crate::ops::nn::{DataFormat, DataShape};

use tract_linalg::VecMatMul;

#[derive(CustomDebug, Clone, new)]
pub struct VecMat<T>
where
    T: Datum + Add + Mul + Zero + Copy,
{
    pub patch: Patch,
    pub output_shape: DataShape,
    pub k: usize,
    pub n: usize,
    pub kernel_fmt: KernelFormat,
    #[debug(skip)]
    pub packed_kernels: Vec<Tensor>,
    pub bias: Option<ArrayD<T>>,
    pub group: usize,
    pub vmm: Box<VecMatMul<T>>,
}

impl<T> VecMat<T>
where
    T: Datum + Add + Mul + Zero + Copy + AddAssign + ndarray::LinalgScalar,
{
    pub(super) fn conv_gemm<'i>(
        &'i self,
        packed_input: &'i ArrayView3<'i, T>,
    ) -> TractResult<ArrayD<T>> {
        let mut output = unsafe { ArrayD::<T>::uninitialized(&*self.output_shape.shape) };
        let packed_b_len = self.vmm.b_pack().len();

        let co_per_group = self.output_shape.c() / self.group;
        for i in 0..self.output_shape.n() {
            unsafe {
                let output_i =
                    output.as_mut_ptr().offset(self.output_shape.n_stride() as isize * i as isize);
                for g in 0..self.group {
                    let a = &self.packed_kernels[g];
                    let output_i_g = output_i.offset(
                        self.output_shape.c_stride() as isize * co_per_group as isize * g as isize,
                    );

                    let stride_output = match self.output_shape.fmt {
                        DataFormat::NHWC => self.group as isize,
                        DataFormat::NCHW => 1,
                    };

                    self.vmm.vec_mat_mul_prepacked(
                        a.as_ptr()?,
                        packed_input
                            .as_ptr()
                            .offset(((self.group * i + g) * packed_b_len) as isize),
                        output_i_g,
                        stride_output,
                    );
                }
            }
        }

        if let Some(ref bias) = self.bias {
            output += &bias;
        }

        Ok(output)
    }
}

impl<D> Op for VecMat<D>
where
    D: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<D> + PartialEq,
{
    fn name(&self) -> Cow<str> {
        "VecMat".into()
    }

    fn info(&self) -> TractResult<Option<String>> {
        Ok(Some(format!("{:?}", self.vmm)))
    }

    fn cost(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<(Cost, TDim)>> {
        let batch = inputs[0].shape.dim(0);
        Ok(tvec!((Cost::FMA(f32::datum_type()), batch * self.group * self.vmm.k() * self.vmm.n())))
    }
}

impl<D> StatelessOp for VecMat<D>
where
    D: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<D> + PartialEq,
{
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let input = args_1!(inputs);
        let output = self.conv_gemm(&input.to_array_view::<D>()?.into_dimensionality()?)?;
        Ok(tvec!(output.into_arc_tensor()))
    }
}

impl<D> InferenceRulesOp for VecMat<D>
where
    D: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<D>,
{
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        _s: &mut Solver<'r>,
        _inputs: &'p [TensorProxy],
        _outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        unreachable!()
    }
}
