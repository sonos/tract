use num_traits::Zero;
use std::ops::{Add, AddAssign, Mul};

use std::sync::Arc;

use ndarray::prelude::*;
use ops::prelude::*;

use ops::nn::conv::KernelFormat;
use ops::nn::{DataFormat, Patch};

use tract_linalg::MatMul;

/*
 * group=1, N=1         N>1             g>1
 *
 * A: kernel
 *  * O rows            * O rows        * O rows
 *  * I*h*w cols        * I*w*h         * I/g*w*h
 * B: data
 *                      * N blocks
 *  * I*w*h rows        * I*w*h         * I*w*h
 *  * H*W cols          * H*W           * H*W
 * Gemm
 *  * 1 iter            * N iter        * g iter
 *  * m=O               * m=O           * m=O/g
 *  * k=I*h*w           * k=I*h*w       * k=I/g*h*w
 *  * n=H*W             * n=H*W         * n=H*W
 *
 *                                +------------+
 *                                | B input    |
 *                                +------------+
 *              +--------------+  +----------------+
 *              | A kernel g=0 |  | C output  g=0  |
 *              +--------------+  +----------------+
 *              | A kernel g=1 |  | C output  g=1  |
 *              +--------------+  +----------------+
 */

#[derive(CustomDebug, Clone, new)]
pub struct ConvGemm<T>
where
    T: Datum + Add + Mul + Zero + Copy,
{
    pub patch: Patch,
    pub full_output_shape: TVec<usize>,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub kernel_fmt: KernelFormat,
    #[debug(skip)]
    pub packed_kernels: Vec<Tensor>,
    pub bias: Option<ArrayD<T>>,
    pub group: usize,
    pub mm: Arc<MatMul<T>>,
}

impl<T> ConvGemm<T>
where
    T: Datum + Add + Mul + Zero + Copy + AddAssign + ndarray::LinalgScalar
{
    pub(super) fn conv_gemm<'i>(
        &'i self,
        packed_input: &'i ArrayView1<'i, T>,
    ) -> TractResult<ArrayD<T>> {
        let mut output = unsafe { ArrayD::<T>::uninitialized(&*self.full_output_shape) };
        let input_shape = &self.patch.input_shape;

        let co_per_group = self.full_output_shape[input_shape.c_axis()] / self.group;

        for i in 0..input_shape.n_dim() {
            unsafe {
                let output_i = output.as_mut_ptr().offset(output.strides()[input_shape.n_axis()]*i as isize);
                for g in 0..self.group {
                    let a = &self.packed_kernels[g];
                    let output_i_g = output_i.offset(output.strides()[input_shape.c_axis()] * co_per_group  as isize * g as isize);

                    let (rsc, csc) = match self.patch.input_shape.fmt {
                        DataFormat::NHWC => (1, self.m as isize),
                        DataFormat::NCHW => (self.n as isize, 1),
                    };
                    self.mm.mat_mul_prepacked(
                        a.as_ptr()?,
                        packed_input.as_ptr().offset(
                            ((self.group * i + g) * self.mm.packed_b_len()) as isize,
                        ),
                        output_i_g,
                        rsc,
                        csc,
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

impl<D> Op for ConvGemm<D>
where
    D: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<D> + PartialEq,
{
    fn name(&self) -> Cow<str> {
        "ConvGemm".into()
    }

    fn info(&self) -> TractResult<Option<String>> {
        Ok(Some(format!("{:?}", self.mm)))
    }
}

impl<D> StatelessOp for ConvGemm<D>
where
    D: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<D> + PartialEq,
{
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let input = args_1!(inputs);
        let output = self.conv_gemm(&input.to_array_view::<D>()?.into_dimensionality()?)?;
        Ok(tvec!(output.into()))
    }
}

impl<D> InferenceRulesOp for ConvGemm<D>
where
    D: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<D>,
{
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p SharedTensorsProxy,
        outputs: &'p SharedTensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 1)?;
        s.equals(&outputs.len, 1)?;
        s.equals(&inputs[0].datum_type, D::datum_type())?;
        s.equals(&outputs[0].datum_type, D::datum_type())?;
        s.equals(&outputs[0].shape, ShapeFact::from(&*self.full_output_shape))?;
        Ok(())
    }
}
