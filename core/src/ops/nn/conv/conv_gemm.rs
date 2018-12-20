use ndarray::prelude::*;
use ops::prelude::*;

use ops::nn::conv::KernelFormat;
use ops::nn::{DataFormat, Patch};

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

#[derive(Debug, Clone, new)]
pub struct ConvGemm<D>
where
    D: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<D> + PartialEq,
{
    pub patch: Patch,
    pub full_output_shape: TVec<usize>,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub kernel_fmt: KernelFormat,
    pub packed_kernels: Vec<Vec<D>>,
    pub bias: Option<ArrayD<D>>,
    pub group: usize,
}

impl<D> ConvGemm<D>
where
    D: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<D> + PartialEq,
{
    pub(super) fn conv_gemm<'i>(
        &'i self,
        packed_input: &'i ArrayView1<'i, D>,
    ) -> TractResult<ArrayD<D>> {
        let mut output = unsafe { ArrayD::<D>::uninitialized(&*self.full_output_shape) };
        let input_shape = &self.patch.input_shape;

        let c_panel_shape = (self.m, self.n);
        let mut c_panel = unsafe { Array2::uninitialized(c_panel_shape) };

        let co_per_group = self.full_output_shape[input_shape.c_axis()] / self.group;
        for i in 0..input_shape.n_dim() {
            for g in 0..self.group {
                let mut output_subview = output.view_mut();
                output_subview.slice_axis_inplace(Axis(input_shape.n_axis()), (i..(i + 1)).into());
                output_subview.slice_axis_inplace(
                    Axis(input_shape.c_axis()),
                    (g * co_per_group..(g + 1) * co_per_group).into(),
                );
                let a = &self.packed_kernels[g];

                use tract_linalg::fallback::Fallback as MM;
                use tract_linalg::Kernel;
                unsafe {
                    MM::mat_mul_prepacked(
                        self.m,
                        self.k,
                        self.n,
                        a.as_ptr() as *const f32,
                        packed_input.as_ptr().offset(
                            ((self.group * i + g) * MM::packed_b_len(self.k, self.n)) as isize,
                        ) as *const f32,
                        c_panel.as_mut_ptr() as *mut f32,
                        c_panel.strides()[0],
                        c_panel.strides()[1],
                    );
                }
                let shape = output_subview.shape().to_vec();
                match self.patch.input_shape.fmt {
                    DataFormat::NHWC => output_subview
                        .iter_mut()
                        .zip(c_panel.t().iter())
                        .for_each(|(o, c)| *o = *c),
                    DataFormat::NCHW => output_subview.assign(&c_panel.view().into_shape(shape)?),
                };
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
