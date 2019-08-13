use num_traits::Zero;
use std::ops::{Add, AddAssign, Mul};

use crate::internal::*;
use ndarray::prelude::*;

use crate::ops::cnn::conv::KernelFormat;
use crate::ops::cnn::Patch;
use crate::ops::nn::{DataFormat, DataShape};

use tract_linalg::{NonLinearSpec, Tile};

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
pub struct MatMat<T>
where
    T: Datum + Add + Mul + Zero + Copy,
{
    pub patch: Patch,
    pub output_shape: DataShape,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub kernel_fmt: KernelFormat,
    #[debug(skip)]
    pub packed_kernels: Vec<Tensor>,
    pub group: usize,
    pub tile: Box<dyn Tile<T>>,
    pub bias: Option<Vec<NonLinearSpec<T>>>,
    pub non_linear: Vec<NonLinearSpec<T>>,
}

impl<T> MatMat<T>
where
    T: Datum + Add + Mul + Zero + Copy + AddAssign + ndarray::LinalgScalar,
{
    pub(super) fn conv_gemm<'i>(
        &'i self,
        packed_input: &'i ArrayView3<'i, T>,
    ) -> TractResult<ArrayD<T>> {
        let mut output = unsafe { ArrayD::<T>::uninitialized(&*self.output_shape.shape) };
        let packed_b_len = self.tile.b_pack().len();

        let co_per_group = self.output_shape.c() / self.group;

        let (rsc, csc) = match self.output_shape.fmt {
            DataFormat::NHWC => (1, (self.m * self.group) as isize),
            DataFormat::NCHW => (self.n as isize, 1),
        };

        for i in 0..*self.output_shape.n() {
            unsafe {
                let output_i =
                    output.as_mut_ptr().offset(*self.output_shape.n_stride() as isize * i as isize);
                for g in 0..self.group {
                    let a = &self.packed_kernels[g];
                    let output_i_g = output_i.offset(
                        *self.output_shape.c_stride() as isize * co_per_group as isize * g as isize,
                    );

                    let mut non_linear = vec![];
                    if let Some(bias) = &self.bias {
                        non_linear.push(bias[g].clone());
                    }
                    for nl in &self.non_linear {
                        non_linear.push(nl.clone());
                    }

                    self.tile.run(
                        &self.tile.a_from_packed(a.as_ptr()?),
                        &self.tile.b_from_packed(
                            packed_input
                                .as_ptr()
                                .offset(((self.group * i + g) * packed_b_len) as isize),
                        ),
                        &mut self.tile.c_from_data_and_strides(output_i_g, rsc, csc),
                        &*non_linear,
                    );
                }
            }
        }
        Ok(output)
    }
}

impl<T> Op for MatMat<T>
where
    T: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<T> + PartialEq,
{
    fn name(&self) -> Cow<str> {
        "MatMat".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("{:?}", self.tile)])
    }

    fn cost(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<(Cost, TDim)>> {
        let batch = inputs[0].shape.dim(0);
        Ok(tvec!((
            Cost::FMA(T::datum_type()),
            batch * self.group * self.tile.m() * self.tile.k() * self.tile.n()
        )))
    }

    fn fuse(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
        if let Some(succ) = model.single_succ(node.id)? {
            let fused_micro_op = (|| -> TractResult<Option<NonLinearSpec<T>>> {
                if let Some(op) = succ.op_as::<crate::ops::math::Mul::UnaryA>() {
                    if op.b.shape() == &[*self.output_shape.c()] {
                        return Ok(Some(NonLinearSpec::PerRowMul(
                            op.b.as_slice::<T>()?.to_vec(),
                        )));
                    }
                } else if let Some(op) = succ.op_as::<crate::ops::math::Add::UnaryA>() {
                    if op.b.shape() == &[*self.output_shape.c()] {
                        return Ok(Some(NonLinearSpec::PerRowAdd(
                            op.b.as_slice::<T>()?.to_vec(),
                        )));
                    }
                } else if succ.op_is::<crate::ops::nn::Relu>() {
                    return Ok(Some(NonLinearSpec::Max(T::zero())));
                }
                Ok(None)
            })()?;
            if let Some(op) = fused_micro_op {
                let mut ops = self.non_linear.clone();
                ops.push(op);
                return Ok(Some(TypedModelPatch::fuse_with_next(
                    model,
                    &node,
                    Self { non_linear: ops, ..self.clone() },
                )?));
            }
        }
        Ok(None)
    }
}

impl<D> StatelessOp for MatMat<D>
where
    D: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<D> + PartialEq,
{
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let output = self.conv_gemm(&input.to_array_view::<D>()?.into_dimensionality()?)?;
        Ok(tvec!(output.into_arc_tensor()))
    }
}
