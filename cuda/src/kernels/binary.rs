use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use std::fmt;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::context::cuda_context;
use crate::kernels::launch_args::LaunchArgsExt;
use crate::kernels::utils::compute_broadcast_strides;
use crate::kernels::{LibraryName, get_cuda_view};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub enum BinOps {
    Mul,
    Add,
    Div,
    Sub,
    Pow,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Equals,
    NotEquals,
    And,
    Or,
}

impl fmt::Display for BinOps {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl BinOps {
    pub const ALL: [BinOps; 13] = [
        Self::Mul,
        Self::Add,
        Self::Div,
        Self::Sub,
        Self::Pow,
        Self::Less,
        Self::LessEqual,
        Self::Greater,
        Self::GreaterEqual,
        Self::Equals,
        Self::NotEquals,
        Self::And,
        Self::Or,
    ];

    pub fn name(&self) -> Cow<str> {
        format!("{}", self).into()
    }

    pub fn output_datum_type(&self, a: DatumType, b: DatumType) -> TractResult<DatumType> {
        ensure!(a == b);
        if self.is_logic() { Ok(DatumType::Bool) } else { Ok(a) }
    }

    pub fn output_shape<D: DimLike>(&self, a: &[D], b: &[D]) -> TractResult<TVec<D>> {
        tract_core::broadcast::multi_broadcast(&[a, b])
            .with_context(|| format!("Error while broadcasting {:?} {:?}", a, b))
    }

    pub fn all_functions() -> Vec<String> {
        Self::ALL
            .into_iter()
            .flat_map(|op| DeviceTensor::SUPPORTED_DT.into_iter().map(move |dt| (op, dt)))
            .flat_map(|(op, dt)| op.kernel_name(dt).into_iter())
            .collect()
    }

    pub fn is_logic(&self) -> bool {
        matches!(
            self,
            Self::Less
                | Self::LessEqual
                | Self::Greater
                | Self::GreaterEqual
                | Self::Equals
                | Self::NotEquals
                | Self::And
                | Self::Or
        )
    }

    pub fn is_supported_dt(&self, dt: DatumType) -> bool {
        (matches!(self, Self::And | Self::Or) && dt == DatumType::Bool)
            || (!matches!(self, Self::And | Self::Or)
                && matches!(
                    dt,
                    DatumType::F32
                        | DatumType::F16
                        | DatumType::U8
                        | DatumType::U16
                        | DatumType::U32
                        | DatumType::U64
                        | DatumType::I8
                        | DatumType::I16
                        | DatumType::I32
                        | DatumType::I64
                ))
    }

    fn reshape_to_rank_4_with_broadcast(
        lhs: &DeviceTensor,
        rhs: &DeviceTensor,
        out: &DeviceTensor,
    ) -> TractResult<(TVec<usize>, TVec<usize>, TVec<usize>)> {
        let rank = lhs.rank();

        if rank <= 4 {
            let pad = |shape: &[usize]| {
                let mut result = [1; 4];
                result[4 - shape.len()..].copy_from_slice(shape);
                result.into()
            };
            return Ok((pad(lhs.shape()), pad(rhs.shape()), pad(out.shape())));
        }

        if lhs.shape() == rhs.shape() {
            let mut shape = vec![lhs.shape()[..rank - 3].iter().product::<usize>()];
            shape.extend(&lhs.shape()[rank - 3..]);

            Ok((shape.clone().into(), shape.clone().into(), shape.into()))
        } else {
            let broadcast_axes: Vec<usize> = (0..lhs.rank())
                .filter(|ix| lhs.shape()[*ix] != rhs.shape()[*ix] || lhs.shape()[*ix] == 1)
                .collect();

            let mut segments = vec![];
            let mut current_segment = vec![0];
            let mut current_is_broadcast = broadcast_axes.contains(&0);

            for i in 1..rank {
                let is_broadcast = broadcast_axes.contains(&i);
                if is_broadcast == current_is_broadcast {
                    current_segment.push(i);
                } else {
                    segments.push((current_is_broadcast, current_segment));
                    current_segment = vec![i];
                    current_is_broadcast = is_broadcast;
                }
            }
            segments.push((current_is_broadcast, current_segment));

            let mut reshaped_groups: Vec<Vec<usize>> = vec![vec![], vec![], vec![], vec![]];
            let mut group_idx = 0;
            for (_, segment) in segments {
                reshaped_groups[group_idx].extend(segment);
                group_idx += 1;
                ensure!(group_idx < 4, "Cannot reshape to rank 4");
            }

            fn compute_shape(shape: &[usize], groups: &[Vec<usize>]) -> TVec<usize> {
                let mut result = [1; 4];
                for (i, group) in groups.iter().enumerate() {
                    result[i] = group.iter().map(|&dim| shape[dim]).product();
                }
                result.into()
            }

            Ok((
                compute_shape(lhs.shape(), &reshaped_groups),
                compute_shape(rhs.shape(), &reshaped_groups),
                compute_shape(out.shape(), &reshaped_groups),
            ))
        }
    }

    pub fn kernel_name(&self, dt: DatumType) -> TractResult<String> {
        ensure!(self.is_supported_dt(dt), "Unsupported dt {:?} for Cuda binary ops: {self}", dt);

        let tname = DeviceTensor::tname(dt)?;

        let kname = match self {
            Self::Mul => "mul",
            Self::Add => "add",
            Self::Div => "div",
            Self::Sub => "sub",
            Self::Pow => "pow",
            Self::Greater => "greater",
            Self::GreaterEqual => "greater_equal",
            Self::Equals => "equals",
            Self::NotEquals => "not_equals",
            Self::Less => "less",
            Self::LessEqual => "less_equal",
            Self::And => "and",
            Self::Or => "or",
        };

        Ok(format!("binary_{kname}_{tname}"))
    }

    pub fn eval(
        &self,
        stream: &CudaStream,
        lhs: &DeviceTensor,
        rhs: &DeviceTensor,
    ) -> TractResult<DeviceTensor> {
        let out_shape = self.output_shape(lhs.shape(), rhs.shape())?;
        let out_dt = self.output_datum_type(lhs.datum_type(), rhs.datum_type())?;
        let output = unsafe { DeviceTensor::uninitialized_dt(out_dt, &out_shape)? };

        self.dispatch_eval(stream, lhs, rhs, &output)?;

        stream.synchronize()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        stream: &CudaStream,
        lhs: &DeviceTensor,
        rhs: &DeviceTensor,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        ensure!(lhs.rank() == rhs.rank());

        let kernel_name = self.kernel_name(lhs.datum_type())?;

        let (lhs_shape, rhs_shape, out_shape) =
            Self::reshape_to_rank_4_with_broadcast(lhs, rhs, output)?;

        let lhs_strides =
            compute_broadcast_strides::<usize>(&lhs_shape, &natural_strides(&lhs_shape))?;
        let rhs_strides =
            compute_broadcast_strides::<usize>(&rhs_shape, &natural_strides(&rhs_shape))?;
        let out_strides =
            compute_broadcast_strides::<usize>(&out_shape, &natural_strides(&out_shape))?;

        let func = cuda_context().load_pipeline(LibraryName::Binary, kernel_name)?;

        let max_threads = 1024;
        let half_inner_ax = (out_shape[3] / 2).max(1);
        let block_dim_x = half_inner_ax.min(max_threads);
        let block_dim_y = out_shape[2].min(max_threads / block_dim_x);
        let block_dim_z =
            (out_shape[1] * out_shape[0]).min(max_threads / (block_dim_x * block_dim_y)).min(64);

        let cfg = LaunchConfig {
            grid_dim: (
                half_inner_ax.div_ceil(block_dim_x) as _,
                out_shape[2].div_ceil(block_dim_y) as _,
                (out_shape[0] * out_shape[1]).div_ceil(block_dim_z) as _,
            ),
            block_dim: (block_dim_x as _, block_dim_y as _, block_dim_z as _),
            shared_mem_bytes: 0,
        };

        let lhs_view = get_cuda_view(lhs);
        let rhs_view = get_cuda_view(rhs);
        let o_view = get_cuda_view(output);

        let mut launch_args = stream.launch_builder(&func);
        launch_args.arg(&lhs_view);
        launch_args.arg(&rhs_view);
        launch_args.arg(&o_view);
        launch_args.set_slice(&rhs_shape);
        launch_args.set_slice(&out_shape);
        launch_args.set_slice(&lhs_strides);
        launch_args.set_slice(&rhs_strides);
        launch_args.set_slice(&out_strides);

        unsafe { launch_args.launch(cfg) }?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use tract_gpu::tensor::IntoDevice;

    use super::*;

    use crate::context::CUDA_STREAM;

    /* Except for And and Or, Binops are proptest for almost all types  */

    fn reference<FI: Datum, FO: Datum>(
        a: &Tensor,
        b: &Tensor,
        cab: impl Fn(&mut FO, &FI, &FI),
    ) -> TractResult<Tensor> {
        let out_shape = tract_core::broadcast::multi_broadcast(&[a.shape(), b.shape()])?;
        let mut out = unsafe { Tensor::uninitialized_dt(FO::datum_type(), &out_shape)? };
        let a_view = a.to_array_view::<FI>()?;
        let b_view = b.to_array_view::<FI>()?;
        let mut c = out.to_array_view_mut::<FO>()?;
        tract_core::ndarray::Zip::from(&mut c)
            .and_broadcast(a_view)
            .and_broadcast(b_view)
            .for_each(cab);
        Ok(out)
    }

    fn run_test_case_logic(
        op: BinOps,
        a_shape: &[usize],
        b_shape: &[usize],
        cab: impl Fn(&mut bool, &bool, &bool),
    ) -> TractResult<()> {
        CUDA_STREAM.with(|stream| {
            let a_len = a_shape.iter().product::<usize>();
            let b_len = b_shape.iter().product::<usize>();

            let a =
                Tensor::from_shape(a_shape, &(0..a_len).map(|f| f % 2 == 0).collect::<Vec<_>>())?
                    .into_device()?;
            let b =
                Tensor::from_shape(b_shape, &(0..b_len).map(|f| f % 4 == 0).collect::<Vec<_>>())?
                    .into_device()?;
            let output = op.eval(stream, &a, &b)?;
            let ref_output = reference::<bool, bool>(
                &a.to_host()?.into_tensor(),
                &b.to_host()?.into_tensor(),
                cab,
            )?;

            assert_eq!(output.to_host()?.into_tensor(), ref_output);
            Ok(())
        })
    }

    #[test]
    fn test_logic() -> TractResult<()> {
        run_test_case_logic(BinOps::And, &[2, 4], &[2, 4], |c, a, b| *c = *a && *b)?;
        run_test_case_logic(BinOps::Or, &[2, 4], &[2, 4], |c, a, b| *c = *a || *b)?;
        Ok(())
    }
}
