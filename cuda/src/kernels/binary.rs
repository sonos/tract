use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use std::fmt;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::launch_args::LaunchArgsExt;
use crate::kernels::utils::compute_broadcast_strides;
use crate::kernels::{LibraryName, MAX_THREADS, get_cuda_view};

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
        write!(f, "{self:?}")
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

    pub fn name(&self) -> Cow<'_, str> {
        format!("{self}").into()
    }

    pub fn output_datum_type(&self, a: DatumType, b: DatumType) -> TractResult<DatumType> {
        ensure!(a == b);
        if self.is_logic() { Ok(DatumType::Bool) } else { Ok(a) }
    }

    pub fn output_shape<D: DimLike>(&self, a: &[D], b: &[D]) -> TractResult<TVec<D>> {
        tract_core::broadcast::multi_broadcast(&[a, b])
            .with_context(|| format!("Error while broadcasting {a:?} {b:?}"))
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

    pub fn reshape_to_rank_4_with_broadcast(
        lhs: &[usize],
        rhs: &[usize],
        out: &[usize],
    ) -> TractResult<(TVec<usize>, TVec<usize>, TVec<usize>)> {
        ensure!(lhs.len() == rhs.len() && lhs.len() == out.len(), "rank mismatch");
        let rank = lhs.len();

        // 1) Drop trivial axes (all ones for lhs/rhs/out).
        let keep: Vec<usize> =
            (0..rank).filter(|&i| !(lhs[i] == 1 && rhs[i] == 1 && out[i] == 1)).collect();

        if keep.is_empty() {
            return Ok((tvec![1, 1, 1, 1], tvec![1, 1, 1, 1], tvec![1, 1, 1, 1]));
        }

        let map = |shape: &[usize]| keep.iter().map(|&i| shape[i]).collect::<Vec<_>>();
        let lhs_k = map(lhs);
        let rhs_k = map(rhs);
        let out_k = map(out);
        let r = lhs_k.len();

        // 2) Fast path: if reduced rank <= 4, just right-align/pad.
        if r <= 4 {
            let pad = |shape: &[usize]| -> TVec<usize> {
                let mut res = [1usize; 4];
                res[4 - shape.len()..].copy_from_slice(shape);
                res.into()
            };
            return Ok((pad(&lhs_k), pad(&rhs_k), pad(&out_k)));
        }

        // 3) If lhs == rhs after filtering and r > 4, compress the prefix into the first group.
        if lhs_k == rhs_k {
            let mut shape = vec![lhs_k[..r - 3].iter().product::<usize>()];
            shape.extend(&lhs_k[r - 3..]);
            return Ok((shape.clone().into(), shape.clone().into(), shape.into()));
        }

        // 4) Build segments on the reduced arrays:
        //    - broadcast axes are singletons
        //    - non-broadcast axes form contiguous runs
        let is_broadcast = |i: usize| {
            (lhs_k[i] == 1 && rhs_k[i] == out_k[i] && out_k[i] != 1)
                || (rhs_k[i] == 1 && lhs_k[i] == out_k[i] && out_k[i] != 1)
        };

        let mut segments: Vec<Vec<usize>> = Vec::new();
        let mut cur = vec![0usize];
        let mut cur_is_b = is_broadcast(0);
        for i in 1..r {
            let b = is_broadcast(i);
            if b == cur_is_b {
                cur.push(i);
            } else {
                segments.push(std::mem::take(&mut cur));
                cur = vec![i];
                cur_is_b = b;
            }
        }
        segments.push(cur);
        ensure!(segments.len() <= 4, "Cannot reshape to rank 4 while isolating broadcasts");

        // 6) Right-align into exactly 4 groups, padding empties on the left.
        let pad = 4usize.saturating_sub(segments.len());
        let mut groups: [Vec<usize>; 4] = [vec![], vec![], vec![], vec![]];
        for (j, seg) in segments.into_iter().enumerate() {
            groups[pad + j] = seg;
        }

        let prod = |shape: &[usize], idxs: &Vec<usize>| {
            if idxs.is_empty() {
                1
            } else {
                idxs.iter().fold(1usize, |acc, &ix| acc.saturating_mul(shape[ix]))
            }
        };

        let lhs4: TVec<usize> = groups.iter().map(|g| prod(lhs, g)).collect();
        let rhs4: TVec<usize> = groups.iter().map(|g| prod(rhs, g)).collect();
        let out4: TVec<usize> = groups.iter().map(|g| prod(out, g)).collect();

        Ok((lhs4, rhs4, out4))
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
        stream: &TractCudaStream,
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
        stream: &TractCudaStream,
        lhs: &DeviceTensor,
        rhs: &DeviceTensor,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        ensure!(lhs.rank() == rhs.rank());

        let kernel_name = self.kernel_name(lhs.datum_type())?;

        let (lhs_shape, rhs_shape, out_shape) =
            Self::reshape_to_rank_4_with_broadcast(lhs.shape(), rhs.shape(), output.shape())?;

        let lhs_strides =
            compute_broadcast_strides::<usize>(&lhs_shape, &natural_strides(&lhs_shape))?;
        let rhs_strides =
            compute_broadcast_strides::<usize>(&rhs_shape, &natural_strides(&rhs_shape))?;
        let out_strides =
            compute_broadcast_strides::<usize>(&out_shape, &natural_strides(&out_shape))?;

        let func = cuda_context().load_pipeline(LibraryName::Binary, kernel_name)?;

        let half_inner_ax = (out_shape[3] / 2).max(1);
        let block_dim_x = half_inner_ax.min(MAX_THREADS);
        let block_dim_y = out_shape[2].min(MAX_THREADS / block_dim_x);
        let block_dim_z =
            (out_shape[1] * out_shape[0]).min(MAX_THREADS / (block_dim_x * block_dim_y)).min(64);

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

    #[test]
    fn isolates_two_broadcast_axes_exact() {
        let lhs = [2, 3, 1, 5];
        let rhs = [2, 1, 4, 5];
        let out = [2, 3, 4, 5];

        let (l4, r4, o4) = BinOps::reshape_to_rank_4_with_broadcast(&lhs, &rhs, &out).unwrap();
        assert_eq!(l4, tvec![2, 3, 1, 5]);
        assert_eq!(r4, tvec![2, 1, 4, 5]);
        assert_eq!(o4, tvec![2, 3, 4, 5]);
    }

    #[test]
    fn no_broadcast_splits_into_singletons_for_four_groups() {
        let lhs = [2, 6, 7, 8];
        let rhs = [2, 6, 7, 8];
        let out = [2, 6, 7, 8];

        let (l4, r4, o4) = BinOps::reshape_to_rank_4_with_broadcast(&lhs, &rhs, &out).unwrap();
        assert_eq!(l4, tvec![2, 6, 7, 8]);
        assert_eq!(r4, tvec![2, 6, 7, 8]);
        assert_eq!(o4, tvec![2, 6, 7, 8]);
    }

    #[test]
    fn split_heaviest_non_segment_around_broadcast() {
        let lhs = [2, 3, 1, 5, 7];
        let rhs = [2, 3, 3, 5, 7];
        let out = [2, 3, 3, 5, 7];

        let (l4, r4, o4) = BinOps::reshape_to_rank_4_with_broadcast(&lhs, &rhs, &out).unwrap();
        assert_eq!(l4, tvec![1, 6, 1, 35]);
        assert_eq!(r4, tvec![1, 6, 3, 35]);
        assert_eq!(o4, tvec![1, 6, 3, 35]);
    }

    #[test]
    fn right_align_with_padding_when_fewer_than_four_groups() {
        let lhs = [10, 1, 9];
        let rhs = [10, 9, 9];
        let out = [10, 9, 9];

        let (l4, r4, o4) = BinOps::reshape_to_rank_4_with_broadcast(&lhs, &rhs, &out).unwrap();
        assert_eq!(l4, tvec![1, 10, 1, 9]);
        assert_eq!(r4, tvec![1, 10, 9, 9]);
        assert_eq!(o4, tvec![1, 10, 9, 9]);
    }

    #[test]
    fn scalar_broadcast() {
        let lhs = [1, 8, 4, 10, 12];
        let rhs = [1, 1, 1, 1, 1];
        let out = [1, 8, 4, 10, 12];

        let (l4, r4, o4) = BinOps::reshape_to_rank_4_with_broadcast(&lhs, &rhs, &out).unwrap();
        assert_eq!(l4, tvec![8, 4, 10, 12]);
        assert_eq!(r4, tvec![1, 1, 1, 1]);
        assert_eq!(o4, tvec![8, 4, 10, 12]);
    }

    #[test]
    fn supports_adjacent_broadcasts_coalesced() {
        let lhs = [2, 3, 4, 4, 4];
        let rhs = [1, 3, 1, 1, 1];
        let out = [2, 3, 4, 4, 4];

        let (l4, r4, o4) = BinOps::reshape_to_rank_4_with_broadcast(&lhs, &rhs, &out).unwrap();

        assert_eq!(l4, tvec![1, 2, 3, 4 * 4 * 4]);
        assert_eq!(r4, tvec![1, 1, 3, 1]);
        assert_eq!(o4, tvec![1, 2, 3, 64]);
    }

    #[test]
    fn too_many_segments_errors() {
        let lhs = [2, 1, 3, 1, 5, 1, 7];
        let rhs = [2, 9, 3, 8, 5, 6, 7];
        let out = [2, 9, 3, 8, 5, 6, 7];
        let err = BinOps::reshape_to_rank_4_with_broadcast(&lhs, &rhs, &out).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("Cannot reshape to rank 4"), "{msg}");
    }
}
