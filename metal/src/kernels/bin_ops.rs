use super::BroadcastKind;
use super::utils::build_metal_grid_and_groups_for_el_wise_op;
use crate::encoder::EncoderExt;
use crate::kernels::utils::compute_broadcast_strides;
use crate::{LibraryName, MetalStream};
use anyhow::{bail, ensure};
use metal::{Device, MTLSize, NSUInteger};
use std::ffi::c_void;
use std::fmt;
use tract_core::internal::tract_smallvec::SmallVec;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

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

    pub fn name(&self) -> StaticName {
        format!("{}", self).into()
    }

    pub fn validation(&self) -> Validation {
        Validation::Accurate
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
            .flat_map(|(op, dt)| [true, false].into_iter().map(move |r| (op, dt, r)))
            .flat_map(|(op, dt, r)| op.kernel_name(dt, r).into_iter())
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

    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(
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
                | DatumType::Bool
        )
    }

    fn reshape_to_rank_4_with_broadcast(
        lhs: &DeviceTensor,
        rhs: &DeviceTensor,
        out: &DeviceTensor,
    ) -> TractResult<(TVec<usize>, TVec<usize>, TVec<usize>)> {
        let rank = lhs.rank();

        if rank <= 4 {
            let mut pad = |shape: &[usize]| {
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

    fn can_use_row_kernel(&self, lhs: &DeviceTensor, rhs: &DeviceTensor) -> bool {
        let compatible_op = matches!(self, Self::Mul | Self::Add | Self::Div | Self::Sub);
        let compatible_type = matches!(lhs.datum_type(), DatumType::F16 | DatumType::F32);
        let rank = lhs.rank();

        compatible_op
            && compatible_type
            && (rank > 0)
            && ((rhs.len() == rhs.shape()[rank - 1])
                || ((lhs.len() == lhs.shape()[rank - 1]) && matches!(self, Self::Mul | Self::Add)))
            && (lhs.shape()[rank - 1] % 4 == 0)
            && (rhs.shape()[rank - 1] % 4 == 0)
    }

    pub fn kernel_name(&self, dt: DatumType, use_row_kernel: bool) -> TractResult<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupported dt {:?} for metal binary ops", dt);

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

        if use_row_kernel {
            Ok(format!("bin_ops::{kname}_1row_{tname}"))
        } else {
            Ok(format!("bin_ops::{kname}_{tname}"))
        }
    }

    pub fn eval(
        &self,
        stream: &MetalStream,
        lhs: &DeviceTensor,
        rhs: &DeviceTensor,
    ) -> TractResult<DeviceTensor> {
        let out_shape = self.output_shape(lhs.shape(), rhs.shape())?;
        let out_dt = self.output_datum_type(lhs.datum_type(), rhs.datum_type())?;
        let output = unsafe { DeviceTensor::uninitialized_dt(out_dt, &out_shape)? };

        self.dispatch_eval(stream, lhs, rhs, &output)?;

        stream.wait_until_completed()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        stream: &MetalStream,
        lhs: &DeviceTensor,
        rhs: &DeviceTensor,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        stream.retain_tensor(lhs);
        stream.retain_tensor(rhs);
        stream.retain_tensor(output);

        ensure!(lhs.rank() == rhs.rank());
        let rank = lhs.rank();
        let out_shape = output.shape();

        let use_row_kernel = self.can_use_row_kernel(lhs, rhs);

        let kernel_name = self.kernel_name(lhs.datum_type(), use_row_kernel)?;

        if use_row_kernel {
            let pipeline = stream.load_pipeline(LibraryName::BinOps, &kernel_name)?;

            let (a, b) = if (rhs.len() == rhs.shape()[rank - 1]) { (lhs, rhs) } else { (rhs, lhs) };
            let command_buffer = stream.command_buffer();
            command_buffer.encode(|encoder| {
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_metal_tensor(0, a, metal::MTLResourceUsage::Read);
                encoder.set_metal_tensor(1, b, metal::MTLResourceUsage::Read);
                encoder.set_metal_tensor(2, output, metal::MTLResourceUsage::Write);
                encoder.set_bytes(
                    3,
                    std::mem::size_of::<usize>() as u64,
                    &b.len() as *const usize as *const c_void,
                );

                let grid_size =
                    MTLSize { width: (output.len() / 4) as NSUInteger, height: 1, depth: 1 };
                let group_size = MTLSize { width: 1, height: 1, depth: 1 };
                encoder.dispatch_thread_groups(grid_size, group_size);
            });
        } else {
            let (lhs_shape, rhs_shape, out_shape) =
                Self::reshape_to_rank_4_with_broadcast(lhs, rhs, output)?;

            let lhs_strides =
                compute_broadcast_strides::<usize>(&lhs_shape, &natural_strides(&lhs_shape))?;
            let rhs_strides =
                compute_broadcast_strides::<usize>(&rhs_shape, &natural_strides(&rhs_shape))?;
            let out_strides =
                compute_broadcast_strides::<usize>(&out_shape, &natural_strides(&out_shape))?;

            let pipeline = stream.load_pipeline(LibraryName::BinOps, &kernel_name)?;
            let command_buffer = stream.command_buffer();
            command_buffer.encode(|encoder| {
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_metal_tensor(0, lhs, metal::MTLResourceUsage::Read);
                encoder.set_slice(1, &lhs_shape);
                encoder.set_slice(2, &lhs_strides);
                encoder.set_metal_tensor(3, rhs, metal::MTLResourceUsage::Read);
                encoder.set_slice(4, &rhs_shape);
                encoder.set_slice(5, &rhs_strides);
                encoder.set_metal_tensor(6, output, metal::MTLResourceUsage::Write);
                encoder.set_slice(7, &out_shape);
                encoder.set_slice(8, &out_strides);

                let (grid_size, group_size) = build_metal_grid_and_groups_for_el_wise_op(
                    &out_shape,
                    pipeline.max_total_threads_per_threadgroup() as _,
                );
                encoder.dispatch_thread_groups(grid_size, group_size);
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::with_borrowed_metal_stream;

    use super::*;
    use tract_gpu::tensor::IntoDevice;

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
        with_borrowed_metal_stream(|stream| {
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
