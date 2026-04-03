use super::BroadcastKind;
use super::utils::build_metal_grid_and_groups_for_el_wise_op;
use crate::encoder::EncoderExt;
use crate::kernels::utils::compute_broadcast_strides;
use crate::{LibraryName, MetalStream};
use anyhow::ensure;
use metal::{MTLSize, NSUInteger};
use std::ffi::c_void;
use tract_core::internal::tract_smallvec::SmallVec;
use tract_core::internal::*;
use tract_core::ops::binary::BinMiniOp;
use tract_gpu::tensor::DeviceTensor;

const ALL_OP_NAMES: &[&str] = &[
    "mul", "add", "div", "sub", "pow", "min", "max", "gt", "gte", "eq", "ne", "lt", "lte", "and",
    "or", "bitor", "bitand", "bitxor",
];

pub fn all_functions() -> Vec<String> {
    ALL_OP_NAMES
        .iter()
        .flat_map(|kname| {
            DeviceTensor::SUPPORTED_DT.into_iter().flat_map(move |dt| {
                let tname = DeviceTensor::tname(dt).ok()?;
                Some([true, false].into_iter().map(move |row| {
                    if row {
                        format!("bin_ops::{kname}_1row_{tname}")
                    } else {
                        format!("bin_ops::{kname}_{tname}")
                    }
                }))
            })
        })
        .flatten()
        .chain(
            ["u8", "u16", "u32", "u64"]
                .into_iter()
                .map(|tname| format!("bin_ops::iff_generic_{tname}")),
        )
        .collect()
}

pub fn is_supported(mini_op: &dyn BinMiniOp, dt: DatumType) -> bool {
    ALL_OP_NAMES.contains(&mini_op.name().to_lowercase().as_str())
        && (dt.is_number() || dt.is::<bool>())
}

fn kernel_name(op_name: &str, dt: DatumType, use_row_kernel: bool) -> TractResult<String> {
    let tname = DeviceTensor::tname(dt)?;
    if use_row_kernel {
        Ok(format!("bin_ops::{op_name}_1row_{tname}"))
    } else {
        Ok(format!("bin_ops::{op_name}_{tname}"))
    }
}

fn can_use_row_kernel(mini_op: &dyn BinMiniOp, lhs: &DeviceTensor, rhs: &DeviceTensor) -> bool {
    let compatible_op = matches!(mini_op.name(), "Mul" | "Add" | "Div" | "Sub");
    let compatible_type = matches!(lhs.datum_type(), DatumType::F16 | DatumType::F32);
    let rank = lhs.rank();

    compatible_op
        && compatible_type
        && (rank > 0)
        && ((rhs.len() == rhs.shape()[rank - 1])
            || ((lhs.len() == lhs.shape()[rank - 1]) && matches!(mini_op.name(), "Mul" | "Add")))
        && (lhs.shape()[rank - 1] % 4 == 0)
        && (rhs.shape()[rank - 1] % 4 == 0)
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

fn natural_strides(shape: &[usize]) -> SmallVec<[isize; 4]> {
    let mut strides = SmallVec::from_elem(1, shape.len());
    for i in (0..shape.len()).rev().skip(1) {
        strides[i] = strides[i + 1] * shape[i + 1] as isize;
    }
    strides
}

pub fn dispatch_eval(
    stream: &MetalStream,
    mini_op: &dyn BinMiniOp,
    lhs: &DeviceTensor,
    rhs: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    stream.retain_tensor(lhs);
    stream.retain_tensor(rhs);
    stream.retain_tensor(output);

    ensure!(lhs.rank() == rhs.rank());
    let rank = lhs.rank();

    let use_row = can_use_row_kernel(mini_op, lhs, rhs);
    let op_name = mini_op.name().to_lowercase();
    let kname = kernel_name(&op_name, lhs.datum_type(), use_row)?;

    if use_row {
        let pipeline = stream.load_pipeline(LibraryName::BinOps, &kname)?;

        let (a, b) = if rhs.len() == rhs.shape()[rank - 1] { (lhs, rhs) } else { (rhs, lhs) };
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
        let (lhs_shape, rhs_shape, out_shape) = reshape_to_rank_4_with_broadcast(lhs, rhs, output)?;

        let lhs_strides =
            compute_broadcast_strides::<usize>(&lhs_shape, &*natural_strides(&lhs_shape))?;
        let rhs_strides =
            compute_broadcast_strides::<usize>(&rhs_shape, &natural_strides(&rhs_shape))?;
        let out_strides =
            compute_broadcast_strides::<usize>(&out_shape, &natural_strides(&out_shape))?;

        let pipeline = stream.load_pipeline(LibraryName::BinOps, &kname)?;
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

pub fn metal_bin_op_dispatch(
    mini_op: &dyn BinMiniOp,
    lhs: &DeviceTensor,
    rhs: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    crate::with_metal_stream(|stream| dispatch_eval(stream, mini_op, lhs, rhs, output))
}

pub fn metal_iff_dispatch(
    cond: &DeviceTensor,
    then_value: &DeviceTensor,
    else_value: &DeviceTensor,
    cond_strides: &[isize],
    then_strides: &[isize],
    else_strides: &[isize],
    output: &DeviceTensor,
    output_shape: &[usize],
    output_strides: &[isize],
) -> TractResult<()> {
    crate::with_metal_stream(|stream| {
        stream.retain_tensor(cond);
        stream.retain_tensor(then_value);
        stream.retain_tensor(else_value);
        stream.retain_tensor(output);

        let tname = tract_gpu::utils::BroadcastKind::copy_tname(output.datum_type());
        let kernel_name = format!("bin_ops::iff_generic_{tname}");
        let total_elems: usize = output_shape.iter().product();

        let pipeline = stream.load_pipeline(LibraryName::BinOps, &kernel_name)?;
        let command_buffer = stream.command_buffer();

        let cond_strides_usize: TVec<usize> = cond_strides.iter().map(|&s| s as usize).collect();
        let then_strides_usize: TVec<usize> = then_strides.iter().map(|&s| s as usize).collect();
        let else_strides_usize: TVec<usize> = else_strides.iter().map(|&s| s as usize).collect();
        let out_strides_usize: TVec<usize> = output_strides.iter().map(|&s| s as usize).collect();

        command_buffer.encode(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_metal_tensor(0, cond, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(1, then_value, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(2, else_value, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(3, output, metal::MTLResourceUsage::Write);
            encoder.set_slice(4, output_shape);
            encoder.set_slice(5, &cond_strides_usize);
            encoder.set_slice(6, &then_strides_usize);
            encoder.set_slice(7, &else_strides_usize);
            encoder.set_slice(8, &out_strides_usize);

            let grid_size = MTLSize { width: total_elems as NSUInteger, height: 1, depth: 1 };
            let group_size = MTLSize { width: 1, height: 1, depth: 1 };
            encoder.dispatch_thread_groups(grid_size, group_size);
        });
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use crate::utils::with_borrowed_metal_stream;

    use super::*;
    use tract_gpu::tensor::IntoDevice;

    fn reference<FI: Datum, FO: Datum>(
        a: &Tensor,
        b: &Tensor,
        cab: impl Fn(&mut FO, &FI, &FI),
    ) -> TractResult<Tensor> {
        let out_shape = tract_core::broadcast::multi_broadcast(&[a.shape(), b.shape()])?;
        let mut out = unsafe { Tensor::uninitialized_dt(FO::datum_type(), &out_shape)? };
        let a_view = a.to_plain_array_view::<FI>()?;
        let b_view = b.to_plain_array_view::<FI>()?;
        let mut plain_out = out.try_as_plain_mut()?;
        let mut c = plain_out.to_array_view_mut::<FO>()?;
        tract_core::ndarray::Zip::from(&mut c)
            .and_broadcast(a_view)
            .and_broadcast(b_view)
            .for_each(cab);
        Ok(out)
    }

    fn run_test_case_logic(
        mini_op: &dyn BinMiniOp,
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

            let out_dt = mini_op.result_datum_type(a.datum_type(), b.datum_type())?;
            let out_shape = tract_core::broadcast::multi_broadcast(&[a.shape(), b.shape()])?;
            let output = unsafe { DeviceTensor::uninitialized_dt(out_dt, &out_shape)? };
            dispatch_eval(stream, mini_op, &a, &b, &output)?;
            stream.wait_until_completed()?;

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
        run_test_case_logic(&tract_core::ops::logic::And, &[2, 4], &[2, 4], |c, a, b| {
            *c = *a && *b
        })?;
        run_test_case_logic(&tract_core::ops::logic::Or, &[2, 4], &[2, 4], |c, a, b| {
            *c = *a || *b
        })?;
        Ok(())
    }
}
