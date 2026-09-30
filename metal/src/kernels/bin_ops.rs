use super::BroadcastKind;
use super::utils::build_metal_grid_and_groups_for_el_wise_op;
use crate::encoder::EncoderExt;
use crate::{LibraryName, MetalStream};
use anyhow::ensure;
use metal::{MTLSize, NSUInteger};
use std::ffi::c_void;
use tract_core::internal::*;
use tract_core::ops::binary::BinMiniOp;
use tract_gpu::tensor::DeviceTensor;
use tract_gpu::utils::Layout;
use tract_gpu::utils::merge_axes_to_fit;

/// The axes the generic binary kernel walks: the grid carries three, its depth
/// folding the two outermost, and the innermost is the kernel's loop.
const BINARY_MAX_RANK: usize = 5;

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
        && lhs.shape()[rank - 1].is_multiple_of(4)
        && rhs.shape()[rank - 1].is_multiple_of(4)
}

/// The two operands and the output as the kernel addresses them: a shape and a
/// stride per axis, an operand broadcast along an axis holding a stride of zero
/// there, merged down to the `BINARY_MAX_RANK` axes the kernel walks.
fn binary_geometry(
    lhs: &DeviceTensor,
    rhs: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<[(TVec<usize>, TVec<usize>); 3]> {
    let rank = lhs.rank();
    let mut shapes: [TVec<usize>; 3] =
        [lhs.shape().into(), rhs.shape().into(), output.shape().into()];
    let mut strides: [TVec<isize>; 3] =
        [lhs.strides().into(), rhs.strides().into(), output.strides().into()];
    // An axis no tensor steps costs a kernel axis and carries nothing, and its
    // stride is arbitrary, which would block an otherwise exact merge.
    for axis in (0..rank).rev() {
        if shapes.iter().all(|shape| shape[axis] == 1) {
            for (shape, strides) in shapes.iter_mut().zip(strides.iter_mut()) {
                shape.remove(axis);
                strides.remove(axis);
            }
        }
    }
    for axis in 0..shapes[2].len() {
        // An operand shorter than the output is read at the same place for
        // every index of that axis.
        for operand in 0..2 {
            if shapes[operand][axis] < shapes[2][axis] {
                strides[operand][axis] = 0;
            }
        }
    }

    let [ref mut lhs_shape, ref mut rhs_shape, ref mut out_shape] = shapes;
    let [ref mut lhs_strides, ref mut rhs_strides, ref mut out_strides] = strides;
    // The output goes last: `merge_axes_to_fit` reads the extent of the pair's
    // inner half from it, and a broadcast operand holds one there.
    let mut layouts = [
        Layout { shape: lhs_shape, strides: lhs_strides },
        Layout { shape: rhs_shape, strides: rhs_strides },
        Layout { shape: out_shape, strides: out_strides },
    ];
    let merged = merge_axes_to_fit(&mut layouts, BINARY_MAX_RANK);
    ensure!(
        merged <= BINARY_MAX_RANK,
        "Binary of rank {rank} has no adjacent axes to merge into {BINARY_MAX_RANK}: \
         lhs {:?} rhs {:?}",
        lhs.shape(),
        rhs.shape(),
    );

    // The kernel takes the innermost axis as its loop and the outer ones as the
    // grid, so a shorter shape is right-aligned into the five.
    let pad = BINARY_MAX_RANK - merged;
    Ok(std::array::from_fn(|i| {
        let mut shape: TVec<usize> = tvec!(1; BINARY_MAX_RANK);
        let mut stride: TVec<usize> = tvec!(0; BINARY_MAX_RANK);
        for axis in 0..merged {
            shape[pad + axis] = shapes[i][axis];
            stride[pad + axis] = strides[i][axis] as usize;
        }
        (shape, stride)
    }))
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
        let [(lhs_shape, lhs_strides), (rhs_shape, rhs_strides), (out_shape, out_strides)] =
            binary_geometry(lhs, rhs, output)?;

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

pub fn metal_bin_op(mini_op: Box<dyn BinMiniOp>) -> tract_gpu::ops::binary::GpuBinOp {
    tract_gpu::ops::binary::GpuBinOp::new(mini_op, "Metal", metal_bin_op_dispatch)
}

crate::register_metal_op!(tract_core::ops::binary::TypedBinOp, |source, node, op| {
    rule_if!(is_supported(&*op.0, source.node_input_facts(node.id)?[0].datum_type));
    Ok(Some(Box::new(metal_bin_op(op.0.clone()))))
});

crate::register_metal_op!(tract_core::ops::logic::Iff, |_source, _node, _op| {
    Ok(Some(Box::new(tract_gpu::ops::iff::GpuIff::new("Metal", metal_iff_dispatch))))
});

#[allow(clippy::too_many_arguments)]
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
        let mut plain_out = out.try_as_plain_ram_mut()?;
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
