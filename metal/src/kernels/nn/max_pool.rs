use crate::encoder::EncoderExt;
use crate::{LibraryName, MetalStream};
use anyhow::ensure;
use metal::MTLSize;
use std::fmt;
use tract_core::internal::*;
use tract_core::ops::cnn::pools::PoolSpec;
use tract_core::ops::cnn::{MaxPool, OptMaxPool};
use tract_gpu::ops::max_pool::{GpuMaxPool, MaxPool2dGeometry};
use tract_gpu::tensor::DeviceTensor;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MaxPool2d;

impl fmt::Display for MaxPool2d {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl MaxPool2d {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16)
    }

    pub fn kernel_name(&self, dt: DatumType) -> TractResult<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupported dt {:?} for metal max pool op", dt);
        let tname = DeviceTensor::tname(dt)?;
        Ok(format!("nn_ops::max_pool_2d_{tname}"))
    }

    pub fn dispatch_eval(
        &self,
        stream: &MetalStream,
        input: &DeviceTensor,
        geo: &MaxPool2dGeometry,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        stream.retain_tensor(input);
        stream.retain_tensor(output);
        ensure!(output.datum_type() == input.datum_type());

        let params: [i32; 22] = [
            geo.batch as i32,
            geo.channels as i32,
            geo.input_hw.0 as i32,
            geo.input_hw.1 as i32,
            geo.output_hw.0 as i32,
            geo.output_hw.1 as i32,
            geo.kernel.0 as i32,
            geo.kernel.1 as i32,
            geo.strides.0 as i32,
            geo.strides.1 as i32,
            geo.dilations.0 as i32,
            geo.dilations.1 as i32,
            geo.padding.0 as i32,
            geo.padding.1 as i32,
            geo.input_strides[0] as i32,
            geo.input_strides[1] as i32,
            geo.input_strides[2] as i32,
            geo.input_strides[3] as i32,
            geo.output_strides[0] as i32,
            geo.output_strides[1] as i32,
            geo.output_strides[2] as i32,
            geo.output_strides[3] as i32,
        ];

        let pipeline =
            stream.load_pipeline(LibraryName::NNOps, &self.kernel_name(input.datum_type())?)?;
        let command_buffer = stream.command_buffer();
        command_buffer.encode(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(1, output, metal::MTLResourceUsage::Write);
            encoder.set_slice(2, &params);
            let grid_size = MTLSize {
                width: geo.output_hw.1 as _,
                height: geo.output_hw.0 as _,
                depth: (geo.batch * geo.channels) as _,
            };
            let group_size = MTLSize { width: 1, height: 1, depth: 1 };
            encoder.dispatch_thread_groups(grid_size, group_size);
        });
        Ok(())
    }
}

pub fn metal_max_pool_dispatch(
    input: &DeviceTensor,
    geo: &MaxPool2dGeometry,
    output: &DeviceTensor,
) -> TractResult<()> {
    crate::with_metal_stream(|stream| MaxPool2d.dispatch_eval(stream, input, geo, output))
}

/// Resolves an `OptMaxPool` to a bakeable two-spatial-axis geometry. Index
/// outputs, ranks other than 4 and symbolic shapes stay on CPU.
fn geometry(
    source: &TypedModel,
    node: &TypedNode,
    pool_spec: &PoolSpec,
    with_index_outputs: Option<DatumType>,
) -> TractResult<Option<GpuMaxPool>> {
    if with_index_outputs.is_some() || pool_spec.kernel_shape.len() != 2 {
        return Ok(None);
    }
    let facts = source.node_input_facts(node.id)?;
    let Some(input_shape) = facts[0].shape.as_concrete() else { return Ok(None) };
    if input_shape.len() != 4 {
        return Ok(None);
    }
    let dims: TVec<TDim> = input_shape.iter().map(|d| d.to_dim()).collect();
    let geo = pool_spec.compute_geo(&dims)?.to_concrete(input_shape)?.into_owned();
    let (input_shape, output_shape) = (&geo.input_shape, &geo.output_shape);
    let patch = &geo.patch;
    if patch.pad_before.len() != 2 {
        return Ok(None);
    }
    // The kernel subtracts one pad per axis, so an asymmetric `pad_after` is
    // only reachable through the bounds check it already does.
    let strides = |shape: &tract_core::ops::nn::DataShape| -> [usize; 4] {
        [
            *shape.n_stride().unwrap_or(&0),
            *shape.c_stride(),
            shape.hw_strides()[0],
            shape.hw_strides()[1],
        ]
    };
    let hw = |shape: &tract_core::ops::nn::DataShape| (shape.hw_dims()[0], shape.hw_dims()[1]);

    let geometry = MaxPool2dGeometry {
        batch: *input_shape.n().unwrap_or(&1),
        channels: *input_shape.c(),
        input_hw: hw(input_shape),
        output_hw: hw(output_shape),
        kernel: (patch.spec.kernel_shape[0], patch.spec.kernel_shape[1]),
        strides: (patch.spec.strides[0], patch.spec.strides[1]),
        dilations: (patch.spec.dilations[0], patch.spec.dilations[1]),
        padding: (patch.pad_before[0], patch.pad_before[1]),
        input_strides: strides(input_shape),
        output_strides: strides(output_shape),
    };
    Ok(Some(GpuMaxPool::new(
        geometry,
        output_shape.shape.clone(),
        "Metal",
        metal_max_pool_dispatch,
    )))
}

crate::register_metal_op!(MaxPool, |source, node, op| {
    let facts = source.node_input_facts(node.id)?;
    rule_if!(facts[0].is_plain());
    rule_if!(MaxPool2d::is_supported_dt(facts[0].datum_type));
    Ok(geometry(source, node, &op.pool_spec, op.with_index_outputs)?
        .map(|op| Box::new(op) as Box<dyn TypedOp>))
});

crate::register_metal_op!(OptMaxPool, |source, node, op| {
    let facts = source.node_input_facts(node.id)?;
    rule_if!(facts[0].is_plain());
    rule_if!(MaxPool2d::is_supported_dt(facts[0].datum_type));
    Ok(geometry(source, node, &op.pool_spec, op.with_index_outputs)?
        .map(|op| Box::new(op) as Box<dyn TypedOp>))
});
