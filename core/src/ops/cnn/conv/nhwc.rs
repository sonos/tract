use crate::internal::*;
use crate::ops::cnn::PoolSpec;
use crate::ops::cnn::conv::Conv;
use crate::ops::nn::DataFormat;
use crate::transform::ModelTransform;

/// Rewrites every non-quantized `Conv` on `NCHW` or `CHW` data to run channel-last: the input
/// is moved channel-last with an `AxisOp::Move`, the conv runs with `DataFormat::NHWC`
/// (resp. `HWC`), and the output is moved back. Depthwise convs are lowered to `DepthWise`
/// inside the same patch — `DepthWise` has no `change_axes`, so the moves cannot be pushed
/// through it; a plain `Conv` sandwich may instead be folded back by `ChangeAxes` in the
/// declutter pass. The rule never fires on a channel-last `Conv`, so the rewrite terminates.
#[derive(Debug)]
pub struct ConvNhwc;

impl ModelTransform for ConvNhwc {
    fn name(&self) -> StaticName {
        "conv_nhwc".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        Rewriter::<()>::default().with_rule_for("conv-nhwc", conv_nhwc).rewrite(&(), model)
    }
}

fn conv_nhwc(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    op: &Conv,
) -> TractResult<Option<TypedModelPatch>> {
    rule_if!(op.q_params.is_none());
    rule_if!(matches!(op.pool_spec.data_format, DataFormat::NCHW | DataFormat::CHW));
    let fact = model.outlet_fact(node.inputs[0])?;
    let r = fact.rank();
    let c_axis = op.pool_spec.data_format.shape(&fact.shape)?.c_axis();
    let data_format = match op.pool_spec.data_format {
        DataFormat::NCHW => DataFormat::NHWC,
        _ => DataFormat::HWC,
    };
    let mut patch = TypedModelPatch::default();
    let inputs = patch.taps(model, &node.inputs)?;
    let x =
        patch.wire_node(format!("{name}.nhwc_in"), AxisOp::Move(c_axis, r - 1), &[inputs[0]])?;
    let conv = Conv { pool_spec: PoolSpec { data_format, ..op.pool_spec.clone() }, ..op.clone() };
    let mut wires = tvec!(x[0]);
    wires.extend_from_slice(&inputs[1..]);
    let is_depthwise = node.inputs.len() == 3
        && op.group != 1
        && op.group == op.input_channels()
        && op.group == op.output_channels()
        && fact.shape.as_concrete().is_some();
    let wire = if is_depthwise {
        conv.wire_as_depth_wise(&mut patch, name, &wires)?
    } else {
        patch.wire_node(name, conv, &wires)?[0]
    };
    let y = patch.wire_node(format!("{name}.nhwc_out"), AxisOp::Move(r - 1, c_axis), &[wire])?;
    patch.shunt_outside(model, node.id.into(), y[0])?;
    Ok(Some(patch))
}

register_simple_model_transform!("conv_nhwc", ConvNhwc);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::cnn::PaddingSpec;
    use crate::ops::cnn::conv::KernelFormat;
    use crate::ops::cnn::conv::depth_wise::DepthWise;

    fn dw_model() -> TypedModel {
        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::fact([1, 4, 8, 8])).unwrap();
        let kernel: Vec<f32> = (0..4 * 9).map(|i| ((i as f32 * 0.091).cos()) * 0.3).collect();
        let bias: Vec<f32> = (0..4).map(|i| (i as f32 * 0.05) - 0.1).collect();
        let k = model.add_const("k", Tensor::from_shape(&[4, 1, 3, 3], &kernel).unwrap()).unwrap();
        let b = model.add_const("b", Tensor::from_shape(&[4], &bias).unwrap()).unwrap();
        let pool_spec = PoolSpec {
            data_format: DataFormat::NCHW,
            kernel_shape: tvec!(3, 3),
            padding: PaddingSpec::SameUpper,
            dilations: None,
            strides: None,
            input_channels: 4,
            output_channels: 4,
        };
        let conv = Conv { pool_spec, kernel_fmt: KernelFormat::OIHW, group: 4, q_params: None };
        let out = model.wire_node("dw", conv, &[x, k, b]).unwrap();
        model.select_output_outlets(&out).unwrap();
        model
    }

    #[test]
    fn transform_lowers_depthwise_conv() {
        let model = ConvNhwc.transform_into(dw_model()).unwrap();
        assert!(model.nodes.iter().all(|n| !n.op_is::<Conv>()));
        assert!(model.nodes.iter().any(|n| n.op_is::<DepthWise>()));
        for name in ["dw.nhwc_in", "dw.nhwc_out"] {
            let node = model.node(model.node_id_by_name(name).unwrap());
            assert!(matches!(node.op_as::<AxisOp>(), Some(AxisOp::Move(_, _))), "{node}");
        }
    }

    fn conv_op(
        model: &mut TypedModel,
        name: &str,
        input: OutletId,
        c_in: usize,
        c_out: usize,
        group: usize,
        kernel: usize,
    ) -> OutletId {
        let k: Vec<f32> = (0..c_out * (c_in / group) * kernel * kernel)
            .map(|i| ((i as f32 * 0.091).cos()) * 0.3)
            .collect();
        let salt = name.bytes().map(|b| b as f32).sum::<f32>() * 0.001;
        let b: Vec<f32> = (0..c_out).map(|i| (i as f32 * 0.05) - 0.1 + salt).collect();
        let k = model
            .add_const(
                format!("{name}.k"),
                Tensor::from_shape(&[c_out, c_in / group, kernel, kernel], &k).unwrap(),
            )
            .unwrap();
        let b = model
            .add_const(format!("{name}.b"), Tensor::from_shape(&[c_out], &b).unwrap())
            .unwrap();
        let pool_spec = PoolSpec {
            data_format: DataFormat::NCHW,
            kernel_shape: tvec!(kernel, kernel),
            padding: PaddingSpec::SameUpper,
            dilations: None,
            strides: None,
            input_channels: c_in,
            output_channels: c_out,
        };
        let conv = Conv { pool_spec, kernel_fmt: KernelFormat::OIHW, group, q_params: None };
        model.wire_node(name, conv, &[input, k, b]).unwrap()[0]
    }

    fn depthwise_in(model: &TypedModel) -> Option<&Conv> {
        model
            .nodes
            .iter()
            .filter_map(|n| n.op_as::<Conv>())
            .find(|c| c.group > 1 && c.group == c.input_channels())
    }

    #[test]
    fn depthwise_flips_channel_last_in_chain() {
        if tract_linalg::routines::depthwise_c_f32().is_none() {
            return;
        }
        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::fact([1, 32, 8, 8])).unwrap();
        let x = conv_op(&mut model, "pw1", x, 32, 32, 1, 1);
        let x = conv_op(&mut model, "dw", x, 32, 32, 32, 3);
        let x = conv_op(&mut model, "pw2", x, 32, 32, 1, 1);
        model.select_output_outlets(&[x]).unwrap();
        let model = model.into_decluttered().unwrap();
        let dw = depthwise_in(&model).expect("depthwise conv was lowered too early");
        assert!(dw.pool_spec.data_format.c_is_last(), "{model}");
        assert!(
            model.nodes.iter().all(|n| !matches!(n.op_as::<AxisOp>(), Some(AxisOp::Move(..)))),
            "{model}"
        );
    }

    /// A regular conv forwards the c-to-last move to its own input instead of
    /// absorbing it, so a depthwise conv sitting directly behind a stem keeps
    /// propagating until it hits the locked model input and the suggestion is
    /// rejected: the depthwise conv stays NCHW.
    #[test]
    fn depthwise_behind_regular_stem_stays_nchw() {
        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::fact([1, 3, 16, 16])).unwrap();
        let x = conv_op(&mut model, "stem", x, 3, 32, 1, 3);
        let x = conv_op(&mut model, "dw", x, 32, 32, 32, 3);
        model.select_output_outlets(&[x]).unwrap();
        let model = model.into_decluttered().unwrap();
        let dw = depthwise_in(&model).expect("depthwise conv was lowered too early");
        assert!(!dw.pool_spec.data_format.c_is_last(), "{model}");
    }

    /// A depthwise conv bounded by pointwise convs flips even when the chain
    /// starts behind a regular conv: the neighbouring 1x1 convs lower to
    /// `EinSum`, which swallows the axis change at its interface, so the move
    /// never reaches the locked model input (the stem stays NCHW).
    #[test]
    fn depthwise_flips_channel_last_between_pointwise() {
        if tract_linalg::routines::depthwise_c_f32().is_none() {
            return;
        }
        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::fact([1, 3, 16, 16])).unwrap();
        let x = conv_op(&mut model, "stem", x, 3, 32, 1, 3);
        let x = conv_op(&mut model, "pw1", x, 32, 32, 1, 1);
        let x = conv_op(&mut model, "dw", x, 32, 32, 32, 3);
        let x = conv_op(&mut model, "pw2", x, 32, 32, 1, 1);
        model.select_output_outlets(&[x]).unwrap();
        let model = model.into_decluttered().unwrap();
        let dw = depthwise_in(&model).expect("depthwise conv was lowered too early");
        assert!(dw.pool_spec.data_format.c_is_last(), "{model}");
        let stem = model
            .nodes
            .iter()
            .filter_map(|n| n.op_as::<Conv>())
            .find(|c| c.group == 1)
            .expect("stem conv was lowered");
        assert!(!stem.pool_spec.data_format.c_is_last(), "{model}");
    }

    #[test]
    fn depthwise_below_16_channels_stays_nchw() {
        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::fact([1, 8, 8, 8])).unwrap();
        let x = conv_op(&mut model, "dw", x, 8, 8, 8, 3);
        model.select_output_outlets(&[x]).unwrap();
        let model = model.into_decluttered().unwrap();
        let dw = depthwise_in(&model).expect("depthwise conv was lowered too early");
        assert!(!dw.pool_spec.data_format.c_is_last(), "{model}");
    }

    #[test]
    fn nhwc_depthwise_matches_nchw() {
        let x: Vec<f32> = (0..4 * 8 * 8).map(|i| ((i as f32 * 0.137).sin()) * 0.7).collect();
        let xv = Tensor::from_shape(&[1, 4, 8, 8], &x).unwrap().into_tvalue();
        let base = dw_model().into_optimized().unwrap().into_runnable().unwrap();
        let want = base.run(tvec![xv.clone()]).unwrap().remove(0);
        let nhwc = ConvNhwc
            .transform_into(dw_model())
            .unwrap()
            .into_decluttered()
            .unwrap()
            .into_optimized()
            .unwrap();
        assert!(
            nhwc.nodes.iter().any(|n| n.op_is::<DepthWise>()),
            "expected a DepthWise node, got {}",
            nhwc.nodes.iter().map(|n| n.op().name()).collect::<Vec<_>>().join(",")
        );
        let nhwc = nhwc.into_runnable().unwrap();
        let got = nhwc.run(tvec![xv]).unwrap().remove(0);
        let want = want.to_plain_array_view::<f32>().unwrap();
        let got = got.to_plain_array_view::<f32>().unwrap();
        let max_abs = want.iter().zip(got.iter()).map(|(w, g)| (w - g).abs()).fold(0f32, f32::max);
        assert!(max_abs < 1e-5, "max_abs={max_abs}");
    }
}
