use tract_core::context::Context;
use tract_core::model::ModelDsl;
use tract_core::ops::nn::ConvUnary;
use tract_core::ops::prelude::*;
use tract_core::optim::OptimizerPass;
use tract_core::*;

#[derive(Debug)]
pub struct TensorflowContext;

impl TensorflowContext {
}

impl Context for TensorflowContext {
    fn optimizer_passes(&self) -> Vec<Box<OptimizerPass>> {
        let mut passes = optim::normalization();
        passes.push(Box::new(UntensorflowConv));
        passes.extend(optim::codegen().into_iter());
        passes
    }
}

#[derive(Debug)]
struct UntensorflowConv;

impl OptimizerPass for UntensorflowConv {
    fn pass(&self, model: &mut Model) -> TractResult<bool> {
        let mut done_something = false;
        done_something = done_something || undo_all_conv1d_as_conv2d(model)?;
        done_something = done_something || undo_all_space_to_batch(model)?;
        Ok(done_something)
    }
}

macro_rules! some_or_ok_false {
    ($option:expr) => {
        match $option {
            Some(prec) => prec,
            None => return Ok(false),
        }
    };
}

fn undo_all_conv1d_as_conv2d(model: &mut Model) -> TractResult<bool> {
    let convs: Vec<usize> = model
        .eval_order()?
        .into_iter()
        .filter(|&node| model.node(node).op_is::<ConvUnary>())
        .collect();
    convs.into_iter().try_fold(
        false,
        |acc, cv| Ok(acc || undo_conv1d_as_conv2d(model, cv)?),
    )
}

fn undo_conv1d_as_conv2d(model: &mut Model, node_id: usize) -> TractResult<bool> {
    use tract_core::ops::array::{AddDims, RmDims};
    let new_op = {
        let prec_node = some_or_ok_false!(model.single_prec(node_id)?);
        let add_dim_op = some_or_ok_false!(prec_node.op_as::<AddDims>());
        let succ_node = some_or_ok_false!(model.single_succ(node_id)?);
        let rm_dim_op = some_or_ok_false!(succ_node.op_as::<RmDims>());
        let conv_op = some_or_ok_false!(model.node(node_id).op_as::<ConvUnary>());
        if add_dim_op.axes.len() == 1 && rm_dim_op.axes == add_dim_op.axes {
            let axis = add_dim_op.axes[0];
            conv_op.rm_dummy_axis(axis)?
        } else {
            None
        }
    };
    if let Some(new_op) = new_op {
        let name = model.node(node_id).name.clone();
        model.replace_nodes(node_id, 1, 1, vec![(name, Box::new(new_op))])?;
    }
    Ok(false)
}

fn undo_all_space_to_batch(model: &mut Model) -> TractResult<bool> {
    let convs: Vec<usize> = model
        .eval_order()?
        .into_iter()
        .filter(|&node| model.node(node).op_is::<ConvUnary>())
        .collect();
    convs
        .into_iter()
        .try_fold(false, |acc, cv| Ok(acc || undo_space_to_batch(model, cv)?))
}

fn undo_space_to_batch(model: &mut Model, node_id: usize) -> TractResult<bool> {
    use ops::nn::s2b::unary::SpaceToBatchUnary;
    let new_op = {
        let prec_node = some_or_ok_false!(model.single_prec(node_id)?);
        let s2b_op = some_or_ok_false!(prec_node.op_as::<SpaceToBatchUnary>());
        let succ_node = some_or_ok_false!(model.single_succ(node_id)?);
        let conv_op = some_or_ok_false!(model.node(node_id).op_as::<ConvUnary>());
        let new_op = ConvUnary {
            data_fmt: conv_op.data_fmt,
            kernel_is_hwio: conv_op.kernel_is_hwio,
            padding: conv_op.padding.clone(), // FIXME
            dilations: s2b_op.block_shape.iter().map(|&i| i as usize).collect(),
            strides: conv_op.strides.clone(),
            kernel: conv_op.kernel.clone(),
            bias: conv_op.bias.clone(),
            full_input_shape: model
                .fact(prec_node.inputs[0])?
                .shape
                .concretize()
                .ok_or("Optimizing an unalized network")?,
            full_output_shape: succ_node.outputs[0]
                .fact
                .shape
                .concretize()
                .ok_or("Optimizing an unalized network")?,
            group: conv_op.group,
        };
        Some(new_op)
    };
    if let Some(new_op) = new_op {
        let name = model.node(node_id).name.clone();
        model.replace_nodes(node_id, 1, 1, vec![(name, Box::new(new_op))])?;
    }
    Ok(false)
}

#[cfg(test)]
mod test {
    use super::*;
    use std::sync::Arc;
    use tract_core::model::*;

    fn mk(sizes: &[usize]) -> Tensor {
        ::ndarray::Array::range(1f32, sizes.iter().product::<usize>() as f32 + 1.0, 1.0)
            .into_shape(sizes)
            .unwrap()
            .into()
    }

    fn make_conv(strides: TVec<usize>, valid: bool) -> Box<Op> {
        use tract_core::ops::nn::*;
        Box::new(Conv::new(
            DataFormat::NHWC,
            true,
            None,
            None,
            if valid {
                PaddingSpec::Valid
            } else {
                PaddingSpec::SameUpper
            },
            Some(strides),
            1,
        ))
    }

    #[test]
    fn conv2d_unarization() {
//        ::setup_test_logger();
        let mut model = Model::default().with_context(Arc::new(TensorflowContext));
        model.add_source_fact(
            "source",
            TensorFact::dt_shape(DatumType::F32, &[1, 10, 10, 3]),
        ).unwrap();
        let conv = model.chain("conv2d", make_conv(tvec!(1, 1), true)).unwrap();
        let kernel = model.add_const("kernel", mk(&[1, 1, 3, 3]).into()).unwrap();
        model.add_edge(OutletId::new(kernel, 0), InletId::new(conv, 1)).unwrap();

        assert_eq!(model.eval_order().unwrap().len(), 3);

        model.analyse().unwrap();
        let optimized = model.into_optimized().unwrap();
        println!("{:#?}", optimized);
        assert_eq!(optimized.eval_order().unwrap().len(), 2);
    }

}
