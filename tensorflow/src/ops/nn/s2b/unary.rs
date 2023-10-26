use tract_hir::internal::*;
use tract_ndarray::prelude::*;

use tract_hir::ops::cnn::{ConvUnary, PoolSpec};

#[derive(Debug, Copy, Clone, Hash)]
pub enum PaddingStrat {
    FlexFixed(usize),
    FixedFlex(usize),
    FixedFixed(usize, usize),
}

#[derive(Debug, Clone, new, Hash)]
pub struct SpaceToBatchUnary {
    pub datum_type: DatumType,
    pub space_shape: TVec<TDim>,
    pub batch_shape: TVec<TDim>,
    pub block_shape: Array1<i32>,
    pub pad: TVec<PaddingStrat>,
}



impl Op for SpaceToBatchUnary {
    fn name(&self) -> Cow<str> {
        "SpaceToBatchUnary".into()
    }

    op_as_typed_op!();
}

impl EvalOp for SpaceToBatchUnary {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let mut paddings = Array2::zeros((self.block_shape.len(), 2));
        for (ax, &strat) in self.pad.iter().enumerate() {
            let spread = (self.batch_shape[2 + ax].clone() * self.block_shape[ax]
                - &self.space_shape[2 + ax])
                .to_usize()?;
            let (bef, aft) = match strat {
                PaddingStrat::FlexFixed(f) => (spread - f, f),
                PaddingStrat::FixedFlex(f) => (f, spread - f),
                PaddingStrat::FixedFixed(a, b) => (a, b),
            };
            paddings[(ax, 0)] = bef as i32;
            paddings[(ax, 1)] = aft as i32;
        }
        let r = dispatch_numbers!(super::space_to_batch(input.datum_type())(
            input,
            &self.block_shape.view(),
            &paddings.view()
        ))?;
        Ok(tvec!(r))
    }
}

impl TypedOp for SpaceToBatchUnary {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].datum_type.fact(&self.batch_shape)))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if let Some(conv_node) = model.single_succ(node.id)? {
            if let Some(b2s_node) = model.single_succ(conv_node.id)? {
                if let (Some(conv_op), Some(_)) =
                    (conv_node.op_as::<ConvUnary>(), b2s_node.op_as::<BatchToSpaceUnary>())
                {
                    let input_channels = node.outputs[0].fact.shape.last().unwrap().to_usize()?;
                    let op = ConvUnary {
                        pool_spec: PoolSpec {
                            data_format: conv_op.pool_spec.data_format,
                            padding: conv_op.pool_spec.padding.clone(), // FIXME
                            strides: conv_op.pool_spec.strides.clone(),
                            kernel_shape: conv_op.pool_spec.kernel_shape.clone(),
                            output_channel_override: conv_op.pool_spec.output_channel_override,
                            dilations: Some(self.block_shape.iter().map(|&i| i as usize).collect()),
                        },
                        kernel_fmt: conv_op.kernel_fmt,
                        kernel: conv_op.kernel.clone(),
                        group: conv_op.group,
                        input_channels,
                        bias: None,
                        q_params: None,
                    };
                    let mut patch = TypedModelPatch::default();
                    let tap = patch.tap_model(model, node.inputs[0])?;
                    let out = patch.model.wire_node(&*conv_node.name, op, &[tap])?[0];
                    patch.shunt_outside(model, OutletId::new(b2s_node.id, 0), out)?;
                    return Ok(Some(patch));
                }
            }
        }
        Ok(None)
    }

    as_op!();
}

#[derive(Debug, Clone, new, Hash)]
pub struct BatchToSpaceUnary {
    datum_type: DatumType,
    batch_shape: TVec<TDim>,
    space_shape: TVec<TDim>,
    block_shape: Array1<i32>,
    pad: Vec<PaddingStrat>,
}



impl Op for BatchToSpaceUnary {
    fn name(&self) -> Cow<str> {
        "BatchToSpaceUnary".into()
    }

    op_as_typed_op!();
}

impl EvalOp for BatchToSpaceUnary {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let mut paddings = Array2::zeros((self.block_shape.len(), 2));
        for (ax, &strat) in self.pad.iter().enumerate() {
            let spread = (self.batch_shape[2 + ax].clone() * self.block_shape[ax]
                - &self.space_shape[2 + ax])
                .to_usize()?;
            let (bef, aft) = match strat {
                PaddingStrat::FlexFixed(f) => (spread - f, f),
                PaddingStrat::FixedFlex(f) => (f, spread - f),
                PaddingStrat::FixedFixed(a, b) => (a, b),
            };
            paddings[(ax, 0)] = bef as i32;
            paddings[(ax, 1)] = aft as i32;
        }
        let r = dispatch_numbers!(super::batch_to_space(input.datum_type())(
            input,
            &self.block_shape.view(),
            &paddings.view()
        ))?;
        Ok(tvec!(r))
    }
}

impl TypedOp for BatchToSpaceUnary {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].datum_type.fact(&self.space_shape)))
    }
}
