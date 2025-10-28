use crate::model::{optional_inputs, ParsingContext};
use crate::pb::*;
use tract_hir::internal::*;
use tract_hir::ops::array;
use tract_hir::prelude::tract_itertools::Itertools;

pub fn pad(
    ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    match ctx.onnx_operator_set_version {
        2..=10 => pad_2(ctx, node),
        11.. => pad_18(ctx, node), // pad 11 is more restrivie that pad 18 (no axes input)
        _ => bail!("Unsupported operator set for Pad operator"),
    }
}

pub fn pad_mode(node: &NodeProto) -> TractResult<array::PadMode> {
    let value: f32 = node.get_attr_opt("value")?.unwrap_or(0.0);
    let mode = match node.get_attr_opt("mode")? {
        None | Some("constant") => None,
        Some(mode) => node.check_value(
            "mode",
            match mode {
                "reflect" => TractResult::Ok(Some(array::PadMode::Reflect)),
                "edge" => Ok(Some(array::PadMode::Edge)),
                _ => bail!("Unsupported mode {mode}"),
            },
        )?,
    }
    .unwrap_or_else(|| tract_hir::ops::array::PadMode::Constant(Arc::new(value.into())));
    Ok(mode)
}

pub fn pad_2(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let pads = node.get_attr_tvec("pads")?;
    let rank = pads.len() / 2;
    let pads = (0..rank).map(|ax| (pads[ax], pads[ax + rank])).collect();
    let mode = pad_mode(node)?;
    Ok((Box::new(tract_hir::ops::array::Pad::new(pads, mode)), vec![]))
}

pub fn pad_18(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let mode = pad_mode(node)?;
    let mut inputs = optional_inputs(node).skip(2);
    let op = Pad18::new(mode, inputs.next().unwrap(), inputs.next().unwrap());
    Ok((expand(op), vec![]))
}

#[derive(Debug, Clone, new, Hash)]
pub struct Pad18 {
    mode: array::PadMode,
    constant_input: Option<usize>,
    axes_input: Option<usize>,
}

impl Expansion for Pad18 {
    fn name(&self) -> StaticName {
        "Pad".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(
            inputs,
            2 + self.constant_input.is_some() as usize + self.axes_input.is_some() as usize,
        )?;
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        s.equals(&inputs[1].rank, 1)?;
        if let Some(constant) = self.constant_input {
            s.equals(&inputs[0].datum_type, &inputs[constant].datum_type)?;
            s.equals(&inputs[constant].rank, 0)?;
        }

        fn do_output_shape<'a, 'r, 'p: 'r>(
            s: &mut Solver<'r>,
            inputs: &'p [TensorProxy],
            outputs: &'p [TensorProxy],
            axes: Vec<usize>,
        ) -> TractResult<()> {
            s.given(&inputs[1].value, move |s, pads| {
                let pads = pads.cast_to::<TDim>()?;
                let pads = pads.as_slice::<TDim>()?;
                let rank = pads.len() / 2;
                for (ix, axis) in axes.iter().enumerate() {
                    let left = pads[ix].clone();
                    let right = pads[ix + rank].clone();
                    s.equals(
                        &outputs[0].shape[*axis],
                        inputs[0].shape[*axis].bex() + left + right,
                    )?;
                }
                Ok(())
            })?;
            Ok(())
        }

        if let Some(axes) = self.axes_input {
            s.equals(&inputs[axes].rank, 1)?;
            s.equals(&inputs[1].shape[0], 2 * inputs[axes].shape[0].bex())?;
            s.given_2(&inputs[0].rank, &inputs[axes].value, move |s, rank, axes| {
                let axes = axes
                    .cast_to::<i64>()?
                    .as_slice::<i64>()?
                    .iter()
                    .map(|x| (if *x < 0 { *x + rank } else { *x }) as usize)
                    .collect_vec();
                do_output_shape(s, inputs, outputs, axes)
            })
        } else {
            s.equals(&inputs[1].shape[0], 2 * inputs[0].rank.bex().to_dim())?;
            s.given(&inputs[0].rank, move |s, rank| {
                let axes = (0..rank as usize).collect_vec();
                do_output_shape(s, inputs, outputs, axes)
            })
        }
    }

    fn wire(
        &self,
        name: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let mode = if let Some(constant) = self.constant_input {
            if let Some(c) = model.outlet_fact(inputs[constant])?.konst.clone() {
                array::PadMode::Constant(c)
            } else {
                bail!("Pad constant input must be constant")
            }
        } else {
            self.mode.clone()
        };
        let rank = model.outlet_fact(inputs[0])?.rank();
        let axes = if let Some(axes) = self.axes_input {
            model
                .outlet_fact(inputs[axes])?
                .konst
                .as_ref()
                .context("Axes must be a constant")?
                .cast_to::<i64>()?
                .as_slice::<i64>()?
                .iter()
                .map(|x| (if *x < 0 { *x + rank as i64 } else { *x }) as usize)
                .collect_vec()
        } else {
            (0..rank).collect_vec()
        };
        let pads = model
            .outlet_fact(inputs[1])?
            .konst
            .as_ref()
            .context("Expect padding to be constant")?
            .cast_to::<i64>()?;
        let pads = pads.as_slice::<i64>()?;

        let mut fixed_pads = vec![(0, 0); rank];
        for (ix, &axis) in axes.iter().enumerate() {
            fixed_pads[axis] = (pads[ix] as usize, pads[ix + pads.len() / 2] as usize);
        }
        model.wire_node(name, array::Pad { mode, pads: fixed_pads }, &inputs[0..1])
    }
}
