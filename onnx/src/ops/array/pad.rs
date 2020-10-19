use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;
use tract_hir::ops::array;

pub fn pad(
    ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    match ctx.onnx_operator_set_version {
        2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 => pad_2(ctx, node),
        v if v >= 10 => pad_11(ctx, node),
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
                "reflect" => Ok(Some(array::PadMode::Reflect)),
                "edge" => Ok(Some(array::PadMode::Edge)),
                _ => Err(mode),
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

pub fn pad_11(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let mode = pad_mode(node)?;
    let op = Pad11::new(mode, if node.input.len() == 3 { Some(2) } else { None });
    Ok((expand(op), vec![]))
}

#[derive(Debug, Clone, new, Hash)]
pub struct Pad11 {
    mode: array::PadMode,
    constant_input: Option<usize>,
}

tract_data::impl_dyn_hash!(Pad11);

impl Expansion for Pad11 {
    fn name(&self) -> Cow<str> {
        "Pad".into()
    }

    op_onnx!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 2 + self.constant_input.is_some() as usize)?;
        check_output_arity(&outputs, 1)?;
        if let Some(input) = self.constant_input {
            s.equals(&inputs[0].datum_type, &inputs[input].datum_type)?;
            s.equals(&inputs[input].rank, 0)?;
        }
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        s.equals(&inputs[1].rank, 1)?;
        s.equals(&inputs[1].shape[0], 2 * inputs[0].rank.bex().to_dim())?;
        s.given(&inputs[1].value, move |s, pads| {
            let pads = pads.as_slice::<i64>()?;
            let rank = pads.len() / 2;
            let pads: Vec<_> =
                (0..rank).map(|ax| (pads[ax] as usize, pads[ax + rank] as usize)).collect();
            for i in 0..rank {
                s.equals(
                    &outputs[0].shape[i],
                    inputs[0].shape[i].bex() + pads[i].0.to_dim() + pads[i].1.to_dim(),
                )?;
            }
            Ok(())
        })?;
        Ok(())
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
        let pads = model
            .outlet_fact(inputs[1])?
            .konst
            .as_ref()
            .context("Expect padding to be constant")?
            .cast_to::<i64>()?;
        let pads = pads.as_slice::<i64>()?;
        let rank = pads.len() / 2;
        let pads = (0..rank).map(|ax| (pads[ax] as usize, pads[ax + rank] as usize)).collect();
        model.wire_node(name, array::Pad { mode, pads }, &inputs[0..1])
    }
}
