use crate::internal::*;
use crate::ops::array::MultiBroadcastTo;
use crate::ops::quant::lookup_table;

pub fn cast(to: DatumType) -> Cast {
    Cast { to }
}

pub fn wire_cast(
    prefix: impl AsRef<str>,
    target: &mut TypedModel,
    inputs: &[OutletId],
    operating_datum_type: DatumType,
) -> TractResult<TVec<OutletId>> {
    let prefix = prefix.as_ref();
    let mut wires = tvec!();
    for mut wire in inputs.iter().copied() {
        if target.outlet_fact(wire)?.datum_type != operating_datum_type {
            wire = target.wire_node(
                target.unique_name(format!("{prefix}.cast")),
                crate::ops::cast::cast(operating_datum_type),
                &[wire],
            )?[0];
        }
        wires.push(wire);
    }
    Ok(wires)
}

#[derive(Debug, Clone, new, Hash, PartialEq, Eq)]
pub struct Cast {
    pub to: DatumType,
}

impl Op for Cast {
    fn name(&self) -> StaticName {
        "Cast".into()
    }

    op_as_typed_op!();
}

impl EvalOp for Cast {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        _node_id: usize,
        state: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        if input.datum_type() == self.to {
            Ok(tvec!(input))
        } else if input.datum_type() == TDim::datum_type() {
            let mut tmp = Tensor::zero_dt(i64::datum_type(), input.shape())?;
            let input_plain = input.try_as_plain()?;
            let mut tmp_plain = tmp.try_as_plain_mut()?;
            for (dim, i) in tract_itertools::izip!(
                input_plain.as_slice::<TDim>()?,
                tmp_plain.as_slice_mut::<i64>()?
            ) {
                *i = dim.eval(&state.resolved_symbols).to_i64()?
            }
            Ok(tvec!(tmp.cast_to_dt(self.to)?.into_owned().into_tvalue()))
        } else {
            Ok(tvec!(input.cast_to_dt(self.to)?.into_owned().into_tvalue()))
        }
    }
}

impl TypedOp for Cast {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = self.to.fact(inputs[0].shape.clone());
        fact.uniform_tdim = inputs[0].uniform_tdim.clone();
        if let Some(u) = &inputs[0].uniform
            && let Ok(cast_u) = u.cast_to_dt(self.to)
        {
            fact.uniform = Some(std::sync::Arc::new(cast_u.into_owned()));
        }
        Ok(tvec!(fact))
    }

    fn input_roi(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TVec<Option<TDim>>>> {
        crate::optim::propagate_roi::bubble_roi(model, node)
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if model.outlet_fact(node.inputs[0])?.datum_type == self.to {
            return TypedModelPatch::shunt_one_op(model, node);
        }
        // linear_prec (fan-in=1, fan-out=1) rather than single_prec: swapping
        // through a fan-out predecessor clones it, and the clone breaks
        // downstream pattern detectors (e.g. Square+Reduce<Sum>+Mul fusion into
        // Reduce<MeanOfSquares>, which then feeds RmsNorm detection).
        //
        // AxisOp is intentionally NOT in the predicate: pulling Cast above an
        // AxisOp (Reshape/Move/Add/Rm) prevents the CUDA conversion from
        // fusing the post-AxisOp Cast into the downstream GEMM-class kernel,
        // leaving ~64 standalone CudaCast ops on OpenELM-270M (TG128 -4%).
        if let Some(prec) = model.linear_prec(node.id)?
            && (prec.op_is::<IntoShape>() || prec.op_is::<MultiBroadcastTo>())
        {
            let mut patch = TypedModelPatch::default();
            let mut wire = tvec!(patch.tap_model(model, prec.inputs[0])?);
            wire = patch.wire_node(&node.name, &node.op, &wire)?;
            wire = patch.wire_node(&prec.name, &prec.op, &wire)?;
            patch.shunt_outside(model, node.id.into(), wire[0])?;
            return Ok(Some(patch));
        }
        Ok(None)
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        AxesMapping::natural(inputs, outputs)
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        Ok(Some(AxisChangeConsequence::new(model, node, None, change)))
    }

    fn slice(
        &self,
        patch: &mut TypedModelPatch,
        _model: &TypedModel,
        node: &TypedNode,
        _prefix: &str,
        inputs: &[OutletId],
        _output_axis: usize,
        _start: &TDim,
        _end: &TDim,
    ) -> TractResult<Option<TVec<OutletId>>> {
        patch.wire_node(&node.name, &node.op, inputs).map(Some)
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let src_dt = model.node_input_facts(node.id)?[0].datum_type;
        if src_dt.is_quantized() && src_dt.size_of() == 1 && self.to.is_float() {
            dequant_to_lut(model, node)
        } else {
            Ok(None)
        }
    }

    as_op!();
}

/// Fuse a dequantization cast (`q8 -> float`), the following chain of
/// elementwise-unary ops, and a requantization cast (`float -> q8`) into a
/// single 256-entry lookup table. `dequant` is the leading dequantization cast.
fn dequant_to_lut(model: &TypedModel, dequant: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
    let mut current = dequant;
    while let Some(requant) = model.single_succ(current.id)? {
        let requant_dt = requant.op_as::<Cast>().map(|c| c.to).filter(|dt| dt.is_quantized());
        if let Some(dst) = requant_dt {
            let (zero_point, scale) = dst.zp_scale();
            let dt = dst.unquantized();
            // first, try to rewrite every op in the chain to operate on quantized data
            let mut patch = TypedModelPatch::default();
            let mut wire = patch.tap_model(model, dequant.inputs[0])?;
            let mut next = model.single_succ(dequant.id)?.unwrap();
            loop {
                if let Some(op) = next
                    .op
                    .quantize(model, dequant, dt, scale, zero_point)
                    .with_context(|| format!("Quantizing {next}"))?
                {
                    wire = patch.wire_node(&*next.name, op, &[wire])?[0];
                } else {
                    break;
                }
                if next.id == current.id {
                    patch.shunt_outside(model, requant.id.into(), wire)?;
                    return Ok(Some(patch));
                }
                next = model.single_succ(next.id)?.unwrap();
            }
            // otherwise, bake the whole chain into a lookup table
            return quant_seq_to_lut(model, dequant, requant).map(Some);
        }
        let (input_facts, output_facts) = model.node_facts(requant.id)?;
        let invariants = requant
            .op
            .axes_mapping(&input_facts, &output_facts)
            .with_context(|| format!("Querying invariants for {requant}"))?;
        if invariants.is_element_wise_unary() {
            current = requant;
        } else {
            break;
        }
    }
    Ok(None)
}

fn quant_seq_to_lut(
    model: &TypedModel,
    dequant: &TypedNode,
    requant: &TypedNode,
) -> TractResult<TypedModelPatch> {
    let incoming_dt = model.node_input_facts(dequant.id)?[0].datum_type;
    let outgoing_dt = requant.op_as::<Cast>().context("requant is not a Cast")?.to;

    let mut adhoc = TypedModel::default();
    let mut wire = adhoc.add_source("source", incoming_dt.fact([256]))?;
    wire = adhoc.wire_node(&*dequant.name, dequant.op.clone(), &[wire])?[0];
    let mut node = model.single_succ(dequant.id)?.unwrap();
    let mut name = None;
    while node.id != requant.id {
        name.get_or_insert(&*node.name);
        wire = adhoc.wire_node(&*node.name, node.op.clone(), &[wire])?[0];
        node = model.single_succ(node.id)?.unwrap();
    }
    wire = adhoc.wire_node(&*requant.name, requant.op.clone(), &[wire])?[0];
    adhoc.select_output_outlets(&[wire])?;

    let raw = (0u8..=255).collect::<Vec<u8>>();
    let mut input = match incoming_dt.unquantized() {
        DatumType::I8 => tensor1(unsafe { std::mem::transmute::<&[u8], &[i8]>(&raw[..]) }),
        DatumType::U8 => tensor1(&raw),
        _ => bail!("Expected a byte-sized quantized type, got {incoming_dt:?}"),
    };
    unsafe { input.set_datum_type(incoming_dt) };
    let output = SimplePlan::new(adhoc)?.run(tvec!(input.into_tvalue()))?.remove(0);

    let mut op = lookup_table((tract_linalg::ops().lut_u8)(output.as_bytes()));
    op.1 = Some(outgoing_dt);
    let mut patch = TypedModelPatch::default();
    let mut wire = patch.tap_model(model, dequant.inputs[0])?;
    wire = patch.wire_node(name.unwrap_or(&*dequant.name), op, &[wire])?[0];
    patch.shunt_outside(model, requant.id.into(), wire)?;
    Ok(patch)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ops::nn::sigmoid;
    use tract_data::itertools::Itertools;

    #[test]
    fn dequant_ew_requant_fuses_to_lut() -> TractResult<()> {
        let dt = i8::datum_type().with_zp_scale(0, 0.03);
        let mut model = TypedModel::default();
        let src = model.add_source("src", dt.fact([10]))?;
        let mut wire = model.wire_node("dq", cast(f32::datum_type()), &[src])?;
        wire = model.wire_node("sigmoid", sigmoid(), &wire)?;
        wire = model.wire_node("q", cast(dt), &wire)?;
        model.select_output_outlets(&wire)?;

        let input =
            tensor1(&(-5i32..5).collect_vec()).cast_to::<f32>()?.cast_to_dt(dt)?.into_owned();
        let reference = model.clone().into_runnable()?.run(tvec!(input.clone().into_tvalue()))?;

        let optimized = model.into_optimized()?;
        assert_eq!(optimized.nodes.len(), 2); // Source then LookupTable
        let output = optimized.into_runnable()?.run(tvec!(input.into_tvalue()))?;
        output[0].close_enough(&reference[0], Approximation::Exact)
    }
}
