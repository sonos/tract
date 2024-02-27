use tract_data::itertools::Itertools;

use crate::internal::*;
use crate::plan::eval;

pub fn cast(to: DatumType) -> Cast {
    Cast { to }
}

#[derive(Debug, Clone, new, Hash)]
pub struct Cast {
    pub to: DatumType,
}

impl Cast {
    fn do_eval(&self, input: &Tensor, symbols: &SymbolValues) -> TractResult<TVec<TValue>> {
        if input.datum_type() == self.to {
            Ok(tvec!(input.clone().into_tvalue()))
        } else if input.datum_type() == TDim::datum_type() {
            unsafe {
                let mut tmp = Tensor::uninitialized_dt(i64::datum_type(), input.shape())?;
                for (dim, i) in
                    tract_itertools::izip!(input.as_slice::<TDim>()?, tmp.as_slice_mut::<i64>()?)
                {
                    *i = dim.eval(symbols).to_i64()?
                }
                Ok(tvec!(tmp.cast_to_dt(self.to)?.into_owned().into_tvalue()))
            }
        } else {
            let out = input.cast_to_dt(self.to)?;
            Ok(tvec!(out.into_owned().into_tvalue()))
        }
    }
}

impl Op for Cast {
    fn name(&self) -> Cow<str> {
        "Cast".into()
    }

    op_as_typed_op!();
}

impl EvalOp for Cast {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        self.do_eval(&inputs[0], &Default::default())
    }

    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(self.clone())))
    }
}

impl OpState for Cast {
    fn eval(
        &mut self,
        session: &mut SessionState,
        _op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        self.do_eval(&inputs[0], &session.resolved_symbols)
    }
}

trivial_op_state_freeeze!(Cast);
impl TypedOp for Cast {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(self.to.fact(inputs[0].shape.clone())))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if model.outlet_fact(node.inputs[0])?.datum_type == self.to {
            TypedModelPatch::shunt_one_op(model, node)
        } else {
            Ok(None)
        }
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

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let src_dt = model.node_input_facts(node.id)?[0].datum_type;
        if src_dt.is_quantized() && src_dt.size_of() == 1 && self.to.is_float() {
            codegen_quant_ew_chain_to_lut(self, model, node)
        } else {
            Ok(None)
        }
    }

    as_op!();
}

fn codegen_quant_ew_chain_to_lut(
    original_dequant: &Cast,
    model: &TypedModel,
    origin: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    let mut current = origin;
    let incoming_dt = model.node_input_facts(origin.id)?[0].datum_type;
    while let Some(next) = model.single_succ(current.id)? {
        /*
        let q_params = if let Some(op) = op.op_as::<ElementWiseOp>() {
        if let Some(mop) = op.0.downcast_ref::<QuantizeLinearU8>() {
        Some((mop.scale, mop.zero_point as i32, u8::datum_type()))
        } else {
        op.0.downcast_ref::<QuantizeLinearI8>()
        .map(|mop| (mop.scale, mop.zero_point as i32, i8::datum_type()))
        }
        } else {
        None
        };
        */
        let q_dt_dst: Option<DatumType> =
            next.op_as::<Cast>().map(|c| c.to).filter(|dt| dt.is_quantized());
        if let Some(dt) = q_dt_dst {
            let (zp, scale) = dt.zp_scale();
            /*
            // first, try Op::quantize() on all ops in the chain
            let mut patch = TypedModelPatch::default();
            let mut wire: OutletId = patch.tap_model(model, origin.inputs[0])?;
            let mut next = model.single_succ(origin.id)?.unwrap();
            loop {
            if let Some(op) = next
            .op
            .quantize(model, dequant, dt, scale, zero_point)
            .with_context(|| format!("Quantizing {next}"))?
            {
            wire = patch.wire_node(&*next.name, op, [wire].as_ref())?[0];
            } else {
            break;
            }
            if next.id == current.id {
            patch.shunt_outside(model, OutletId::new(op.id, 0), wire)?;
            return Ok(Some(patch));
            } else {
            next = model.single_succ(next.id)?.unwrap();
            }
            }
            */
            // or else make a lookup table
            if incoming_dt.is_quantized() && incoming_dt.size_of() == 1 {
                return Ok(Some(
                    transform_quant_seq_to_lut(model, origin.inputs[0], next.id.into())
                        .context("Transforming sequence to LUT")?,
                ));
                /*
                   let mut adhoc_model = TypedModel::default();
                   let mut wire = adhoc_model.add_source("ad-hoc", dt.fact([256]))?;
                   let mut next = model.single_succ(dequant.id)?.unwrap();
                   let mut name = None;
                // plug in dequant
                wire =
                adhoc_model.wire_node(&*dequant.name, dequant.op.clone(), [wire].as_ref())?[0];
                while next.id != op.id {
                name.get_or_insert(&*next.name);
                wire = adhoc_model.wire_node(&*next.name, next.op.clone(), [wire].as_ref())?[0];
                next = model.single_succ(next.id)?.unwrap();
                }
                // plug in quant
                wire = adhoc_model.wire_node(&*op.name, op.op.clone(), [wire].as_ref())?[0];
                adhoc_model.set_output_outlets(&[wire])?;
                let input = (0u8..=255).collect::<Vec<u8>>();
                let input = match dt {
                DatumType::I8 => unsafe {
                tensor1(std::mem::transmute::<&[u8], &[i8]>(&*input))
                },
                DatumType::U8 => tensor1(&input),
                _ => unreachable!(),
                };
                let output =
                SimplePlan::new(adhoc_model)?.run(tvec!(input.into_tvalue()))?.remove(0);
                let table: &[u8] = match dt {
                DatumType::I8 => unsafe { std::mem::transmute(output.as_slice::<i8>()?) },
                DatumType::U8 => output.as_slice::<u8>()?,
                _ => unreachable!(),
                };
                let op = lookup_table((tract_linalg::ops().lut_u8)(table));
                let mut patch = TypedModelPatch::default();
                let mut wire: OutletId = patch.tap_model(model, dequant.inputs[0])?;

                wire = patch.wire_node(name.unwrap_or(&*dequant.name), op, [wire].as_ref())?[0];
                patch.shunt_outside(model, OutletId::new(op.id, 0), wire)?;
                return Ok(Some(patch));
                */
            }
        }
        let (input_facts, output_facts) = model.node_facts(next.id)?;
        let invariants = next
            .op
            .axes_mapping(&input_facts, &output_facts)
            .with_context(|| format!("Querying invariants for {next}"))?;
        if invariants.is_element_wise_unary() {
            current = next;
        } else {
            break;
        }
    }
    Ok(None)
}

fn transform_quant_seq_to_lut(
    model: &TypedModel,
    src: OutletId, // wire before the dequant cast
    dst: OutletId, // wire after the requant cast
) -> TractResult<TypedModelPatch> {
    let incoming_dt = model.outlet_fact(src)?.datum_type;
    let outgoing_dt = model.outlet_fact(dst)?.datum_type;
    ensure!(incoming_dt.is_quantized() && incoming_dt.size_of() == 1);

    let mut adhoc_model = TypedModel::default();
    let wire = adhoc_model.add_source("ad-hoc", incoming_dt.fact([256]))?;
    let mut next = model.single_succ(src.node)?.unwrap();
    let mut name = None;
    // plug in dequant
    let dequant = model.node(src.node);
    let mut wire = tvec!(wire);
    while next.id != dst.node {
        name.get_or_insert(&*next.name);
        wire = adhoc_model.wire_node(&*next.name, next.op.clone(), &wire)?;
        next = model.single_succ(next.id)?.unwrap();
    }
    // plug in quant
    wire = adhoc_model.wire_node(&*next.name, next.op.clone(), &wire)?;
    adhoc_model.set_output_outlets(&wire)?;

    let input = tensor1(&(0u8..=255).collect_vec());
    let input = input.cast_to_dt(incoming_dt.unquantized())?.cast_to_dt(incoming_dt)?.into_owned();
    let output = SimpleState::new(SimplePlan::new(adhoc_model)?)?
        .run_plan_with_eval(tvec!(input.into_tvalue()), |s, op, node, inputs| {
            eprintln!("{node} {inputs:?}");
            eval(s, op, node, inputs)
        })?
        .remove(0);
    dbg!(&output);

    let table: &[u8] = match incoming_dt.unquantized() {
        DatumType::I8 => unsafe { std::mem::transmute(output.as_slice::<i8>()?) },
        DatumType::U8 => output.as_slice::<u8>()?,
        _ => unreachable!(),
    };
    let op = crate::ops::quant::lookup_table((tract_linalg::ops().lut_u8)(table));
    let mut patch = TypedModelPatch::default();
    let mut wire: OutletId = patch.tap_model(model, src)?;

    wire = patch.wire_node(name.unwrap_or(&*dequant.name), op, [wire].as_ref())?[0];
    patch.shunt_outside(model, dst, wire)?;
    Ok(patch)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ops::nn::sigmoid;

    #[test]
    fn test_lut() -> TractResult<()> {
        let mut model = TypedModel::default();
        let dt = i8::datum_type().with_zp_scale(0, 0.03);
        let src = model.add_source("src", dt.fact(&[10]))?;
        let mut wire = model.wire_node("dq", cast(f32::datum_type()), &[src])?;
        wire = model.wire_node("sigmoid", sigmoid(), &wire)?;
        wire = model.wire_node("q", cast(dt), &wire)?;
        model.set_output_outlets(&wire)?;

        let input =
            tensor1(&(-5i32..5i32).collect_vec()).cast_to::<f32>()?.cast_to_dt(dt)?.into_owned();
        let ref_output = model.clone().into_runnable()?.run(tvec!(input.clone().into_tvalue()))?;
        dbg!(&input);
        dbg!(&ref_output);

        let codegen = model.into_optimized()?;
        assert!(codegen.nodes.len() == 2); // Source then LookupTable
        let output = codegen.into_runnable()?.run(tvec!(input.into_tvalue()))?;
        output[0].close_enough(&ref_output[0], Approximation::Exact)
    }
}
