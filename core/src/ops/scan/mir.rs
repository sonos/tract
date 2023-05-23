use super::lir::{LirScan, LirScanOpParams};
use tract_data::internal::*;

use super::*;

#[derive(Debug, Clone, Default)]
pub struct Scan {
    pub skip: usize,
    pub iters: TDim,
    pub body: TypedModel,
    pub input_mapping: Vec<InputMapping>,
    pub output_mapping: Vec<OutputMapping<TDim>>,
}

impl Scan {
    pub fn to_codegen_op(&self, optimize_inner: bool) -> TractResult<LirScan> {
        let mut model = self.body.clone();
        if optimize_inner {
            model = model.into_optimized()?;
        }
        let plan = SimplePlan::new(model)?;

        Ok(LirScan::new(Arc::new(LirScanOpParams::new(
            self.skip,
            self.iters.clone(),
            Arc::new(plan),
            self.input_mapping.clone(),
            self.output_mapping.clone(),
        ))))
    }

    pub fn new(
        body: TypedModel,
        input_mapping: Vec<InputMapping>,
        output_mapping: Vec<OutputMapping<TDim>>,
        skip: usize,
        iters: TDim,
    ) -> TractResult<Scan> {
        body.check_consistency()?;
        ensure!(input_mapping.len() == body.input_outlets()?.len());
        ensure!(output_mapping.len() == body.output_outlets()?.len());
        Ok(Scan { skip, iters, body, input_mapping, output_mapping })
    }
}

impl Op for Scan {
    fn name(&self) -> Cow<str> {
        "Scan".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        let mut lines = vec![format!("iters: {:?}", self.iters)];
        for (ix, im) in self.input_mapping.iter().enumerate() {
            lines.push(format!("Model input  #{ix}: {im:?}"));
        }
        for (ix, om) in self.output_mapping.iter().enumerate() {
            lines.push(format!("Model output #{ix}: {om:?}"));
        }
        Ok(lines)
    }

    fn validation(&self) -> Validation {
        Validation::Rounding
    }

    op_as_typed_op!();
}

impl EvalOp for Scan {
    fn is_stateless(&self) -> bool {
        false
    }
    fn state(
        &self,
        session: &mut SessionState,
        node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        self.to_codegen_op(false)?.state(session, node_id)
    }
}

impl TypedOp for Scan {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        anyhow::ensure!(inputs.len() == self.body.inputs.len());
        anyhow::ensure!(self.input_mapping.len() == self.body.inputs.len());
        anyhow::ensure!(
            self.input_mapping.iter().filter(|m| m.is_state()).count()
                == self.output_mapping.iter().filter(|m| m.state).count()
        );
        for (i, o) in
            self.input_mapping.iter().enumerate().filter(|(_, m)| m.is_state()).map(|(i, _)| i).zip(
                self.output_mapping.iter().enumerate().filter(|(_, m)| m.state).map(|(o, _)| o),
            )
        {
            anyhow::ensure!(
                self.body.outlet_fact(self.body.inputs[i])?
                    == self.body.outlet_fact(self.body.outputs[o])?
            )
        }
        let mut outputs = tvec!();
        for (ix, output) in self.output_mapping.iter().enumerate() {
            let fact = self.body.output_fact(ix)?;
            if let Some((slot, info)) = output.scan {
                let mut shape = fact.shape.clone();
                let scanning_dim =
                    output.full_dim_hint.clone().unwrap_or(shape[info.axis].clone() * &self.iters);
                shape.set(info.axis, scanning_dim);
                outputs.push((slot, fact.datum_type.fact(shape)));
            }
            if let Some(slot) = output.last_value_slot {
                outputs.push((slot, fact.datum_type.fact(fact.shape.clone())));
            }
        }
        outputs.sort_by_key(|a| a.0);
        anyhow::ensure!(outputs.iter().enumerate().all(|(ix, (slot, _))| ix == *slot));
        let outputs: TVec<_> = outputs.into_iter().map(|(_slot, v)| v).collect();
        Ok(outputs)
    }

    fn concretize_dims(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
        values: &SymbolValues,
    ) -> TractResult<TVec<OutletId>> {
        let inputs = node.inputs.iter().map(|o| mapping[o]).collect::<TVec<_>>();
        let op = Self {
            output_mapping: self
                .output_mapping
                .iter()
                .map(|om| om.concretize_dims(values))
                .collect::<TractResult<Vec<_>>>()?,
            body: self.body.concretize_dims(values)?,
            ..self.clone()
        };
        target.wire_node(&node.name, op, &inputs)
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        Ok(Some(TypedModelPatch::replace_single_op(
            model,
            node,
            &node.inputs,
            self.to_codegen_op(true)?,
        )?))
    }
}
