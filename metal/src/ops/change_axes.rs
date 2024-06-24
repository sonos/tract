use crate::tensor::MetalTensorExt;
use crate::IntoMetal;
use std::fmt::Debug;
use tract_core::internal::*;
use tract_itertools::Itertools;

#[derive(Clone, Hash, PartialEq)]
#[allow(clippy::large_enum_variant)] // FIXME ?
#[allow(clippy::derived_hash_with_manual_eq)] // FIXME. this one may be pretty bad. how about a.canonical() == b.canonical() ? need proper canonicalizeation of Reshape
pub struct MetalAxisOp(pub AxisOp);

impl MetalAxisOp {
    pub fn new(op: AxisOp) -> Option<Self> {
        if !matches!(op, AxisOp::Move(_, _)) {
            Some(Self(op))
        } else {
            None
        }
    }
}

impl Debug for MetalAxisOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            AxisOp::Add(a) => write!(f, "MetalAdd({a})"),
            AxisOp::Rm(a) => write!(f, "MetalRm({a})"),
            AxisOp::Move(_, _) => {
                unimplemented!("Unsupported Metal AxisOp {:?}", self)
            }
            AxisOp::Reshape(at, from, to) => {
                write!(
                    f,
                    "MetalReshape({at}, [{}], [{}])",
                    from.iter().join(","),
                    to.iter().join(",")
                )
            }
        }
    }
}

impl Op for MetalAxisOp {
    fn name(&self) -> Cow<str> {
        format!("Metal{}", self.0.name()).into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        self.0.info()
    }

    op_as_typed_op!();
}

impl EvalOp for MetalAxisOp {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        session: &SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let mut input = args_1!(inputs).into_tensor();
        fn _eval(op: &MetalAxisOp, session: &SessionState, t: &mut Tensor) -> TractResult<()> {
            match &op.0 {
                AxisOp::Reshape(skip, from, to) => {
                    let from = from.iter().map(|d| d.eval(&session.resolved_symbols)).collect();
                    let to = to.iter().map(|d| d.eval(&session.resolved_symbols)).collect();
                    AxisOp::Reshape(*skip, from, to).change_tensor(t, false)?
                }
                _ => op.0.change_tensor(t, false)?,
            }
            Ok(())
        }
        match input.as_opaque_metal_tensor() {
            Some(metal_tensor) => {
                // TODO avoid this copy
                let mut t = metal_tensor.tensor().clone();
                _eval(self, session, &mut t)?;
                Ok(tvec![t.into_metal()?.into_opaque_tensor().into()])
            }
            None => {
                _eval(self, session, &mut input)?;
                Ok(tvec!(input.into_tvalue()))
            }
        }
    }
}

impl TypedOp for MetalAxisOp {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::metal_output_facts(inputs, |facts| {
            let mut shape = facts[0].shape.clone();
            self.0
                .change_shape(&mut shape, false)
                .with_context(|| format!("Applying {self:?} to {:?}", facts[0]))?;
            Ok(tvec!(facts[0].datum_type.fact(shape)))
        })
        .with_context(|| anyhow::anyhow!("Error while computing facts for {:?}", self.name()))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        let ref_inputs = crate::utils::metal_facts(inputs, |facts| Ok(facts.to_vec()))?;
        let ref_outputs = crate::utils::metal_facts(outputs, |facts| Ok(facts.to_vec()))?;
        self.0.axes_mapping(&ref_inputs, &ref_outputs)
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        self.0.declutter(model, node)
    }

    fn suggested_axis_changes(&self) -> TractResult<TVec<(InOut, AxisOp)>> {
        self.0.suggested_axis_changes()
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        self.0.change_axes(model, node, io, change)
    }

    fn concretize_dims(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
        values: &SymbolValues,
    ) -> TractResult<TVec<OutletId>> {
        let op = if let MetalAxisOp(AxisOp::Reshape(axis, from, to)) = self {
            MetalAxisOp(AxisOp::Reshape(
                *axis,
                from.iter().map(|d| d.eval(values)).collect(),
                to.iter().map(|d| d.eval(values)).collect(),
            ))
        } else {
            self.clone()
        };
        target.wire_node(&node.name, op, &[mapping[&node.inputs[0]]])
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let conc_shape =
            crate::utils::metal_fact(&node.outputs[0].fact, |fact| Ok(fact.shape.as_concrete()))?;
        if let Some(shape) = conc_shape {
            if !matches!(self, MetalAxisOp(AxisOp::Move(_, _))) {
                let (inputs, outputs) = model.node_facts(node.id)?;
                let mapping = self.axes_mapping(&inputs, &outputs)?;
                let op = MetalIntoShape {
                    mapping,
                    len: shape.iter().product(),
                    strides: Tensor::natural_strides(shape),
                    dims: shape.into(),
                };
                return Ok(Some(TypedModelPatch::replace_single_op(
                    model,
                    node,
                    &node.inputs,
                    op,
                )?));
            }
        }
        Ok(None)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MetalIntoShape {
    pub(crate) mapping: AxesMapping,
    pub(crate) len: usize,
    pub(crate) dims: TVec<usize>,
    pub(crate) strides: TVec<isize>,
}

impl Op for MetalIntoShape {
    fn name(&self) -> Cow<str> {
        "MetalIntoShape".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("{}", self.mapping)])
    }

    op_as_typed_op!();
    impl_op_same_as!();
}

impl EvalOp for MetalIntoShape {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let mut input = args_1!(inputs).into_tensor();
        match input.as_opaque_metal_tensor() {
            Some(metal_tensor) => {
                ensure!(metal_tensor.len() == self.len);
                // TODO avoid this copy
                let mut t = metal_tensor.tensor().clone();
                unsafe { t.set_geometry_unchecked(&self.dims, &self.strides) };
                Ok(tvec![t.into_metal()?.into_opaque_tensor().into()])
            }
            None => {
                ensure!(input.len() == self.len);
                unsafe { input.set_geometry_unchecked(&self.dims, &self.strides) };
                Ok(tvec!(input.into_tvalue()))
            }
        }
    }
}

impl TypedOp for MetalIntoShape {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::metal_output_facts(inputs, |facts| {
            Ok(tvec!(facts[0].datum_type.fact(&self.dims)))
        })
        .with_context(|| anyhow::anyhow!("Error while computing facts for {:?}", self.name()))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if let Some(succ) = model.single_succ(node.id)? {
            if let Some(into_shape) = succ.op_as::<MetalIntoShape>() {
                let op = Self {
                    mapping: self.mapping.compose(&into_shape.mapping)?,
                    ..into_shape.clone()
                };
                return Ok(Some(TypedModelPatch::fuse_with_next(model, node, op)?));
            }
        }
        Ok(None)
    }

    as_op!();
}
