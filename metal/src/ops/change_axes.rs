use crate::kernels::{Memcpy, PermuteAxes};
use crate::tensor::MetalTensorExt;
use std::fmt::Debug;
use tract_core::internal::*;
use tract_itertools::Itertools;

#[derive(Clone, Hash, PartialEq)]
#[allow(clippy::large_enum_variant)] // FIXME ?
#[allow(clippy::derived_hash_with_manual_eq)] // FIXME. this one may be pretty bad. how about a.canonical() == b.canonical() ? need proper canonicalizeation of Reshape
pub struct MetalAxisOp(pub AxisOp);

impl MetalAxisOp {
    pub fn new(op: AxisOp) -> Option<Self> {
        Some(Self(op))
    }
}

impl Debug for MetalAxisOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            AxisOp::Add(a) => write!(f, "MetalAdd({a})"),
            AxisOp::Rm(a) => write!(f, "MetalRm({a})"),
            AxisOp::Move(from, to) => {
                write!(f, "MetalMove({from}, {to})")
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

        if let AxisOp::Move(from, to) = &self.0 {
            if let Some(t) = input.as_metal_tensor() {
                let output = objc::rc::autoreleasepool(|| {
                    crate::METAL_CONTEXT.with_borrow(|context| -> TractResult<_> {
                        let mut permutation: Vec<usize> = (0..t.rank()).collect();
                        permutation.remove(*from);
                        permutation.insert(*to, *from);
                        PermuteAxes.dispatch_eval(context, t, &permutation)
                    })
                })?;
                return Ok(tvec!(output.into_opaque_tensor().into_tvalue()));
            } else {
                self.0
                    .change_tensor(&mut input, false)
                    .with_context(|| anyhow!("Error while doing metal move axis"))?;
                return Ok(tvec!(input.into_tvalue()));
            }
        }

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

        if let Some(t) = input.as_metal_tensor_mut().and_then(|t| t.tensor_mut()) {
            _eval(self, session, t)?;
            Ok(tvec!(input.into_tvalue()))
        } else if let Some(t) = input.as_metal_tensor() {
            objc::rc::autoreleasepool(|| {
                crate::METAL_CONTEXT.with_borrow(|context| -> TractResult<_> {
                    // Copy perform by GPU
                    let mut cpy_t = Memcpy.dispatch_eval(context, t, 0)?;
                    let ref_mut_tensor = cpy_t.tensor_mut().ok_or_else(|| {
                        anyhow!(
                            "Could not take mutable reference of inner tensor after a metal copy."
                        )
                    })?;
                    _eval(self, session, ref_mut_tensor)?;
                    Ok(tvec![cpy_t.into_opaque_tensor().into_tvalue()])
                })
            })
        } else {
            _eval(self, session, &mut input)?;
            Ok(tvec!(input.into_tvalue()))
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
    pub mapping: AxesMapping,
    pub len: usize,
    pub dims: TVec<usize>,
    pub strides: TVec<isize>,
}

impl MetalIntoShape {
    pub fn new(core_op: IntoShape) -> Self {
        MetalIntoShape {
            mapping: core_op.mapping,
            len: core_op.len,
            strides: core_op.strides,
            dims: core_op.dims,
        }
    }
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

        if let Some(t) = input.as_metal_tensor_mut().and_then(|t| t.tensor_mut()) {
            ensure!(t.len() == self.len);
            unsafe { t.set_geometry_unchecked(&self.dims, &self.strides) };
            Ok(tvec!(input.into_tvalue()))
        } else if let Some(t) = input.as_metal_tensor() {
            ensure!(t.len() == self.len);
            objc::rc::autoreleasepool(|| {
                crate::METAL_CONTEXT.with_borrow(|context| -> TractResult<_>{
                    // Copy perform by GPU
                    let mut cpy_t = Memcpy.dispatch_eval(context, t, 0)?;
                    unsafe {
                        cpy_t.tensor_mut()
                          .ok_or_else(|| {
                             anyhow!("Could not take mutable reference of inner tensor after a metal copy.")
                          })?
                          .set_geometry_unchecked(&self.dims, &self.strides)
                    };
                    Ok(tvec![cpy_t.into_opaque_tensor()
                        .into_tvalue()])

                })
            })
        } else {
            ensure!(input.len() == self.len);
            unsafe { input.set_geometry_unchecked(&self.dims, &self.strides) };
            Ok(tvec!(input.into_tvalue()))
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

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::MetalTransform;
    use tract_core::transform::ModelTransform;

    #[test]
    fn test_change_axes() -> TractResult<()> {
        let input = Tensor::from_shape(
            &[1, 1, 4, 6],
            &[
                0.0f32, 6.0, 1.0, 7.0, 2.0, 8.0, 12.0, 18.0, 13.0, 19.0, 14.0, 20.0, 3.0, 9.0, 4.0,
                10.0, 5.0, 11.0, 15.0, 21.0, 16.0, 22.0, 17.0, 23.0,
            ],
        )?;
        let mut model = TypedModel::default();

        let x = model.add_source("x", f32::fact([1, 1, 4, 6]))?;
        let y_0 = model.wire_node(
            "y.0",
            AxisOp::Reshape(
                2,
                tvec![4.into(), 6.into()],
                tvec![2.into(), 2.into(), 3.into(), 2.into()],
            ),
            &[x],
        )?[0];
        let y_1 = model.wire_node("y.1", AxisOp::Move(3, 1), &[y_0])?[0];

        let y_2 = model.wire_node("y.2", AxisOp::Move(5, 2), &[y_1])?[0];

        let y_3_0 = model.wire_node("y.3.0", AxisOp::Rm(3), &[y_2])?[0];

        let _y_3_1 = model.wire_node(
            "y.3.1",
            AxisOp::Reshape(1, tvec![2.into(), 2.into()], tvec![4.into()]),
            &[y_3_0],
        )?[0];

        model.auto_outputs()?;

        let expected = model.clone().into_runnable()?.run(tvec![input.clone().into()])?;

        let metal_model = MetalTransform.transform_into(&model)?;
        let output = metal_model.clone().into_runnable()?.run(tvec![input.clone().into()])?;

        let _ = &output[0].close_enough(&expected[0], Approximation::Close)?;

        Ok(())
    }
}
