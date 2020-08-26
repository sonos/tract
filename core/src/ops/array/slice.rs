use crate::internal::*;
use ndarray::prelude::*;

#[derive(Debug, Clone, new, Default, PartialEq, Hash)]
pub struct Slice<D: DimLike + ToDim> {
    pub axis: usize,
    pub start: D,
    pub end: D,
}

impl<D: DimLike + ToDim + Hash> DynHash for Slice<D> {
    fn dyn_hash(&self, hasher: &mut dyn std::hash::Hasher) {
        tract_linalg::hash::dyn_hash(&self, hasher)
    }
}

impl<D: DimLike + ToDim + Hash> Slice<D> {
    unsafe fn eval_t<T: Datum>(&self, input: &Tensor) -> TractResult<Tensor> {
        let mut input = input.to_array_view_unchecked::<T>();
        input.slice_axis_inplace(
            Axis(self.axis),
            ::ndarray::Slice::from(
                (self.start.to_integer()? as i32)..(self.end.to_integer()? as i32),
            ),
        );
        Ok(Tensor::from(input.to_owned()).into())
    }
}

impl<D: DimLike + ToDim + Hash> Op for Slice<D> {
    fn name(&self) -> Cow<str> {
        "Slice".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {}, {}..{}", self.axis, self.start, self.end)])
    }

    canonic!();
    op_core_lir_mir!();
    op_as_typed_op!();
    op_as_pulsed_op!();

    fn same_as(&self, other: &dyn Op) -> bool {
        if let Some(other) = other.downcast_ref::<Self>() {
            other == self
        } else {
            false
        }
    }
}

impl<D: DimLike + ToDim + Hash> StatelessOp for Slice<D> {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        unsafe {
            let mut tensor =
                dispatch_datum_by_size!(Self::eval_t(input.datum_type())(self, &input))?;
            tensor.set_datum_type(input.datum_type());
            Ok(tvec!(tensor.into_arc_tensor()))
        }
    }
}

impl<D: DimLike + ToDim + Hash> TypedOp for Slice<D> {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = inputs[0].clone();
        fact.shape.set_dim(self.axis, (self.end.clone() - &self.start).to_dim())?;
        Ok(tvec!(fact))
    }

    fn invariants(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Invariants> {
        let fact = model.outlet_fact(node.inputs[0])?;
        let axes = (0..fact.shape.rank())
            .filter(|&ax| self.axis != ax)
            .map(|axis| AxisInfo::simple(axis))
            .collect();
        Ok(axes)
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        if let Some(axis) = change.transform_axis(self.axis) {
            Ok(Some(AxisChangeConsequence::new(
                model,
                node,
                Some(Box::new(Slice { axis, ..self.clone() }) as _),
                change,
            )))
        } else {
            Ok(None)
        }
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let prec = model.node(node.inputs[0].node);
        if let Some(tdim) = node.op_as::<Slice<TDim>>() {
            if let (Ok(start), Ok(end)) = (tdim.start.to_integer(), tdim.end.to_integer()) {
                return Ok(Some(TypedModelPatch::replace_single_op(
                    model,
                    node,
                    &node.inputs,
                    Slice { start: start as usize, end: end as usize, axis: self.axis },
                )?));
            }
        }
        if self.start == D::zero()
            && (self.end.clone().to_dim()
                == model.outlet_fact(node.inputs[0])?.shape.dim(self.axis))
        {
            return Ok(Some(TypedModelPatch::shunt_one_op(model, node)?));
        }
        let (start, end) = if let (Ok(s), Ok(e)) = (self.start.to_integer(), self.end.to_integer())
        {
            (s as usize, e as usize)
        } else {
            return Ok(None);
        };
        let mut patch = TypedModelPatch::default();
        //        println!("declutter slice {}", node);
        if let Some(wire) = prec.op().as_typed().unwrap().slice_output(
            model,
            prec,
            &mut patch,
            node.inputs[0].slot,
            self.axis,
            start,
            end,
        )? {
            patch.shunt_outside(model, OutletId::new(node.id, 0), wire)?;
            //            dbg!(&patch);
            //            dbg!(&self);
            if patch.model.nodes.len() == 2 && patch.model.node(1).op().same_as(self) {
                return Ok(None);
            }
            return Ok(Some(patch));
        }
        Ok(None)
    }

    fn pulsify(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
        _pulse: usize,
    ) -> TractResult<TVec<OutletId>> {
        let input = mapping[&node.inputs[0]];
        let fact = target.outlet_fact(input)?.clone();
        let op: Box<dyn PulsedOp> = if self.axis == fact.axis {
            let skip = self.start.to_integer()? as usize;
            let take = (self.end.clone() - &self.start).to_dim();
            PulsedAxisSlice::new(self.axis, skip, take).into()
        } else {
            dyn_clone::clone_box(self)
        };
        target.wire_node(&*node.name, op, &[input])
    }

    fn slice_output(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        patch: &mut TypedModelPatch,
        _output_slot: usize,
        axis: usize,
        start: usize,
        end: usize,
    ) -> TractResult<Option<OutletId>> {
        let prec = model.node(node.inputs[0].node);
        if axis != self.axis {
            return prec
                .op()
                .as_typed()
                .unwrap()
                .slice_output(model, &prec, patch, node.inputs[0].slot, axis, start, end)?
                .map(|w| Ok(patch.wire_node(&node.name, self.clone(), &[w])?[0]))
                .transpose();
        }
        Ok(None)
    }

    as_op!();
}

impl<D: DimLike + ToDim + Hash> PulsedOp for Slice<D> {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        if self.axis == fact.axis {
            fact.delay += self.start.to_integer()? as usize;
            fact.dim = (self.end.clone() - &self.start).to_dim();
        } else {
            fact.shape[self.axis] = (self.end.to_integer()? - self.start.to_integer()?) as usize;
        }
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}

#[derive(Debug, Clone, new, Default, Hash)]
pub struct PulsedAxisSlice {
    pub axis: usize,
    pub skip: usize,
    pub take: TDim,
}

tract_linalg::impl_dyn_hash!(PulsedAxisSlice);

impl Op for PulsedAxisSlice {
    fn name(&self) -> Cow<str> {
        "PulsedAxisSlice".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis:{}, skip:{} take:{}", self.axis, self.skip, self.take)])
    }

    op_core_lir_mir!();
    not_a_typed_op!();
    op_as_pulsed_op!();
}

impl StatelessOp for PulsedAxisSlice {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        Ok(inputs)
    }
}

impl PulsedOp for PulsedAxisSlice {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        fact.delay += self.skip;
        fact.dim = self.take.clone();
        Ok(tvec!(fact))
    }

    fn to_typed(&self) -> Box<dyn TypedOp> {
        Box::new(crate::ops::identity::Identity::default())
    }

    as_op!();
}
