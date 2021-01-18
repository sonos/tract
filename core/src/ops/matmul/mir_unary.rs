use super::lir_unary::LirMatMulUnary;
use super::*;
use crate::internal::*;
use tract_ndarray::prelude::*;

/// The pseudo Unary matrix multiplier. A is constant, B is the input
#[derive(Debug, Clone, new, Hash)]
pub struct MatMulUnary {
    pub a: Arc<Tensor>,
    pub a_trans: bool,
    pub b_trans: bool,
    pub c_trans: bool,
}

impl_dyn_hash!(MatMulUnary);

impl Op for MatMulUnary {
    fn name(&self) -> Cow<str> {
        "MatMulUnary".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![
            format!(
                "a_trans:{:?} b_trans:{:?} c_trans:{:?}",
                self.a_trans, self.b_trans, self.c_trans
            ),
            format!("A: {:?}", self.a),
        ])
    }

    op_core_mir!();
    op_as_typed_op!();
}

impl EvalOp for MatMulUnary {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let t = eval(&self.a, &inputs[0], self.a_trans, self.b_trans, self.c_trans)?;
        Ok(tvec!(t.into_arc_tensor()))
    }
}

impl TypedOp for MatMulUnary {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        if inputs[0].rank() != self.a.rank() {
            bail!(
                "Inconsistent matmul between input {:?} and attribute {:?} (rank mismatch)",
                inputs[0],
                self.a
            );
        }
        let (_m, _k, _n, c_shape) = compute_shape(
            &self.a.shape().iter().map(|d| d.to_dim()).collect::<TVec<_>>(),
            &inputs[0].shape,
            self.a_trans,
            self.b_trans,
            self.c_trans,
        )?;
        let c_dt = output_type(inputs[0].datum_type);
        Ok(tvec!(TypedFact::dt_shape(c_dt, c_shape)))
    }

    fn invariants(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Invariants> {
        let input_fact = model.outlet_fact(node.inputs[0])?;
        if input_fact.shape.rank() != node.outputs[0].fact.shape.rank() {
            return Ok(Invariants::none());
        }
        let mut broadcasted_a_shape: TVec<_> = self.a.shape().into();
        while broadcasted_a_shape.len() < input_fact.shape.rank() {
            broadcasted_a_shape.insert(0, 1);
        }
        let mut invars = broadcasted_a_shape[..broadcasted_a_shape.len() - 2]
            .into_iter()
            .enumerate()
            .map(|(axis, &period)| AxisInfo::simple(axis).with_period(period))
            .collect::<Vec<_>>();
        if self.b_trans && self.c_trans && input_fact.rank() >= 2 {
            invars.push(AxisInfo::simple(input_fact.shape.rank() - 2))
        }
        if !self.b_trans && !self.c_trans {
            invars.push(AxisInfo::simple(input_fact.shape.rank() - 1))
        };
        Ok(invars.into_iter().collect())
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        let b = &model.outlet_fact(node.inputs[0])?;
        match change {
            AxisOp::Move(from, to) => {
                if *from == b.rank() - 2 && *to == b.rank() - 1 {
                    let op = MatMulUnary {
                        b_trans: !self.b_trans,
                        c_trans: !self.c_trans,
                        ..self.clone()
                    };
                    Ok(Some(AxisChangeConsequence::new(model, node, Some(Box::new(op)), change)))
                } else {
                    Ok(None)
                }
            }
            AxisOp::Add(axis) if *axis < b.rank() - 1 => {
                let mut a = self.a.clone().into_tensor();
                a.insert_axis(*axis)?;
                let op =
                    Some(Box::new(MatMulUnary { a: a.into_arc_tensor(), ..self.clone() }) as _);
                Ok(Some(AxisChangeConsequence::new(model, node, op, change)))
            }
            // b is [.. 1, n], can add axis to the right and transpose
            AxisOp::Add(axis) if *axis == b.rank() && b.shape[b.rank() - 2] == 1.to_dim() => {
                let mut a = self.a.clone().into_tensor();
                a.insert_axis(*axis - 2)?;
                let op = MatMulUnary {
                    b_trans: !self.b_trans,
                    c_trans: !self.c_trans,
                    a: a.into_arc_tensor(),
                    ..self.clone()
                };
                Ok(Some(AxisChangeConsequence::new(model, node, Some(Box::new(op)), change)))
            }
            AxisOp::Rm(axis) if b.rank() - axis > 2 => {
                let mut a = self.a.clone().into_tensor();
                a.remove_axis(*axis)?;
                let op =
                    Some(Box::new(MatMulUnary { a: a.into_arc_tensor(), ..self.clone() }) as _);
                Ok(Some(AxisChangeConsequence::new(model, node, op, change)))
            }
            _ => return Ok(None),
        }
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        use crate::ops::array::concat::ConcatSlice;
        use crate::ops::array::TypedConcat;
        let input_fact = model.outlet_fact(node.inputs[0])?;
        if let Some(concat) = model.nodes()[node.inputs[0].node].op().downcast_ref::<TypedConcat>()
        {
            let mut patch = TypedModelPatch::new("split over k-concatenated input");
            let k_axis = self.a.rank() - 1 - self.a_trans as usize;
            if concat.axis == input_fact.shape.rank() - 1 && self.b_trans {
                let mut input = 0;
                let concat_node = model.node(node.inputs[0].node);
                let offsets = concat
                    .offsets(&model.node_input_facts(concat_node.id)?)?
                    .iter()
                    .map(|x| x.to_usize())
                    .collect::<TractResult<Vec<usize>>>()?;
                let mut wires = vec![];
                for (ix, slice) in concat.slices.iter().enumerate() {
                    let wire = match slice {
                        ConcatSlice::Const(t) => patch.add_const(
                            format!("{}.const-{}", node.name, ix),
                            t.clone().into_arc_tensor(),
                        )?,
                        ConcatSlice::Var => {
                            input += 1;
                            patch.tap_model(model, concat_node.inputs[input - 1])?
                        }
                    };
                    let mut a = self.a.slice(k_axis, offsets[ix], offsets[ix + 1])?;
                    while a.rank() > 0 && a.shape()[0] == 1 {
                        a.remove_axis(0)?;
                    }
                    let wire = patch.wire_node(
                        format!("{}.k-{}-{}", node.name, offsets[ix], offsets[ix + 1]),
                        MatMulUnary { a: a.into_arc_tensor(), ..self.clone() },
                        &[wire],
                    )?[0];
                    wires.push(wire)
                }
                let mut wire = wires[0];
                for (ix, w) in wires[1..].iter().enumerate() {
                    wire = patch.wire_node(
                        format!("{}.k-add-{}", node.name, ix),
                        crate::ops::binary::TypedBinOp(Box::new(crate::ops::math::Add)),
                        &[wire, *w],
                    )?[0];
                }
                patch.shunt_outside(model, OutletId::new(node.id, 0), wire)?;
                return Ok(Some(patch));
            }
        }
        Ok(None)
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
        let b_fact = model.outlet_fact(node.inputs[0])?;
        let c_fact = &self.output_facts(&[b_fact])?[0];
        if axis + self.c_trans as usize == c_fact.shape.rank() {
            let a_split_axis = self.a.rank() - 1 - !self.a_trans as usize;
            let a = self.a.slice(a_split_axis, start, end)?.into_arc_tensor();
            let wire = patch.tap_model(model, node.inputs[0])?;
            return Ok(Some(
                patch.wire_node(
                    format!("{}.sliced-m-{}-{}", node.name, start, end),
                    Self { a, ..self.clone() },
                    &[wire],
                )?[0],
            ));
        }
        return Ok(None);
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let mut cost = super::mir::cost(
            self.a.shape(),
            &inputs[0].shape.to_tvec(),
            self.a.datum_type(),
            self.a_trans,
            self.b_trans,
        )?;
        cost.push((Cost::Params(self.a.datum_type()), self.a.len().to_dim()));
        Ok(cost)
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let b = args_1!(model.node_input_facts(node.id)?);
        if let Some(b_shape) = b.shape.as_concrete() {
            return Ok(Some(self.new_mat_mul_unary_finite(model, node, &b_shape, b.datum_type)?));
        }
        Ok(None)
    }

    as_op!();
}

impl MatMulUnary {
    fn new_mat_mul_unary_finite(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        b_shape: &[usize],
        b_dt: DatumType,
    ) -> TractResult<TypedModelPatch> {
        let mut patch = TypedModelPatch::default();
        let mut wire = patch.tap_model(model, node.inputs[0])?;

        let c_dt = output_type(self.a.datum_type());
        let (m, k, n, c_shape) =
            compute_shape(&self.a.shape(), b_shape, self.a_trans, self.b_trans, self.c_trans)?;

        let mm = tract_linalg::ops().mmm(self.a.datum_type(), b_dt, c_dt, m, k, n).with_context(
            || {
                format!(
                    "No matrix multiplier for {:?}x{:?} to {:?}",
                    self.a.datum_type(),
                    b_dt,
                    c_dt
                )
            },
        )?;

        let packed_as =
            Array::from_shape_fn(&self.a.shape()[0..self.a.rank() - 2], |a_prefix| unsafe {
                let mut pa = Tensor::uninitialized_aligned_dt(
                    self.a.datum_type(),
                    &[mm.a_pack().len(m)],
                    mm.a_pack().alignment(),
                )
                .unwrap();
                mm.a_pack().pack(
                    &mut pa.view_mut(),
                    &self.a.view_at_prefix(a_prefix.slice()).unwrap(),
                    !self.a_trans as usize,
                    self.a_trans as usize,
                );
                pa.into_arc_tensor()
            });
        unsafe {
            let b_storage = if n == 1 {
                mm.b_vec_from_data_and_stride(if self.b_trans {
                    1
                } else {
                    *b_shape.last().unwrap() as isize
                })
            } else {
                let mut packed_b_shape: TVec<usize> = b_shape[..b_shape.len() - 2].into();
                packed_b_shape.push(mm.b_pack().len(n));
                wire = patch.wire_node(
                    format!("{}.pack", &*node.name),
                    super::MatMatMulPack {
                        packer: mm.b_pack(),
                        trans: self.b_trans,
                        output_shape: packed_b_shape,
                    },
                    &[wire],
                )?[0];
                mm.b_packed()
            };
            let rank = c_shape.len();
            let mut strides = natural_strides(&c_shape);
            let mut overrided_shape = c_shape.clone();
            if self.c_trans {
                overrided_shape.swap(rank - 2, rank - 1);
                strides.swap(rank - 2, rank - 1);
            }
            wire = patch.wire_node(
                format!("{}.matmatmul", &*node.name),
                LirMatMulUnary {
                    b_storage,
                    c_fact: TypedFact::dt_shape(c_dt, &c_shape),
                    packed_as,
                    fused_ops: None,
                    mmm: mm,
                    k,
                    m,
                    c_m_axis: rank - 2 + self.c_trans as usize,
                    c_n_axis: rank - 2 + !self.c_trans as usize,
                    c_final_shape: c_shape.into(),
                },
                &[wire],
            )?[0];
            patch.shunt_outside(model, OutletId::new(node.id, 0), wire)?;
        }
        Ok(patch)
    }
}
