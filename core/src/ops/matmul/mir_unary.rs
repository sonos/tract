use super::lir_unary::{AddMatMulGeometry, LirMatMulUnary, MapOutputAxisToInput, ProtoFusedSpec};
use super::*;
use crate::internal::*;
use crate::ops::array::TypedConcat;
use num_integer::Integer;
use tract_data::itertools::izip;

/// The pseudo Unary matrix multiplier. A is constant, B is the input
#[derive(Debug, Clone, new, Hash)]
pub struct MatMulUnary {
    pub a: Arc<Tensor>,
    pub axes: MatMulAxes,
}

impl Op for MatMulUnary {
    fn name(&self) -> Cow<str> {
        "MatMulUnary".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("{:?}", self.axes), format!("A: {:?}", self.a)])
    }

    op_as_typed_op!();
}

impl EvalOp for MatMulUnary {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let t = eval(&self.a, &inputs[0], self.axes)?;
        Ok(tvec!(t.into()))
    }
}

impl TypedOp for MatMulUnary {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(
            inputs[0].rank() == self.a.rank(),
            "Inconsistent matmul between input {:?} and attribute {:?} (rank mismatch)",
            inputs[0],
            self.a
        );
        let (_m, _k, _n, c_shape) = compute_shape(
            &self.a.shape().iter().map(|d| d.to_dim()).collect::<TVec<_>>(),
            &inputs[0].shape,
            self.axes,
        )?;
        let c_dt = output_type(inputs[0].datum_type);
        Ok(tvec!(c_dt.fact(c_shape)))
    }

    fn invariants(&self, inputs: &[&TypedFact], outputs: &[&TypedFact]) -> TractResult<Invariants> {
        mir_unary_invariants(inputs[0], outputs[0], self.axes)
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        if let Some((a, axes, wire_changes)) =
            mir_unary_change_axes(model, node, io, change, &self.axes, &self.a)?
        {
            let op = Self { axes, a: a.into_arc_tensor() };
            Ok(Some(AxisChangeConsequence { substitute_op: Some(Box::new(op)), wire_changes }))
        } else {
            Ok(None)
        }
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if let Some(patch) = self
            .declutter_precusor_is_concat(model, node)
            .context("declutter precursor is concat")?
        {
            return Ok(Some(patch));
        }
        /*
        if let Some(patch) = self
        .declutter_successors_are_slices(model, node)
        .context("declutter successor are slice")?
        {
        return Ok(Some(patch));
        }
        */
        Ok(None)
    }

    fn slice(
        &self,
        patch: &mut TypedModelPatch,
        prefix: &str,
        inputs: &[OutletId],
        output_axis: usize,
        start: usize,
        end: usize,
    ) -> TractResult<Option<TVec<OutletId>>> {
        if output_axis == self.axes.c_m {
            let a = self.a.slice(self.axes.a_m, start, end)?.into_arc_tensor();
            patch.wire_node(prefix, Self { a, ..self.clone() }, inputs).map(Some)
        } else {
            patch.wire_node(prefix, self.clone(), inputs).map(Some)
        }
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let mut cost = super::cost(
            self.a.shape(),
            &inputs[0].shape.to_tvec(),
            self.a.datum_type(),
            self.axes,
        )?;
        cost.push((Cost::Params(self.a.datum_type().unquantized()), self.a.len().to_dim()));
        Ok(cost)
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let b = args_1!(model.node_input_facts(node.id)?);
        if let Some(b_shape) = b.shape.as_concrete() {
            Ok(Some(self.new_mat_mul_unary_finite(model, node, b_shape, b.datum_type)?))
        } else {
            Ok(None)
        }
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
        let (m, k, n, c_shape) = compute_shape(self.a.shape(), b_shape, self.axes)?;

        let mmm = tract_linalg::ops()
            .mmm(self.a.datum_type(), b_dt, c_dt, Some(m), Some(k), Some(n))
            .with_context(|| {
                format!(
                    "No matrix multiplier for {:?}x{:?} to {:?}",
                    self.a.datum_type(),
                    b_dt,
                    c_dt
                )
            })?;

        unsafe {
            let mut packed_a_shape: TVec<usize> = self.a.shape().into();
            packed_a_shape[self.axes.a_k] = 1;
            packed_a_shape[self.axes.a_m] = 1;
            let packer_a = mmm.a_pack();
            packed_a_shape.push(Integer::next_multiple_of(
                &packer_a.len(k, m),
                &(packer_a.alignment() / self.a.datum_type().size_of()),
            ));

            let mut pa = Tensor::uninitialized_aligned_dt(
                self.a.datum_type(),
                &packed_a_shape,
                mmm.a_pack().alignment(),
            )?;
            for a_iter in tract_ndarray::indices(&packed_a_shape[..(packed_a_shape.len() - 1)]) {
                let mut pa_coords = a_iter.slice().to_vec();
                pa_coords.push(0);
                packer_a.pack(
                    &mut pa.view_offsetting_mut(&pa_coords)?,
                    &self.a.view_offsetting(a_iter.slice())?,
                    self.axes.a_k,
                    self.axes.a_m,
                )
            }

            let mut packed_b_shape: TVec<usize> = b_shape.into();
            packed_b_shape.remove(self.axes.b_k.max(self.axes.b_n));
            packed_b_shape.remove(self.axes.b_k.min(self.axes.b_n));
            packed_b_shape.push(mmm.b_pack().len(k, n));
            wire = patch.wire_node(
                format!("{}.pack", &*node.name),
                super::MatMatMulPack {
                    packer: mmm.b_pack(),
                    k_axis: self.axes.b_k,
                    mn_axis: self.axes.b_n,
                },
                &[wire],
            )?[0];

            let a_storage = mmm.a_packed(self.a.datum_type().size_of(), k);
            let b_storage = mmm.b_packed(b_dt.size_of(), k);
            let c_to_a: TVec<(usize, usize)> = izip!(
                (0..c_shape.len()).filter(|&c| c != self.axes.c_m && c != self.axes.c_n),
                (0..self.a.rank()).filter(|&a| a != self.axes.a_k && a != self.axes.a_m),
            )
            .filter(|(_, a)| self.a.shape()[*a] > 1)
            .collect();
            let c_to_b: TVec<(usize, usize)> = izip!(
                (0..c_shape.len()).filter(|&c| c != self.axes.c_m && c != self.axes.c_n),
                (0..b_shape.len()).filter(|&b| b != self.axes.b_k && b != self.axes.b_n),
            )
            .filter(|(_, b)| b_shape[*b] > 1)
            .collect();
            let micro_ops = vec![
                ProtoFusedSpec::AddMatMul(
                    AddMatMulGeometry {
                        a_storage,
                        b_storage,
                        k: k.to_dim(),
                        c_to_a_axis_mapping: MapOutputAxisToInput(c_to_a),
                        c_to_b_axis_mapping: MapOutputAxisToInput(c_to_b),
                    },
                    AttrOrInput::Attr(pa.into_arc_tensor()),
                    AttrOrInput::Input(0),
                ),
                ProtoFusedSpec::Store(mmm.c_view(self.axes.c_m, self.axes.c_n)),
            ];
            wire = patch.wire_node(
                format!("{}.matmatmul", &*node.name),
                LirMatMulUnary::new(
                    mmm,
                    c_dt.fact(&c_shape),
                    self.axes.c_m,
                    self.axes.c_n,
                    micro_ops,
                )?,
                &[wire],
            )?[0];
            patch.shunt_outside(model, OutletId::new(node.id, 0), wire)?;
            patch.obliterate(node.id)?;
        }
        Ok(patch)
    }

    fn declutter_precusor_is_concat(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if let Some(concat) = model.nodes()[node.inputs[0].node].op().downcast_ref::<TypedConcat>()
        {
            let mut patch = TypedModelPatch::new("split over k-concatenated input");
            if concat.axis == self.axes.b_k {
                let concat_node = model.node(node.inputs[0].node);
                let offsets = concat
                    .offsets(&model.node_input_facts(concat_node.id)?)?
                    .iter()
                    .map(|x| x.to_usize())
                    .collect::<TractResult<Vec<usize>>>()?;
                let mut wires = vec![];
                for (ix, input) in concat_node.inputs.iter().enumerate() {
                    let wire = patch.tap_model(model, *input)?;
                    let a = self.a.slice(self.axes.a_k, offsets[ix], offsets[ix + 1])?;
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
}

pub(super) fn mir_unary_invariants(
    input_fact: &TypedFact,
    output_fact: &TypedFact,
    axes: MatMulAxes,
) -> TractResult<Invariants> {
    anyhow::ensure!(input_fact.shape.rank() == output_fact.shape.rank());
    let axes = (0..input_fact.rank())
        .filter(|ax| *ax != axes.b_k)
        .zip((0..output_fact.rank()).filter(|ax| *ax != axes.c_m))
        .map(|(b, c)| AxisInfo {
            inputs: tvec!(Some(b)),
            outputs: tvec!(Some(c)),
            disposable: true,
            period: 1,
        })
        .collect();
    Ok(axes)
}

#[allow(clippy::type_repetition_in_bounds, clippy::type_complexity)]
pub(super) fn mir_unary_change_axes(
    model: &TypedModel,
    node: &TypedNode,
    io: InOut,
    change: &AxisOp,
    old_axes: &MatMulAxes,
    old_a: &Arc<Tensor>,
) -> TractResult<Option<(Arc<Tensor>, MatMulAxes, TVec<(InOut, AxisOp)>)>> {
    let b_fact = model.outlet_fact(node.inputs[0])?;
    let result = if io == InOut::In(0) {
        old_axes.change_axis_from_b(change, b_fact.rank())
    } else if io == InOut::Out(0) {
        old_axes.change_axis_from_c(change, b_fact.rank())
    } else {
        unreachable!();
    };
    if let Ok((axes, change_a, change_b, change_c)) = result {
        let new_a = if let Some(change_a) = change_a {
            let mut new_a = old_a.clone().into_tensor();
            if change_a.change_tensor(&mut new_a, false).is_err() {
                return Ok(None); // can not apply change to A (Rm on non-trivial axis ?)
            }
            new_a.into_arc_tensor()
        } else {
            old_a.clone()
        };
        let mut wires = tvec!();
        if let Some(change_b) = change_b {
            wires.push((InOut::In(0), change_b));
        }
        if let Some(change_c) = change_c {
            wires.push((InOut::Out(0), change_c));
        }
        Ok(Some((new_a, axes, wires)))
    } else {
        Ok(None) // is it right ? or return error ?
    }
}
