use super::lir_unary::{
    ConcreteMatMulGeometry, LirMatMulUnary, MatMulGeometry, ProtoFusedSpec, SymbolicMatMulGeometry,
};
use super::*;
use crate::internal::*;
use tract_ndarray::prelude::*;

/// The pseudo Unary matrix multiplier. A is constant, B is the input
#[derive(Debug, Clone, new, Hash)]
pub struct MatMulUnary {
    pub a: Arc<Tensor>,
    pub axes: MatMulAxes,
}

impl_dyn_hash!(MatMulUnary);

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
        if let Some(patch) = self
            .declutter_successors_are_slices(model, node)
            .context("declutter successor are slice")?
        {
            return Ok(Some(patch));
        }
        Ok(None)
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
        Ok(Some(new_mat_mul_unary_finite(model, node, &self.a, 0, &self.axes)?))
    }

    as_op!();
}

pub fn new_mat_mul_unary_finite(
    model: &TypedModel,
    node: &TypedNode,
    a: &Arc<Tensor>,
    input: usize,
    axes: &MatMulAxes,
) -> TractResult<TypedModelPatch> {
    let mut patch = TypedModelPatch::default();
    let mut wire = patch.tap_model(model, node.inputs[input])?;
    let b = model.outlet_fact(node.inputs[input])?;
    let c_dt = output_type(a.datum_type());
    let a_shape = ShapeFact::from(a.shape());
    let (m, k, n, c_shape) = compute_shape(&a_shape, &b.shape, *axes)?;
    let concrete_m = m.to_usize().unwrap();
    let concrete_k = k.to_usize().unwrap();

    let mmm = tract_linalg::ops()
        .mmm(
            a.datum_type(),
            b.datum_type,
            c_dt,
            Some(concrete_m),
            Some(concrete_k),
            n.to_usize().ok(),
        )
        .with_context(|| {
            format!(
                "No matrix multiplier for {:?}x{:?} to {:?}",
                a.datum_type(),
                b.datum_type,
                c_dt
            )
        })?;

    let mut a_iter_shape: TVec<usize> = a.shape().into();
    a_iter_shape[axes.a_m] = 1;
    a_iter_shape[axes.a_k] = 1;
    let packed_as = Array::from_shape_fn(&*a_iter_shape, |a_prefix| unsafe {
        let offset = a_prefix
            .as_array_view()
            .iter()
            .zip(a.strides())
            .map(|(x, s)| *x as isize * s)
            .sum::<isize>()
            * a.datum_type().size_of() as isize;
        let mut pa = Tensor::uninitialized_aligned_dt(
            a.datum_type(),
            &[mmm.a_pack().len(concrete_k, concrete_m)],
            mmm.a_pack().alignment(),
        )
        .unwrap();
        mmm.a_pack().pack(
            &mut pa.view_mut(),
            TensorView::from_bytes(a, offset, a.shape(), a.strides()),
            axes.a_k,
            axes.a_m,
        );
        (pa.into_arc_tensor(), vec![ProtoFusedSpec::Store])
    });
    unsafe {
        let mut packed_b_shape = b.shape.to_tvec();
        packed_b_shape.remove(axes.b_k.max(axes.b_n));
        packed_b_shape.remove(axes.b_k.min(axes.b_n));
        packed_b_shape.push(mmm.b_pack().len(k.clone(), n.clone()));
        wire = patch.wire_node(
            format!("{}.pack", &*node.name),
            super::MatMatMulPack { packer: mmm.b_pack(), k_axis: axes.b_k, mn_axis: axes.b_n },
            &[wire],
        )?[0];
        let geometry = if let Ok(n) = n.to_usize() {
            MatMulGeometry::Concrete(ConcreteMatMulGeometry {
                m: concrete_m,
                k: concrete_k,
                n,
                b_storage: mmm.b_packed(b.datum_type.size_of(), concrete_k),
            })
        } else {
            MatMulGeometry::Symbolic(SymbolicMatMulGeometry {
                m,
                k,
                n,
                mmm: mmm.clone(),
                b_datum_type: b.datum_type,
            })
        };
        wire = patch.wire_node(
            format!("{}.matmatmul", &*node.name),
            LirMatMulUnary {
                c_fact: c_dt.fact(&c_shape),
                geometry,
                micro_ops: packed_as,
                c_m_axis: axes.c_m,
                c_n_axis: axes.c_n,
                c_final_shape: c_shape.into(),
                reshape_post: vec![],
                mmm,
            },
            &[wire],
        )?[0];
        patch.shunt_outside(model, OutletId::new(node.id, 0), wire)?;
        patch.obliterate(node.id)?;
    }
    Ok(patch)
}

impl MatMulUnary {
    fn declutter_precusor_is_concat(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        use crate::ops::array::concat::ConcatSlice;
        use crate::ops::array::TypedConcat;
        if let Some(concat) = model.nodes()[node.inputs[0].node].op().downcast_ref::<TypedConcat>()
        {
            let mut patch = TypedModelPatch::new("split over k-concatenated input");
            if concat.axis == self.axes.b_k {
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

    // FIXME: should this be the general case for slice_output mecanism ?
    fn declutter_successors_are_slices(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        use crate::ops::array::Slice;
        let m_axis = self.axes.c_m;
        if let Some(slice) = node.outputs[0].successors.iter().find_map(|inlet| {
            if model
                .node(inlet.node)
                .op_as::<Slice>()
                .filter(|slice| slice.axis == m_axis)
                .is_some()
            {
                Some(inlet.node)
            } else {
                None
            }
        }) {
            let slice_op = model.node(slice).op_as::<Slice>().unwrap();
            let axis = slice_op.axis;
            let mut boundaries = tvec!();
            for succ in &node.outputs[0].successors {
                if let Some(slice) = model.node(succ.node).op_as::<Slice>() {
                    if slice.axis == axis {
                        boundaries.push(slice.start.clone());
                        boundaries.push(slice.end.clone());
                    }
                }
            }
            let mut boundaries: TVec<usize> = if let Ok(boundaries) =
                boundaries.iter().map(|x| x.to_usize()).collect::<TractResult<TVec<_>>>()
            {
                boundaries
            } else {
                return Ok(None);
            };
            let end = if let Ok(x) = node.outputs[0].fact.shape[axis].to_usize() {
                x
            } else {
                return Ok(None);
            };
            boundaries.push(end);
            boundaries.retain(|x| *x > 0);
            boundaries.sort();
            boundaries.dedup();
            let mut patch = TypedModelPatch::new("split over m-concatenated output");
            let input = patch.tap_model(model, node.inputs[0])?;

            let mut done = 0;
            let mut splits = tvec!();
            let a_m_axis = self.axes.a_m;
            for &up in &boundaries {
                let spliced_a = self.a.slice(a_m_axis, done, up)?;
                let wire = patch.wire_node(
                    format!("{}.split-m.{}..{}", node.name, done, up),
                    Self { a: spliced_a.into_arc_tensor(), ..self.clone() },
                    &[input],
                )?;
                splits.push(wire[0]);
                done = up;
            }
            let full = patch.wire_node(
                format!("{}.concat-m.full", node.name),
                crate::ops::array::TypedConcat::concat_vars(axis, splits.len()),
                &splits,
            )?[0];
            patch.shunt_outside(model, node.id.into(), full)?;
            for (ix, succ) in node.outputs[0].successors.iter().enumerate() {
                if let Some(slice) =
                    model.node(succ.node).op_as::<Slice>().filter(|slice| slice.axis == axis)
                {
                    // example: boundaries: 2, 3, wanted: 0..2 -> [0]
                    let slices: TVec<OutletId> = boundaries
                        .iter()
                        .zip(splits.iter())
                        .filter_map(|(up, split)| {
                            if *up > slice.start.to_usize().unwrap()
                                && *up <= slice.end.to_usize().unwrap()
                            {
                                Some(*split)
                            } else {
                                None
                            }
                        })
                        .collect();
                    let wire = if slices.len() > 1 {
                        patch.wire_node(
                            format!("{}.concat-m{}..{}..{}", node.name, ix, slice.start, slice.end),
                            crate::ops::array::TypedConcat::concat_vars(axis, slices.len()),
                            &slices,
                        )?[0]
                    } else {
                        slices[0]
                    };
                    patch.shunt_outside(model, succ.node.into(), wire)?;
                }
            }
            Ok(Some(patch))
        } else {
            Ok(None)
        }
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
