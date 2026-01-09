use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::NodeProto;
use tract_core::ops::math::{abs, div, max, mul, rsqrt, square};
use tract_core::ops::nn::{Reduce, Reducer};
use tract_hir::internal::*;
use tract_hir::ops::logic::wire_with_rank_broadcast;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("Normalizer", normalizer);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum NormKind {
    Max,
    L1,
    L2,
}

fn parse_norm_kind(s: &str) -> TractResult<NormKind> {
    match s.to_ascii_uppercase().as_str() {
        "MAX" => Ok(NormKind::Max),
        "L1" => Ok(NormKind::L1),
        "L2" => Ok(NormKind::L2),
        other => bail!("Invalid norm kind: {}", other),
    }
}

#[derive(Debug, Clone, Hash)]
struct Normalizer {
    kind: NormKind,
}

fn normalizer(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let norm: String = node.get_attr_opt("norm")?.unwrap_or_else(|| "MAX".to_string());
    let kind = parse_norm_kind(&norm)?;
    Ok((expand(Normalizer { kind }), vec![]))
}

impl Expansion for Normalizer {
    fn name(&self) -> StaticName {
        "Normalizer".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 1)?;
        check_output_arity(outputs, 1)?;
        s.equals(&outputs[0].datum_type, DatumType::F32)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let input_fact = model.outlet_fact(inputs[0])?.clone();
        let rank = input_fact.rank();
        ensure!(rank >= 1, "Normalizer expects rank 1 or 2 inputs");
        let axis = rank - 1;

        let mut x = inputs[0];
        let x_fact = model.outlet_fact(x)?.clone();

        if x_fact.datum_type != f32::datum_type() {
            x = model.wire_node(
                format!("{prefix}.to_f32"),
                tract_core::ops::cast::cast(f32::datum_type()),
                &[x],
            )?[0];
        }

        let eps = model.add_const(format!("{prefix}.eps"), rctensor0(1e-12f32))?;

        let y = match self.kind {
            NormKind::Max => {
                let ax = model.wire_node(format!("{prefix}.abs"), abs(), &[x])?;
                let d0 = model.wire_node(
                    format!("{prefix}.max"),
                    Reduce { axes: tvec![axis], reducer: Reducer::Max },
                    &ax,
                )?[0];
                let d = wire_with_rank_broadcast(
                    format!("{prefix}.clamp_max"),
                    model,
                    max(),
                    &[d0, eps],
                )?[0];
                wire_with_rank_broadcast(format!("{prefix}.div_max"), model, div(), &[x, d])?[0]
            }
            NormKind::L1 => {
                let ax = model.wire_node(format!("{prefix}.abs"), abs(), &[x])?;
                let d0 = model.wire_node(
                    format!("{prefix}.sum_abs"),
                    Reduce { axes: tvec![axis], reducer: Reducer::Sum },
                    &ax,
                )?[0];
                let d = wire_with_rank_broadcast(
                    format!("{prefix}.clamp_l1"),
                    model,
                    max(),
                    &[d0, eps],
                )?[0];
                wire_with_rank_broadcast(format!("{prefix}.div_sum"), model, div(), &[x, d])?[0]
            }
            NormKind::L2 => {
                let x2 = model.wire_node(format!("{prefix}.square"), square(), &[x])?;
                let ss0 = model.wire_node(
                    format!("{prefix}.sum_sq"),
                    Reduce { axes: tvec![axis], reducer: Reducer::Sum },
                    &x2,
                )?[0];
                let ss = wire_with_rank_broadcast(
                    format!("{prefix}.clamp_l2"),
                    model,
                    max(),
                    &[ss0, eps],
                )?[0];
                let inv = model.wire_node(format!("{prefix}.rsqrt"), rsqrt(), &[ss])?[0];
                wire_with_rank_broadcast(format!("{prefix}.mul_invnorm"), model, mul(), &[x, inv])?
                    [0]
            }
        };

        Ok(tvec!(y))
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(1)
    }
}
