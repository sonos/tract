use crate::model::{optional_outputs, ParsingContext};
use crate::pb::NodeProto;
use tract_core::ops::cast::cast;
use tract_core::ops::math::{add, div, mul, rsqrt, square, sub};
use tract_core::ops::nn::{Reduce, Reducer};
use tract_hir::internal::*;
use tract_hir::ops::logic::wire_with_rank_broadcast;

pub fn layer_norm(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(-1);
    let epsilon = node.get_attr_opt("epsilon")?.unwrap_or(1e-5);
    let datum_type = node.get_attr_opt("stash_type")?.unwrap_or(DatumType::F32);
    let have_bias = node.input.len() == 3;
    let mut optional_outputs = optional_outputs(node).skip(1);
    let mean_output = optional_outputs.next().unwrap();
    let invstddev_output = optional_outputs.next().unwrap();
    Ok((
        expand(LayerNorm { axis, epsilon, datum_type, have_bias, mean_output, invstddev_output }),
        vec![],
    ))
}

#[derive(Debug, Clone)]
pub struct LayerNorm {
    axis: isize,
    epsilon: f32,
    datum_type: DatumType,
    have_bias: bool,
    mean_output: Option<usize>,
    invstddev_output: Option<usize>,
}

impl Expansion for LayerNorm {
    fn name(&self) -> Cow<str> {
        "LayerNorm".into()
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(1 + self.mean_output.is_some() as usize + self.invstddev_output.is_some() as usize)
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 2 + self.have_bias as usize)?;
        check_output_arity(outputs, self.nboutputs()?)?;
        s.equals(&inputs[0].datum_type, &inputs[1].datum_type)?;
        if self.have_bias {
            s.equals(&inputs[0].datum_type, &inputs[2].datum_type)?;
        }
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;

        if let Some(mean) = self.mean_output {
            s.equals(&outputs[mean].datum_type, self.datum_type)?;
            s.equals(&inputs[0].rank, &outputs[mean].rank)?;
        }
        if let Some(invstddev) = self.invstddev_output {
            s.equals(&outputs[invstddev].datum_type, self.datum_type)?;
            s.equals(&inputs[0].rank, &outputs[invstddev].rank)?;
        }
        s.given(&inputs[0].rank, move |s, rank| {
            let axis = if self.axis < 0 {
                (self.axis + rank as isize) as usize
            } else {
                self.axis as usize
            };
            for ax in 0..axis {
                if let Some(mean) = self.mean_output {
                    s.equals(&inputs[0].shape[ax], &outputs[mean].shape[ax])?;
                }
                if let Some(invstddev) = self.invstddev_output {
                    s.equals(&inputs[0].shape[ax], &outputs[invstddev].shape[ax])?;
                }
            }
            for ax in axis..rank as usize {
                if let Some(mean) = self.mean_output {
                    s.equals(&outputs[mean].shape[ax], 1.to_dim())?;
                }
                if let Some(invstddev) = self.invstddev_output {
                    s.equals(&outputs[invstddev].shape[ax], 1.to_dim())?;
                }
            }
            Ok(())
        })?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        // Mean = ReduceMean<axes=normalized_axes>(X) D = Sub(X, Mean) DD = Mul(D, D) Var = ReduceMean<axes=normalized_axes>(DD) VarEps = Add(Var, epsilon) StdDev = Sqrt(VarEps) InvStdDev = Reciprocal(StdDev) Normalized = Mul(D, InvStdDev) }
        let fact = model.outlet_fact(inputs[0])?.clone();
        let axis = if self.axis < 0 {
            (self.axis + fact.rank() as isize) as usize
        } else {
            self.axis as usize
        };
        let cast_x =
            model.wire_node(format!("{prefix}.cast_x"), cast(self.datum_type), &[inputs[0]])?;
        let cast_scale =
            model.wire_node(format!("{prefix}.cast_scale"), cast(self.datum_type), &[inputs[1]])?;
        let cast_bias = if self.have_bias {
            Some(model.wire_node(
                format!("{prefix}.cast_bias"),
                cast(self.datum_type),
                &[inputs[2]],
            )?)
        } else {
            None
        };
        let axes: TVec<usize> = (axis..fact.rank()).collect();
        let reduced_sum_x = model.wire_node(
            format!("{prefix}.reduced_sum"),
            Reduce { axes: axes.clone(), reducer: Reducer::Sum },
            &cast_x,
        )?;
        let len = axes.iter().map(|ax| fact.shape[*ax].clone()).product::<TDim>();
        let len = model.add_const(format!("{prefix}.len"), tensor0(len))?;
        let cast_len =
            model.wire_node(format!("{prefix}.cast_len"), cast(self.datum_type), &[len])?;
        let reduced_mean_x = wire_with_rank_broadcast(
            format!("{prefix}.reduced_mean_x"),
            model,
            div(),
            &[reduced_sum_x[0], cast_len[0]],
        )?;
        let d = model.wire_node(format!("{prefix}.d"), sub(), &[cast_x[0], reduced_mean_x[0]])?;
        let dd = model.wire_node(format!("{prefix}.dd"), square(), &d)?;
        let reduced_sum_dd = model.wire_node(
            format!("{prefix}.reduced_sum_dd"),
            Reduce { axes, reducer: Reducer::Sum },
            &dd,
        )?;
        let var = wire_with_rank_broadcast(
            format!("{prefix}.var"),
            model,
            div(),
            &[reduced_sum_dd[0], cast_len[0]],
        )?;
        let epsilon = model.add_const(
            format!("{prefix}.epsilon"),
            tensor0(self.epsilon).cast_to_dt(self.datum_type)?.into_owned(),
        )?;
        let var_eps = wire_with_rank_broadcast(
            format!("{prefix}.var_eps"),
            model,
            add(),
            &[var[0], epsilon],
        )?;
        let inv_std_dev = model.wire_node(format!("{prefix}.inv_std_dev"), rsqrt(), &var_eps)?;
        let normalized =
            model.wire_node(format!("{prefix}.normalized"), mul(), &[d[0], inv_std_dev[0]])?;
        // NormalizedScaled = Mul(Normalized, Scale) Y = Add(NormalizedScaled, B)
        let cast_normalized = model.wire_node(
            format!("{prefix}.cast_normalized"),
            cast(fact.datum_type),
            &normalized,
        )?;
        let normalized_scaled = wire_with_rank_broadcast(
            format!("{prefix}.normalized_scaled"),
            model,
            mul(),
            &[cast_normalized[0], cast_scale[0]],
        )?;
        let y = if let Some(bias) = cast_bias {
            wire_with_rank_broadcast(
                format!("{prefix}.y"),
                model,
                add(),
                &[normalized_scaled[0], bias[0]],
            )?
        } else {
            normalized_scaled
        };
        let mut outputs = tvec!(y[0]);
        if self.mean_output.is_some() {
            outputs.push(reduced_mean_x[0]);
        }
        if self.invstddev_output.is_some() {
            outputs.push(inv_std_dev[0]);
        }
        Ok(outputs)
    }
}
