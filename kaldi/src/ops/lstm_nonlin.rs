use crate::model::NodeLine;
use crate::model::ParsingContext;
use tract_hir::internal::*;

pub fn lstm_nonlin(ctx: &ParsingContext, name: &str) -> TractResult<Box<dyn InferenceOp>> {
    let node = &ctx.proto_model.config_lines.nodes.iter().find(|l| l.0 == name);
    let line = if let Some((_, NodeLine::Component(line))) = node {
        line
    } else {
        bail!("Could not find component {}", name);
    };
    let component = &ctx.proto_model.components[&line.component];
    let params: &Tensor = component.attributes.get("Params").context("missing attribute Params")?;
    Ok(expand(LstmNonlin { peepholes_params: params.to_owned() }))
}

#[derive(Clone, Debug, new, Hash)]
pub struct LstmNonlin {
    peepholes_params: Tensor,
}

tract_data::impl_dyn_hash!(LstmNonlin);

impl Expansion for LstmNonlin {
    fn name(&self) -> std::borrow::Cow<str> {
        "LstmNonlin".into()
    }

    op_kaldi!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, f32::datum_type())?;
        s.equals(&outputs[0].datum_type, f32::datum_type())?;
        s.equals(&inputs[0].rank, 2)?;
        s.equals(&outputs[0].rank, 2)?;
        s.equals(&outputs[0].shape[0], &inputs[0].shape[0])?;
        s.equals(5 * outputs[0].shape[1].bex(), 2 * inputs[0].shape[1].bex())?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        use math::add::bin_typed as add;
        use math::mul::bin_typed as mul;
        use tract_hir::ops::{array, math, nn};

        let params = self
            .peepholes_params
            .to_array_view::<f32>()?
            .into_dimensionality::<tract_ndarray::Ix2>()?;
        let w_ic: OutletId = target
            .add_const(
                format!("{}.w_ic", prefix),
                params.slice_axis(tract_ndarray::Axis(0), (0..1).into()).to_owned(),
            )?
            .into();
        let w_fc: OutletId = target
            .add_const(
                format!("{}.w_fc", prefix),
                params.slice_axis(tract_ndarray::Axis(0), (1..2).into()).to_owned(),
            )?
            .into();
        let w_oc: OutletId = target
            .add_const(
                format!("{}.w_oc", prefix),
                params.slice_axis(tract_ndarray::Axis(0), (2..3).into()).to_owned(),
            )?
            .into();

        let cell_hidden_dim = params.shape()[1];

        let mut five_parts = (0..5)
            .map(|ix| {
                Ok(target.wire_node(
                    format!("{}.part-{}", prefix, ix),
                    array::Slice::new(1, cell_hidden_dim * ix, cell_hidden_dim * (ix + 1)),
                    inputs,
                )?[0])
            })
            .collect::<TractResult<Vec<_>>>()?;
        let (i_part, f_part, c_part, o_part, c_prev) = args_5!(five_parts);

        macro_rules! wire {
            ($name: ident = $op: expr, $($param: expr),*) => {
                let $name = target.wire_node(
                    format!("{}.{}", prefix, stringify!($name)),
                    $op, [$($param),*].as_ref())?[0];
            }
        };

        // let i_t = sigmoid_f32(i_part + w_ic * c_prev);
        wire!(w_ic_c_prev = mul(), w_ic, c_prev);
        wire!(i_part_w_ic_c_prev = add(), i_part, w_ic_c_prev);
        wire!(i_t = nn::sigmoid(), i_part_w_ic_c_prev);

        // let f_t = sigmoid_f32(f_part + w_fc * c_prev);
        wire!(w_fc_c_prev = mul(), w_fc, c_prev);
        wire!(f_part_w_fc_c_prev = add(), f_part, w_fc_c_prev);
        wire!(f_t = nn::sigmoid(), f_part_w_fc_c_prev);

        // let c_t = f_t * c_prev + i_t * tanh_f32(c_part);
        wire!(tanh_c_part = math::tanh(), c_part);
        wire!(i_t_tanh_c_part = mul(), i_t, tanh_c_part);
        wire!(f_t_c_prev = mul(), f_t, c_prev);
        wire!(c_t = add(), f_t_c_prev, i_t_tanh_c_part);

        // let o_t = sigmoid_f32(o_part + w_oc * c_t);
        wire!(w_oc_c_t = mul(), w_oc, c_t);
        wire!(o_part_w_oc_c_t = add(), o_part, w_oc_c_t);
        wire!(o_t = nn::sigmoid(), o_part_w_oc_c_t);

        // let m_t = o_t * tanh_f32(c_t);
        wire!(tanh_c_t = math::tanh(), c_t);
        wire!(m_t = mul(), o_t, tanh_c_t);

        wire!(output = array::TypedConcat::concat_vars(1, 2), c_t, m_t);

        Ok(tvec!(output))
    }
}
