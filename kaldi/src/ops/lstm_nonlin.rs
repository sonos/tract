use crate::model::NodeLine;
use crate::model::ParsingContext;
use tract_core::internal::*;
use tract_core::infer::*;

pub fn lstm_nonlin(ctx: &ParsingContext, name: &str) -> TractResult<Box<dyn InferenceOp>> {
    let node = &ctx.proto_model.config_lines.nodes.iter().find(|l| l.0 == name);
    let line = if let Some((_, NodeLine::Component(line))) = node {
        line
    } else {
        bail!("Could not find component {}", name);
    };
    let component = &ctx.proto_model.components[&line.component];
    let params: &Tensor = component.attributes.get("Params").ok_or("missing attribute Params")?;
    Ok(Box::new(LstmNonlin { peepholes_params: params.to_owned() }))
}

#[derive(Clone, Debug, new)]
pub struct LstmNonlin {
    peepholes_params: Tensor,
}

impl Op for LstmNonlin {
    fn name(&self) -> std::borrow::Cow<str> {
        "kaldi.LstmNonlin".into()
    }

    op_as_typed_op!();
}

impl StatelessOp for LstmNonlin {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        use tract_core::ndarray::*;

        let sigmoid = (tract_linalg::ops().ssigmoid)();
        let tanh = (tract_linalg::ops().stanh)();

        let input = args_1!(inputs);
        let input = input.to_array_view::<f32>()?.into_dimensionality()?;
        let params = self.peepholes_params.to_array_view::<f32>()?.into_dimensionality()?;
        let t_len = input.shape()[0];
        let cell_dim = input.shape()[1] / 5;
        let mut output = Array2::<f32>::zeros((t_len, 2 * cell_dim));
        let mut i_t = vec![0f32; cell_dim];
        let mut f_t = vec![0f32; cell_dim];
        let mut tanh_c_part = vec![0f32; cell_dim];
        let mut c_t = vec![0f32; cell_dim];
        let mut tanh_c_t = vec![0f32; cell_dim];
        let mut o_t = vec![0f32; cell_dim];
        for t in 0..t_len {
            for x in 0..cell_dim {
                let i_part = input[(t, 0 * cell_dim + x)];
                let f_part = input[(t, 1 * cell_dim + x)];
                tanh_c_part[x] = input[(t, 2 * cell_dim + x)];
                let c_prev = input[(t, 4 * cell_dim + x)];

                let w_ic = params[(0, x)];
                let w_fc = params[(1, x)];
                i_t[x] = i_part + w_ic * c_prev;
                f_t[x] = f_part + w_fc * c_prev;
            }
            sigmoid.run(&mut i_t);
            sigmoid.run(&mut f_t);
            tanh.run(&mut tanh_c_part);

            for x in 0..cell_dim {
                let w_oc = params[(2, x)];
                let o_part = input[(t, 3 * cell_dim + x)];
                let c_prev = input[(t, 4 * cell_dim + x)];
                c_t[x] = f_t[x] * c_prev + i_t[x] * tanh_c_part[x];
                o_t[x] = o_part + w_oc * c_t[x];
            }
            tanh_c_t.as_mut_slice().copy_from_slice(&c_t);
            tanh.run(&mut tanh_c_t);
            sigmoid.run(&mut o_t);

            for x in 0..cell_dim {
                let m_t = o_t[x] * tanh_c_t[x];

                output[(t, x)] = c_t[x];
                output[(t, cell_dim + x)] = m_t;
            }
        }
        Ok(tvec!(output.into_arc_tensor()))
    }
}

impl InferenceRulesOp for LstmNonlin {
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

    inference_op_as_op!();

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        use tract_core::ndarray;
        use tract_core::ops::math::add::bin_typed as add;
        use tract_core::ops::math::mul::bin_typed as mul;
        use tract_core::ops::{array, math, nn};
        use tract_core::ops::array::ConcatSlice;

        let params =
            self.peepholes_params.to_array_view::<f32>()?.into_dimensionality::<ndarray::Ix2>()?;
        let w_ic: OutletId = target
            .add_const(
                format!("{})-w_ic", node.name),
                params.index_axis(ndarray::Axis(0), 0).to_owned(),
            )?
            .into();
        let w_fc: OutletId = target
            .add_const(
                format!("{}-w_fc", node.name),
                params.index_axis(ndarray::Axis(0), 1).to_owned(),
            )?
            .into();
        let w_oc: OutletId = target
            .add_const(
                format!("{}-w_oc", node.name),
                params.index_axis(ndarray::Axis(0), 2).to_owned(),
            )?
            .into();

        let cell_hidden_dim = params.shape()[1];

        let input = mapping[&node.inputs[0]];

        let mut five_parts = (0..5)
            .map(|ix| {
                Ok(target.wire_node(
                    format!("{}-part-{}", node.name, ix),
                    array::Slice::new(1, cell_hidden_dim * ix, cell_hidden_dim * (ix + 1)),
                    &*tvec!(input),
                )?[0])
            })
            .collect::<TractResult<Vec<_>>>()?;
        let (i_part, f_part, c_part, o_part, c_prev) = args_5!(five_parts);

        macro_rules! wire {
            ($name: ident = $op: expr, $($param: expr),*) => {
                let $name = target.wire_node(
                    format!("{}-{}", node.name, stringify!($name)),
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

        wire!(
            output = array::Concat::new(1, tvec!(ConcatSlice::Var, ConcatSlice::Var)),
            c_t,
            m_t
        );

        Ok(tvec!(output))
    }
}

impl TypedOp for LstmNonlin {
    typed_op_as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(
            inputs[0].datum_type,
            [inputs[0].shape.dim(0), inputs[0].shape.dim(1) * 2 / 5].as_ref()
        )?))
    }
}
