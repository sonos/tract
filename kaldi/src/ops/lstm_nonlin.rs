use crate::model::NodeLine;
use crate::model::ParsingContext;
use tract_core::internal::*;

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

    fn translation_invariants(
        &self,
        _model: &TypedModel,
        _node: &TypedNode,
    ) -> TractResult<Vec<TranslationInvariant>> {
        Ok(vec![TranslationInvariant { axis: 0, period: 1 }])
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        use std::convert::TryInto;
        use tract_core::ndarray;
        use tract_core::ops::{array, math, nn};

        let input_fact = model.outlet_fact(node.inputs[0])?;
        let ref t_len = input_fact.shape.dim(0);
        let cell_hidden_dim = input_fact.shape.dim(1).to_integer()? as usize / 5;

        let params =
            self.peepholes_params.to_array_view::<f32>()?.into_dimensionality::<ndarray::Ix2>()?;

        let mut patch = TypedModelPatch::default();
        let input = patch.tap_model(&model, node.inputs[0])?;

        let fact: TypedTensorInfo =
            TensorFact::dt_shape(f32::datum_type(), tvec!(t_len.clone(), cell_hidden_dim.to_dim()))
                .try_into()?;

        let mut five_parts = (0..5)
            .map(|ix| {
                patch.add_node_simple(
                    format!("{}-part-{}", node.name, ix),
                    array::Slice::new(
                        vec![1],
                        vec![cell_hidden_dim * ix],
                        vec![cell_hidden_dim * (ix + 1)],
                    ),
                    tvec!(input.node),
                    fact.clone(),
                )
            })
            .collect::<TractResult<Vec<_>>>()?;
        let (i, f, c, o, c_prev) = args_5!(five_parts);
        // let i_t = sigmoid_f32(i_part + w_ic * c_prev);
        let i_t = patch.add_node_simple(
            format!("{}-i_t-mul", node.name),
            math::Mul::UnaryA::new(
                f32::datum_type().into(),
                params.index_axis(ndarray::Axis(0), 0).to_owned().into_arc_tensor(),
            ),
            tvec!(c_prev),
            fact.clone(),
        )?;
        let i_t = patch.add_node_simple(
            format!("{}-i_t-add", node.name),
            math::Add::default(),
            tvec!(i_t, i),
            fact.clone(),
        )?;
        let i_t = patch.add_node_simple(
            format!("{}-i_t-sigmoid", node.name),
            nn::Sigmoid::default(),
            tvec!(i_t),
            fact.clone(),
        )?;

        // let f_t = sigmoid_f32(f_part + w_fc * c_prev);
        let f_t = patch.add_node_simple(
            format!("{}-f_t-mul", node.name),
            math::Mul::UnaryA::new(
                f32::datum_type().into(),
                params.index_axis(ndarray::Axis(0), 1).to_owned().into_arc_tensor(),
            ),
            tvec!(c_prev),
            fact.clone(),
        )?;
        let f_t = patch.add_node_simple(
            format!("{}-f_t-add", node.name),
            math::Add::default(),
            tvec!(f_t, f),
            fact.clone(),
        )?;
        let f_t = patch.add_node_simple(
            format!("{}-f_t-sigmoid", node.name),
            nn::Sigmoid::default(),
            tvec!(f_t),
            fact.clone(),
        )?;

        // let c_t = f_t * c_prev + i_t * tanh_f32(c_part);
        let tanh_c = patch.add_node_simple(
            format!("{}-c_t-tanh", node.name),
            math::Tanh::default(),
            tvec!(c),
            fact.clone(),
        )?;
        let i_t_tanh_c = patch.add_node_simple(
            format!("{}-i_t_c_t-tanh", node.name),
            math::Mul::default(),
            tvec!(i_t, tanh_c),
            fact.clone(),
        )?;
        let f_t_c_prev = patch.add_node_simple(
            format!("{}-f_t_c_prev", node.name),
            math::Mul::default(),
            tvec!(f_t, c_prev),
            fact.clone(),
        )?;
        let c_t = patch.add_node_simple(
            format!("{}-c_t", node.name),
            math::Add::default(),
            tvec!(f_t_c_prev, i_t_tanh_c),
            fact.clone(),
        )?;

        // let o_t = sigmoid_f32(o_part + w_oc * c_t);
        let o_t = patch.add_node_simple(
            format!("{}-o_t-mul", node.name),
            math::Mul::UnaryA::new(
                f32::datum_type().into(),
                params.index_axis(ndarray::Axis(0), 2).to_owned().into_arc_tensor(),
            ),
            tvec!(c_t),
            fact.clone(),
        )?;
        let o_t = patch.add_node_simple(
            format!("{}-o_t-add", node.name),
            math::Add::default(),
            tvec!(o_t, o),
            fact.clone(),
        )?;
        let o_t = patch.add_node_simple(
            format!("{}-o_t-sigmoid", node.name),
            nn::Sigmoid::default(),
            tvec!(o_t),
            fact.clone(),
        )?;

        // let m_t = o_t * tanh_f32(c_t);
        let tanh_c_t = patch.add_node_simple(
            format!("{}-tanh_c_t", node.name),
            math::Tanh::default(),
            tvec!(c_t),
            fact.clone(),
        )?;

        let m_t = patch.add_node_simple(
            format!("{}-m_t", node.name),
            math::Mul::default(),
            tvec!(o_t, tanh_c_t),
            fact.clone(),
        )?;

        let output = patch.add_node_simple(
            format!("{}-ouput", node.name),
            array::Concat::new(1),
            tvec!(c_t, m_t),
            node.outputs[0].fact.clone(),
        )?;

        patch.shunt_outside(OutletId::new(node.id, 0), OutletId::new(output, 0))?;
        Ok(Some(patch))
    }
}

impl StatelessOp for LstmNonlin {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        use tract_core::ndarray::*;

        let sigmoid = (tract_linalg::ops().ssigmoid)();
        let sigmoid_f32 = |f: f32| -> f32 {
            let mut f = [f];
            sigmoid.run(&mut f);
            f[0]
        };

        let tanh = (tract_linalg::ops().stanh)();
        let tanh_f32 = |f: f32| -> f32 {
            let mut f = [f];
            tanh.run(&mut f);
            f[0]
        };

        let input = args_1!(inputs);
        let input = input.to_array_view::<f32>()?.into_dimensionality()?;
        let params = self.peepholes_params.to_array_view::<f32>()?.into_dimensionality()?;
        let t_len = input.shape()[0];
        let cell_dim = input.shape()[1] / 5;
        let mut output = Array2::<f32>::zeros((t_len, 2 * cell_dim));
        for t in 0..t_len {
            for x in 0..cell_dim {
                let i_part = input[(t, 0 * cell_dim + x)];
                let f_part = input[(t, 1 * cell_dim + x)];
                let c_part = input[(t, 2 * cell_dim + x)];
                let o_part = input[(t, 3 * cell_dim + x)];
                let c_prev = input[(t, 4 * cell_dim + x)];

                let w_ic = params[(0, x)];
                let w_fc = params[(1, x)];
                let w_oc = params[(2, x)];

                let i_t = sigmoid_f32(i_part + w_ic * c_prev);
                let f_t = sigmoid_f32(f_part + w_fc * c_prev);
                let c_t = f_t * c_prev + i_t * tanh_f32(c_part);
                let o_t = sigmoid_f32(o_part + w_oc * c_t);
                let m_t = o_t * tanh_f32(c_t);

                output[(t, x)] = c_t;
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
}
