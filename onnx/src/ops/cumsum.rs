use tract_hir::internal::*;

use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::*;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("CumSum", cumsum);
}

fn cumsum(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let reverse = node.get_attr_opt::<i64>("reverse")? == Some(1);
    let exclusive = node.get_attr_opt::<i64>("exclusive")? == Some(1);
    Ok((expand(CumSum { reverse, exclusive }), vec![]))
}

#[derive(Debug, Clone, Hash)]
struct CumSum {
    reverse: bool,
    exclusive: bool,
}

impl_dyn_hash!(CumSum);

impl Expansion for CumSum {
    fn name(&self) -> Cow<str> {
        "CumSum".into()
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        use tract_core::ops::scan;
        let axis =
            model.outlet_fact(inputs[1])?.konst.as_ref().context("Axis expected to be a const")?;
        let axis = axis.cast_to_scalar::<i64>()?;
        let data = model.outlet_fact(inputs[0])?.clone();
        let mut var_shape = data.shape.clone();
        let axis = if axis < 0 { (axis + data.rank() as i64) as usize } else { axis as usize };
        let zero = model.add_const(
            format!("{}.zero", prefix),
            Tensor::zero_dt(data.datum_type, &[])?.into_arc_tensor(),
        )?;
        var_shape.set(axis, 1.to_dim());
        let init = model.wire_node(
            format!("{}.init", prefix),
            tract_core::ops::array::MultiBroadcastTo::new(var_shape.clone().into()),
            &[zero],
        )?[0];
        let chunk = if self.reverse { -1 } else { 1 };
        let input_mapping = vec![
            scan::InputMapping::Scan { slot: 0, axis, chunk },
            scan::InputMapping::State { initializer: scan::StateInitializer::FromInput(1) },
        ];
        let output_mapping = vec![
            scan::OutputMapping {
                full_slot: Some(0),
                axis,
                chunk,
                full_dim_hint: None,
                last_value_slot: None,
                state: false,
            },
            scan::OutputMapping {
                full_slot: None,
                axis,
                chunk,
                full_dim_hint: None,
                last_value_slot: None,
                state: true,
            },
        ];
        let mut body = TypedModel::default();
        let var_fact = data.datum_type.fact(var_shape);
        let a = body.add_source("scan_input", var_fact.clone())?;
        let b = body.add_source("acc_input", var_fact)?;
        let sum = body.wire_node("add", tract_core::ops::math::add::bin_typed(), &[a, b])?[0];
        if self.exclusive {
            body.set_output_outlets(&[b, sum])?;
        } else {
            body.set_output_outlets(&[sum, sum])?;
        }
        let scan = scan::Scan::new(body, input_mapping, output_mapping, None, 0)?;
        model.wire_node(prefix, scan, &[inputs[0], init])
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 2)?;
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        s.equals(&inputs[1].rank, 0)?;
        Ok(())
    }

    op_onnx!();
}
