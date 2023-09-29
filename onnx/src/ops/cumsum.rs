use tract_hir::internal::*;
use tract_hir::tract_core::ops::scan::ScanInfo;

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
pub struct CumSum {
    pub reverse: bool,
    pub exclusive: bool,
}

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
            format!("{prefix}.zero"),
            Tensor::zero_dt(data.datum_type, &[])?.into_arc_tensor(),
        )?;
        var_shape.set(axis, 1.to_dim());
        let init = model.wire_node(
            format!("{prefix}.init"),
            tract_core::ops::array::MultiBroadcastTo::new(var_shape.clone()),
            &[zero],
        )?[0];
        let chunk = if self.reverse { -1 } else { 1 };
        let input_mapping =
            vec![scan::InputMapping::Scan(ScanInfo { axis, chunk }), scan::InputMapping::State];
        // outputs will be
        // acc + x (!exclusive)
        // acc input (exclusive)
        let output_mapping = vec![
            scan::OutputMapping {
                scan: Some((0, ScanInfo { axis, chunk })),
                full_dim_hint: None,
                last_value_slot: None,
                state: true,
            },
            scan::OutputMapping {
                scan: Some((1, ScanInfo { axis, chunk })),
                full_dim_hint: None,
                last_value_slot: None,
                state: false,
            },
        ];
        let mut body = TypedModel::default();
        let var_fact = data.datum_type.fact(var_shape);
        let x = body.add_source("scan_input", var_fact.clone())?;
        let acc = body.add_source("acc_input", var_fact)?;
        let sum = body.wire_node("add", tract_core::ops::math::add(), &[x, acc])?[0];
        body.set_output_outlets(&[sum, acc])?;
        let scan = scan::Scan::new(body, input_mapping, output_mapping, 0)?;
        let wires = model.wire_node(prefix, scan, &[inputs[0], init])?;
        let output = wires[self.exclusive as usize];
        Ok(tvec![output])
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
}
