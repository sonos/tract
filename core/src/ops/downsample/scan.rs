use super::Downsample;
use crate::internal::*;
use crate::ops;
use crate::ops::scan::*;

pub fn pull_downsample_over_scan(
    model: &TypedModel,
    scan_node: &TypedNode,
    scan_op: &ops::scan::Typed,
    down_node: &TypedNode,
    down_op: &Downsample,
) -> TractResult<Option<TypedModelPatch>> {
    let mut inner_model = scan_op.body.clone();

    let outputs = inner_model.output_outlets()?.to_owned();
    let downsample_outputs = outputs
        .into_iter()
        .enumerate()
        .map(|(ix, oo)| {
            Ok(inner_model.wire_node(
                format!("{}-{}", &down_node.name, ix),
                down_op.clone(),
                &[oo],
            )?[0])
        })
        .collect::<TractResult<Vec<_>>>()?;
    inner_model.set_output_outlets(&*downsample_outputs)?;
    let mut inner_model = inner_model.declutter()?;

    for input in inner_model.input_outlets()? {
        let input = inner_model.node(input.node);
        if input.outputs[0].successors.len() > 1
            || !inner_model.node(input.outputs[0].successors[0].node).op().same_as(down_op)
        {
            return Ok(None);
        }
    }

    let inputs = inner_model.input_outlets()?.to_vec();
    for input in inputs {
        let ref mut fact = inner_model.node_mut(input.node).outputs[0].fact;
        *fact = down_op.transform_fact(fact)?;
        let ds = inner_model.node(input.node).outputs[0].successors[0].node;
        TypedModelPatch::shunt_one_op(&inner_model as _, inner_model.node(ds))?
            .apply(&mut inner_model)?;
    }

    let inner_model = crate::model::compact::compact(&inner_model.declutter()?)?;

    let mut new_scan = scan_op.clone();
    new_scan.body = inner_model;
    for input in &mut new_scan.input_mapping {
        match input {
            InputMapping::State { ref mut initializer } => {
                if let StateInitializer::Value(ref v) = initializer {
                    let new_v = down_op.as_stateless().unwrap().eval(tvec!(v.clone()))?;
                    *initializer = StateInitializer::Value(new_v[0].clone())
                }
            }
            InputMapping::Scan { ref mut chunk, .. } => {
                if chunk.to_integer()? as usize % down_op.stride != 0 {
                    return Ok(None);
                }
                *chunk = chunk.div_ceil(down_op.stride.to_dim())
            }
            _ => (),
        }
    }
    for output in &mut new_scan.output_mapping {
        if output.chunk.to_integer()? as usize % down_op.stride != 0 {
            return Ok(None);
        }
        output.full_dim_hint.as_mut().map(|d| *d = down_op.transform_dim(d));
        output.chunk = output.chunk.div_ceil(down_op.stride.to_dim());
    }

    let mut patch = TypedModelPatch::default();
    let mut inputs = tvec!();
    for (ix, &i) in scan_node.inputs.iter().enumerate() {
        let tap = patch.tap_model(model, i)?;
        let ds = patch.wire_node(format!("{}-{}", down_node.name, ix), down_op.clone(), &[tap])?[0];
        inputs.push(ds);
    }
    let scan = patch.wire_node(&*scan_node.name, new_scan, &inputs)?;
    for ix in 0..scan_node.outputs.len() {
        // FIXME need to check earlier on that all output are followed by a ds
        let succ = scan_node.outputs[ix].successors[0].node;
        patch.shunt_outside(OutletId::new(succ, 0), scan[ix])?;
    }
    Ok(Some(patch))
}
