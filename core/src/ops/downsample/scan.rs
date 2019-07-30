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
            let ds = inner_model.chain_after(
                oo,
                format!("{}-{}", &down_node.name, ix),
                down_op.clone(),
                tvec!(down_op.transform_fact(inner_model.outlet_fact(oo)?)?),
            )?;
            Ok(OutletId::new(ds, 0))
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
    for mut input in &mut new_scan.input_mapping {
        match input {
            InputMapping::State { ref mut initializer } => {
                if let StateInitializer::Value(ref v) = initializer {
                    let new_v = down_op.as_stateless().unwrap().eval(tvec!(v.clone()))?;
                    *initializer = StateInitializer::Value(new_v[0].clone())
                }
            }
            // FIXME: check chunk multiple of stride
            InputMapping::Scan { ref mut chunk, .. } => *chunk = chunk.clone() / down_op.stride,
            _ => (),
        }
    }
    for mut output in &mut new_scan.output_mapping {
        match output {
            // FIXME: check chunk multiple of stride
            OutputMapping::Scan { ref mut chunk, .. } => *chunk = chunk.clone() / down_op.stride,
            _ => (),
        }
    }

    let mut patch = TypedModelPatch::default();
    let scan_id = patch.add_node(
        scan_node.name.clone(),
        new_scan,
        model
            .node_input_facts(scan_node.id)?
            .into_iter()
            .map(|f| down_op.transform_fact(f))
            .collect::<TractResult<_>>()?,
    )?;
    for (ix, &i) in scan_node.inputs.iter().enumerate() {
        patch.tap_model(model, i)?;
        let ds = patch.chain(
            format!("{}-{}", down_node.name, ix),
            down_op.clone(),
            tvec!(down_op.transform_fact(model.outlet_fact(i)?)?),
        )?;
        patch.add_edge(OutletId::new(ds, 0), InletId::new(scan_id, ix))?;
    }
    for ix in 0..scan_node.outputs.len() {
        // FIXME need to check earlier on that all output are followed by a ds
        let succ = scan_node.outputs[ix].successors[0].node;
        patch.shunt_outside(OutletId::new(succ, 0), OutletId::new(scan_id, ix))?;
    }
    Ok(Some(patch))
}
