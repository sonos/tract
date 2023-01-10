use super::Downsample;
use crate::internal::*;
use crate::ops;
use crate::ops::identity::Identity;
use crate::ops::scan::*;

pub fn pull_downsample_over_scan(
    model: &TypedModel,
    scan_node: &TypedNode,
    scan_op: &ops::scan::Scan,
    down_node: &TypedNode,
    down_op: &Downsample,
) -> TractResult<Option<TypedModelPatch>> {
    if down_op.stride < 0 {
        return Ok(None);
    }

    // introduce downsample at end of body
    let mut downsampled_body = scan_op.body.clone();
    downsampled_body.check_consistency()?;
    let outputs = downsampled_body.output_outlets()?.to_owned();
    let downsample_outputs = outputs
        .into_iter()
        .enumerate()
        .map(|(ix, oo)| {
            Ok(downsampled_body.wire_node(
                format!("{}-{}", &down_node.name, ix),
                down_op.clone(),
                &[oo],
            )?[0])
        })
        .collect::<TractResult<Vec<_>>>()?;
    downsampled_body.set_output_outlets(&downsample_outputs)?;
    downsampled_body.declutter()?;
    downsampled_body.check_consistency()?;

    // check if downsample ops introduced at end have swimmed up to scan inputs during declutter
    for input in downsampled_body.input_outlets()? {
        let input = downsampled_body.node(input.node);
        if input.outputs[0]
            .successors
            .iter()
            .any(|succ| !downsampled_body.node(succ.node).op().same_as(down_op))
        {
            return Ok(None);
        }
    }

    let inputs = downsampled_body.input_outlets()?.to_vec();
    for input in inputs {
        let node = &mut downsampled_body.node_mut(input.node);
        let fact = &mut node.outputs[0].fact;
        *fact = down_op.transform_fact(fact)?;
        node.op_as_mut::<crate::ops::source::TypedSource>().unwrap().fact = fact.clone();
        let downsamples = downsampled_body.node(input.node).outputs[0].successors.clone();
        for ds in downsamples {
            TypedModelPatch::replace_single_op(
                &downsampled_body,
                downsampled_body.node(ds.node),
                &downsampled_body.node(ds.node).inputs,
                Identity,
            )?
            .apply(&mut downsampled_body)?;
        }
    }

    downsampled_body.check_consistency()?;
    let inner_model = downsampled_body.into_decluttered()?;

    let mut new_scan = scan_op.clone();
    new_scan.body = inner_model;
    for input in &mut new_scan.input_mapping {
        match input {
            InputMapping::State { ref mut initializer } => {
                if let StateInitializer::Value(ref v) = initializer {
                    let mut new_v = down_op.eval(tvec!(v.clone().into_tvalue()))?;
                    *initializer = StateInitializer::Value(new_v.remove(0).into_arc_tensor());
                }
            }
            InputMapping::Scan(info) => {
                if info.chunk > 0 && info.chunk as usize % down_op.stride as usize != 0 {
                    return Ok(None);
                }
                info.chunk = info.chunk.unsigned_abs().divceil(down_op.stride as usize) as isize
                    * info.chunk.signum()
            }
            _ => (),
        }
    }
    for output in &mut new_scan.output_mapping {
        if let Some(d) = output.full_dim_hint.as_mut() {
            *d = down_op.transform_dim(d)
        }
        if let Some(info) = &mut output.scan {
            if info.chunk as usize % down_op.stride as usize != 0 {
                return Ok(None);
            }
            info.chunk = info.chunk.unsigned_abs().divceil(down_op.stride as usize) as isize
                * info.chunk.signum()
        }
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
        patch.shunt_outside(model, OutletId::new(succ, 0), scan[ix])?;
    }
    Ok(Some(patch))
}
