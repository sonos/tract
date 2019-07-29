use super::Downsample;
use crate::internal::*;
use crate::ops;

pub fn pull_downsample_over_scan(
    model: &TypedModel,
    scan_node: &TypedNode,
    scan_op: &ops::scan::Typed,
    down_node: &TypedNode,
    down_op: &Downsample,
) -> TractResult<Option<TypedModelPatch>> {
    let mut inner_model = scan_op.body.clone();
    println!("ORIGINAL\n{:#?}", inner_model);

    let outputs = inner_model.output_outlets()?.to_owned();
    let downsample_outputs = outputs.into_iter().enumerate().map(|(ix, oo)| {
        let ds = inner_model.chain_after(oo, 
            format!("{}-{}", &down_node.name, ix),
            down_op.clone(),
            tvec!(down_op.transform_fact(inner_model.outlet_fact(oo)?)?),
        )?;
        Ok(OutletId::new(ds, 0))
    }).collect::<TractResult<Vec<_>>>()?;
    inner_model.set_output_outlets(&*downsample_outputs)?;

    println!("DOWNSAMPLED\n{:#?}", inner_model);

    let inner_model = inner_model.declutter()?;
    println!("DECLUTTERED\n{:#?}", inner_model);

    panic!();
    let mut new_scan = scan_op.clone();
    new_scan.body = inner_model;

    let mut patch = TypedModelPatch::default();
    patch.tap_model(model, scan_node.inputs[0])?;
    let id = patch.chain(
        scan_node.name.clone(),
        new_scan,
        model.node_input_facts(scan_node.id)?.into_iter().cloned().collect(),
    )?;
    patch.shunt_outside(OutletId::new(down_node.id, 2), OutletId::new(id, 2))?;
    Ok(Some(patch))
}
