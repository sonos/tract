use crate::internal::*;
use crate::ops;
use super::Downsample;

pub fn pull_downsample_over_crop(
    model: &TypedModel,
    crop_node: &TypedNode,
    crop_op: &ops::array::Crop,
    down_node: &TypedNode,
    down_op: &Downsample,
) -> TractResult<Option<TypedModelPatch>> {
    let modulo = (down_op.modulo + crop_op.prune[down_op.axis].0) % down_op.stride;
    let left = (down_op.modulo + crop_op.prune[down_op.axis].0) / down_op.stride;
    let mut patch = TypedModelPatch::default();
    patch.tap_model(model, crop_node.inputs[0])?;
    let input_outlet = crop_node.inputs[0].clone();
    let input_fact = model.outlet_fact(input_outlet).unwrap();
    let final_len = down_node.outputs[0].fact.shape.dim(down_op.axis);
    let new_down = Downsample::new(down_op.axis, down_op.stride, modulo);
    let downed = new_down.transform_shape(&input_fact)?;
    let midway_len = downed.shape.dim(down_op.axis);
    patch.chain(&*down_node.name, new_down, tvec!(downed))?;
    let mut new_prunes = crop_op.prune.clone();
    new_prunes[down_op.axis].0 = left;
    new_prunes[down_op.axis].1 =
        (midway_len.to_dim() - final_len.to_dim() - left).to_integer()? as usize;
    let new_crop = patch.chain(
        &*crop_node.name,
        ops::array::Crop::new(new_prunes),
        tvec!(down_node.outputs[0].fact.clone()),
    )?;
    patch.shunt_outside(OutletId::new(down_node.id, 0), OutletId::new(new_crop, 0))?;
    return Ok(Some(patch));
}

pub fn pull_downsample_over_adddims(
    model: &TypedModel,
    add_node: &TypedNode,
    add_op: &ops::array::AddDims,
    down_node: &TypedNode,
    down_op: &Downsample,
) -> TractResult<Option<TypedModelPatch>> {
    let mut patch = TypedModelPatch::default();
    patch.tap_model(model, add_node.inputs[0])?;
    let input_outlet = add_node.inputs[0].clone();
    let input_fact = model.outlet_fact(input_outlet).unwrap();
    let mut new_down = down_op.clone();
    new_down.axis -= add_op.axes.iter().filter(|&ax| *ax <= down_op.axis).count();
    let downed = new_down.transform_shape(&input_fact)?;
    patch.chain(&*down_node.name, new_down, tvec!(downed))?;
    let new_node =
        patch.chain(&*add_node.name, add_op.clone(), tvec!(down_node.outputs[0].fact.clone()))?;
    patch.shunt_outside(OutletId::new(down_node.id, 0), OutletId::new(new_node, 0))?;
    return Ok(Some(patch));
}

pub fn pull_downsample_over_rmdims(
    model: &TypedModel,
    rm_node: &TypedNode,
    rm_op: &ops::array::RmDims,
    down_node: &TypedNode,
    down_op: &Downsample,
) -> TractResult<Option<TypedModelPatch>> {
    let mut patch = TypedModelPatch::default();
    patch.tap_model(model, rm_node.inputs[0])?;
    let input_outlet = rm_node.inputs[0].clone();
    let input_fact = model.outlet_fact(input_outlet).unwrap();
    let mut new_down = down_op.clone();
    new_down.axis += rm_op.axes.iter().filter(|&ax| *ax <= down_op.axis).count();
    let downed = new_down.transform_shape(&input_fact)?;
    patch.chain(&*down_node.name, new_down, tvec!(downed))?;
    let new_rm =
        patch.chain(&*rm_node.name, rm_op.clone(), tvec!(down_node.outputs[0].fact.clone()))?;
    patch.shunt_outside(OutletId::new(down_node.id, 0), OutletId::new(new_rm, 0))?;
    return Ok(Some(patch));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops;
    use proptest::prelude::*;
    use proptest::test_runner::TestCaseResult;

    fn crop_then_down_strat() -> BoxedStrategy<(usize, usize, usize, usize, usize)> {
        (1usize..5, 1usize..5)
            .prop_flat_map(|(cropped, stride)| {
                (Just(cropped), 0..=cropped, Just(stride), (cropped + 15)..=(cropped + 15))
            })
            .prop_flat_map(|(cropped, left, stride, len)| {
                (Just(len), Just(left), Just(cropped - left), Just(stride), 0..stride)
            })
            .boxed()
    }

    fn crop_then_down(
        len: usize,
        left: usize,
        right: usize,
        stride: usize,
        modulo: usize,
    ) -> TestCaseResult {
        let model = {
            let mut model = InferenceModel::default();
            model.add_source("input", TensorFact::dt_shape(i32::datum_type(), vec![len]))?;
            model.chain_default("crop", ops::array::Crop::new(vec![(left, right)]))?;
            model.chain_default("down", Downsample::new(0, stride, modulo))?;
            model.auto_outputs()?;
            model
        };
        prop_assert!(model.node(model.output_outlets().unwrap()[0].node).op_is::<Downsample>());
        let typed = model.into_typed()?;
        let input = tensor1(&(0i32..len as _).collect::<Vec<_>>());
        let expected = SimplePlan::new(&typed)?.run(tvec!(input.clone()))?;

        let typed = typed.declutter()?;
        prop_assert!(!typed.node(typed.output_outlets().unwrap()[0].node).op_is::<Downsample>());
        let found = SimplePlan::new(&typed)?.run(tvec!(input))?;
        prop_assert_eq!(found, expected);
        Ok(())
    }

    proptest! {
        #[test]
        fn crop_then_down_prop((len, left, right, stride, modulo) in crop_then_down_strat()) {
            crop_then_down(len, left, right, stride, modulo).unwrap()
        }
    }

    #[test]
    fn crop_then_down_1() {
        crop_then_down(1, 0, 0, 2, 0).unwrap()
    }

    #[test]
    fn crop_then_down_2() {
        crop_then_down(2, 0, 1, 2, 0).unwrap()
    }

    #[test]
    fn crop_then_down_3() {
        crop_then_down(0, 0, 0, 2, 1).unwrap()
    }

    #[test]
    fn crop_then_down_4() {
        crop_then_down(1, 0, 1, 2, 1).unwrap()
    }

    #[test]
    fn crop_then_down_5() {
        crop_then_down(16, 0, 1, 2, 1).unwrap()
    }
}
