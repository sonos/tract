//! Loading a model without reading its weights, then reading only what survives pruning.

use tract_core::internal::*;
use tract_core::ops::konst::{Const, LazyConst, materialize_lazy_consts};
use tract_nnef::internal::*;

/// Non-uniform, so declutter cannot fold the weight away and delete the node with it.
fn ramp(base: f32) -> Tensor {
    tensor1(&(0..64).map(|i| base + i as f32).collect::<Vec<f32>>())
}

/// `x -> add(w0) -> add(w1) -> y`, with two weights that survive decluttering.
fn model_with_two_weights() -> TractResult<TypedModel> {
    let mut m = TypedModel::default();
    let x = m.add_source("x", f32::fact([64]))?;
    let w0 = m.add_const("w0", ramp(1.0))?;
    let a = m.wire_node("a", tract_core::ops::math::add(), &[x, w0])?[0];
    let w1 = m.add_const("w1", ramp(100.0))?;
    let y = m.wire_node("y", tract_core::ops::math::add(), &[a, w1])?[0];
    m.select_output_outlets(&[y])?;
    m.into_decluttered()
}

fn write_dir(model: &TypedModel) -> TractResult<temp_dir::TempDir> {
    let dir = temp_dir::TempDir::new()?;
    tract_nnef::nnef().write_to_dir(model, dir.path().join("model"))?;
    Ok(dir)
}

fn lazy_nodes(model: &TypedModel) -> usize {
    model.nodes().iter().filter(|n| n.op_is::<LazyConst>()).count()
}

#[test]
fn a_lazy_load_types_the_graph_without_reading_weights() -> TractResult<()> {
    let model = model_with_two_weights()?;
    let dir = write_dir(&model)?;
    let lazy = tract_nnef::nnef().model_for_dir_lazy(dir.path().join("model"))?;

    assert_eq!(lazy_nodes(&lazy), 2, "expected both weights to be lazy");
    for node in lazy.nodes() {
        if node.op_is::<LazyConst>() {
            let fact = &node.outputs[0].fact;
            assert!(fact.konst.is_none(), "{} has a value before materializing", node.name);
            assert_eq!(fact.datum_type, f32::datum_type());
            assert_eq!(fact.shape.as_concrete(), Some(&[64usize][..]));
        }
    }
    Ok(())
}

#[test]
fn materializing_reproduces_an_eager_load() -> TractResult<()> {
    let model = model_with_two_weights()?;
    let dir = write_dir(&model)?;
    let path = dir.path().join("model");

    let eager = tract_nnef::nnef().model_for_path(&path)?.into_decluttered()?;
    let mut lazy = tract_nnef::nnef().model_for_dir_lazy(&path)?;
    assert_eq!(materialize_lazy_consts(&mut lazy)?, 2);
    assert_eq!(lazy_nodes(&lazy), 0);
    let lazy = lazy.into_decluttered()?;

    assert_eq!(lazy.nodes().len(), eager.nodes().len());
    let x = tensor1(&[1.0f32; 64]);
    let want = eager.into_runnable()?.run(tvec!(x.clone().into_tvalue()))?;
    let got = lazy.into_runnable()?.run(tvec!(x.into_tvalue()))?;
    assert_eq!(got[0], want[0]);
    Ok(())
}

/// The point of the exercise: prune first, and the discarded weights are never read.
#[test]
fn pruning_before_materializing_skips_the_discarded_weights() -> TractResult<()> {
    let model = model_with_two_weights()?;
    let dir = write_dir(&model)?;
    let mut lazy = tract_nnef::nnef().model_for_dir_lazy(dir.path().join("model"))?;
    assert_eq!(lazy_nodes(&lazy), 2);

    // Keep the cone of the first weight only; the second is dropped with the rest. The
    // serializer inserts casts and renames, so find the consumer rather than name it.
    let first_weight = lazy.nodes().iter().find(|n| n.op_is::<LazyConst>()).unwrap().id;
    let consumer = lazy.node(first_weight).outputs[0].successors[0].node;
    let adder = lazy.node(consumer).outputs[0].successors[0].node;
    lazy.select_output_outlets(&[OutletId::new(adder, 0)])?;
    lazy.compact()?;
    assert_eq!(lazy_nodes(&lazy), 1, "the discarded weight should be gone");

    assert_eq!(materialize_lazy_consts(&mut lazy)?, 1, "only the surviving weight is read");
    assert_eq!(lazy.nodes().iter().filter(|n| n.op_is::<Const>()).count(), 1);
    Ok(())
}

/// A lazy constant carries no value, so it must not be evaluated: `PropConst` and
/// `compute_const_facts` both see a node whose inputs are vacuously all-constant.
#[test]
fn declutter_does_not_evaluate_a_lazy_constant() -> TractResult<()> {
    let model = model_with_two_weights()?;
    let dir = write_dir(&model)?;
    let lazy = tract_nnef::nnef().model_for_dir_lazy(dir.path().join("model"))?;
    let decluttered = lazy.into_decluttered()?;
    assert_eq!(lazy_nodes(&decluttered), 2, "declutter consumed the lazy constants");
    Ok(())
}
