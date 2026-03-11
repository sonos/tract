use std::str::FromStr;

use tract_nnef::internal::*;
use tract_nnef::prelude::tract_itertools::Itertools;
use tract_nnef::ser::{datum_type, tdims};
use tract_nnef::tract_core::ops::OpStateFreeze;
use tract_nnef::tract_core::ops::array::TypedConcat;
use tract_nnef::tract_core::ops::source::TypedSource;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_dyn_kv_cache);
    registry.register_primitive(
        "tract_transformers_dyn_kv_cache",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::String.named("name"),
            TypeName::Integer.named("axis"),
            TypeName::String.named("datum_type"),
            TypeName::Integer.array().named("past_sequence_shape"),
            TypeName::Integer.array().named("input_sequence_shape"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_dyn_kv_cache,
    );
}

fn ser_dyn_kv_cache(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &DynKeyValueCache,
) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    Ok(Some(invocation(
        "tract_transformers_dyn_kv_cache",
        &[input],
        &[
            ("name", string(&op.name)),
            ("axis", numeric(op.axis)),
            ("datum_type", datum_type(op.past_sequence_fact.datum_type)),
            ("past_sequence_shape", tdims(op.past_sequence_fact.shape.dims())),
            ("input_sequence_shape", tdims(op.input_sequence_fact.shape.dims())),
        ],
    )))
}

fn de_dyn_kv_cache(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let name: String = invocation.named_arg_as(builder, "name")?;
    let axis: usize = invocation.named_arg_as(builder, "axis")?;
    let dt = DatumType::from_str(&invocation.named_arg_as::<String>(builder, "datum_type")?)?;
    let past_sequence_shape: TVec<TDim> = builder
        .allowing_new_symbols(|builder| invocation.named_arg_as(builder, "past_sequence_shape"))?;
    let input_sequence_shape: TVec<TDim> = builder
        .allowing_new_symbols(|builder| invocation.named_arg_as(builder, "input_sequence_shape"))?;
    builder.wire(
        DynKeyValueCache {
            name,
            axis,
            past_sequence_fact: dt.fact(&*past_sequence_shape),
            input_sequence_fact: dt.fact(&*input_sequence_shape),
        },
        &[input],
    )
}

#[derive(Debug, Clone)]
pub struct DynKeyValueCacheState {
    name: String,
    axis: usize,
    past_sequence_fact: TypedFact,
    kv_cache: Option<TValue>,
}

impl DynKeyValueCacheState {
    pub fn resolve_symbols(
        state: &mut TurnState,
        fact: TypedFact,
        concrete_shape: Option<&[usize]>,
    ) -> TractResult<()> {
        let unresolved = fact
            .shape
            .iter()
            .enumerate()
            .filter_map(|(ax, symb)| match symb {
                TDim::Sym(s) if state.resolved_symbols.get(s).is_none() => Some((ax, s)),
                _ => None,
            })
            .collect_vec();

        if unresolved.is_empty() {
            return Ok(());
        }

        ensure!(unresolved.len() == 1);
        let (ax, sym) = unresolved[0];
        if let Some(shape) = concrete_shape {
            ensure!(ax < shape.len());
            state.resolved_symbols.set(sym, shape[ax] as i64);
        } else {
            state.resolved_symbols.set(sym, 0);
        }

        if state.scenario.is_none() {
            state.scenario = sym.scope().unwrap().guess_scenario(&state.resolved_symbols)?;
        }
        Ok(())
    }

    pub fn truncate(&mut self, len: usize) -> TractResult<()> {
        if let Some(t) = self.kv_cache.as_mut() {
            *t = t.slice(self.axis, 0, len)?.into_tvalue();
        } else {
            bail!("Can not truncate a zero-len kv-cache value");
        }
        Ok(())
    }
}

impl OpState for DynKeyValueCacheState {
    fn load_from(
        &mut self,
        state: &mut TurnState,
        states: &mut dyn Iterator<Item = tract_nnef::prelude::TValue>,
    ) -> TractResult<()> {
        // KV Cache fact is always at index 0
        let kv_cache_init = states.next().context("Not enough state initializers")?;
        Self::resolve_symbols(state, self.past_sequence_fact.clone(), Some(kv_cache_init.shape()))?;
        self.kv_cache = Some(kv_cache_init.clone());

        Ok(())
    }

    fn save_to(&self, states: &mut Vec<TValue>) -> TractResult<()> {
        if let Some(kv_cache) = &self.kv_cache {
            states.push(kv_cache.clone());
            Ok(())
        } else {
            bail!("KV cache {} was never initialized", self.name)
        }
    }

    fn init_tensor_fact(&self) -> Option<(String, TypedFact)> {
        Some((self.name.clone(), self.past_sequence_fact.clone()))
    }

    fn resolve_symbols(&mut self, state: &mut TurnState) -> TractResult<()> {
        let shape = self.kv_cache.as_ref().map(|kv_cache| kv_cache.shape());
        Self::resolve_symbols(state, self.past_sequence_fact.clone(), shape)
    }

    fn eval(
        &mut self,
        _state: &mut TurnState,
        _op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        // build output
        let output = if let Some(curr) = self.kv_cache.take() {
            TypedConcat { axis: self.axis }.eval(tvec![curr, input])?.remove(0)
        } else {
            input
        };
        self.kv_cache = Some(output.clone());

        Ok(tvec!(output))
    }
}

#[derive(Clone, Debug)]
pub struct DynKeyValueCache {
    pub name: String,
    pub axis: usize,
    pub past_sequence_fact: TypedFact,
    pub input_sequence_fact: TypedFact,
}

impl Op for DynKeyValueCache {
    fn name(&self) -> StaticName {
        "DynamicKeyValueCache".to_string().into()
    }

    op_as_typed_op!();
}

impl EvalOp for DynKeyValueCache {
    fn is_stateless(&self) -> bool {
        false
    }

    fn state(
        &self,
        _session: &TurnState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(DynKeyValueCacheState {
            name: self.name.clone(),
            axis: self.axis,
            past_sequence_fact: self.past_sequence_fact.clone(),
            kv_cache: None,
        })))
    }
}

impl TypedOp for DynKeyValueCache {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 1);
        let input = inputs[0];
        let mut fact = input.without_value();

        fact.shape.set(
            self.axis,
            self.past_sequence_fact.shape.dims()[self.axis].clone()
                + self.input_sequence_fact.shape.dims()[self.axis].clone(),
        );
        Ok(tvec!(fact))
    }

    fn cost(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let token_volume = self
            .past_sequence_fact
            .shape
            .iter()
            .enumerate()
            .filter(|(axis, _d)| *axis != self.axis)
            .map(|(_axis, d)| d)
            .product::<TDim>();
        Ok(tvec!((Cost::Custom(false, "KVCacheValuesPerToken".to_string()), token_volume)))
    }

    as_op!();
}

#[derive(Debug, Clone)]
pub struct FrozenDynKeyValueCacheState {
    name: String,
    axis: usize,
    past_sequence_fact: TypedFact,
    kv_cache: Option<Tensor>,
}

impl OpStateFreeze for DynKeyValueCacheState {
    fn freeze(&self) -> Box<dyn FrozenOpState> {
        Box::new(FrozenDynKeyValueCacheState {
            name: self.name.clone(),
            axis: self.axis,
            past_sequence_fact: self.past_sequence_fact.clone(),
            kv_cache: self.kv_cache.clone().map(|t| t.into_tensor()),
        })
    }
}

impl FrozenOpState for FrozenDynKeyValueCacheState {
    fn unfreeze(&self) -> Box<dyn OpState> {
        Box::new(DynKeyValueCacheState {
            axis: self.axis,
            name: self.name.clone(),
            past_sequence_fact: self.past_sequence_fact.clone(),
            kv_cache: self.kv_cache.clone().map(|t| t.into_tvalue()),
        })
    }
}

/// Reverse of `replace_kv_cache`: replaces a DynKeyValueCache node with Source + Concat,
/// restoring KV cache state as explicit model inputs and outputs.
pub fn unfold_kv_cache(target: &mut TypedModel, kv_node_id: usize) -> TractResult<()> {
    let node = target.node(kv_node_id);
    let op = node.op_as::<DynKeyValueCache>().context("Not a DynKeyValueCache node")?;
    let name = op.name.clone();
    let axis = op.axis;
    let past_fact = op.past_sequence_fact.clone();
    let input_fact = op.input_sequence_fact.clone();
    let existing_input = node.inputs[0];

    // Add a new Source node for the past KV cache
    let source_outlet = target.add_source(&name, past_fact)?;

    // Compute output fact for the Concat
    let mut output_fact = input_fact.clone();
    output_fact.shape.set(
        axis,
        target.outlet_fact(source_outlet)?.shape.dims()[axis].clone()
            + input_fact.shape.dims()[axis].clone(),
    );

    // Replace DynKeyValueCache op with TypedConcat
    let kv_node = target.node_mut(kv_node_id);
    kv_node.name = format!("{name}_concat");
    kv_node.op = Box::new(TypedConcat { axis });
    kv_node.outputs[0].fact = output_fact;

    // Rewire: Concat takes [source, existing_input] as inputs
    // Currently the node has [existing_input] at slot 0
    // We need [source_outlet, existing_input] at slots [0, 1]
    kv_node.inputs = vec![source_outlet, existing_input];

    // Update successor info on the source node
    target.nodes[source_outlet.node].outputs[source_outlet.slot]
        .successors
        .push(InletId::new(kv_node_id, 0));

    // Update the existing input's successor slot from 0 to 1
    target.nodes[existing_input.node].outputs[existing_input.slot].successors.iter_mut().for_each(
        |succ| {
            if succ.node == kv_node_id && succ.slot == 0 {
                succ.slot = 1;
            }
        },
    );

    // Add the Concat output to model outputs
    target.outputs.push(OutletId::new(kv_node_id, 0));

    Ok(())
}

/// Search pattern => Input -> Concat -> Output
/// Return type is for using rule-ensure macro
pub fn replace_kv_cache(target: &mut TypedModel, source_node_id: usize) -> TractResult<Option<()>> {
    assert!(target.node(source_node_id).op_is::<TypedSource>());
    let (concat_node_id, non_source_input_id, axis, input_facts) = {
        rule_if_some!(concat_node = target.next_node(target.node(source_node_id)));

        // Check KV Cache Pattern
        rule_if!(
            concat_node.op_is::<TypedConcat>()
                && concat_node.inputs.len() == 2
                && concat_node.outputs.len() == 1
                && target.outputs.contains(&concat_node.id.into())
        );

        let concat_in_facts = target.node_input_facts(concat_node.id)?;

        // Check on shapes
        let concat_in_shapes = [concat_in_facts[0].shape.dims(), concat_in_facts[1].shape.dims()];
        let rank = concat_in_shapes[0].len();
        let axes = (0..rank)
            .filter(|ax| concat_in_shapes[0][*ax] != concat_in_shapes[1][*ax])
            .collect_vec();
        ensure!(axes.len() == 1);

        let axis = axes[0];
        rule_if!(
            matches!(concat_in_shapes[0][axis], TDim::Sym(_))
                && matches!(concat_in_shapes[1][axis], TDim::Sym(_))
        );
        let mut facts = [concat_in_facts[0].clone(), concat_in_facts[1].clone()];
        if concat_node.inputs[0].node == source_node_id {
            (concat_node.id, concat_node.inputs[1].node, axis, facts)
        } else if concat_node.inputs[1].node == source_node_id {
            facts.swap(0, 1);
            (concat_node.id, concat_node.inputs[0].node, axis, facts)
        } else {
            return Ok(None);
        }
    };

    {
        // Replace Concat by KVCache
        let name = target.node_names().collect_vec()[source_node_id].to_string();
        let concat_node = target.node_mut(concat_node_id);
        concat_node.op = Box::new(DynKeyValueCache {
            name: name.clone(),
            axis,
            past_sequence_fact: input_facts[0].clone(),
            input_sequence_fact: input_facts[1].clone(),
        });
        concat_node.name = name;
        concat_node.inputs.retain(|input| input != &source_node_id.into());
    }

    {
        // Replace Source by Dummy Op for it to be cleaned later
        let dummy_op = target.create_dummy();
        let source_node = target.node_mut(source_node_id);
        source_node.outputs[0].successors.clear();
        source_node.op = dummy_op;
    }
    {
        // Non-source input is usually the second input of Concat. Rewire it to the only input of the new KVCache Op
        let non_source_input = target.node_mut(non_source_input_id);
        non_source_input.outputs.iter_mut().for_each(|output| {
            output.successors.iter_mut().for_each(|succ| {
                if succ.node == concat_node_id {
                    succ.slot = 0
                }
            })
        });
    }

    // Clean model I/Os
    target.outputs.retain(|output| output.node != concat_node_id);
    target.inputs.retain(|input| input.node != source_node_id);
    target.outlet_labels.remove(&concat_node_id.into());
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tract_num_traits::AsPrimitive;
    use tract_num_traits::Zero;

    fn run_test_case<F: Datum + Zero + Copy>(
        input_shapes: &[Vec<usize>],
        axis: usize,
    ) -> TractResult<()>
    where
        usize: AsPrimitive<F>,
    {
        let first_shape = &input_shapes[0];
        ensure!(input_shapes.iter().all(|shape| (shape.len() == first_shape.len())
            && (shape[..axis] == first_shape[..axis])
            && (if axis != (shape.len() - 1) {
                shape[(axis + 1)..] == first_shape[(axis + 1)..]
            } else {
                true
            })));

        let op_name = "test".to_string();
        let dummy_model = TypedModel::default();

        let make_shape =
            |sym: &str| {
                input_shapes[0]
                    .iter()
                    .enumerate()
                    .map(|(i, &dim)| {
                        if i == axis {
                            TDim::Sym(dummy_model.sym(sym))
                        } else {
                            TDim::Val(dim as _)
                        }
                    })
                    .collect::<TVec<TDim>>()
            };

        let past_shape = make_shape("P");
        let input_shape = make_shape("S");

        let op = DynKeyValueCache {
            name: op_name.clone(),
            past_sequence_fact: TypedFact::dt_shape(F::datum_type(), past_shape),
            input_sequence_fact: TypedFact::dt_shape(F::datum_type(), input_shape),
            axis,
        };

        let mut session_state = TurnState::default();
        let mut state = op.state(&mut session_state, 0)?.unwrap();

        let mut inputs = tvec![];

        // Init state with first shape
        let shape = &input_shapes[0];
        let len = shape.iter().product::<usize>();
        let input = Tensor::from_shape(shape, &(0..len).map(|f| f.as_()).collect::<Vec<F>>())?;
        inputs.push(input.clone().into_tvalue());

        let mut state_initializers = vec![input.into()].into_iter();

        state.load_from(&mut session_state, &mut state_initializers)?;

        for shape in input_shapes {
            let len = shape.iter().product::<usize>();
            let input = Tensor::from_shape(&shape, &(0..len).map(|f| f.as_()).collect::<Vec<F>>())?;
            inputs.push(input.clone().into_tvalue());
            state.eval(&mut session_state, &op, tvec!(input.clone().into()))?[0]
                .clone()
                .into_tensor();
        }

        let mut curr_states = vec![];
        state.save_to(&mut curr_states)?;
        let output = curr_states.remove(0);

        let reference = &TypedConcat { axis }.eval(inputs)?[0];
        output.close_enough(&reference.clone().into_tensor(), Approximation::Close)?;
        Ok(())
    }

    #[test]
    fn test_dyn_kv_cache() -> TractResult<()> {
        run_test_case::<f32>(&[vec![2, 2]], 0)?;
        run_test_case::<f32>(&[vec![2, 2], vec![4, 2]], 0)?;
        run_test_case::<f32>(&[vec![2, 2], vec![2, 1], vec![2, 3]], 1)?;
        Ok(())
    }

    #[test]
    fn test_unfold_kv_cache() -> TractResult<()> {
        // Build a model with DynKeyValueCache
        let mut model = TypedModel::default();
        let s = model.sym("S");
        let p = model.sym("P");

        let input_shape: TVec<TDim> = tvec![1.to_dim(), s.into(), 64.to_dim()];
        let past_shape: TVec<TDim> = tvec![1.to_dim(), p.into(), 64.to_dim()];

        let input = model.add_source("input", f32::fact(&input_shape))?;
        let op = DynKeyValueCache {
            name: "kv_cache_0".to_string(),
            axis: 1,
            past_sequence_fact: f32::fact(&past_shape),
            input_sequence_fact: f32::fact(&input_shape),
        };
        let out = model.wire_node("kv_cache", op, &[input])?;
        model.set_output_outlets(&out)?;

        // Model should have 1 input (input), 1 output (kv_cache)
        assert_eq!(model.inputs.len(), 1);
        assert_eq!(model.outputs.len(), 1);
        assert!(model.node(1).op_is::<DynKeyValueCache>());

        // Unfold
        unfold_kv_cache(&mut model, 1)?;

        // After unfold: 2 inputs (input + kv_cache_0 source), 2 outputs (original + concat)
        assert_eq!(model.inputs.len(), 2);
        assert_eq!(model.outputs.len(), 2);

        // The KV cache node should now be a Concat
        assert!(model.node(1).op_is::<TypedConcat>());
        let concat = model.node(1).op_as::<TypedConcat>().unwrap();
        assert_eq!(concat.axis, 1);

        // The new source node should exist
        let source_node_id = model.inputs[1].node;
        assert!(model.node(source_node_id).op_is::<TypedSource>());
        assert_eq!(model.node(source_node_id).name, "kv_cache_0");

        // Concat should have 2 inputs: [source, input]
        assert_eq!(model.node(1).inputs.len(), 2);
        assert_eq!(model.node(1).inputs[0].node, source_node_id);
        assert_eq!(model.node(1).inputs[1].node, 0); // original input

        Ok(())
    }

    #[test]
    fn test_fold_unfold_round_trip() -> TractResult<()> {
        use crate::rewriter::KeyValueCacheTransform;
        use tract_nnef::tract_core::transform::ModelTransform;

        // Build a model with Source + Concat (the pre-fold pattern)
        let mut model = TypedModel::default();
        let s = model.sym("S");
        let p = model.sym("P");

        let input_shape: TVec<TDim> = tvec![1.to_dim(), s.into(), 64.to_dim()];
        let past_shape: TVec<TDim> = tvec![1.to_dim(), p.into(), 64.to_dim()];

        let past = model.add_source("kv_past", f32::fact(&past_shape))?;
        let input = model.add_source("input", f32::fact(&input_shape))?;
        let concat = model.wire_node("concat", TypedConcat { axis: 1 }, &[past, input])?;
        model.set_output_outlets(&concat)?;

        let orig_input_count = model.inputs.len();
        let orig_output_count = model.outputs.len();

        // Fold: Source + Concat -> DynKeyValueCache
        KeyValueCacheTransform.transform(&mut model)?;
        assert_eq!(model.inputs.len(), orig_input_count - 1); // past source removed
        assert_eq!(model.outputs.len(), orig_output_count - 1); // concat output removed

        // Find the DynKeyValueCache node
        let kv_node_id = model.nodes().iter().find(|n| n.op_is::<DynKeyValueCache>()).unwrap().id;

        // Unfold: DynKeyValueCache -> Source + Concat
        unfold_kv_cache(&mut model, kv_node_id)?;

        // Should be back to original structure
        assert_eq!(model.inputs.len(), orig_input_count);
        assert_eq!(model.outputs.len(), orig_output_count);

        // Verify it's a Concat again
        let concat_node = model.nodes().iter().find(|n| n.op_is::<TypedConcat>()).unwrap();
        assert_eq!(concat_node.op_as::<TypedConcat>().unwrap().axis, 1);
        assert_eq!(concat_node.inputs.len(), 2);

        Ok(())
    }

    #[test]
    fn test_dyn_kv_cache_nnef_round_trip() -> TractResult<()> {
        use crate::WithTractTransformers;

        let mut model = TypedModel::default();
        let s = model.sym("S");
        let p = model.sym("P");

        let input_shape: TVec<TDim> = tvec![1.to_dim(), s.into(), 64.to_dim()];
        let past_shape: TVec<TDim> = tvec![1.to_dim(), p.into(), 64.to_dim()];

        let input = model.add_source("input", f32::fact(&input_shape))?;
        let op = DynKeyValueCache {
            name: "kv_cache_0".to_string(),
            axis: 1,
            past_sequence_fact: f32::fact(&past_shape),
            input_sequence_fact: f32::fact(&input_shape),
        };
        let out = model.wire_node("kv_cache", op, &[input])?;
        model.set_output_outlets(&out)?;

        let nnef = tract_nnef::nnef().with_tract_transformers();
        let mut buffer = vec![];
        nnef.write_to_tar(&model, &mut buffer)?;
        let reloaded = nnef.model_for_read(&mut &*buffer)?;

        assert_eq!(reloaded.nodes().len(), model.nodes().len());
        let reloaded_kv = reloaded.node(1);
        let reloaded_op = reloaded_kv.op_as::<DynKeyValueCache>().unwrap();
        assert_eq!(reloaded_op.name, "kv_cache_0");
        assert_eq!(reloaded_op.axis, 1);
        assert_eq!(reloaded_op.past_sequence_fact.datum_type, DatumType::F32);
        assert_eq!(reloaded_op.past_sequence_fact.shape.rank(), 3);
        assert_eq!(reloaded_op.input_sequence_fact.datum_type, DatumType::F32);
        assert_eq!(reloaded_op.input_sequence_fact.shape.rank(), 3);
        Ok(())
    }
}
