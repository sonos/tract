use tract_nnef::internal::*;
use tract_nnef::prelude::tract_itertools::Itertools;
use tract_nnef::tract_core::ops::array::TypedConcat;
use tract_nnef::tract_core::ops::source::TypedSource;
use tract_nnef::tract_core::ops::OpStateFreeze;

use crate::rule_ensure;

use super::next_node;

#[derive(Debug, Clone)]
pub struct DynKeyValueCacheState {
    io_name: [String; 2],
    input_facts: [TypedFact; 2],
    kv_cache: Option<Tensor>,
}

impl DynKeyValueCacheState {
    /// # Safety
    ///
    /// Input and Ouput tensors shape must be compatible with this operator, otherwise it could lead
    /// to an undefined behaviour.
    pub unsafe fn apply_delay_unchecked(
        &mut self,
        op: &DynKeyValueCache,
        input: &Tensor,
        output: &mut Tensor,
    ) {
        let old_cache = self.kv_cache.as_mut().unwrap();
        let old_cache_len = old_cache.shape()[op.axis];

        unsafe { 
            output.assign_slice_unchecked(..old_cache_len, old_cache, ..old_cache_len, op.axis);
            output.assign_slice_unchecked(old_cache_len.., input, .., op.axis);
        }
    }
}

impl DynKeyValueCacheState {
    pub fn resolve_symbols(
        state: &mut SessionState,
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
            state.resolved_symbols.set(sym, shape[ax] as i64);
        } else {
            state.resolved_symbols.set(sym, 0);
        }

        if state.scenario.is_none() {
            state.scenario = sym.scope().unwrap().guess_scenario(&state.resolved_symbols)?;
        }
        Ok(())
    }
}

impl OpState for DynKeyValueCacheState {
    fn load_from(
        &mut self,
        state: &mut SessionState,
        states: &mut HashMap<String, Tensor>,
    ) -> TractResult<()> {
        if let Some(kv_cache) = states.remove(&self.io_name[0]) {
            // KV Cache fact is always at index 0
            Self::resolve_symbols(state, self.input_facts[0].clone(), Some(kv_cache.shape()))?;
            self.kv_cache = Some(kv_cache.into_tensor());

            Ok(())
        } else {
            bail!("KV cache input {} not found in given states", self.io_name[0])
        }
    }

    fn save_to(&mut self, states: &mut HashMap<String, Tensor>) -> TractResult<()> {
        if let Some(kv_cache) = &self.kv_cache {
            states.insert(self.io_name[1].clone(), kv_cache.clone());
            Ok(())
        } else {
            bail!("KV cache {} was never initialized", self.io_name[1])
        }
    }

    fn init_tensor_fact(&self) -> Option<TypedFact> {
        Some(self.input_facts[0].clone())       
    }

    fn resolve_symbols(&mut self, state: &mut SessionState) -> TractResult<()> {
        let shape = self.kv_cache.as_ref().map(|kv_cache| kv_cache.shape());
        Self::resolve_symbols(state, self.input_facts[0].clone(), shape)
    }

    fn eval(
        &mut self,
        _state: &mut SessionState,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let op =
            op.downcast_ref::<DynKeyValueCache>().ok_or_else(|| format_err!("Wrong Op type"))?;

        let input_num_tokens = input.shape()[op.axis];

        // build output
        unsafe {
            let output = if let Some(curr) = self.kv_cache.as_ref() {
                let mut shape = curr.shape().to_owned();
                shape[op.axis] += input_num_tokens;
                let mut output = Tensor::uninitialized_dt(input.datum_type(), &shape)?;
                self.apply_delay_unchecked(op, &input, &mut output);
                output
            } else {
                input.into_tensor()
            };
            self.kv_cache = Some(output.clone());

            Ok(tvec!(output.into()))
        }
    }
}

#[derive(Clone, Debug)]
pub struct DynKeyValueCache {
    pub io_name: [String; 2],
    pub axis: usize,
    pub input_facts: [TypedFact; 2],
}

impl Op for DynKeyValueCache {
    fn name(&self) -> Cow<str> {
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
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(DynKeyValueCacheState {
            io_name: self.io_name.clone(),
            input_facts: self.input_facts.clone(),
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
            self.input_facts[0].shape.dims()[self.axis].clone()
                + self.input_facts[0].shape.dims()[self.axis].clone(),
        );
        Ok(tvec!(fact))
    }

    as_op!();
}

impl OpStateFreeze for DynKeyValueCacheState {
    fn freeze(&self) -> Box<dyn FrozenOpState> {
        Box::new(self.clone())
    }
}

impl FrozenOpState for DynKeyValueCacheState {
    fn unfreeze(&self) -> Box<dyn OpState> {
        Box::new(self.clone())
    }
}

/// Search pattern => Input -> Concat -> Output
/// Return type is for using rule-ensure macro
pub fn replace_kv_cache(target: &mut TypedModel, source_node_id: usize) -> TractResult<Option<()>> {
    assert!(target.node(source_node_id).op_is::<TypedSource>());
    let (concat_node_id, non_source_input_id, axis, input_facts) = {
        let concat_node = if let Some(n_node) = next_node(target, target.node(source_node_id)) {
            n_node
        } else {
            return Ok(None);
        };

        // Check KV Cache Pattern
        rule_ensure!(
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
        rule_ensure!(
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
            io_name: [name.clone(), concat_node.name.to_string()],
            axis,
            input_facts,
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

        let symb_shape = make_shape("P");
        let second_shape = make_shape("S");

        let op = DynKeyValueCache {
            io_name: [op_name.clone(), op_name.clone()],
            input_facts: [
                TypedFact::dt_shape(F::datum_type(), symb_shape),
                TypedFact::dt_shape(F::datum_type(), second_shape),
            ],
            axis,
        };

        let mut session_state = SessionState::default();
        let mut state = op.state(&mut session_state, 0)?.unwrap();

        let mut inputs = tvec![];

        // Init state with first shape
        let shape = &input_shapes[0];
        let len = shape.iter().product::<usize>();
        let input = Tensor::from_shape(shape, &(0..len).map(|f| f.as_()).collect::<Vec<F>>())?;
        inputs.push(input.clone().into_tvalue());

        let mut hashmap = HashMap::new();
        hashmap.insert(op_name.clone(), input);

        state.load_from(&mut session_state, &mut hashmap)?;

        for shape in input_shapes {
            let len = shape.iter().product::<usize>();
            let input = Tensor::from_shape(&shape, &(0..len).map(|f| f.as_()).collect::<Vec<F>>())?;
            inputs.push(input.clone().into_tvalue());
            state.eval(&mut session_state, &op, tvec!(input.clone().into()))?[0]
                .clone()
                .into_tensor();
        }
        state.save_to(&mut hashmap)?;
        let output = hashmap.get(&op_name).unwrap();

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
}
