use tract_nnef::internal::*;
use tract_nnef::prelude::tract_itertools::Itertools;
use tract_nnef::tract_core::ops::array::TypedConcat;
use tract_nnef::tract_core::ops::source::TypedSource;
use tract_nnef::tract_core::ops::OpStateFreeze;

use crate::rule_ensure;

use super::next_node;

#[derive(Debug, Clone)]
pub struct DynKeyValueCacheState {
    pub stored_kv_cache: Option<Tensor>,
}

impl DynKeyValueCacheState {
    pub unsafe fn apply_delay_unchecked(
        &mut self,
        op: &DynKeyValueCache,
        input: &Tensor,
        output: &mut Tensor,
    ) {
        let old_cache = self.stored_kv_cache.as_mut().unwrap();
        let old_cache_len = old_cache.shape()[op.axis];
        
        output.assign_slice_unchecked(..old_cache_len, old_cache, ..old_cache_len, op.axis);
        output.assign_slice_unchecked(old_cache_len.., input, .., op.axis);
    }
}

impl OpState for DynKeyValueCacheState {
    fn eval(
        &mut self,
        _state: &mut SessionState,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let op = op.downcast_ref::<DynKeyValueCache>().ok_or_else(|| format_err!("Wrong Op type"))?;

        let input_num_tokens = input.shape()[op.axis];

        // build output
        unsafe {
            let output = if let Some(curr) = self.stored_kv_cache.as_ref() {
                let mut shape = curr.shape().to_owned();
                shape[op.axis] += input_num_tokens;
                let mut output = Tensor::uninitialized_dt(input.datum_type(), &shape)?;
                self.apply_delay_unchecked(op, &input, &mut output);
                output
            } else {
                input.into_tensor()
            };
            self.stored_kv_cache = Some(output.clone());
            Ok(tvec!(output.into()))
        }
    }
}

#[derive(Default, Clone, Debug)]
pub struct DynKeyValueCache {
    axis: usize,
    symbols: [TDim; 2]
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
        Ok(Some(Box::new(DynKeyValueCacheState { stored_kv_cache: None })))
    }
}

impl TypedOp for DynKeyValueCache {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 1);
        let input = inputs[0];
        let mut fact = input.without_value();

        let dim = self.symbols[0].clone() + &self.symbols[1];
        fact.shape.set(self.axis, dim);
        dbg!(&fact);
        Ok(tvec!(fact))
    }

    as_op!();
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
struct FrozenDynKeyValueCacheState {
    stored_kv_cache: Option<Arc<Tensor>>,
}

impl OpStateFreeze for DynKeyValueCacheState {
    fn freeze(&self) -> Box<dyn FrozenOpState> {
        Box::new(FrozenDynKeyValueCacheState { stored_kv_cache: self.stored_kv_cache.as_ref().map(|t| t.clone().into_arc_tensor()) })
    }
}

impl FrozenOpState for FrozenDynKeyValueCacheState {
    fn unfreeze(&self) -> Box<dyn OpState> {
        Box::new(DynKeyValueCacheState { stored_kv_cache: self.stored_kv_cache.as_ref().map(|t| t.clone().into_tensor()) })
    }
}


/// Search pattern => Input -> Concat -> Output
pub fn replace_kv_cache(
    target: &mut TypedModel,
    source_node_id: usize
) -> TractResult<Option<()>> {
    assert!(target.node(source_node_id).op_is::<TypedSource>());
    let (concat_node_id, non_source_input_id, axis, symbols) = {
        let source_node = target.node(source_node_id);
        let n_node = next_node(target, source_node);
        rule_ensure!(n_node.is_some());

        let concat_node = n_node.unwrap();

        rule_ensure!(concat_node.op_is::<TypedConcat>() &&
            concat_node.inputs.len() == 2 &&
            concat_node.outputs.len() == 1 && target.outputs.contains(&concat_node.id.into()));

        let concat_in_shapes = target.node_input_facts(concat_node.id)?.iter().map(|fact| fact.shape.dims()).collect_vec();
                                                                        
        let concat_out_shape = concat_node.outputs[0].fact.shape.dims();
        let rank = concat_out_shape.len();
        let axes = (0..rank).filter(|ax| 
            concat_out_shape[*ax] == (concat_in_shapes[0][*ax].clone() + concat_in_shapes[1][*ax].clone())).collect_vec();
        ensure!(axes.len() == 1);

        let axis = axes[0];
        rule_ensure!(matches!(concat_in_shapes[0][axis], TDim::Sym(_)) 
                && matches!(concat_in_shapes[1][axis], TDim::Sym(_)));

        let symbols = [concat_in_shapes[0][axis].clone(), concat_in_shapes[1][axis].clone()];
        
        let non_source_input = concat_node.inputs.iter().find(|o| o.node != source_node_id);
        rule_ensure!(non_source_input.is_some());
        (concat_node.id, non_source_input.unwrap().node, axis, symbols)
    };

    {
        let concat_node = target.node_mut(concat_node_id);
        concat_node.op = Box::new(DynKeyValueCache { axis, symbols });
        concat_node.inputs.retain(|input| input != &source_node_id.into());
    }

    {   
        let dummy_op = target.create_dummy();
        let source_node = target.node_mut(source_node_id);
        source_node.outputs[0].successors.clear();
        source_node.op = dummy_op;
    }
    {
        let non_source_input = target.node_mut(non_source_input_id);
        non_source_input.outputs.iter_mut().for_each(|output| output.successors.iter_mut().for_each(|succ| if succ.node == concat_node_id {succ.slot = 0}));
    }
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

    fn run_test_case<F: Datum + Zero + Copy>(input_shapes: &[Vec<usize>], axis: usize) -> TractResult<()>
    where
        usize: AsPrimitive<F>,
    {   
        let mut state = DynKeyValueCacheState { stored_kv_cache: None};
        let op = DynKeyValueCache{ axis, symbols: [TDim::Val(0), TDim::Val(0)]};
        let mut session_state = SessionState::default();

        let first_shape = &input_shapes[0];
        ensure!(input_shapes.iter().all(|shape| (shape.len() == first_shape.len()) && (shape[..axis] == first_shape[..axis]) && (if dbg!(axis != (shape.len() - 1)) {shape[(axis + 1)..] == first_shape[(axis + 1)..]} else {true})));

        let mut output = tensor0(0);
        let mut inputs = tvec![];
        for shape in input_shapes {
            let len = shape.iter().product::<usize>();
            let input = Tensor::from_shape(&shape, &(0..len).map(|f| f.as_()).collect::<Vec<F>>())?;
            inputs.push(input.clone().into_tvalue());
            output = state.eval(&mut session_state, &op,  tvec!(input.clone().into()))?[0].clone().into_tensor();
        }
        let reference = &TypedConcat { axis }.eval(inputs)?[0];

        output.close_enough(&reference.clone().into_tensor(), Approximation::Close)?;
        Ok(())
    }

    #[test]
    fn test_dyn_kv_cache() -> TractResult<()> {
        run_test_case::<f32>(&[vec!(2, 2)], 0)?;
        run_test_case::<f32>(&[vec!(2, 2), vec!(4,2)], 0)?;
        run_test_case::<f32>(&[vec!(2, 2), vec!(2,1), vec!(2,3)], 1)?;
        Ok(())
    }
}