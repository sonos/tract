use crate::kernels::array::Concat;
use crate::ops::CudaConcat;
use derive_new::new;
use tract_core::internal::*;
use tract_core::ops::OpStateFreeze;
use tract_gpu::fact::DeviceTypedFactExt;
use tract_gpu::tensor::{DeviceTensor, DeviceTensorExt, IntoDevice};
use tract_transformers::ops::dyn_kv_cache::{DynKeyValueCache, DynKeyValueCacheState};

#[derive(Debug, Clone, new)]
pub struct CudaDynKVCacheState {
    node_id: usize,
    name: String,
    axis: usize,
    past_sequence_fact: TypedFact,
    kv_cache: Option<TValue>,
}

impl OpState for CudaDynKVCacheState {
    fn load_from(&mut self, state: &mut SessionState, states: &mut Vec<TValue>) -> TractResult<()> {
        let kv_cache = states.remove(0);
        // KV Cache fact is always at index 0
        DynKeyValueCacheState::resolve_symbols(
            state,
            self.past_sequence_fact.clone(),
            Some(kv_cache.shape()),
        )?;
        self.kv_cache =
            Some(kv_cache.into_tensor().into_device()?.into_opaque_tensor().into_tvalue());
        Ok(())
    }

    fn save_to(&self, states: &mut Vec<TValue>) -> TractResult<()> {
        if let Some(kv_cache) = &self.kv_cache {
            states.push(kv_cache.to_device_tensor()?.to_host()?.into_tensor().into_tvalue());
            Ok(())
        } else {
            bail!("KV cache {} was never initialized", self.name)
        }
    }

    fn init_tensor_fact(&self) -> Option<(String, TypedFact)> {
        Some((self.name.clone(), self.past_sequence_fact.clone()))
    }

    fn resolve_symbols(&mut self, state: &mut SessionState) -> TractResult<()> {
        let shape = self
            .kv_cache
            .as_ref()
            .map(|kv_cache| kv_cache.to_device_tensor().expect("Expected Cuda Tensor").shape());
        DynKeyValueCacheState::resolve_symbols(state, self.past_sequence_fact.clone(), shape)
    }

    fn eval(
        &mut self,
        session: &mut SessionState,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == 1);
        let mut op_inputs = TVec::new();

        if let Some(kv_cache) = self.kv_cache.take() {
            op_inputs.push(kv_cache);
        }

        op_inputs.push(inputs.into_iter().next().unwrap());

        let concat = &op
            .downcast_ref::<CudaDynKVCache>()
            .ok_or_else(|| format_err!("Wrong Op type"))?
            .concat;
        let res = concat.eval_with_session(self.node_id, session, op_inputs)?.remove(0);

        self.kv_cache = Some(res.clone());

        Ok(tvec!(res))
    }
}

impl CudaDynKVCacheState {
    pub fn truncate(&mut self, len: usize) -> TractResult<()> {
        if let Some(v) = &mut self.kv_cache {
            let mut t: Tensor = v.to_device_tensor()?.to_host()?.into_tensor();
            t = t.slice(self.axis, 0, len)?;
            *v = t.into_device()?.into_opaque_tensor().into_tvalue();
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct FrozenCudaDynKVCacheState {
    node_id: usize,
    name: String,
    axis: usize,
    past_sequence_fact: TypedFact,
    kv_cache: Option<DeviceTensor>,
}

impl OpStateFreeze for CudaDynKVCacheState {
    fn freeze(&self) -> Box<dyn FrozenOpState + 'static> {
        Box::new(FrozenCudaDynKVCacheState {
            node_id: self.node_id,
            name: self.name.clone(),
            axis: self.axis,
            past_sequence_fact: self.past_sequence_fact.clone(),
            kv_cache: self.kv_cache.clone().map(|t| t.to_device_tensor().cloned().unwrap()),
        })
    }
}

impl FrozenOpState for FrozenCudaDynKVCacheState {
    fn unfreeze(&self) -> Box<dyn OpState> {
        Box::new(CudaDynKVCacheState {
            node_id: self.node_id,
            name: self.name.clone(),
            axis: self.axis,
            past_sequence_fact: self.past_sequence_fact.clone(),
            kv_cache: self.kv_cache.clone().map(|t| t.into_opaque_tensor().into_tvalue()),
        })
    }
}

#[derive(new, Debug, Clone, Hash)]
pub struct CudaDynKVCache {
    name: String,
    past_sequence_fact: TypedFact,
    input_sequence_fact: TypedFact,
    concat: CudaConcat,
}

impl CudaDynKVCache {
    pub fn from_tract_transformers(op: &DynKeyValueCache) -> Self {
        Self {
            name: op.name.clone(),
            concat: CudaConcat { kernel: Concat { axis: op.axis } },
            past_sequence_fact: op.past_sequence_fact.clone(),
            input_sequence_fact: op.input_sequence_fact.clone(),
        }
    }

    pub fn axis(&self) -> usize {
        self.concat.kernel.axis
    }
}

impl Op for CudaDynKVCache {
    fn name(&self) -> StaticName {
        "CudaDynKVCache".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {}", self.axis())])
    }

    op_as_typed_op!();
}

impl EvalOp for CudaDynKVCache {
    fn is_stateless(&self) -> bool {
        false
    }

    #[allow(unused_variables)]
    fn state(
        &self,
        session: &mut tract_core::internal::SessionState,
        node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(CudaDynKVCacheState::new(
            node_id,
            self.name.clone(),
            self.axis(),
            self.past_sequence_fact.clone(),
            None,
        ))))
    }
}

impl TypedOp for CudaDynKVCache {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 1);
        let mut facts = tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            let mut fact = facts[0].without_value();
            fact.shape.set(
                self.axis(),
                self.past_sequence_fact.shape.dims()[self.axis()].clone()
                    + self.input_sequence_fact.shape.dims()[self.axis()].clone(),
            );
            Ok(tvec!(fact))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))?;
        facts[0].as_device_fact_mut().unwrap().state_owned = true;
        Ok(facts)
    }
}

#[cfg(test)]
mod tests {
    use crate::context::CUDA_STREAM;
    use crate::CudaTransform;

    use super::*;
    use tract_core::ops::array::TypedConcat;
    use tract_core::transform::ModelTransform;
    use tract_num_traits::AsPrimitive;
    use tract_num_traits::Zero;

    fn run_test_case<F: Datum + Zero + Copy>(
        input_shapes: &[Vec<usize>],
        axis: usize,
    ) -> TractResult<()>
    where
        usize: AsPrimitive<F>,
    {
        CUDA_STREAM.with(|_| {
            let op_name = "test".to_string();

            let mut model = TypedModel::default();

            let make_shape =
                |sym: &str| {
                    input_shapes[0]
                        .iter()
                        .enumerate()
                        .map(|(i, &dim)| {
                            if i == axis {
                                TDim::Sym(model.sym(sym))
                            } else {
                                TDim::Val(dim as _)
                            }
                        })
                        .collect::<TVec<TDim>>()
                };

            let cache_shape = make_shape("P");
            let input_shape = make_shape("S");

            let source_shape = ShapeFact::from_dims(cache_shape.clone());
            let typed_fact = TypedFact {
                datum_type: F::datum_type(),
                shape: source_shape,
                konst: None,
                uniform: None,
                opaque_fact: None,
            };

            let op = DynKeyValueCache {
                name: op_name.clone(),
                past_sequence_fact: TypedFact::dt_shape(F::datum_type(), cache_shape),
                input_sequence_fact: TypedFact::dt_shape(F::datum_type(), input_shape),
                axis,
            };

            let x = model.add_source("x", typed_fact)?;
            model.wire_node("kv_cache", op, &[x])?;
            model.auto_outputs()?;

            let cuda_model = CudaTransform::default().transform_into(model)?;
            let mut state = TypedSimpleState::new(cuda_model.into_runnable()?)?;

            let first_shape = &input_shapes[0];
            ensure!(input_shapes.iter().all(|shape| (shape.len() == first_shape.len())
                && (shape[..axis] == first_shape[..axis])
                && (if axis != (shape.len() - 1) {
                    shape[(axis + 1)..] == first_shape[(axis + 1)..]
                } else {
                    true
                })));

            let mut inputs = tvec![];
            for shape in input_shapes {
                let len = shape.iter().product::<usize>();
                let input =
                    Tensor::from_shape(&shape, &(0..len).map(|f| f.as_()).collect::<Vec<F>>())?;
                inputs.push(input.clone().into_tvalue());

                state.run(tvec!(input.clone().into()))?;
            }
            let kv_cache_state = state.states[2].clone().unwrap();

            let mut curr_state = vec![];
            kv_cache_state.save_to(&mut curr_state)?;
            let output = curr_state.remove(0);

            let reference = &TypedConcat { axis }.eval(inputs)?[0];
            output.close_enough(&reference.clone().into_tensor(), Approximation::Close)?;
            Ok(())
        })
    }

    #[test]
    fn test_dyn_kv_cache() -> TractResult<()> {
        run_test_case::<f32>(&[vec![2, 2]], 0)?;
        run_test_case::<f32>(&[vec![2, 2], vec![4, 2]], 0)?;
        run_test_case::<f32>(&[vec![2, 2], vec![2, 1], vec![2, 3]], 1)?;
        Ok(())
    }
}
