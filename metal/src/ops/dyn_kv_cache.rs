use crate::kernels::array::Concat;
use crate::ops::MetalEvalOp;
use crate::utils::with_borrowed_metal_stream;
use crate::MetalStream;
use derive_new::new;
use tract_core::internal::*;
use tract_core::ops::OpStateFreeze;
use tract_gpu::tensor::{DeviceTensor, DeviceTensorExt, IntoDevice};
use tract_transformers::ops::dyn_kv_cache::{DynKeyValueCache, DynKeyValueCacheState};

#[derive(Debug, Clone, new)]
pub struct MetalDynKVCacheState<O: MetalEvalOp> {
    node_id: usize,
    op: O,
    io_name: [String; 2],
    input_facts: [TypedFact; 2],
    kv_cache: Option<DeviceTensor>,
}

impl<O: MetalEvalOp> OpStateFreeze for MetalDynKVCacheState<O> {
    fn freeze(&self) -> Box<(dyn FrozenOpState + 'static)> {
        Box::new(self.clone())
    }
}

impl<O: MetalEvalOp> FrozenOpState for MetalDynKVCacheState<O> {
    fn unfreeze(&self) -> Box<dyn OpState> {
        Box::new(self.clone())
    }
}

impl<O: MetalEvalOp> OpState for MetalDynKVCacheState<O> {
    fn load_from(
        &mut self,
        state: &mut SessionState,
        states: &mut HashMap<String, TValue>,
    ) -> TractResult<()> {
        if let Some(kv_cache) = states.remove(&self.io_name[0]) {
            // KV Cache fact is always at index 0
            DynKeyValueCacheState::resolve_symbols(
                state,
                self.input_facts[0].clone(),
                Some(kv_cache.shape()),
            )?;
            self.kv_cache = Some(kv_cache.into_tensor().into_device()?);
            Ok(())
        } else {
            bail!("KV cache input {} not found in given states", self.io_name[0])
        }
    }

    fn save_to(&mut self, states: &mut HashMap<String, TValue>) -> TractResult<()> {
        if let Some(kv_cache) = &self.kv_cache {
            states.insert(self.io_name[1].clone(), kv_cache.to_host()?.into_tensor().into_tvalue());
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
        DynKeyValueCacheState::resolve_symbols(state, self.input_facts[0].clone(), shape)
    }

    fn eval(
        &mut self,
        session: &mut SessionState,
        _op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == 1);
        with_borrowed_metal_stream(|stream| {
            let mut op_inputs = TVec::new();

            if let Some(kv_cache) = self.kv_cache.take() {
                op_inputs.push(kv_cache.into_opaque_tensor().into_tvalue());
            }

            op_inputs.push(inputs.into_iter().next().unwrap());

            let res = self.op.metal_eval(stream, self.node_id, session, op_inputs)?;

            let kv_tensor = res[0].to_device_tensor()?;
            self.kv_cache = Some(kv_tensor.clone());

            Ok(res)
        })
    }
}

#[derive(new, Debug, Clone, Hash)]
pub struct MetalDynKVCache {
    io_name: [String; 2],
    input_facts: [TypedFact; 2],
    kernel: Concat,
}

impl MetalDynKVCache {
    pub fn from_tract_transformers(op: &DynKeyValueCache) -> Self {
        Self {
            io_name: op.io_name.clone(),
            kernel: Concat { axis: op.axis },
            input_facts: op.input_facts.clone(),
        }
    }

    pub fn axis(&self) -> usize {
        self.kernel.axis
    }
}

impl Op for MetalDynKVCache {
    fn name(&self) -> Cow<str> {
        "MetalDynKVCache".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {}", self.axis())])
    }

    op_as_typed_op!();
}

impl EvalOp for MetalDynKVCache {
    fn is_stateless(&self) -> bool {
        false
    }

    #[allow(unused_variables)]
    fn state(
        &self,
        session: &mut tract_core::internal::SessionState,
        node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(MetalDynKVCacheState::new(
            node_id,
            self.clone(),
            self.io_name.clone(),
            self.input_facts.clone(),
            None,
        ))))
    }
}

impl MetalEvalOp for MetalDynKVCache {
    fn metal_eval(
        &self,
        stream: &MetalStream,
        node_id: usize,
        session: &mut SessionState,
        opaque_inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        ensure!(opaque_inputs.len() <= 2);
        let inputs = opaque_inputs
            .iter()
            .map(|it| it.to_device_tensor())
            .collect::<TractResult<TVec<_>>>()?;

        let mut output_shape = inputs[0].shape().to_vec();
        output_shape[self.axis()] = inputs.iter().map(|inp| inp.shape()[self.axis()]).sum();

        let output = crate::ops::make_tensor_for_node(
            session,
            node_id,
            inputs[0].datum_type(),
            &output_shape,
        )?;
        ensure!(matches!(output, DeviceTensor::Owned(_)));

        self.kernel.dispatch_eval(stream, &inputs, &output)?;
        Ok(tvec!(output.into_opaque_tensor().into_tvalue()))
    }
}

impl TypedOp for MetalDynKVCache {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 1);
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            let mut fact = facts[0].without_value();
            fact.shape.set(
                self.axis(),
                self.input_facts[0].shape.dims()[self.axis()].clone()
                    + self.input_facts[1].shape.dims()[self.axis()].clone(),
            );
            Ok(tvec!(fact))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }
}

#[cfg(test)]
mod tests {
    use crate::MetalTransform;

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
        with_borrowed_metal_stream(|_| {
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
                io_name: [op_name.clone(), op_name.clone()],
                input_facts: [
                    TypedFact::dt_shape(F::datum_type(), cache_shape),
                    TypedFact::dt_shape(F::datum_type(), input_shape),
                ],
                axis,
            };

            let x = model.add_source("x", typed_fact)?;
            model.wire_node("kv_cache", op, &[x])?;
            model.auto_outputs()?;

            let metal_model = MetalTransform::default().transform_into(model)?;
            let mut state = TypedSimpleState::new(metal_model.into_runnable()?)?;

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
            let mut kv_cache_state = state.states[2].clone().unwrap();

            let mut hashmap = HashMap::new();
            kv_cache_state.save_to(&mut hashmap)?;
            let output = hashmap.get(&op_name).unwrap();

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
