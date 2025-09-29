#![cfg(test)]

use std::sync::Arc;
use tract_core::internal::*;

use pastey::paste;
use tract_core::runtime::Runtime;
use tract_core::tract_data::itertools::Itertools;

#[path = "../suite.rs"]
mod suite;

#[derive(Debug)]
struct CudaTestTransformState {
    state: TypedSimpleState<TypedModel, Arc<TypedRunnableModel<TypedModel>>>,
    transpose_inputs: bool,
    use_arena: bool,
}

impl State for CudaTestTransformState {
    fn run(&mut self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let mut state = if self.use_arena {
            let session_handler = tract_gpu::session_handler::DeviceSessionHandler::from_plan(
                self.state.plan(),
                &self.state.session_state.resolved_symbols,
            )?;

            let plan = self.state.plan().clone().with_session_handler(session_handler);
            TypedSimpleState::new(Arc::new(plan))?
        } else {
            self.state.clone()
        };

        if self.transpose_inputs {
            let inputs = inputs
                .into_iter()
                .map(|input| {
                    let input = input.into_tensor();
                    let rank = input.rank();
                    let perms = (0..rank).rev().collect_vec();
                    Ok(input.permute_axes(&perms)?.into_tvalue())
                })
                .collect::<TractResult<TVec<TValue>>>()?;

            state
                .run(inputs)?
                .into_iter()
                .map(|t| {
                    let t = t.into_tensor();
                    let rank = t.rank();
                    let perms = (0..rank).rev().collect_vec();
                    Ok(t.permute_axes(&perms)?.into_tvalue())
                })
                .collect()
        } else {
            state.run(inputs)
        }
    }
}

#[derive(Debug)]
struct CudaTestTransformRunnable {
    runnable: Arc<TypedRunnableModel<TypedModel>>,
    transpose_inputs: bool,
    use_arena: bool,
}

impl Runnable for CudaTestTransformRunnable {
    fn spawn(&self) -> TractResult<Box<dyn State>> {
        Ok(Box::new(CudaTestTransformState {
            state: TypedSimpleState::new(self.runnable.clone())?,
            transpose_inputs: self.transpose_inputs,
            use_arena: self.use_arena,
        }))
    }
}

#[derive(Debug)]
struct CudaTestRuntime {
    name: &'static str,
    phase: usize,
    optimize: bool,
    transpose_inputs: bool,
    use_arena: bool,
}

impl Runtime for CudaTestRuntime {
    fn name(&self) -> StaticName {
        self.name.into()
    }

    fn prepare(&self, mut model: TypedModel) -> TractResult<Box<dyn Runnable>> {
        if self.transpose_inputs {
            for ix in 0..model.inputs.len() {
                let input = model.input_outlets()?[ix];
                let in_fact = model.outlet_fact(input)?;
                let rank = in_fact.rank();
                let shape = in_fact.shape.dims().into_iter().rev().collect::<TVec<_>>();
                let fact = in_fact.datum_type.fact(shape);

                let transposed_input = model.add_source(format!("transposed_input_{ix}"), fact)?;

                let mut patch = TypedModelPatch::default();
                let mut wire = patch.tap_model(&model, transposed_input)?;

                let perms = (0..rank).rev().collect_vec();
                let axis_ops = perm_to_ops(&perms);

                for (ax, op) in axis_ops.into_iter().enumerate() {
                    wire = patch.wire_node(format!("transposed_input.{ix}_{ax}"), op, &[wire])?[0];
                }
                patch.shunt_outside(&model, input, wire)?;
                patch.apply(&mut model)?;
            }

            // Delete old inputs
            for _ in 0..model.inputs.len() / 2 {
                let input = model.inputs.remove(0);
                model.node_mut(input.node).op = model.create_dummy();
            }

            for (ix, output) in model.outputs.clone().iter().enumerate() {
                let rank = model.outlet_fact(*output)?.rank();
                let mut wire = *output;
                let perms = (0..rank).rev().collect_vec();
                let axis_ops = perm_to_ops(&perms);

                for (ax, op) in axis_ops.into_iter().enumerate() {
                    wire = model.wire_node(format!("transposed_output.{ix}_{ax}"), op, &[wire])?[0];
                }
                model.outputs[ix] = wire;
            }
        }

        tract_cuda::CudaTransform.transform_up_to_phase(&mut model, self.phase)?;
        if self.optimize {
            model = model.into_optimized()?;
        }

        let runnable = CudaTestTransformRunnable {
            runnable: Arc::new(model.into_runnable()?),
            transpose_inputs: self.transpose_inputs,
            use_arena: self.use_arena,
        };
        Ok(Box::new(runnable))
    }
}

macro_rules! cuda_test_suite {
    ($id: ident, $phase: expr, $optimize: expr, $transpose_inputs: ident, $use_arena: ident) => {
        paste! {
            mod [<$id>] {
                use super::*;

                fn runtime() -> &'static CudaTestRuntime {
                    lazy_static::lazy_static! {
                        static ref RT: CudaTestRuntime = CudaTestRuntime {
                            name: stringify!([<$id>]),
                            phase: $phase,
                            optimize: $optimize,
                            transpose_inputs: $transpose_inputs,
                            use_arena: $use_arena,
                        };
                    };
                    &RT
                }

                include!(concat!(env!("OUT_DIR"), "/tests/tests.rs"));
            }
        }
    };
}

//cuda_test_suite!(cuda_phase_2_translate, 2, false, , false, false);
//cuda_test_suite!(cuda_phase_3_post_translate, 3, false, , false, false);
cuda_test_suite!(optimized_cuda, usize::MAX, true, false, false);
cuda_test_suite!(optimized_cuda_with_arena, usize::MAX, true, false, true);
cuda_test_suite!(optimized_cuda_transpose, usize::MAX, true, true, false);
