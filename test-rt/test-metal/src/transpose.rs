#![cfg(all(test, any(target_os = "macos", target_os = "ios")))]

use std::borrow::Cow;
use std::sync::Arc;
use std::usize;
use tract_core::internal::*;

use paste::paste;
use tract_core::runtime::Runtime;
use tract_core::tract_data::itertools::Itertools;
use tract_metal::MetalGemmImplKind;

#[path = "../suite.rs"]
mod suite;

#[derive(Debug)]
struct MetalTestTransposeRuntime {
    name: &'static str,
    gemm_impl: MetalGemmImplKind,
}

impl Runtime for MetalTestTransposeRuntime {
    fn name(&self) -> Cow<str> {
        self.name.into()
    }

    fn prepare(&self, mut model: TypedModel) -> TractResult<Box<dyn Runnable>> {
        for ix in 0..model.inputs.len() {
            let input = model.input_outlets()?[ix];
            let in_fact =  model.outlet_fact(input)?;
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

        tract_metal::transform::MetalTransform { gemm_impl: Some(self.gemm_impl) }
            .transform_up_to_phase(&mut model, usize::MAX)?;
        let runnable = MetalTestTransformRunnable { runnable: Arc::new(model.into_runnable()?) };
        Ok(Box::new(runnable))
    }
}

macro_rules! metal_runtime {
    ($gemm_impl: ident) => {
        paste! {
            mod [<$gemm_impl:lower>] {
                use super::*;

                fn runtime() -> &'static MetalTestTransposeRuntime {
                    lazy_static::lazy_static! {
                        static ref RT: MetalTestTransposeRuntime = MetalTestTransposeRuntime { name: stringify!([<$gemm_impl:lower>]), gemm_impl: MetalGemmImplKind::$gemm_impl };
                    };
                    &RT
                }

                include!(concat!(env!("OUT_DIR"), "/tests/",  stringify!([<$gemm_impl:lower>]), ".rs"));
            }
        }
    };
}


metal_runtime!(Mlx);
metal_runtime!(Mfa);
metal_runtime!(Ggml);


#[derive(Debug)]
struct MetalTestTransformState {
    state: TypedSimpleState<TypedModel, Arc<TypedRunnableModel<TypedModel>>>,
}

impl State for MetalTestTransformState {
    fn run(&mut self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let inputs= inputs.into_iter().map(|input| {
            let input = input.into_tensor();
            let rank = input.rank();
            let perms = (0..rank).rev().collect_vec();
            Ok(input.permute_axes(&perms)?.into_tvalue())
        }).collect::<TractResult<TVec<TValue>>>()?;

        self.state.run(inputs)?.into_iter().map(|t| {
            let t = t.into_tensor();
            let rank = t.rank();
            let perms = (0..rank).rev().collect_vec();
            Ok(t.permute_axes(&perms)?.into_tvalue())
        }).collect()
    }
}

#[derive(Debug)]
struct MetalTestTransformRunnable {
    runnable: Arc<TypedRunnableModel<TypedModel>>,
}

impl Runnable for MetalTestTransformRunnable {
    fn spawn(&self) -> TractResult<Box<dyn State>> {
        Ok(Box::new(MetalTestTransformState { state: TypedSimpleState::new(self.runnable.clone())? }))
    }
}