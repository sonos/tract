#![cfg(test)]
use log::*;
use tract_tflite::Tflite;
use tract_tflite::internal::*;

#[path = "../suite.rs"]
mod suite;

mod tflite_runtime;

mod tflite_predump {
    use super::*;
    #[derive(Debug)]
    #[allow(dead_code)]
    struct TflitePredump(Tflite);

    impl Runtime for TflitePredump {
        fn name(&self) -> StaticName {
            "tflite-predump".into()
        }

        fn prepare_with_options(
            &self,
            mut model: TypedModel,
            options: &PlanOptions,
        ) -> TractResult<Box<dyn Runnable>> {
            tract_tflite::rewriter::rewrite_for_tflite(&mut model).context("Preparing model")?;
            Ok(Box::new(model.into_runnable_with_options(&options)?))
        }
    }

    fn runtime() -> &'static TflitePredump {
        lazy_static::lazy_static! {
            static ref RT: TflitePredump = TflitePredump(Tflite::default());
        };
        &RT
    }

    include!(concat!(env!("OUT_DIR"), "/tests/tests.rs"));
}

mod tflite_cycle {
    use tract_tflite::internal::tract_core::ops::dummy::Dummy;

    use super::*;
    #[derive(Debug)]
    struct TfliteCyclingRuntime(Tflite);

    impl Runtime for TfliteCyclingRuntime {
        fn name(&self) -> StaticName {
            "tflite-cycle".into()
        }

        fn prepare_with_options(
            &self,
            model: TypedModel,
            options: &PlanOptions,
        ) -> TractResult<Box<dyn Runnable>> {
            info!("Store to Tflite");
            let mut buffer = vec![];
            self.0.write(&model, &mut buffer).context("Translating model to tflite")?;
            info!("Reload from Tflite");
            let mut reloaded =
                self.0.model_for_read(&mut &*buffer).context("Reloading model from tflite")?;
            for i in 0..model.inputs.len() {
                if model.input_fact(i)? != reloaded.input_fact(i)?
                    && model.input_fact(i)?.datum_type.unquantized()
                        == reloaded.input_fact(i)?.datum_type.unquantized()
                {
                    let old_source_outlet = reloaded.inputs[i];
                    let name = reloaded.node(old_source_outlet.node).name.clone();
                    let new_source = reloaded.add_source(&name, model.input_fact(i)?.clone())?;
                    let wire = reloaded.wire_node(
                        format!("{name}.qp"),
                        tract_core::ops::cast::cast(reloaded.input_fact(i)?.datum_type),
                        &[new_source],
                    )?[0];
                    reloaded.inputs.pop();
                    reloaded.inputs[i] = new_source;
                    let succs = reloaded.node(old_source_outlet.node).outputs[0].successors.clone();
                    for succ in succs {
                        reloaded.add_edge(wire, succ)?;
                    }
                    for output in &mut reloaded.outputs {
                        if *output == old_source_outlet {
                            *output = new_source;
                        }
                    }
                    reloaded.nodes[old_source_outlet.node].name.push_str(".old");
                    reloaded.nodes[old_source_outlet.node].op = Box::new(Dummy);
                }
            }
            Ok(Box::new(
                reloaded
                    .into_optimized()
                    .context("Optimising post-cycle model")?
                    .into_runnable_with_options(options)?,
            ))
        }
    }

    fn runtime() -> &'static TfliteCyclingRuntime {
        lazy_static::lazy_static! {
            static ref RT: TfliteCyclingRuntime = TfliteCyclingRuntime(Tflite::default());
        };
        &RT
    }

    include!(concat!(env!("OUT_DIR"), "/tests/tests.rs"));
}
