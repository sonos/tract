#![cfg(test)]

#[path = "../suite.rs"]
mod suite;

mod run_as_f16 {
    use super::*;
    use tract_core::internal::*;
    use tract_core::model::translator::Translate;

    #[derive(Debug)]
    struct RunAsF16;

    impl Runtime for RunAsF16 {
        fn name(&self) -> StaticName {
            "run_as_f16".into()
        }

        fn prepare_with_options(
            &self,
            model: TypedModel,
            options: &RunOptions,
        ) -> TractResult<Box<dyn Runnable>> {
            let outputs_dt =
                model.outputs.iter().map(|o| model.outlet_fact(*o).unwrap().datum_type).collect();
            let tr = tract_core::floats::FloatPrecisionTranslator::<f32, f16>::default();
            let model = tr.translate_model(&model)?;
            Ok(Box::new(RunnableAsF16(
                model.into_optimized()?.into_runnable_with_options(options)?,
                outputs_dt,
            )))
        }
    }

    #[derive(Debug)]
    pub struct RunnableAsF16(pub Arc<TypedRunnableModel>, pub TVec<DatumType>);

    impl Runnable for RunnableAsF16 {
        fn spawn(&self) -> TractResult<Box<dyn State>> {
            Ok(Box::new(StateAsF16(self.0.spawn()?, self.1.clone())))
        }

        fn input_count(&self) -> usize {
            self.0.input_count()
        }

        fn output_count(&self) -> usize {
            self.0.output_count()
        }

        fn typed_model(&self) -> Option<&Arc<TypedModel>> {
            self.0.typed_model()
        }

        fn typed_plan(&self) -> Option<&Arc<TypedSimplePlan>> {
            self.0.typed_plan()
        }
    }

    #[derive(Debug)]
    struct StateAsF16(TypedSimpleState, TVec<DatumType>);

    impl State for StateAsF16 {
        fn run(&mut self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
            let inputs = inputs
                .into_iter()
                .map(|v| {
                    if v.datum_type() == DatumType::F32 {
                        v.into_tensor()
                            .cast_to_dt(f16::datum_type())
                            .unwrap()
                            .into_owned()
                            .into_tvalue()
                    } else {
                        v
                    }
                })
                .collect();
            let outputs = self.0.run(inputs)?;
            Ok(outputs
                .into_iter()
                .zip(self.1.iter())
                .map(|(t, dt)| t.into_tensor().cast_to_dt(*dt).unwrap().into_owned().into_tvalue())
                .collect())
        }

        fn initializable_states_count(&self) -> usize {
            self.0.initializable_states_count()
        }

        fn get_states_facts(&self) -> Vec<TypedFact> {
            self.0.get_states_facts()
        }

        fn init_state(&mut self, states: &[TValue]) -> TractResult<()> {
            self.0.init_state(states)
        }

        fn get_states(&self) -> TractResult<Vec<TValue>> {
            self.0.get_states()
        }

        fn input_count(&self) -> usize {
            self.0.input_count()
        }

        fn output_count(&self) -> usize {
            self.0.output_count()
        }

        fn runnable(&self) -> &dyn Runnable {
            self.0.runnable()
        }

        fn freeze(&self) -> Box<dyn FrozenState> {
            Box::new(self.0.freeze())
        }
    }

    fn runtime() -> &'static RunAsF16 {
        static RUN_AS_F16: RunAsF16 = RunAsF16;
        &RUN_AS_F16
    }

    include!(concat!(env!("OUT_DIR"), "/tests/tests.rs"));
}

mod nnef_f16 {
    use std::fmt::Debug;

    use super::run_as_f16::RunnableAsF16;
    use super::*;
    use tract_core::internal::*;
    use tract_core::model::translator::Translate;
    use tract_nnef::internal::Nnef;
    use tract_onnx_opl::WithOnnx;
    use tract_transformers::WithTractTransformers;

    struct NnefF16(Nnef);

    impl Debug for NnefF16 {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "NnefF16")
        }
    }

    impl Runtime for NnefF16 {
        fn name(&self) -> StaticName {
            "nnef_f16".into()
        }

        fn prepare_with_options(
            &self,
            model: TypedModel,
            options: &RunOptions,
        ) -> TractResult<Box<dyn Runnable>> {
            let outputs_dt =
                model.outputs.iter().map(|o| model.outlet_fact(*o).unwrap().datum_type).collect();
            let tr = tract_core::floats::FloatPrecisionTranslator::<f32, f16>::default();
            let model = tr.translate_model(&model)?;
            let mut buf = vec![];
            self.0.write_to_tar(&model, &mut buf)?;
            let reloaded = self.0.model_for_read(&mut &*buf)?;
            Ok(Box::new(RunnableAsF16(
                reloaded.into_optimized()?.into_runnable_with_options(&options)?,
                outputs_dt,
            )))
        }
    }

    fn runtime() -> &'static NnefF16 {
        lazy_static::lazy_static! {
            static ref RT: NnefF16 = NnefF16(tract_nnef::nnef().with_onnx().with_tract_transformers());
        };
        &RT
    }

    include!(concat!(env!("OUT_DIR"), "/tests/tests.rs"));
}
