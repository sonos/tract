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
        fn name(&self) -> Cow<str> {
            "run_as_f16".into()
        }

        fn prepare(&self, model: TypedModel) -> TractResult<Box<dyn Runnable>> {
            let outputs_dt =
                model.outputs.iter().map(|o| model.outlet_fact(*o).unwrap().datum_type).collect();
            let tr = tract_core::floats::FloatPrecisionTranslator::<f32, f16>::default();
            let model = tr.translate_model(&model)?;
            Ok(Box::new(RunnableAsF16(
                Arc::new(model.into_optimized()?.into_runnable()?),
                outputs_dt,
            )))
        }
    }

    #[derive(Debug)]
    struct RunnableAsF16(Arc<TypedRunnableModel<TypedModel>>, TVec<DatumType>);

    impl Runnable for RunnableAsF16 {
        fn spawn(&self) -> TractResult<Box<dyn State>> {
            Ok(Box::new(StateAsF16(SimpleState::new(self.0.clone())?, self.1.clone())))
        }
    }

    #[derive(Debug)]
    struct StateAsF16(
        TypedSimpleState<TypedModel, Arc<TypedRunnableModel<TypedModel>>>,
        TVec<DatumType>,
    );

    impl State for StateAsF16 {
        fn run(&mut self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
            let inputs = inputs
                .into_iter()
                .map(|v| {
                    if v.datum_type().is_float() {
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
    }

    fn runtime() -> RunAsF16 {
        RunAsF16
    }

    include!(concat!(env!("OUT_DIR"), "/tests/tests.rs"));
}
