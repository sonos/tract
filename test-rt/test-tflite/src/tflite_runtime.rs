use super::*;
use tflite::ops::builtin::BuiltinOpResolver;
use tflite::FlatBufferModel;
use tflite::Interpreter;
use tflite::InterpreterBuilder;

struct TfliteRuntime(Tflite);

impl Runtime for TfliteRuntime {
    fn name(&self) -> Cow<str> {
        "tflite".into()
    }

    fn prepare(&self, model: TypedModel) -> TractResult<Box<dyn Runnable>> {
        let mut buffer = vec![];
        self.0.write(&model, &mut buffer)?;
        Ok(Box::new(TfliteRunnable(buffer)))
    }
}

struct TfliteRunnable(Vec<u8>);

impl Runnable for TfliteRunnable {
    fn spawn(&self) -> TractResult<Box<dyn State>> {
        let fb = FlatBufferModel::build_from_buffer(self.0.clone())?;
        let resolver = BuiltinOpResolver::default();
        let builder = InterpreterBuilder::new(fb, resolver)?;
        let mut interpreter = builder.build()?;
        interpreter.allocate_tensors()?;
        Ok(Box::new(TfliteState(interpreter)))
    }
}

struct TfliteState(Interpreter<'static, BuiltinOpResolver>);

impl State for TfliteState {
    fn run(&mut self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == self.0.inputs().len());
        for (ix, input) in inputs.iter().enumerate() {
            let input_ix = self.0.inputs()[ix];
            let input_tensor = self.0.tensor_info(input_ix).unwrap();
            assert_eq!(input_tensor.element_kind as u32, 1);
            assert_eq!(input_tensor.dims, input.shape());
            self.0.tensor_buffer_mut(input_ix).unwrap().copy_from_slice(unsafe { input.as_bytes() })
        }
        self.0.invoke()?;
        let mut outputs = tvec![];
        for ix in 0..self.0.outputs().len() {
            let output_ix = self.0.outputs()[ix];
            let output_tensor = self.0.tensor_info(output_ix).unwrap();
            assert_eq!(output_tensor.element_kind as u32, 1);
            let tensor = unsafe {
                Tensor::from_raw_dt(
                    f32::datum_type(),
                    &output_tensor.dims,
                    self.0.tensor_buffer(output_ix).unwrap(),
                )?
            };
            outputs.push(tensor.into_tvalue());
        }
        Ok(outputs)
    }
}

fn runtime() -> &'static TfliteRuntime {
    lazy_static::lazy_static! {
        static ref RT: TfliteRuntime = TfliteRuntime(Tflite::default());
    };
    &RT
}

include!(concat!(env!("OUT_DIR"), "/tests/tests.rs"));
