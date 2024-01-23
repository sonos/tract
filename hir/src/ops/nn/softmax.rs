use crate::internal::*;

#[derive(Debug, Clone, new, Hash)]
pub struct Softmax {
    axis: isize,
}

impl Expansion for Softmax {
    fn name(&self) -> Cow<str> {
        "Softmax".into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {:?}", self.axis)])
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 1)?;
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;

        Ok(())
    }

    fn wire(
        &self,
        name: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let axis = if self.axis < 0 {
            (target.outlet_fact(inputs[0])?.rank() as isize + self.axis) as usize
        } else {
            self.axis as usize
        };

        let input = target.outlet_fact(inputs[0])?.clone();
        let input_dt = input.datum_type;
        let quant_output_dt = if input_dt.is_quantized() {
            // Quantization parameters are not specified in ONNX (v13) so we set this value as default
            // in order to maximize the precision of the output.
            Some(DatumType::QU8(QParams::ZpScale { zero_point: 0, scale: 0.0078125 }))
        } else {
            None
        };

        target.wire_node(
            name,
            tract_core::ops::nn::Softmax {
                axes: tvec![axis],
                quant_output_dt,
                ..tract_core::ops::nn::Softmax::default()
            },
            inputs,
        )
    }
}
