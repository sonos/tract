use super::*;

/// Infers basic properties in the case of unary or binary operators.
pub fn infer_forward_basic(op: &Op, inputs: Vec<&ATensor>) -> Result<Vec<ATensor>> {
    let input_values: Result<Vec<_>> = inputs
        .iter()
        .map(|t| t.value.concretize())
        .collect();

    let output = if let Ok(v) = input_values {
        // If we know the value of all the inputs, we can deduce everything.
        let input_inputs: Vec<_> = v
            .into_iter()
            .map(|v| v.clone().into())
            .collect();

        let output_value = op.eval(input_inputs)?.pop().unwrap();

        ATensor {
            datatype: inputs[0].datatype.clone(),
            shape: output_value.shape().into(),
            value: avalue!(output_value.into_matrix())
        }
    } else {
        // Otherwise we can only deduce the type and shape of the output.
        let input_shapes: Vec<_> = inputs
            .iter()
            .map(|t| &t.shape)
            .collect();

        ATensor {
            datatype: inputs[0].datatype.clone(),
            shape: infer_shape_broadcasting(input_shapes)?,
            value: avalue!(_)
        }
    };

    Ok(vec![output])
}