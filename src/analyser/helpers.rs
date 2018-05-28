use super::*;

/// Infers every property when all the values are concrete.
pub fn infer_forward_concrete(op: &Op, inputs: &Vec<&ATensor>) -> Result<Vec<ATensor>> {
    let input_values: Result<Vec<_>> = inputs
        .iter()
        .map(|t| t.value.concretize())
        .collect();

    match input_values {
        Ok(v) => {
            // If we know the value of all the inputs, we can deduce everything.
            let input_inputs: Vec<_> = v
                .into_iter()
                .map(|v| v.clone().into())
                .collect();

            let output_value = op.eval(input_inputs)?.pop().unwrap();
            let output = ATensor {
                datatype: inputs[0].datatype.clone(),
                shape: output_value.shape().into(),
                value: avalue!(output_value.into_matrix())
            };

            Ok(vec![output])
        },

        _ => bail!("Can't infer value: some inputs are not concrete.")
    }
}

#[macro_export]
macro_rules! try_infer_forward_concrete {
    ($op:expr, $inputs:expr) => ({
        if let Ok(output) = $crate::analyser::helpers::infer_forward_concrete($op, $inputs) {
            return Ok(output);
        }
    })
}

/// Infers basic properties in the case of unary or binary operators.
pub fn infer_forward_basic(op: &Op, inputs: Vec<&ATensor>) -> Result<Vec<ATensor>> {
    try_infer_forward_concrete!(op, &inputs);

    // Otherwise we can only deduce the type and shape of the output.
    let input_shapes: Vec<_> = inputs
        .iter()
        .map(|t| &t.shape)
        .collect();

    let output = ATensor {
        datatype: inputs[0].datatype.clone(),
        shape: infer_shape_broadcasting(input_shapes)?,
        value: avalue!(_)
    };

    Ok(vec![output])
}

/// Infers basic shape properties in the case of broadcasting operators.
pub fn infer_shape_broadcasting(shapes: Vec<&AShape>) -> Result<AShape> {
    if shapes.iter().any(|s| s.is_open()) {
        bail!("Can't infer shape for broadcasting operators when some inputs have an open shape.");
    }

    let shapes: Vec<_> = shapes.iter()
        .map(|s| s.inner())
        .collect();
    let bound = shapes.iter()
        .map(|s| s.len())
        .max()
        .unwrap();

    let mut output_shape = vec![];

    for i in 1..bound {
        let mut previous = None;
        let mut unknown = 0;

        for shape in &shapes {
            if shape.len() < i {
                continue;
            }

            match &shape[shape.len() - i] {
                ADimension::Any => unknown += 1,
                ADimension::Only(j) => match previous {
                    Some(k) if k != j => bail!("Invalid shape (broadcasting): {} is not compatible with {}.", j, k),
                    _ => previous = Some(j)
                }
            };
        }

        if unknown > 1 {
            bail!("Can't infer shape (broadcasting): there are multiple unknown values at same index.");
        } else if unknown == 1 && previous != None {
            bail!("Can't infer shape (broadcasting): there are both unknown and known values at same index.");
        } else if unknown == 1 && previous == None {
            output_shape.push(ADimension::Any);
        } else {
            output_shape.push(ADimension::Only(*previous.unwrap()));
        }
    }

    Ok(AShape::Closed(output_shape))
}