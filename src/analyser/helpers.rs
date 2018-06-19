use tensor::Tensor;
use super::*;

/// Build a TensorFact from a Tensor.
pub fn tensor_to_fact(tensor: Tensor) -> TensorFact {
    TensorFact {
        datatype: typefact!(tensor.datatype()),
        shape: tensor.shape().into(),
        value: valuefact!(tensor),
    }
}

/// Infers every possible fact when all the values are concrete.
pub fn infer_forward_concrete(
    op: &Op,
    inputs: &Vec<&TensorFact>,
) -> Result<Option<Vec<TensorFact>>> {
    let input_values: Vec<_> = inputs
        .iter()
        .filter_map(|t| t.value.concretize())
        .map(|v| v.clone().into())
        .collect();

    if input_values.len() < inputs.len() {
        debug!("Can't infer value: some inputs are still unknown.");
        return Ok(None);
    }

    // If we know the value of all the inputs, we can deduce everything.
    let output_value = op.eval(input_values)?.pop().unwrap();

    Ok(Some(vec![
        tensor_to_fact(output_value.into_tensor())
    ]))
}

/// Infers basic shape facts in the case of broadcasting operators.
pub fn infer_shape_broadcasting(shapes: Vec<&ShapeFact>) -> Result<Option<ShapeFact>> {
    if shapes.iter().any(|s| s.open) {
        debug!("Can't infer shape for broadcasting operators when some inputs have an open shape.");
        return Ok(None);
    }

    let dims: Vec<_> = shapes.iter().map(|s| &s.dims).collect();
    let bound = dims.iter().map(|s| s.len()).max().unwrap();

    let mut output_shape = vec![];

    for i in 1..(bound + 1) {
        let mut previous = None;
        let mut unknown = 0;

        for shape in &dims {
            if shape.len() < i {
                continue;
            }

            match &shape[shape.len() - i] {
                DimFact::Any => unknown += 1,
                DimFact::Only(j) => match previous {
                    Some(k) if k != j => bail!(
                        "Invalid shape (broadcasting): {} is not compatible with {}.",
                        j,
                        k
                    ),
                    _ => previous = Some(j),
                },
            };
        }

        if unknown > 1 {
            debug!("Can't infer shape (broadcasting): there are multiple unknown values at same index.");
            return Ok(None);
        } else if unknown == 1 && previous != None {
            debug!("Can't infer shape (broadcasting): there are both unknown and known values at same index.");
            return Ok(None);
        } else if unknown == 1 && previous == None {
            output_shape.push(DimFact::Any);
        } else {
            output_shape.push(DimFact::Only(*previous.unwrap()));
        }
    }

    output_shape.reverse();

    Ok(Some(ShapeFact::closed(output_shape)))
}

/// Infers basic facts in the case of unary or binary operators.
pub fn infer_forward_basic(op: &Op, inputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>> {
    if let Some(output) = infer_forward_concrete(op, &inputs)? {
        return Ok(Some(output));
    }

    // Otherwise we can only deduce the type and shape of the output.
    let input_shapes: Vec<_> = inputs.iter().map(|t| &t.shape).collect();

    let datatype = inputs
        .iter()
        .filter_map(|i| i.datatype.concretize())
        .next()
        .map(|t| typefact!(t))
        .unwrap_or(typefact!(_));

    let output = TensorFact {
        datatype,
        shape: infer_shape_broadcasting(input_shapes)?.unwrap_or(shapefact![..]),
        value: valuefact!(_),
    };

    Ok(Some(vec![output]))
}

/// Returns the most specific closed shape out of an iterator.
pub fn most_specific_shape<'a, I: IntoIterator<Item = &'a ShapeFact>>(
    iter: I,
) -> Result<Option<&'a ShapeFact>> {
    let mut prev_rank = None;
    let mut prev_concrete = None;
    let mut best = None;

    for shape in iter {
        if !shape.open {
            let rank = shape.dims.len();

            if prev_rank.is_some() && rank != prev_rank.unwrap() {
                bail!("Rank mismatch between different shapes.");
            } else {
                prev_rank = Some(rank);
            }

            let concrete = shape.dims.iter().filter(|d| d.is_concrete()).count();

            if prev_concrete.is_none() || concrete > prev_concrete.unwrap() {
                prev_concrete = Some(concrete);
                best = Some(shape)
            }
        }
    }

    Ok(best)
}
