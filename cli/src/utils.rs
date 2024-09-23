use crate::params::Parameters;
use tract_hir::internal::*;
use tract_libcli::model::Model;

/// Compares the outputs of a node in tract and tensorflow.
pub fn check_outputs(got: &[Vec<TValue>], params: &Parameters) -> TractResult<()> {
    let mut error = None;
    // iter over all possible tract model outputs
    for (ix, output) in params.tract_model.output_outlets().iter().enumerate() {
        // get either name from outlet_label or from node_name
        let name = if let Some(label) = params.tract_model.outlet_label(*output) {
            label
        } else {
            params.tract_model.node_name(output.node)
        };
        // pick expected tensor values for this output
        let exp = params
            .tensors_values
            .by_name(name)
            .with_context(|| format!("Do not have reference value for output {name:?}"))?;
        debug!("Output {}, expects {:?}", ix, exp);
        let mut exp: TValue = exp.values.as_ref().with_context(|| {
            format!("Output {name:?}: found reference info without value: {exp:?}")
        })?[0]
            .clone();
        let got: TValue = if got[ix].len() > 1 {
            let props = params.tract_model.properties();
            let axis = props
                .get("pulse.output_axes")
                .context("multiple turn without pulse.output_axes property")?
                .as_slice::<i64>()?[ix] as usize;
            let delay = props
                .get("pulse.delay")
                .context("multiple turn without pulse.delay properties")?
                .as_slice::<i64>()?[ix] as usize;
            let stacked = Tensor::stack_tensors(axis, &got[ix])?;
            stacked.slice(axis, delay, delay + exp.shape()[axis])?.into()
        } else {
            got[ix][0].clone()
        };
        if (params.allow_float_casts
            && exp.datum_type() == f32::datum_type()
            && got.datum_type() == f16::datum_type())
            || exp.datum_type().unquantized() == got.datum_type().unquantized()
        {
            exp = exp.cast_to_dt(got.datum_type())?.into_owned().into_tvalue();
        }
        if let Err(e) = exp
            .close_enough(&got, params.assertions.approximation)
            .context(format!("Checking output {ix}"))
        {
            if error.is_some() {
                error!("{:?}", e);
            } else {
                error = Some(e);
            }
        } else {
            info!("Checked output #{}, ok.", ix);
        }
    }

    if let Some(e) = error {
        Err(e)
    } else {
        Ok(())
    }
}

/// Compares the outputs of a node in tract and tensorflow.
pub fn check_inferred(got: &[InferenceFact], expected: &[InferenceFact]) -> TractResult<()> {
    if got.len() != expected.len() {
        bail!("Number of output differ: got:{}, expected:{}", got.len(), expected.len())
    }

    for (got, exp) in got.iter().zip(expected.iter()) {
        if exp.datum_type != got.datum_type {
            bail!("Failed to infer datum type: expected {:?}, got {:?}", exp, got);
        }
        if exp.shape != got.shape {
            bail!("Failed to infer shape: expected {:?}, got {:?}", exp, got);
        }
    }

    Ok(())
}

pub fn count_op(model: &dyn Model, name: &str) -> TractResult<usize> {
    Ok(model
        .eval_order()
        .context("Cannot assert op count without an eval order")?
        .into_iter()
        .map(|i| {
            if model.node_op_name(i) == name {
                1
            } else {
                model.nested_models(i).into_iter().flat_map(|(_, m)| count_op(m, name)).sum()
            }
        })
        .sum())
}
