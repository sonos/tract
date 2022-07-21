use crate::model::Model;
use crate::params::Parameters;
use crate::CliResult;
use tract_hir::internal::*;

/// Compares the outputs of a node in tract and tensorflow.
pub fn check_outputs(got: &[Arc<Tensor>], params: &Parameters) -> CliResult<()> {
    for (ix, output) in params.tract_model.output_outlets().iter().enumerate() {
        let mut names = vec![params.tract_model.node_name(output.node)];
        if let Some(label) = params.tract_model.outlet_label(*output) {
            names.push(label);
        }
        let exp = names
            .iter()
            .find_map(|n| params.tensors_values.by_name(n))
            .with_context(|| format!("Do not have reference value for output {:?}", names))?;
        let exp = &exp.values.as_ref().with_context(|| {
            format!("Output {:?}: found reference info without value: {:?}", names, exp)
        })?[0];
        let got = &got[ix];
        if (params.allow_float_casts
            && exp.datum_type() == f32::datum_type()
            && got.datum_type() == f16::datum_type())
            || exp.datum_type().unquantized() == got.datum_type().unquantized()
        {
            let exp = exp.cast_to_dt(got.datum_type())?;
            exp.close_enough(&got, true).with_context(|| {
                format!("Checking output {} (expected {:?}, got {:?}", ix, exp, got)
            })?;
            info!("Checked output #{}, ok.", ix);
        } else {
            bail!("Checking output {} (expected {:?}, got {:?}", ix, exp, got)
        }
    }

    Ok(())
}

/// Compares the outputs of a node in tract and tensorflow.
pub fn check_inferred(got: &[InferenceFact], expected: &[InferenceFact]) -> CliResult<()> {
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

pub fn count_op(model: &dyn Model, name: &str) -> CliResult<usize> {
    Ok(model
        .eval_order()
        .context("Cannot assert op count without an eval order")?
        .into_iter()
        .map(|i| {
            if model.node_op(i).name() == name {
                1
            } else {
                model.nested_models(i).into_iter().flat_map(|(_, m)| count_op(m, name)).sum()
            }
        })
        .sum())
}
