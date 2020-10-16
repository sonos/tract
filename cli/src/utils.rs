use crate::CliResult;
use tract_hir::internal::*;

/// Compares the outputs of a node in tract and tensorflow.
pub fn check_outputs(got: &[Arc<Tensor>], expected: &[Option<Arc<Tensor>>]) -> CliResult<()> {
    if got.len() != expected.len() {
        error_chain::bail!(
            "Number of output differ: got:{}, expected:{}",
            got.len(),
            expected.len()
        )
    }

    for (ix, (got, exp)) in got.iter().zip(expected.iter()).enumerate() {
        if let Some(exp) = exp {
            if let Ok(exp) = exp.to_array_view::<f32>() {
                debug!("Exp: {:?}", exp);
            }
            if let Ok(got) = got.to_array_view::<f32>() {
                debug!("Got: {:?}", got);
            }
            if exp.shape() != got.shape() {
                error_chain::bail!(
                    "Checking output {}, expected shape: {:?}, got {:?}",
                    ix,
                    exp.shape(),
                    got.shape()
                )
            } else if let Err(e) = exp.close_enough(got, true) {
                error_chain::bail!("Checking output {}, {:?}", ix, e);
            } else {
                info!("Checked output #{}, ok.", ix);
            }
        }
    }

    Ok(())
}

/// Compares the outputs of a node in tract and tensorflow.
pub fn check_inferred(got: &[InferenceFact], expected: &[InferenceFact]) -> CliResult<()> {
    if got.len() != expected.len() {
        error_chain::bail!(
            "Number of output differ: got:{}, expected:{}",
            got.len(),
            expected.len()
        )
    }

    for (got, exp) in got.iter().zip(expected.iter()) {
        if exp.datum_type != got.datum_type {
            error_chain::bail!("Failed to infer datum type: expected {:?}, got {:?}", exp, got);
        }
        if exp.shape != got.shape {
            error_chain::bail!("Failed to infer shape: expected {:?}, got {:?}", exp, got);
        }
    }

    Ok(())
}
