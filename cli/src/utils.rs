use crate::CliResult;
use tract_core::ops::prelude::*;

/// Compares the outputs of a node in tract and tensorflow.
pub fn check_outputs(got: &[SharedTensor], expected: &[TensorFact]) -> CliResult<()> {
    if got.len() != expected.len() {
        bail!(
            "Number of output differ: got:{}, expected:{}",
            got.len(),
            expected.len()
        )
    }

    for (got, exp) in got.iter().zip(expected.iter()) {
        exp.datum_type.unify(&got.datum_type().into())?;
        exp.shape.unify(&got.shape().into())?;
        if let Some(t) = exp.value.concretize() {
            if !t.close_enough(got, true) {
                bail!("Values are not close enough")
            }
        }
    }

    Ok(())
}

/// Compares the outputs of a node in tract and tensorflow.
pub fn check_inferred(got: &[TensorFact], expected: &[TensorFact]) -> CliResult<()> {
    if got.len() != expected.len() {
        bail!(
            "Number of output differ: got:{}, expected:{}",
            got.len(),
            expected.len()
        )
    }

    for (got, exp) in got.iter().zip(expected.iter()) {
        if exp.datum_type != got.datum_type {
            bail!(
                "Failed to infer datum type: expected {:?}, got {:?}",
                exp,
                got
            );
        }
        if exp.shape != got.shape {
            bail!("Failed to infer shape: expected {:?}, got {:?}", exp, got);
        }
    }

    Ok(())
}
