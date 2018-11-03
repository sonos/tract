use tract_core::ops::prelude::*;
use CliResult;

/// Compares the outputs of a node in tract and tensorflow.
pub fn check_outputs(got: &[Value], expected: &[TensorFact]) -> CliResult<()> {
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
