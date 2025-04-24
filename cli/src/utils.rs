use std::iter::once;

use crate::params::Parameters;
use tract_hir::internal::*;
use tract_itertools::Itertools;
use tract_libcli::model::Model;

/// Compares the outputs of a node in tract and tensorflow.
pub fn check_outputs(got: &[Vec<TValue>], params: &Parameters) -> TractResult<()> {
    let mut error = None;
    // iter over all possible tract model outputs
    for (ix, output) in params.tract_model.output_outlets().iter().enumerate() {
        // get either name from outlet_label or from node_name
        let lookup_names = params
            .tract_model
            .outlet_label(*output)
            .into_iter()
            .chain(once(params.tract_model.node_name(output.node)))
            .collect_vec();
        let exp = lookup_names.iter().find_map(|name| params.tensors_values.by_name(name));
        if exp.is_none() {
            if params.assertions.allow_missing_outputs {
                warn!("Missing reference output in bundle for {}", lookup_names.join(" or "));
                continue;
            } else {
                bail!("Missing reference output in bundle for {}", lookup_names.join(" or "));
            }
        }
        let exp = exp.unwrap();
        debug!("Output {ix}, expects {exp:?}");
        let mut exp: TValue = exp.values.as_ref().with_context(|| {
            format!("Output {lookup_names:?}: found reference info without value: {exp:?}")
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
                error!("{e:?}");
            } else {
                error = Some(e);
            }
        } else {
            info!("Checked output #{ix}, ok.");
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

pub fn clarify_tvalues(values: &TVec<TValue>) -> TractResult<TVec<TValue>> {
    values
        .iter()
        .map(|t| {
            if t.datum_type().is_opaque() && t.volume() == 1 {
                if let Some(clarified) = t.to_scalar::<Opaque>()?.clarify_to_tensor()? {
                    return Ok(clarified.into_tvalue());
                }
            }
            Ok(t.clone())
        })
        .collect()
}

pub fn clarify_typed_fact<'a>(fact: impl Into<Cow<'a, TypedFact>>) -> Cow<'a, TypedFact> {
    let fact = fact.into();
    if fact.datum_type == DatumType::Opaque {
        fact.opaque_fact
            .as_ref()
            .and_then(|it| it.clarify_dt_shape())
            .map(|(dt, s)| Cow::Owned(TypedFact::dt_shape(dt, s)))
            .unwrap_or_else(|| fact)
    } else {
        fact
    }
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
