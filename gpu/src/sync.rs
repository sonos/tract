use crate::fact::{DeviceFact, DeviceTypedFactExt};
use crate::tensor::{DeviceTensorExt, IntoDevice};
use derive_new::new;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use tract_core::internal::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceSyncKind {
    ToHost,
    ToDevice,
}

impl fmt::Display for DeviceSyncKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

#[derive(Debug, Clone, new, Copy, PartialEq, Eq, Hash)]
pub struct DeviceSync {
    pub kind: DeviceSyncKind,
}

impl Op for DeviceSync {
    fn name(&self) -> StaticName {
        format!("DeviceSync{}", self.kind).into()
    }

    op_as_typed_op!();
}

impl EvalOp for DeviceSync {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        match self.kind {
            DeviceSyncKind::ToHost => {
                let device_tensor = input.to_device_tensor()?;

                let tensor = device_tensor
                    .to_host()
                    .with_context(|| "Error while syncing device tensor to host")?;
                Ok(tvec![tensor.into_tvalue()])
            }
            DeviceSyncKind::ToDevice => {
                // Already-device tensors (e.g. cache views fed back by the
                // caller) pass through untouched: uploading them would read
                // opaque storage as host bytes and panic.
                if input.to_device_tensor().is_ok() {
                    return Ok(tvec![input]);
                }
                let device_input = if let Some(t) = input.as_arc_tensor() {
                    Arc::clone(t).into_device()?
                } else {
                    input.into_tensor().into_device()?
                };
                Ok(tvec![device_input.into_tensor().into()])
            }
        }
    }
}

impl TypedOp for DeviceSync {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let input = inputs[0];
        match self.kind {
            DeviceSyncKind::ToHost => {
                let mut typed_fact = input
                    .to_device_fact()
                    .with_context(|| {
                        "Cannot sync to Host a tensor without DeviceFact as metadata in its TypedFact"
                    })?
                    .clone()
                    .into_typed_fact();
                if let Some(konst) = input.konst.clone() {
                    if let Some(dt) = konst.as_device_tensor() {
                        typed_fact.konst = Some(dt.to_host()?);
                    } else {
                        typed_fact.konst = Some(konst);
                    }
                }
                Ok(tvec!(typed_fact))
            }
            DeviceSyncKind::ToDevice => {
                ensure!(
                    input.as_device_fact().is_none(),
                    "Cannot sync to Device a tensor already on Device"
                );
                Ok(tvec![DeviceFact::from_host(input.clone())?.into_exotic_fact()])
            }
        }
    }

    as_op!();
}

/// Map node inputs through the translation mapping, inserting DeviceSync nodes
/// where needed to move tensors to/from the device.
pub fn sync_inputs_if_required(
    model: &mut TypedModel,
    node: &TypedNode,
    mapping: &HashMap<OutletId, OutletId>,
    sync_kind: DeviceSyncKind,
) -> TractResult<TVec<OutletId>> {
    let mut mapped_inputs = tvec![];
    for (i_idx, i) in node.inputs.iter().enumerate() {
        let in_fact = model.outlet_fact_mut(mapping[i])?;
        match sync_kind {
            DeviceSyncKind::ToHost if in_fact.as_device_fact().is_some() => {
                mapped_inputs.push(
                    model.wire_node(
                        format!("{}.to-cpu-{i_idx}", node.name),
                        DeviceSync::new(sync_kind),
                        &[mapping[i]],
                    )?[0],
                );
            }
            DeviceSyncKind::ToDevice if in_fact.as_device_fact().is_none() => {
                if let Some(ref konst) = in_fact.konst
                    && konst.as_device_tensor().is_none()
                {
                    let device_konst = konst.as_ref().clone().into_device()?.into_tensor();
                    let device_fact = DeviceFact::from_host(in_fact.clone())?;

                    *in_fact = device_fact.into_exotic_fact();

                    in_fact.konst = Some(Arc::new(device_konst));
                    mapped_inputs.push(mapping[i]);
                    continue;
                }
                ensure!(
                    in_fact.datum_type.is_copy(),
                    "Only copy DatumType can be sync to Device: {:?}",
                    in_fact.datum_type
                );

                mapped_inputs.push(
                    model.wire_node(
                        format!("{}.to-device-{i_idx}", node.name),
                        DeviceSync::new(sync_kind),
                        &[mapping[i]],
                    )?[0],
                );
            }
            _ => mapped_inputs.push(mapping[i]),
        }
    }
    Ok(mapped_inputs)
}

/// Model property key declaring device-resident outputs: a rank-1 i64 tensor
/// of model-output indexes, set through [`declare_device_resident_outputs`]
/// (or `Model::declare_device_resident_outputs` in the rust embedding API).
/// Living on the model, it survives clones and serialization and reaches the
/// GPU runtime transform without any side channel.
pub const DEVICE_RESIDENT_OUTPUTS_PROPERTY: &str = "gpu.device_resident_outputs";

/// Declare model outputs the caller keeps device-resident: they are fed back
/// verbatim as next-step inputs (recurrent/conv states, unfolded KV caches)
/// and never read on host, so the final ToHost sync (a full GPU pipeline
/// stall each) is pure waste and GPU runtimes skip it. The matching outputs
/// then yield opaque device tensors; the ToDevice sync on the paired input
/// passes device tensors through untouched, closing the loop without any host
/// round trip. CPU runtimes ignore the declaration. Declaring an empty set
/// clears a previous declaration. The `TRACT_GPU_DEVICE_RESIDENT_OUTPUTS` env
/// var, when set, overrides the declaration in both directions (see
/// [`is_device_resident_output`]).
pub fn declare_device_resident_outputs(
    model: &mut TypedModel,
    outputs: impl IntoIterator<Item = usize>,
) -> TractResult<()> {
    let mut ixes: Vec<i64> = outputs.into_iter().map(|ix| ix as i64).collect();
    ixes.sort_unstable();
    ixes.dedup();
    let output_count = model.outputs.len();
    if let Some(&last) = ixes.last() {
        ensure!(
            (last as usize) < output_count,
            "device-resident output index {last} out of range (model has {output_count} outputs)"
        );
    }
    if ixes.is_empty() {
        model.properties.remove(DEVICE_RESIDENT_OUTPUTS_PROPERTY);
    } else {
        model
            .properties
            .insert(DEVICE_RESIDENT_OUTPUTS_PROPERTY.to_string(), tensor1(&ixes).into_arc_tensor());
    }
    Ok(())
}

/// Env override for device-resident outputs (escape hatch, highest
/// precedence): `TRACT_GPU_DEVICE_RESIDENT_OUTPUTS` lists model-output
/// indexes as comma-separated entries, `a-b` inclusive ranges allowed (e.g.
/// `1-80,82`). Setting it (even to the empty string, which forces every
/// output back to host) fully replaces any model-level declaration; unset
/// defers to the model property.
fn parse_device_resident_output_spec(spec: &str) -> TractResult<Vec<(usize, usize)>> {
    let mut ranges = vec![];
    for entry in spec.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let range = if let Some((a, b)) = entry.split_once('-') {
            (a.trim().parse::<usize>(), b.trim().parse::<usize>())
        } else {
            let ix = entry.parse::<usize>();
            (ix.clone(), ix)
        };
        let (Ok(a), Ok(b)) = range else {
            bail!(
                "TRACT_GPU_DEVICE_RESIDENT_OUTPUTS: cannot parse entry {entry:?} in {spec:?} \
                 (expected comma-separated output indexes, `a-b` inclusive ranges allowed)"
            );
        };
        ensure!(
            a <= b,
            "TRACT_GPU_DEVICE_RESIDENT_OUTPUTS: empty range {entry:?} in {spec:?} (start > end)"
        );
        ranges.push((a, b));
    }
    Ok(ranges)
}

/// Resolve the effective device-resident output ranges and their source.
/// Precedence: env spec (when set, even empty) > model-level declaration >
/// nothing. Split out from the env read so precedence is unit-testable.
fn resolve_device_resident_output_ranges(
    env_spec: Option<&str>,
    model: &TypedModel,
) -> TractResult<Option<(Vec<(usize, usize)>, &'static str)>> {
    if let Some(spec) = env_spec {
        return Ok(Some((
            parse_device_resident_output_spec(spec)?,
            "TRACT_GPU_DEVICE_RESIDENT_OUTPUTS env override",
        )));
    }
    let Some(t) = model.properties.get(DEVICE_RESIDENT_OUTPUTS_PROPERTY) else {
        return Ok(None);
    };
    let ixes = t.cast_to::<i64>()?;
    let mut ranges = vec![];
    for &ix in ixes.try_as_plain()?.as_slice::<i64>()? {
        ensure!(
            ix >= 0,
            "{DEVICE_RESIDENT_OUTPUTS_PROPERTY}: negative output index {ix} in declaration"
        );
        ranges.push((ix as usize, ix as usize));
    }
    Ok(Some((ranges, "model declaration")))
}

/// True when the caller declared this src-model output device-resident,
/// either through [`declare_device_resident_outputs`] (the supported API) or
/// the `TRACT_GPU_DEVICE_RESIDENT_OUTPUTS` env var (escape hatch, wins over
/// the declaration in both directions when set). Errors (instead of silently
/// keeping the ToHost sync, which would strip the loop-closing behavior for
/// e.g. the logits output) on an unparseable spec or on indexes outside the
/// model's output range.
pub fn is_device_resident_output(src: &TypedModel, outlet: OutletId) -> TractResult<bool> {
    let env_spec = std::env::var("TRACT_GPU_DEVICE_RESIDENT_OUTPUTS").ok();
    is_device_resident_output_with_env(env_spec.as_deref(), src, outlet)
}

fn is_device_resident_output_with_env(
    env_spec: Option<&str>,
    src: &TypedModel,
    outlet: OutletId,
) -> TractResult<bool> {
    let Some((ranges, source)) = resolve_device_resident_output_ranges(env_spec, src)? else {
        return Ok(false);
    };
    if ranges.is_empty() {
        return Ok(false);
    }
    let output_count = src.outputs.len();
    for &(_, b) in &ranges {
        ensure!(
            b < output_count,
            "device-resident outputs ({source}): output index {b} out of range \
             (model has {output_count} outputs)"
        );
    }
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| {
        log::info!("device-resident outputs resolved to ranges {ranges:?} ({source})");
    });
    Ok(src
        .outputs
        .iter()
        .position(|o| *o == outlet)
        .is_some_and(|ix| ranges.iter().any(|(a, b)| (*a..=*b).contains(&ix))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_device_resident_output_spec() -> TractResult<()> {
        assert_eq!(parse_device_resident_output_spec("")?, vec![]);
        assert_eq!(parse_device_resident_output_spec("2")?, vec![(2, 2)]);
        assert_eq!(parse_device_resident_output_spec("1-80,82")?, vec![(1, 80), (82, 82)]);
        assert_eq!(parse_device_resident_output_spec(" 1 - 3 , 5 ")?, vec![(1, 3), (5, 5)]);
        assert!(parse_device_resident_output_spec("1-x").is_err());
        assert!(parse_device_resident_output_spec("abc").is_err());
        assert!(parse_device_resident_output_spec("1;2").is_err());
        assert!(parse_device_resident_output_spec("5-2").is_err());
        Ok(())
    }

    fn model_with_outputs(n: usize) -> TractResult<TypedModel> {
        let mut m = TypedModel::default();
        let mut outs = tvec![];
        for i in 0..n {
            outs.push(m.add_source(format!("s{i}"), f32::fact([2]))?);
        }
        m.select_output_outlets(&outs)?;
        Ok(m)
    }

    #[test]
    fn test_declared_outputs_resolve_device_resident() -> TractResult<()> {
        let mut m = model_with_outputs(3)?;
        // Nothing declared, no env: everything syncs to host.
        assert!(!is_device_resident_output_with_env(None, &m, m.outputs[1])?);
        declare_device_resident_outputs(&mut m, [1, 2])?;
        assert!(!is_device_resident_output_with_env(None, &m, m.outputs[0])?);
        assert!(is_device_resident_output_with_env(None, &m, m.outputs[1])?);
        assert!(is_device_resident_output_with_env(None, &m, m.outputs[2])?);
        // Declaring the empty set clears the previous declaration.
        declare_device_resident_outputs(&mut m, [])?;
        assert!(!m.properties.contains_key(DEVICE_RESIDENT_OUTPUTS_PROPERTY));
        assert!(!is_device_resident_output_with_env(None, &m, m.outputs[1])?);
        Ok(())
    }

    #[test]
    fn test_declare_device_resident_outputs_validates_range() -> TractResult<()> {
        let mut m = model_with_outputs(2)?;
        assert!(declare_device_resident_outputs(&mut m, [2]).is_err());
        Ok(())
    }

    #[test]
    fn test_env_overrides_declaration_both_ways() -> TractResult<()> {
        let mut m = model_with_outputs(3)?;
        declare_device_resident_outputs(&mut m, [1])?;
        // Env set: fully replaces the declaration (force-resident output 2,
        // force-host the declared output 1).
        assert!(!is_device_resident_output_with_env(Some("2"), &m, m.outputs[1])?);
        assert!(is_device_resident_output_with_env(Some("2"), &m, m.outputs[2])?);
        // Env set but empty: forces every output back to host.
        assert!(!is_device_resident_output_with_env(Some(""), &m, m.outputs[1])?);
        // Env unset: the declaration applies.
        assert!(is_device_resident_output_with_env(None, &m, m.outputs[1])?);
        // Out-of-range env index errors instead of silently syncing to host.
        assert!(is_device_resident_output_with_env(Some("5"), &m, m.outputs[0]).is_err());
        Ok(())
    }
}

/// For model outputs that are on device, insert DeviceSync nodes to move them back to host.
pub fn sync_model_outputs_if_required(
    src: &TypedModel,
    node: &TypedNode,
    target: &mut TypedModel,
    target_node_outlet_ids: TVec<OutletId>,
) -> TractResult<TVec<OutletId>> {
    let mut outputs = tvec![];
    for (o_idx, o) in target_node_outlet_ids.into_iter().enumerate() {
        let src_outlet = OutletId::new(node.id, o_idx);
        let is_src_output = src.outputs.contains(&src_outlet);
        if target.outlet_fact(o)?.as_device_fact().is_some()
            && is_src_output
            && !is_device_resident_output(src, src_outlet)?
        {
            let sync_output = target.wire_node(
                format!("{}.to-host-{o_idx}-out", node.name),
                DeviceSync::new(DeviceSyncKind::ToHost),
                &[o],
            )?[0];
            outputs.push(sync_output);
        } else {
            outputs.push(o)
        }
    }
    Ok(outputs)
}
