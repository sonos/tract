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

/// Model outputs the caller keeps device-resident: they are fed back verbatim
/// as next-step inputs (recurrent/conv states, unfolded KV caches) and never
/// read on host, so the ToHost sync (a full GPU pipeline stall each) is pure
/// waste. `TRACT_GPU_DEVICE_RESIDENT_OUTPUTS` lists model-output indexes as
/// comma-separated entries, `a-b` inclusive ranges allowed (e.g. `1-80,82`).
/// The matching outputs then yield opaque device tensors; the ToDevice sync
/// on the paired input passes device tensors through untouched, closing the
/// loop without any host round trip. Empty/unset keeps every output on host.
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

fn device_resident_output_indexes() -> TractResult<Vec<(usize, usize)>> {
    let Ok(spec) = std::env::var("TRACT_GPU_DEVICE_RESIDENT_OUTPUTS") else {
        return Ok(vec![]);
    };
    parse_device_resident_output_spec(&spec)
}

/// True when the caller declared this src-model output device-resident.
/// Errors (instead of silently keeping the ToHost sync, which would strip
/// the loop-closing behavior for e.g. the logits output) on an unparseable
/// spec or on indexes outside the model's output range.
pub fn is_device_resident_output(src: &TypedModel, outlet: OutletId) -> TractResult<bool> {
    let ranges = device_resident_output_indexes()?;
    if ranges.is_empty() {
        return Ok(false);
    }
    let output_count = src.outputs.len();
    for &(_, b) in &ranges {
        ensure!(
            b < output_count,
            "TRACT_GPU_DEVICE_RESIDENT_OUTPUTS: output index {b} out of range \
             (model has {output_count} outputs)"
        );
    }
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| {
        log::info!("TRACT_GPU_DEVICE_RESIDENT_OUTPUTS resolved to output ranges {ranges:?}");
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
