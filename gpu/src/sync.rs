use crate::fact::{DeviceFact, DeviceTypedFactExt};
use crate::tensor::{DeviceTensorExt, IntoDevice};
use derive_new::new;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use tract_core::internal::*;
use tract_core::transform::ModelTransform;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceSyncKind {
    ToHost,
    ToDevice,
    ToDeviceOrPass,
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
    op_out_of_plan!();

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
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
                let device_input = if let Some(t) = input.as_arc_tensor() {
                    Arc::clone(t).into_device()?
                } else {
                    input.into_tensor().into_device()?
                };
                Ok(tvec![device_input.into_tensor().into()])
            }
            DeviceSyncKind::ToDeviceOrPass => {
                if input.to_device_tensor().is_ok() {
                    Ok(tvec![input])
                } else {
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
            DeviceSyncKind::ToDeviceOrPass => {
                if input.as_device_fact().is_some() {
                    Ok(tvec![input.clone()])
                } else {
                    Ok(tvec![DeviceFact::from_host(input.clone())?.into_exotic_fact()])
                }
            }
        }
    }

    as_op!();
}

/// Map node inputs through the translation mapping, inserting DeviceSync nodes
/// where needed to move tensors to/from the device.
pub fn sync_inputs_if_required(
    src: &TypedModel,
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
                let sync_kind = if is_device_resident_input(src, *i)? {
                    DeviceSyncKind::ToDeviceOrPass
                } else {
                    DeviceSyncKind::ToDevice
                };
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

pub const DEVICE_RESIDENT_OUTPUTS_PROPERTY: &str = "gpu.device_resident_outputs";
pub const DEVICE_RESIDENT_INPUTS_PROPERTY: &str = "gpu.device_resident_inputs";

#[derive(Debug, Default, serde::Deserialize)]
pub struct DeviceResidentOutputsConfig {
    #[serde(default)]
    pub inputs: Vec<usize>,
    #[serde(default)]
    pub outputs: Vec<usize>,
}

#[derive(Debug)]
struct DeviceResidentOutputsTransform(DeviceResidentOutputsConfig);

impl ModelTransform for DeviceResidentOutputsTransform {
    fn name(&self) -> StaticName {
        "gpu_device_resident_outputs".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        declare_device_resident_io(
            model,
            self.0.inputs.iter().copied(),
            self.0.outputs.iter().copied(),
        )
    }
}

register_model_transform!("gpu_device_resident_outputs", DeviceResidentOutputsConfig, |config| Ok(
    Box::new(DeviceResidentOutputsTransform(config))
));
register_model_transform!("gpu_device_resident_io", DeviceResidentOutputsConfig, |config| Ok(
    Box::new(DeviceResidentOutputsTransform(config))
));

fn declare_device_resident_indexes(
    model: &mut TypedModel,
    property: &str,
    len: usize,
    ixes: impl IntoIterator<Item = usize>,
    kind: &str,
) -> TractResult<()> {
    let mut ixes: Vec<i64> = ixes
        .into_iter()
        .map(|ix| {
            i64::try_from(ix).with_context(|| format!("{kind} index {ix} does not fit in i64"))
        })
        .collect::<TractResult<_>>()?;
    ixes.sort_unstable();
    ixes.dedup();
    if let Some(&last) = ixes.last() {
        ensure!(
            (last as usize) < len,
            "device-resident {kind} index {last} out of range (model has {len} {kind}s)"
        );
    }
    if ixes.is_empty() {
        model.properties.remove(property);
    } else {
        model.properties.insert(property.to_string(), tensor1(&ixes).into_arc_tensor());
    }
    Ok(())
}

pub fn declare_device_resident_io(
    model: &mut TypedModel,
    inputs: impl IntoIterator<Item = usize>,
    outputs: impl IntoIterator<Item = usize>,
) -> TractResult<()> {
    declare_device_resident_inputs(model, inputs)?;
    declare_device_resident_outputs(model, outputs)?;
    Ok(())
}

pub fn declare_device_resident_inputs(
    model: &mut TypedModel,
    inputs: impl IntoIterator<Item = usize>,
) -> TractResult<()> {
    declare_device_resident_indexes(
        model,
        DEVICE_RESIDENT_INPUTS_PROPERTY,
        model.inputs.len(),
        inputs,
        "input",
    )
}

pub fn declare_device_resident_outputs(
    model: &mut TypedModel,
    outputs: impl IntoIterator<Item = usize>,
) -> TractResult<()> {
    declare_device_resident_indexes(
        model,
        DEVICE_RESIDENT_OUTPUTS_PROPERTY,
        model.outputs.len(),
        outputs,
        "output",
    )
}

fn is_device_resident_outlet(
    src: &TypedModel,
    property: &str,
    outlets: &[OutletId],
    outlet: OutletId,
) -> TractResult<bool> {
    let Some(t) = src.properties.get(property) else { return Ok(false) };
    let ixes = t.cast_to::<i64>()?;
    let count = outlets.len();
    let ixes = ixes.try_as_plain()?.as_slice::<i64>()?;
    for &ix in ixes {
        ensure!(
            ix >= 0 && (ix as usize) < count,
            "{property}: index {ix} out of range (model has {count} entries)"
        );
    }
    Ok(outlets.iter().enumerate().any(|(ix, o)| *o == outlet && ixes.contains(&(ix as i64))))
}

pub fn is_device_resident_input(src: &TypedModel, outlet: OutletId) -> TractResult<bool> {
    is_device_resident_outlet(src, DEVICE_RESIDENT_INPUTS_PROPERTY, &src.inputs, outlet)
}

pub fn is_device_resident_output(src: &TypedModel, outlet: OutletId) -> TractResult<bool> {
    is_device_resident_outlet(src, DEVICE_RESIDENT_OUTPUTS_PROPERTY, &src.outputs, outlet)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(!is_device_resident_output(&m, m.outputs[1])?);
        declare_device_resident_outputs(&mut m, [1, 2])?;
        assert!(!is_device_resident_output(&m, m.outputs[0])?);
        assert!(is_device_resident_output(&m, m.outputs[1])?);
        assert!(is_device_resident_output(&m, m.outputs[2])?);
        declare_device_resident_outputs(&mut m, [])?;
        assert!(!m.properties.contains_key(DEVICE_RESIDENT_OUTPUTS_PROPERTY));
        assert!(!is_device_resident_output(&m, m.outputs[1])?);
        Ok(())
    }

    #[test]
    fn test_declared_inputs_resolve_device_resident() -> TractResult<()> {
        let mut m = model_with_outputs(3)?;
        assert!(!is_device_resident_input(&m, m.inputs[1])?);
        declare_device_resident_inputs(&mut m, [1, 2])?;
        assert!(!is_device_resident_input(&m, m.inputs[0])?);
        assert!(is_device_resident_input(&m, m.inputs[1])?);
        assert!(is_device_resident_input(&m, m.inputs[2])?);
        declare_device_resident_inputs(&mut m, [])?;
        assert!(!m.properties.contains_key(DEVICE_RESIDENT_INPUTS_PROPERTY));
        assert!(!is_device_resident_input(&m, m.inputs[1])?);
        Ok(())
    }

    #[test]
    fn test_declared_duplicate_output_resolves_by_any_declared_index() -> TractResult<()> {
        let mut m = model_with_outputs(2)?;
        let duplicated = m.outputs[0];
        m.select_output_outlets(&[duplicated, duplicated])?;
        declare_device_resident_outputs(&mut m, [1])?;
        assert!(is_device_resident_output(&m, duplicated)?);
        Ok(())
    }

    #[test]
    fn test_declare_device_resident_outputs_validates_range() -> TractResult<()> {
        let mut m = model_with_outputs(2)?;
        assert!(declare_device_resident_outputs(&mut m, [2]).is_err());
        Ok(())
    }

    #[test]
    fn test_declare_device_resident_inputs_validates_range() -> TractResult<()> {
        let mut m = model_with_outputs(2)?;
        assert!(declare_device_resident_inputs(&mut m, [2]).is_err());
        Ok(())
    }

    #[test]
    fn test_device_resident_outputs_transform() -> TractResult<()> {
        let mut m = model_with_outputs(3)?;
        let mut config = <dyn erased_serde::Deserializer>::erase(serde_json::json!({
            "inputs": [1],
            "outputs": [0, 2],
        }));
        let transform = tract_core::transform::get_transform_with_params(
            "gpu_device_resident_outputs",
            &mut config,
        )?
        .context("device-resident output transform was not registered")?;
        transform.transform(&mut m)?;
        assert!(!is_device_resident_input(&m, m.inputs[0])?);
        assert!(is_device_resident_input(&m, m.inputs[1])?);
        assert!(is_device_resident_output(&m, m.outputs[0])?);
        assert!(!is_device_resident_output(&m, m.outputs[1])?);
        assert!(is_device_resident_output(&m, m.outputs[2])?);
        Ok(())
    }

    #[test]
    fn test_to_device_or_pass_accepts_device_facts() -> TractResult<()> {
        let host = f32::fact([2]);
        let device = DeviceFact::from_host(host.clone())?.into_exotic_fact();
        let sync = DeviceSync::new(DeviceSyncKind::ToDeviceOrPass);

        let from_host = sync.output_facts(&[&host])?;
        assert!(from_host[0].as_device_fact().is_some());

        let from_device = sync.output_facts(&[&device])?;
        assert_eq!(from_device[0], device);
        Ok(())
    }

    #[test]
    fn test_strict_to_device_rejects_device_facts() -> TractResult<()> {
        let host = f32::fact([2]);
        let device = DeviceFact::from_host(host)?.into_exotic_fact();
        let sync = DeviceSync::new(DeviceSyncKind::ToDevice);

        assert!(sync.output_facts(&[&device]).is_err());
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
