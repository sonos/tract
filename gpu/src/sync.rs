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

/// For model outputs that are on device, insert DeviceSync nodes to move them back to host.
pub fn sync_model_outputs_if_required(
    src: &TypedModel,
    node: &TypedNode,
    target: &mut TypedModel,
    target_node_outlet_ids: TVec<OutletId>,
) -> TractResult<TVec<OutletId>> {
    let mut outputs = tvec![];
    for (o_idx, o) in target_node_outlet_ids.into_iter().enumerate() {
        let is_src_output = src.outputs.contains(&OutletId::new(node.id, o_idx));
        if target.outlet_fact(o)?.as_device_fact().is_some() && is_src_output {
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
