use tract_core::internal::*;
use tract_linalg::block_quant::{BlockQuantFact, BlockQuantValue, Q4_0};

use crate::fact::*;

pub fn facts_to_device_facts(
    facts: &[&TypedFact],
    resolve_facts: impl Fn(&[&TypedFact]) -> TractResult<TVec<TypedFact>>,
) -> TractResult<TVec<TypedFact>> {
    if facts.iter().all(|it| it.datum_type == DatumType::Opaque) {
        let device_facts = facts
            .iter()
            .map(|it| it.to_device_fact().map(|it| it.as_ref()))
            .collect::<TractResult<TVec<_>>>()?;
        let output_facts = (resolve_facts)(device_facts.as_slice())?;
        Ok(output_facts
            .into_iter()
            .map(|it| Ok(DeviceFact::new(DeviceTensorOrigin::FromDevice, it)?.into_opaque_fact()))
            .collect::<TractResult<_>>()?)
    } else if facts.iter().all(|it| it.datum_type != DatumType::Opaque) {
        (resolve_facts)(facts)
    } else {
        bail!(
            "Inconsistent facts datum type: {:?}",
            facts.iter().map(|it| it.datum_type).collect::<TVec<_>>()
        );
    }
}

pub fn get_device_facts<'a, 'b: 'a, T>(
    facts: &'a [&'b TypedFact],
    map_facts: impl Fn(&[&'b TypedFact]) -> TractResult<T>,
) -> TractResult<T> {
    if facts.iter().all(|it| it.datum_type == DatumType::Opaque) {
        let device_facts = facts
            .iter()
            .map(|it| it.to_device_fact().map(|it| it.as_ref()))
            .collect::<TractResult<TVec<_>>>()?;
        (map_facts)(device_facts.as_slice())
    } else if facts.iter().all(|it| it.datum_type != DatumType::Opaque) {
        (map_facts)(facts)
    } else {
        bail!(
            "Inconsistent facts datum type: {:?}",
            facts.iter().map(|it| it.datum_type).collect::<Vec<_>>()
        );
    }
}

pub fn get_device_fact<'a, T: 'a>(
    fact: &'a TypedFact,
    map_fact: impl Fn(&'a TypedFact) -> TractResult<T>,
) -> TractResult<T> {
    if fact.datum_type == DatumType::Opaque {
        (map_fact)(fact.to_device_fact()?)
    } else {
        (map_fact)(fact)
    }
}

pub fn as_q40_fact(fact: &TypedFact) -> Option<&BlockQuantFact> {
    fact.opaque_fact
        .as_ref()
        .and_then(|of| of.downcast_ref::<BlockQuantFact>())
        .and_then(|bqf| if bqf.format.same_as(&Q4_0) { Some(bqf) } else { None })
        .or_else(|| {
            fact.konst
                .as_ref()
                .and_then(|k| k.to_scalar::<Opaque>().ok())
                .and_then(|o| o.downcast_ref::<BlockQuantValue>())
                .map(|v| &v.fact)
                .and_then(|bqf| if bqf.format.same_as(&Q4_0) { Some(bqf) } else { None })
        })
}

pub fn as_q40_tensor(a: &Tensor) -> Option<&BlockQuantValue> {
    a.to_scalar::<Opaque>().ok().and_then(|od| {
        od.downcast_ref::<BlockQuantValue>().and_then(|bqv| {
            if bqv.fact.format.same_as(&Q4_0) {
                Some(bqv)
            } else {
                None
            }
        })
    })
}

pub fn check_strides_validity(shape: TVec<usize>, strides: TVec<isize>) -> TractResult<()> {
    let mut zipped_shape_strides: Vec<_> = shape.into_iter().zip(strides).collect();
    zipped_shape_strides.sort_by_key(|&(_, stride)| stride);

    let mut prev_stride = 1;
    for (dim, stride) in zipped_shape_strides {
        ensure!((stride == prev_stride) || (dim == 1), "Invalid strides");
        prev_stride *= dim as isize;
    }
    Ok(())
}
