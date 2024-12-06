use crate::fact::{MetalFact, MetalOrigin, MetalTypedFactExt};
use num_traits::{AsPrimitive, Zero};
use tract_core::internal::*;

#[macro_export]
macro_rules! impl_eval_op_for_metal_op {
    ($op:ty) => {
        impl tract_core::internal::EvalOp for $op {
            fn is_stateless(&self) -> bool {
                false
            }

            #[allow(unused_variables)]
            fn state(
                &self,
                session: &mut tract_core::internal::SessionState,
                node_id: usize,
            ) -> TractResult<Option<Box<dyn OpState>>> {
                Ok(Some(Box::new($crate::ops::MetalOpState::new(node_id, self.clone()))))
            }
        }
    };
}

pub fn metal_facts_from_gpu(
    facts: &[&TypedFact],
    resolve_facts: impl Fn(&[&TypedFact]) -> TractResult<TVec<TypedFact>>,
) -> TractResult<TVec<TypedFact>> {
    if facts.iter().all(|it| it.datum_type == DatumType::Opaque) {
        let metal_facts = facts
            .iter()
            .map(|it| it.to_metal_fact().map(|it| it.as_ref()))
            .collect::<TractResult<TVec<_>>>()?;
        let output_facts = (resolve_facts)(metal_facts.as_slice())?;
        Ok(output_facts
            .into_iter()
            .map(|it| Ok(MetalFact::new(MetalOrigin::FromGpu, it)?.into_opaque_fact()))
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

pub fn metal_facts<'a, 'b: 'a, T>(
    facts: &'a [&'b TypedFact],
    map_facts: impl Fn(&[&'b TypedFact]) -> TractResult<T>,
) -> TractResult<T> {
    if facts.iter().all(|it| it.datum_type == DatumType::Opaque) {
        let metal_facts = facts
            .iter()
            .map(|it| it.to_metal_fact().map(|it| it.as_ref()))
            .collect::<TractResult<TVec<_>>>()?;
        (map_facts)(metal_facts.as_slice())
    } else if facts.iter().all(|it| it.datum_type != DatumType::Opaque) {
        (map_facts)(facts)
    } else {
        bail!(
            "Inconsistent facts datum type: {:?}",
            facts.iter().map(|it| it.datum_type).collect::<Vec<_>>()
        );
    }
}

pub fn metal_fact<'a, T: 'a>(
    fact: &'a TypedFact,
    map_fact: impl Fn(&'a TypedFact) -> TractResult<T>,
) -> TractResult<T> {
    if fact.datum_type == DatumType::Opaque {
        (map_fact)(fact.to_metal_fact()?)
    } else {
        (map_fact)(fact)
    }
}

pub fn compute_broadcast_strides<T: Zero + Copy + 'static>(
    shape: &[usize],
    strides: &[isize],
) -> TractResult<TVec<T>>
where
    isize: AsPrimitive<T>,
{
    ensure!(
        shape.len() == strides.len(),
        "Mistmach between shape and strides length while computing broadcast strides"
    );
    Ok(strides
        .iter()
        .zip(shape)
        .map(|(s, dim)| if *dim == 1 { T::zero() } else { s.as_() })
        .collect::<TVec<T>>())
}

pub fn rescale_gpu_duration(
    pass_duration: u64,
    cpu_start: u64,
    cpu_end: u64,
    gpu_start: u64,
    gpu_end: u64,
) -> u64 {
    let cpu_time_span = cpu_end - cpu_start;
    let gpu_time_span = gpu_end - gpu_start;

    pass_duration * cpu_time_span / gpu_time_span
}
