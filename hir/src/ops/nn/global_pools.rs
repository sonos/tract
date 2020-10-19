use crate::infer::*;
use crate::internal::*;

#[derive(Clone, Debug, new, Hash)]
pub struct GlobalAvgPool;
tract_data::impl_dyn_hash!(GlobalAvgPool);

impl Expansion for GlobalAvgPool {
    fn name(&self) -> Cow<str> {
        "GlobalAvgPool".into()
    }

    op_hir!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        rules(solver, inputs, outputs)
    }

    fn wire(
        &self,
        name: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let input = inputs[0];
        let input_fact = target.outlet_fact(input)?.clone();
        let axes = (2..input_fact.rank()).collect();
        let wire = target.wire_node(
            name.to_string() + ".sum",
            tract_core::ops::nn::Reduce::new(axes, tract_core::ops::nn::Reducer::Sum),
            &[input],
        )?;
        let div =
            tensor0((input_fact.shape.iter().skip(2).maybe_product()?.to_i64()? as f64).recip())
                .cast_to_dt(input_fact.datum_type)?
                .into_owned()
                .broadcast_into_rank(input_fact.rank())?;

        target.wire_node(
            name.to_string() + ".norm",
            tract_core::ops::math::mul::unary(div.into_arc_tensor()),
            &wire,
        )
    }
}

#[derive(Clone, Debug, new, Hash)]
pub struct GlobalLpPool(usize);
tract_data::impl_dyn_hash!(GlobalLpPool);

impl Expansion for GlobalLpPool {
    fn name(&self) -> Cow<str> {
        format!("GlobalL{}Pool", self.0).into()
    }
    op_hir!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        rules(solver, inputs, outputs)
    }

    fn wire(
        &self,
        name: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let input = inputs[0];
        let input_fact = target.outlet_fact(input)?.clone();
        let axes = (2..input_fact.rank()).collect();
        let mut wire = tvec!(input);
        if self.0 == 2 {
            wire = target.wire_node(
                name.to_string() + ".sqr",
                tract_core::ops::math::square(),
                &wire,
            )?;
        } else {
            let pow = tensor0(self.0 as f64)
                .cast_to_dt(input_fact.datum_type)?
                .into_owned()
                .broadcast_into_rank(input_fact.rank())?
                .into_arc_tensor();
            wire = target.wire_node(
                name.to_string() + ".pow",
                tract_core::ops::math::flipped_pow::unary(pow),
                &wire,
            )?;
        }
        wire = target.wire_node(
            name.to_string() + ".sum",
            tract_core::ops::nn::Reduce::new(axes, tract_core::ops::nn::Reducer::Sum),
            &wire,
        )?;
        let div =
            tensor0((input_fact.shape.iter().skip(2).maybe_product()?.to_i64()? as f64).recip())
                .cast_to_dt(input_fact.datum_type)?
                .into_owned()
                .broadcast_into_rank(input_fact.rank())?;
        wire = target.wire_node(
            name.to_string() + ".norm",
            tract_core::ops::math::mul::unary(div.into_arc_tensor()),
            &wire,
        )?;
        if self.0 == 2 {
            wire = target.wire_node(
                name.to_string() + ".sqrt",
                tract_core::ops::math::sqrt(),
                &wire,
            )?;
        } else {
            let anti_pow = tensor0((self.0 as f64).recip())
                .cast_to_dt(input_fact.datum_type)?
                .into_owned()
                .broadcast_into_rank(input_fact.rank())?
                .into_arc_tensor();
            wire = target.wire_node(
                name.to_string() + ".antipow",
                tract_core::ops::math::flipped_pow::unary(anti_pow),
                &wire,
            )?;
        }
        Ok(wire)
    }
}

#[derive(Clone, Debug, new, Hash)]
pub struct GlobalMaxPool;
tract_data::impl_dyn_hash!(GlobalMaxPool);

impl Expansion for GlobalMaxPool {
    fn name(&self) -> Cow<str> {
        "GlobalMaxPool".into()
    }

    op_hir!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        rules(solver, inputs, outputs)
    }

    fn wire(
        &self,
        name: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let input = inputs[0];
        let input_fact = target.outlet_fact(input)?.clone();
        let axes = (2..input_fact.rank()).collect();
        target.wire_node(
            name.to_string() + ".max",
            tract_core::ops::nn::Reduce::new(axes, tract_core::ops::nn::Reducer::Max),
            &[input],
        )
    }
}

fn rules<'r, 'p: 'r, 's: 'r>(
    s: &mut Solver<'r>,
    inputs: &'p [TensorProxy],
    outputs: &'p [TensorProxy],
) -> InferenceResult {
    check_input_arity(&inputs, 1)?;
    check_output_arity(&outputs, 1)?;
    s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
    s.equals(&outputs[0].rank, &inputs[0].rank)?;
    s.equals(&outputs[0].shape[0], &inputs[0].shape[0])?;
    s.equals(&outputs[0].shape[1], &inputs[0].shape[1])?;
    s.given(&inputs[0].rank, move |s, rank| {
        for i in 2..rank {
            s.equals(&outputs[0].shape[i as usize], TDim::from(1))?;
        }
        Ok(())
    })
}
