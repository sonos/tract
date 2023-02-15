use crate::model::ParsingContext;
use crate::ops::OnnxOpRegister;
use crate::pb::*;
use tract_hir::internal::*;
use tract_onnx_opl::random::Dist;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("RandomUniform", random);
    reg.insert("RandomUniformLike", random);
    reg.insert("RandomNormal", random);
    reg.insert("RandomNormalLike", random);
}

pub fn random(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let dt: Option<DatumType> = node.get_attr_opt("dtype")?;
    let seed = node.get_attr_opt::<f32>("seed")?;

    let dist = if node.op_type.starts_with("RandomNormal") {
        Dist::Normal {
            mean: rctensor0(node.get_attr::<f32>("mean").unwrap_or(0.0)),
            dev: rctensor0(node.get_attr::<f32>("scale").unwrap_or(1.0)),
        }
    } else {
        Dist::Uniform {
            low: rctensor0(node.get_attr::<f32>("low").unwrap_or(0.0)),
            high: rctensor0(node.get_attr::<f32>("high").unwrap_or(1.0)),
        }
    };

    if node.name.ends_with("Like") {
        Ok((expand(RandomLike { dt, dist, seed }), vec![]))
    } else {
        let shape = node.get_attr_slice::<i64>("shape")?.iter().map(|i| i.to_dim()).collect();
        Ok((expand(Random { dt: dt.unwrap_or(DatumType::F32), dist, shape, seed }), vec![]))
    }
}

#[derive(Debug, Clone)]
struct Random {
    dt: DatumType,
    dist: Dist,
    shape: TVec<TDim>,
    seed: Option<f32>,
}

impl Expansion for Random {
    fn name(&self) -> Cow<str> {
        "Random".into()
    }

    fn validation(&self) -> Validation {
        Validation::Random
    }

    fn is_stateless(&self) -> bool {
        false
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 0)?;
        check_output_arity(outputs, 1)?;

        s.equals(&outputs[0].shape, self.shape.clone())?;
        s.equals(&outputs[0].datum_type, self.dt)?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        _inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        model.wire_node(
            prefix,
            tract_onnx_opl::random::Random {
                dist: self.dist.clone(),
                fact: self.dt.fact(&self.shape),
                seed: self.seed.map(|f| f.to_bits() as u64),
            },
            &[],
        )
    }
}

#[derive(Debug, Clone)]
struct RandomLike {
    dt: Option<DatumType>,
    dist: Dist,
    seed: Option<f32>,
}

impl Expansion for RandomLike {
    fn name(&self) -> Cow<str> {
        "RandomLike".into()
    }

    fn validation(&self) -> Validation {
        Validation::Random
    }

    fn is_stateless(&self) -> bool {
        false
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 1)?;
        check_output_arity(outputs, 1)?;

        s.equals(&outputs[0].shape, &inputs[0].shape)?;
        if let Some(dt) = self.dt {
            s.equals(&outputs[0].datum_type, dt)?;
        } else {
            s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        }
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let mut fact = model.outlet_fact(inputs[0])?.without_value();
        if let Some(dt) = self.dt {
            fact.datum_type = dt;
        }
        model.wire_node(
            prefix,
            tract_onnx_opl::random::Random {
                dist: self.dist.clone(),
                fact,
                seed: self.seed.map(|f| f.to_bits() as u64),
            },
            &[],
        )
    }
}
