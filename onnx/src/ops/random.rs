use crate::ops::OnnxOpRegister;
use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;
use tract_onnx_opl::random::Dist;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("RandomUniform", random);
    reg.insert("RandomNormal", random);
}

pub fn random(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let dt: Option<DatumType> = node.get_attr_opt("dtype")?;
    let seed = node.get_attr_opt::<f32>("seed")?;

    let dist = if node.name.starts_with("RandomNormal") {
        Dist::Normal {
            mean: tensor0(node.get_attr::<f32>("mean").unwrap_or(0.0)),
            dev: tensor0(node.get_attr::<f32>("scale").unwrap_or(1.0)),
        }
    } else {
        Dist::Uniform {
            low: tensor0(node.get_attr::<f32>("low").unwrap_or(0.0)),
            high: tensor0(node.get_attr::<f32>("high").unwrap_or(1.0)),
        }
    };

    if node.name.ends_with("Like") {
        todo!();
        // Ok((expand(Random { dt, dist, shape }), vec![]))
    } else {
        let shape = node.get_attr_slice::<i64>("shape")?.iter().map(|i| i.to_dim()).collect();
        Ok((expand(Random { dt: dt.unwrap_or(f32::datum_type()), dist, shape, seed }), vec![]))
    }
}

#[derive(Debug, Clone, Educe)]
#[educe(Hash)]
struct Random {
    dt: DatumType,
    dist: Dist,
    shape: TVec<TDim>,
    #[educe(Hash(method = "hash_opt_f32"))]
    seed: Option<f32>,
}

impl_dyn_hash!(Random);

impl Expansion for Random {
    fn name(&self) -> Cow<str> {
        "Random".into()
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
