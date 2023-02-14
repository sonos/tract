use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::*;
use tract_hir::internal::*;
use tract_onnx_opl::ml::*;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("CategoryMapper", category_mapper);
}

#[derive(Debug, Clone, Hash)]
struct CategoryMapper {
    pub from: Arc<Tensor>,
    pub to: Arc<Tensor>,
    pub fallback: Arc<Tensor>,
}



impl Expansion for CategoryMapper {
    fn name(&self) -> Cow<str> {
        "CategoryMapper".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 1)?;
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        s.equals(&inputs[0].datum_type, self.from.datum_type())?;
        s.equals(&outputs[0].datum_type, self.to.datum_type())?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let wire = model.wire_node(
            format!("{prefix}.reverse"),
            ReverseLookup::new(self.from.clone(), -1)?,
            inputs,
        )?;
        model.wire_node(
            format!("{prefix}.direct"),
            DirectLookup::new(self.to.clone(), self.fallback.clone())?,
            &wire,
        )
    }
}

fn category_mapper(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let ints: Vec<i64> = node.get_attr_vec("cats_int64s")?;
    let strings: Vec<String> = node.get_attr_vec("cats_strings")?;
    let default_int: Option<i64> = node.get_attr_opt("default_int64")?;
    let default_string: Option<String> = node.get_attr_opt("default_string")?;
    let op: Box<dyn InferenceOp> = match (default_int, default_string.as_ref()) {
        (None, None) | (Some(_), Some(_)) => bail!(
            "CategoryMapper requires exactly one of default_int64 and default_string (found {:?})",
            (default_int, default_string)
        ),
        (Some(def), None) => expand(CategoryMapper {
            from: rctensor1(&strings),
            to: rctensor1(&ints),
            fallback: rctensor0(def),
        }),
        (None, Some(def)) => expand(CategoryMapper {
            from: rctensor1(&ints),
            to: rctensor1(&strings),
            fallback: rctensor0(def.clone()),
        }),
    };
    Ok((op, vec![]))
}
