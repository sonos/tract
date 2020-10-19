use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::*;
use std::hash::Hash;
use tract_hir::internal::*;
use tract_hir::tract_core::itertools::Itertools;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("CategoryMapper", category_mapper);
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
        (Some(def), None) => {
            Box::new(CategoryMapper::new(tract_itertools::zip(strings, ints).collect(), def))
        }
        (None, Some(def)) => Box::new(CategoryMapper::new(
            tract_itertools::zip(ints, strings).collect(),
            def.to_string(),
        )),
    };
    Ok((op, vec![]))
}

#[derive(Clone, new, Debug)]
struct CategoryMapper<Src: Datum + Hash + Eq, Dst: Datum + Hash> {
    hash: HashMap<Src, Dst>,
    default: Dst,
}

impl<Src: Datum + Hash + Eq + Ord, Dst: Datum + Hash> DynHash for CategoryMapper<Src, Dst> {
    fn dyn_hash(&self, hasher: &mut dyn std::hash::Hasher) {
        dyn_hash(self, hasher)
    }
}

impl<Src: Datum + Hash + Eq + Ord, Dst: Datum + Hash> Hash for CategoryMapper<Src, Dst> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.hash.iter().sorted_by_key(|s| s.0).for_each(|v| std::hash::Hash::hash(&v, state));
        std::hash::Hash::hash(&self.default, state)
    }
}

impl<Src: Datum + Hash + Eq + Ord, Dst: Datum + Hash> Op for CategoryMapper<Src, Dst> {
    fn name(&self) -> Cow<str> {
        format!("CategoryMapper<{:?},{:?}>", Src::datum_type(), Dst::datum_type()).into()
    }

    fn op_families(&self) -> &'static [&'static str] {
        &["onnx-ml"]
    }
    op_as_typed_op!();
}

impl<Src: Datum + Hash + Eq + Ord, Dst: Datum + Hash> EvalOp for CategoryMapper<Src, Dst> {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let input = input.to_array_view::<Src>()?;
        let output = input.map(|v| self.hash.get(v).unwrap_or(&self.default).clone());
        Ok(tvec!(output.into_arc_tensor()))
    }
}

impl<Src: Datum + Hash + Eq + Ord, Dst: Datum + Hash> InferenceRulesOp
    for CategoryMapper<Src, Dst>
{
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        s.equals(&inputs[0].datum_type, Src::datum_type())?;
        s.equals(&outputs[0].datum_type, Dst::datum_type())?;
        Ok(())
    }

    as_op!();
    to_typed!();
}

impl<Src: Datum + Hash + Eq + Ord, Dst: Datum + Hash> TypedOp for CategoryMapper<Src, Dst> {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(Dst::datum_type(), inputs[0].shape.clone())?))
    }

    as_op!();
}
