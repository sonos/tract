use crate::internal::*;
use tract_core::internal::*;
use tract_core::ops::konst::Const;

/// `tract_core_named_weight(label, shape, datum_type)`: a weight known by its name. The bundle's
/// tensor of that label when it carries one, else a [`NamedStorage`] of the declared datum type
/// and shape with no source: typed but unread, left for a weight source to be linked to it.
pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_core_named_weight",
        &[
            TypeName::String.named("label"),
            TypeName::Integer.array().named("shape"),
            TypeName::String.named("datum_type"),
        ],
        &[("output", TypeName::Any.tensor())],
        named_weight_load,
    );
}

fn named_weight_load(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let label: String = invocation.named_arg_as(builder, "label")?;
    let shape: TVec<usize> = invocation.named_arg_as(builder, "shape")?;
    let dt: DatumType = invocation.named_arg_as::<String>(builder, "datum_type")?.parse()?;
    let tensors = &builder.proto_model.tensors;
    let tensor = match tensors
        .get(&Identifier(label.clone()))
        .or_else(|| tensors.get(&Identifier(label.trim_start_matches('/').to_owned())))
    {
        Some(t) => {
            ensure!(
                t.datum_type() == dt && t.shape() == &*shape,
                "Weight {label:?} is {:?} {:?} in the bundle, the graph declares {dt:?} {shape:?}",
                t.datum_type(),
                t.shape()
            );
            Arc::clone(t)
        }
        None => NamedStorage::new(&label, dt, &shape, None).into_tensor().into_arc_tensor(),
    };
    builder.wire(Const::new(tensor)?, &[])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::ProtoModel;
    use crate::ast::parse::parse_document;

    const GRAPH: &str = "version 1.0;\nextension tract_registry tract_core;\n\
        graph network() -> (w) {\n  w = tract_core_named_weight(label = \"w\", shape = [2], datum_type = \"F16\");\n}\n";

    fn load(tensors: Vec<(&str, Tensor)>) -> TractResult<TypedModel> {
        let tensors =
            tensors.into_iter().map(|(n, t)| (Identifier::from(n), t.into_arc_tensor())).collect();
        let proto = ProtoModel {
            doc: parse_document(GRAPH)?,
            tensors,
            quantization: None,
            resources: Default::default(),
        };
        crate::nnef().model_for_proto_model(&proto)
    }

    fn weight(model: &TypedModel) -> Arc<Tensor> {
        model.nodes().iter().find_map(|n| n.op_as::<Const>()).unwrap().val().clone()
    }

    #[test]
    fn bundle_data_is_used() -> TractResult<()> {
        let w = tensor1(&[f16::from_f32(1.5), f16::from_f32(-2.0)]);
        assert_eq!(*weight(&load(vec![("w", w.clone())])?), w);
        Ok(())
    }

    #[test]
    fn bundle_data_must_match_the_declaration() {
        assert!(load(vec![("w", tensor1(&[1.5f32, -2.0]))]).is_err());
    }

    #[test]
    fn missing_data_is_a_named_weight() -> TractResult<()> {
        let model = load(vec![])?;
        let w = weight(&model);
        let named = w.storage_as::<NamedStorage>().context("not a named weight")?;
        assert_eq!(named.name(), "w");
        assert!(!named.is_materialized());
        assert_eq!((w.datum_type(), w.shape()), (f16::datum_type(), &[2usize][..]));
        Ok(())
    }
}
