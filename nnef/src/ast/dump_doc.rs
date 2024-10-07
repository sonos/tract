use crate::ast::dump::Dumper;
use crate::ast::*;
use std::path::Path;
use tract_core::internal::*;

pub struct DocDumper<'a> {
    w: &'a mut dyn std::io::Write,
}

impl DocDumper<'_> {
    pub fn new(w: &mut dyn std::io::Write) -> DocDumper {
        DocDumper { w }
    }

    pub fn registry(&mut self, registry: &Registry) -> TractResult<()> {
        // Write registry docstrings.
        for d in registry.docstrings.iter().flatten() {
            writeln!(self.w, "# {d}")?;
        }
        writeln!(self.w)?;
        // Generate and write unit element wise op.
        for unit_el_wise_op in registry.unit_element_wise_ops.iter() {
            // we are assuming function names will not exhibit crazy node name weirdness, so we can
            // dispense with escaping
            writeln!(
                self.w,
                "fragment {}( x: tensor<scalar> ) -> (y: tensor<scalar>);",
                &unit_el_wise_op.0 .0
            )?;
        }
        writeln!(self.w)?;

        // Generate and write element wise op.
        for el_wise_op in registry.element_wise_ops.iter() {
            let fragment_decl = FragmentDecl {
                id: el_wise_op.0.clone(),
                generic_decl: None,
                parameters: el_wise_op.3.clone(),
                results: vec![Result_ { id: "output".into(), spec: TypeName::Any.tensor() }],
            };
            Dumper::new(&Nnef::default(), self.w).with_doc().fragment_decl(&fragment_decl)?;
        }
        // Generate and write Primitive declarations.
        for primitive in registry.primitives.values().sorted_by_key(|v| &v.decl.id) {
            primitive.docstrings.iter().flatten().try_for_each(|d| writeln!(self.w, "# {d}"))?;

            Dumper::new(&Nnef::default(), self.w).with_doc().fragment_decl(&primitive.decl)?;
            writeln!(self.w, ";\n")?;
        }

        // Generate and write fragment declarations
        Dumper::new(&Nnef::default(), self.w)
            .with_doc()
            .fragments(registry.fragments.values().cloned().collect::<Vec<_>>().as_slice())?;

        Ok(())
    }

    pub fn registry_to_path(path: impl AsRef<Path>, registry: &Registry) -> TractResult<()> {
        let mut file = std::fs::File::create(path.as_ref())
            .with_context(|| anyhow!("Error while creating file at path: {:?}", path.as_ref()))?;
        DocDumper::new(&mut file).registry(registry)
    }

    pub fn to_directory(path: impl AsRef<Path>, nnef: &Nnef) -> TractResult<()> {
        for registry in nnef.registries.iter() {
            let registry_file = path.as_ref().join(format!("{}.nnef", registry.id.0));
            let mut file = std::fs::File::create(&registry_file).with_context(|| {
                anyhow!("Error while creating file at path: {:?}", registry_file)
            })?;
            DocDumper::new(&mut file).registry(registry)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use temp_dir::TempDir;

    #[test]
    fn doc_example() -> TractResult<()> {
        let d = TempDir::new()?;
        let nnef = crate::nnef().with_tract_core().with_tract_resource();
        DocDumper::to_directory(d.path(), &nnef)?;
        Ok(())
    }

    #[test]
    fn doc_registry() -> TractResult<()> {
        let mut registry = Registry::new("test_doc")
            .with_doc("test_doc registry gather all the needed primitives")
            .with_doc("to test the documentation dumper");
        registry.register_primitive(
            "tract_primitive",
            &[TypeName::Integer.tensor().named("input")],
            &[("output", TypeName::Scalar.tensor())],
            |_, _| panic!("No deserialization needed"),
        );
        let mut docbytes = vec![];
        let mut dumper = DocDumper::new(&mut docbytes);
        dumper.registry(&registry)?;
        let docstring = String::from_utf8(docbytes)?;
        assert_eq!(
            docstring,
            r#"# test_doc registry gather all the needed primitives
# to test the documentation dumper


fragment tract_primitive(
    input: tensor<integer>
) -> (output: tensor<scalar>);

"#
        );
        Ok(())
    }
}
