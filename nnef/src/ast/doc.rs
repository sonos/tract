use crate::ast::dump::Dumper;
use std::path::Path;
use crate::ast::*;
use tract_core::internal::*;

pub struct DocDumper<'a> {
    w: &'a mut dyn std::io::Write,
}

impl<'a> DocDumper<'a> {
    pub fn new(w: &'a mut dyn std::io::Write) -> DocDumper {
        DocDumper { w }
    }

    pub fn registry(&mut self, registry: &Registry) -> TractResult<()> {
        // Generate fragment declarations
        Dumper::new(self.w)
            .fragments(registry.fragments.values().cloned().collect::<Vec<_>>().as_slice())?;

        for primitive in registry.primitives.iter() {
            let fragment = FragmentDef {
                decl: FragmentDecl {
                    id: primitive.0.clone(),
                generic_decl: None,
                parameters: primitive.1.0.clone(),
                results: vec![], // we need to expose the results.
            }, body: None};
            Dumper::new(self.w).fragment_def(&fragment)?;
        }
        Ok(())
    }

    pub fn to_directory(path: impl AsRef<Path>, nnef: &Nnef) -> TractResult<()> {
        for registry in nnef.registries.iter() {
            let registry_file = path.as_ref().join(format!("{}.nnef", registry.id));
            let mut file = std::fs::File::create(&registry_file)
                .with_context(|| anyhow!("Error while creating file at path: {:?}", registry_file))?;
            DocDumper::new(&mut file).registry(registry)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn doc_example() -> TractResult<()> {
        let nnef = crate::nnef().with_tract_core().with_tract_resource();
        DocDumper::to_directory(".", &nnef)?;
        Ok(())
    }
}

