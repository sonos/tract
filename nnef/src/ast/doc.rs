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

        // Generate Primitive declarations
        for primitive in registry.primitives.values().sorted_by_key(|v| &v.decl.id) {
            primitive.doc.iter().flatten()
                .try_for_each(|d| writeln!(self.w, "# {}", d))?;
            
            Dumper::new(self.w).fragment_decl(&primitive.decl)?;
            writeln!(self.w, ";\n")?;
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
    use temp_dir::TempDir;

    #[test]
    fn doc_example() -> TractResult<()> {
        let d = TempDir::new()?;
        let nnef = crate::nnef().with_tract_core().with_tract_resource();
        DocDumper::to_directory(d.path(), &nnef)?;
        Ok(())
    }
}

