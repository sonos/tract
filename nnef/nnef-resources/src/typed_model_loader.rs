use tract_nnef::internal::*;
use std::path::Path;

pub struct TypedModelLoader {
    pub nnef: tract_nnef::framework::Nnef,
    pub optimized_model: bool,
}

impl TypedModelLoader {
    pub fn new(optimized_model: bool) -> Self {
        Self {
            nnef: tract_nnef::nnef(),
            optimized_model,
        }
    }

    pub fn with_tract_core(mut self) -> Self {
        self.nnef.registries.push(tract_nnef::ops::tract_core());
        self
    }

    pub fn with_tract_resource(mut self) -> Self {
        self.nnef.registries.push(tract_nnef::ops::tract_resource());
        self
    } 

    pub fn with_registry(mut self, registry: Registry) -> Self {
        self.nnef.registries.push(registry);
        self
    }
}

impl ResourceLoader for TypedModelLoader {
    fn name(&self) -> Cow<str> {
        "TypedModelLoader".into()
    }

    fn try_load(
        &self,
        path: &Path,
        reader: &mut dyn std::io::Read,
    ) -> TractResult<Option<(String, Arc<dyn Resource>)>> {
        const NNEF_TGZ: &str = ".nnef.tgz";

        if path.to_str().unwrap_or("").ends_with(NNEF_TGZ) {
            let model = if self.optimized_model {
                self.nnef
                    .model_for_read(reader)?
                    .into_optimized()?
            } else {
                self.nnef
                    .model_for_read(reader)?
            };

            let label = path
                .to_str()
                .ok_or_else(|| anyhow!("invalid model resource path"))?
                .trim_end_matches(NNEF_TGZ);
            Ok(Some((
                tract_nnef::resource::resource_path_to_id(label)?,
                Arc::new(TypedModelResource(model)),
            )))
        } else {
            Ok(None)
        }
    }
}

#[derive(Debug, Clone)]
pub struct TypedModelResource(pub TypedModel);

impl Resource for TypedModelResource {}