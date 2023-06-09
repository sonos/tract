#![allow(dead_code)]

mod model;
mod ops;
mod registry;
mod tensors;

#[allow(unused_imports,clippy::extra_unused_lifetimes,clippy::missing_safety_doc,clippy::derivable_impls,clippy::needless_lifetimes)]
mod tflite_generated;
pub use tflite_generated::tflite;
    
pub use model::Tflite;

pub mod prelude {
}

pub mod internal {
    pub use tract_hir::internal::*;
    pub use crate::model::TfliteProtoModel;
}

pub fn tflite() -> Tflite {
    Tflite::default()
}

/*
use crate::tflite_generated::tflite::Model as ModelBuffer;
impl ModelBuffer {
    pub fn from_file(path: P) -> Result<ModelBuffer, Error> {
        let model_file = &*fs::read(model_file_path)?;
        let mut buffer = Vec::new();
        let table = unsafe { flatbuffers::Table::new(&model_file, 28) };
        let model = unsafe { Model::init_from_table(table) };
        Ok(model)
    }
}

//#[derive(Clone, Default)]
//pub struct TFLiteOpRegister(pub HashMap<u8, OpBuilder)
pub struct TFLiteModel<'model> {
    pub model: TFLiteOpRegister,
}

impl TFLite<'_> {}
*/
