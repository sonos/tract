mod tflite_generated;
use crate::tflite_generated::tflite::Model;
pub struct TFLiteModel<'model> {
    pub model: Model<'model>,
}

impl TFLiteModel<'_> {
    pub fn new(model_file: &[u8]) -> TFLiteModel {
        unsafe {
            let table = flatbuffers::Table::new(model_file, 28);
            let model = Model::init_from_table(table);
            TFLiteModel { model }
        }
    }
}
