pub mod json_loader;
pub mod typed_model_loader;

pub mod internal {
    pub use crate::json_loader::{JsonLoader, JsonResource};
    pub use crate::typed_model_loader::{TypedModelLoader, TypedModelResource};
}
