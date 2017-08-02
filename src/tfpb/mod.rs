#![allow(unknown_lints)]
#![allow(clippy)]

#![cfg_attr(rustfmt, rustfmt_skip)]

#![allow(box_pointers)]
#![allow(dead_code)]
#![allow(missing_docs)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(trivial_casts)]
#![allow(unsafe_code)]
#![allow(unused_imports)]
#![allow(unused_results)]

pub mod attr_value { include!(concat!(env!("OUT_DIR"), "/attr_value.rs")); }
pub mod function { include!(concat!(env!("OUT_DIR"), "/function.rs")); }
pub mod graph { include!(concat!(env!("OUT_DIR"), "/graph.rs")); }
pub mod node_def { include!(concat!(env!("OUT_DIR"), "/node_def.rs")); }
pub mod op_def { include!(concat!(env!("OUT_DIR"), "/op_def.rs")); }
pub mod resource_handle { include!(concat!(env!("OUT_DIR"), "/resource_handle.rs")); }
pub mod tensor { include!(concat!(env!("OUT_DIR"), "/tensor.rs")); }
pub mod tensor_shape { include!(concat!(env!("OUT_DIR"), "/tensor_shape.rs")); }
pub mod types { include!(concat!(env!("OUT_DIR"), "/types.rs")); }
pub mod versions { include!(concat!(env!("OUT_DIR"), "/versions.rs")); }

