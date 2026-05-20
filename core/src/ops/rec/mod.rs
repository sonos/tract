//! Recurrent ops — fused implementations that avoid per-timestep plan
//! re-entry.
//!
//! Today this hosts `OptGru` (A2 from the ORT-Web-vs-tract investigation).
//! Future additions: `OptLstm`, fused `OptRnn`.

mod opt_gru;

pub use opt_gru::OptGru;
