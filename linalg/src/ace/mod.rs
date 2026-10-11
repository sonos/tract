//! ACE (AI Compute Extensions) v1 support.
//!
//! ACE is the joint AMD/Intel x86 matrix ISA standardized by the x86 Ecosystem
//! Advisory Group (specification v1.15): an **outer-product** unit that consumes
//! two ZMM registers and accumulates a 16×16 tile, plus a Block Scale Register for
//! OCP MX formats. No silicon ships it yet; Intel SDE 10.13.1 emulates it with
//! `-future-ag`.
//!
//! ACE int8 operands are laid out exactly like tract's [`crate::pack::PackedI8K4`]
//! (K=4-inner, `lane = row*4 + k`), so the int8 kernel reuses that packing.
//!
//! Layout:
//!   * [`isa`] — bit-exact reference model of the tile instructions, used as the
//!     compute path when [`has_ace`] is false and as the oracle for [`hw`].
//!   * `hw` (x86_64) — the same tile computations on real ACE instructions.
//!   * [`format`] — OCP MX numeric decodes/encoders and the `VUNPACKB`/`VPERM`
//!     marshalling models.
//!   * [`packers`] — bf16 K=2-inner and MX scaled-block packers.
//!   * [`mmm`] — the matrix-multiply kernels, dispatching to `hw` or `isa`.

pub mod detect;
pub mod format;
#[cfg(target_arch = "x86_64")]
pub mod hw;
pub mod isa;
pub mod mmm;
pub mod packers;

pub use detect::has_ace;
pub use mmm::plug;
