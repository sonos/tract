//! ACE (AI Compute Extensions) — future-looking, emulated support.
//!
//! ACE is the joint AMD/Intel x86 matrix-acceleration ISA standardized through the
//! x86 Ecosystem Advisory Group (whitepaper v1.0, 2026-04-15; spec v1.15). It is an
//! **outer-product** matrix unit — architecturally the x86 cousin of ARM SME's
//! SMOPA and IBM Power10 MMA, *not* a tile×tile engine like Intel AMX. Hardware is
//! not expected before ~2028 (AMD's Zen 7 "Matrix Engine").
//!
//! Because no assembler can encode ACE yet, this module provides a **portable,
//! bit-exact software model** so the surrounding tract integration (packing layout,
//! 16×16 tile geometry, dispatch tier, fused epilogues, correctness tests) is built
//! and validated today and the inner compute swaps for real instructions later.
//!
//! Key integration finding: ACE's int8 operands are consumed from two ZMM registers
//! laid out exactly like tract's existing [`crate::pack::PackedI8K4`] (K=4-inner,
//! `lane = row*4 + kr`). So — unlike AMX, which needed the bespoke `PackedAmxA` —
//! **ACE int8 reuses tract's int8 packing unchanged on both A and B sides.**
//!
//! Layout:
//!   * [`isa`] — the emulated ACE ISA primitives (tile registers, `top4bssd`,
//!     `top2bf16ps`, the MX block-scaled outer products), each annotated with the
//!     real intrinsic it stands in for.
//!   * [`format`] — OCP MX numeric decodes and the `VUNPACKB`/`VPERM` data
//!     marshalling primitives ACE relies on for low-precision format conversion.
//!   * [`mmm`] — matrix-multiply microkernels registered via `MMMRustKernel!`.

pub mod detect;
pub mod format;
pub mod isa;
pub mod mmm;
pub mod packers;

pub use detect::has_ace;
pub use mmm::plug;
