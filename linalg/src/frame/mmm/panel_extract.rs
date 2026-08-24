use std::fmt::{Debug, Display};
use tract_data::internal::*;

use super::{EagerPackedInput, MMMInputFormat, MMMInputValue};
use crate::pack::PackedFormat;

type Kernel = unsafe fn(input: *const u8, output: *mut u8, k: usize);

#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Hash, Clone)]
pub struct PanelExtractor {
    pub name: String,
    pub from: Box<dyn MMMInputFormat>,
    pub to: PackedFormat,
    pub kernel: Kernel,
    /// False when this build did not compile the extractor's body, its arch not being the one
    /// it was written for. The struct still exists, so it stays enumerable, but it is never
    /// runnable here and calling it bails.
    pub built: bool,
    /// What the instruction set must offer for this extractor to run here at all. Runnability
    /// only: a preference spelled here would also skip the extractor's tests.
    pub isa: crate::isa::IsaReq,
}

impl Debug for PanelExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({:?} -> {:?})", self.name, self.from, self.to)
    }
}

impl Display for PanelExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl PartialEq for PanelExtractor {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && *self.from == *other.from && self.to == other.to
    }
}
impl Eq for PanelExtractor {}

impl PanelExtractor {
    pub fn runnable(&self) -> bool {
        self.built && self.isa.satisfied_by(crate::isa::native())
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct PanelExtractInput {
    pub format: PanelExtractor,
    pub data: EagerPackedInput,
}

impl MMMInputValue for PanelExtractInput {
    fn scratch_panel_buffer_layout(&self) -> Option<std::alloc::Layout> {
        Some(self.format.to.single_panel_layout(self.data.k(), self.format.to.dt.size_of()))
    }
    fn panel_bytes(&self, i: usize, buffer: Option<*mut u8>) -> TractResult<*const u8> {
        let scratch = buffer.unwrap();
        unsafe {
            let source = self.data.packed.as_ptr().add(self.data.panel_bytes * i);
            (self.format.kernel)(source, scratch, self.data.k());
        }
        Ok(scratch)
    }
    fn mn(&self) -> usize {
        self.data.mn()
    }
    fn k(&self) -> usize {
        self.data.k()
    }
    fn format(&self) -> &dyn MMMInputFormat {
        &self.format.to
    }
    fn exotic_fact(&self) -> &dyn ExoticFact {
        self.data.exotic_fact()
    }
    fn extract_at_mn_f16(&self, mn: usize, slice: &mut [f16]) -> TractResult<()> {
        self.data.extract_at_mn_f16(mn, slice)
    }
    fn extract_at_mn_f32(&self, mn: usize, slice: &mut [f32]) -> TractResult<()> {
        self.data.extract_at_mn_f32(mn, slice)
    }
}

impl Display for PanelExtractInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PanelExtract({})", self.data)
    }
}

impl Debug for PanelExtractInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PanelExtract({})", self.data)
    }
}

// A panel extractor whose body is arch asm or intrinsics. The leading ident names the arch it
// was written for, `built` recording whether this build compiled it; `isa(..)` declares what
// the instruction set has to offer on top, in the same vocabulary as the mmm kernels.
#[macro_export]
macro_rules! panel_extractor {
    (arm; $($rest:tt)*) => { panel_extractor!(@ target_arch = "arm"; $($rest)*); };
    (aarch64; $($rest:tt)*) => { panel_extractor!(@ target_arch = "aarch64"; $($rest)*); };
    (x86_64; $($rest:tt)*) => { panel_extractor!(@ target_arch = "x86_64"; $($rest)*); };
    (riscv64; $($rest:tt)*) => { panel_extractor!(@ target_arch = "riscv64"; $($rest)*); };
    (wasm32; $($rest:tt)*) => {
        panel_extractor!(@ all(target_arch = "wasm32", target_feature = "simd128"); $($rest)*);
    };

    ( @ $built:meta;
        $func:path as $id:ident($from:expr, $to: expr)
            $(isa($($isa:ident),+))?
     ) => {
        paste! {
            lazy_static::lazy_static! {
                pub static ref $id: $crate::mmm::PanelExtractor = {
                    use $crate::mmm::MMMInputFormat;
                    let (from, to) = ($from, $to);
                    assert!(from.r() == to.r());
                    $crate::mmm::PanelExtractor {
                        name: stringify!($id).to_string(),
                        from,
                        to,
                        kernel: $func,
                        built: cfg!($built),
                        isa: $crate::isa::IsaReq::ANY
                            $(.needing(&[$($crate::isa::Isa::$isa),+]))?,
                    }
                };
            }

            #[cfg(test)]
            mod [<test_$id>] {
                use super::$id;
                #[test]
                fn repack_0block_1panel() {
                    $crate::frame::mmm::panel_extract::test::test_packing(&$id, 0, 1).unwrap();
                }

                #[test]
                fn repack_1block_0panel() {
                    $crate::frame::mmm::panel_extract::test::test_packing(&$id, 1, 0).unwrap();
                }

                #[test]
                fn repack_1block_1panel() {
                    $crate::frame::mmm::panel_extract::test::test_packing(&$id, 1, 1).unwrap();
                }

                #[test]
                fn repack_2block_1panel() {
                    $crate::frame::mmm::panel_extract::test::test_packing(&$id, 2, 1).unwrap();
                }

                #[test]
                fn repack_1block_2panel() {
                    $crate::frame::mmm::panel_extract::test::test_packing(&$id, 1, 2).unwrap();
                }

                #[test]
                fn repack_2block_2panel() {
                    $crate::frame::mmm::panel_extract::test::test_packing(&$id, 2, 2).unwrap();
                }
            }
        }
    };
}

#[cfg(test)]
pub mod test {
    use crate::frame::block_quant::PackedBlockQuantFormat;
    use crate::mmm::PackedMatrixStorage;
    use tract_ndarray::Array2;

    use super::*;

    pub fn test_packing(
        extractor: &PanelExtractor,
        blocks: usize,
        panels: usize,
    ) -> TractResult<()> {
        if !extractor.runnable() {
            return Ok(());
        }
        assert!(extractor.from.r() == extractor.to.r());
        assert!(extractor.to.dt == f32::datum_type() || extractor.to.dt == f16::datum_type());
        if let Some(from) = extractor.from.downcast_ref::<PackedBlockQuantFormat>() {
            test_packing_bq(extractor, from, blocks, panels)
        } else if let Some(from) = extractor.from.downcast_ref() {
            test_packing_plain(extractor, from, blocks, panels)
        } else {
            todo!()
        }
    }

    pub fn test_packing_plain(
        extractor: &PanelExtractor,
        from: &PackedFormat,
        blocks: usize,
        panels: usize,
    ) -> TractResult<()> {
        let m = from.r * panels;
        let k = 8 * blocks; // 8 is arbitrary
        let to = &extractor.to;
        let weights_orig =
            Array2::from_shape_fn((m, k), |(m, k)| ((m * 31 + k * 17) % 20) as f32 - 10.)
                .into_tensor()
                .cast_to_dt(from.dt)?
                .into_owned();
        let packed_orig = from.prepare_tensor(&weights_orig, 1, 0)?;
        let packed_orig_storage = packed_orig.try_storage_as::<PackedMatrixStorage>()?;
        let packed_orig = packed_orig_storage.value().downcast_ref::<EagerPackedInput>().unwrap();

        for panel in 0..panels {
            let orig_panel = &packed_orig.packed[packed_orig.panel_bytes * panel..]
                [..k * from.r * from.dt.size_of()];
            let mut reference_panel = Tensor::zero_dt(from.dt, &[k, from.r])?;
            reference_panel.as_bytes_mut().copy_from_slice(orig_panel);
            reference_panel = reference_panel.cast_to_dt(to.dt)?.into_owned();

            let mut tested_panel = Tensor::zero_dt(to.dt, &[k, from.r])?;
            unsafe {
                (extractor.kernel)(
                    orig_panel.as_ptr(),
                    tested_panel.as_bytes_mut().as_mut_ptr(),
                    k,
                );
            }
            compare_panels(&tested_panel, &reference_panel, from.r, k);
        }
        Ok(())
    }

    pub fn test_packing_bq(
        extractor: &PanelExtractor,
        from: &PackedBlockQuantFormat,
        blocks: usize,
        panels: usize,
    ) -> TractResult<()> {
        let m = from.r * panels;
        let k = from.bq.block_len() * blocks;
        let to = &extractor.to;
        let weights_orig =
            Array2::from_shape_fn((m, k), |(m, k)| ((m * 31 + k * 17) % 20) as f32 - 10.)
                .into_tensor()
                .cast_to_dt(to.dt)?
                .into_owned();
        let weights = if to.dt == f32::datum_type() {
            from.bq
                .dequant_f32(&from.bq.quant_f32(weights_orig.try_as_plain()?.as_slice::<f32>()?)?)?
                .into_shape(&[m, k])?
        } else {
            from.bq
                .dequant_f16(&from.bq.quant_f16(weights_orig.try_as_plain()?.as_slice::<f16>()?)?)?
                .into_shape(&[m, k])?
        };
        let block_quant = if to.dt == f32::datum_type() {
            from.bq.quant_f32(weights.try_as_plain()?.as_slice::<f32>()?)?
        } else {
            from.bq.quant_f16(weights.try_as_plain()?.as_slice::<f16>()?)?
        };
        let packed_block_quant =
            from.bq.pack(&block_quant, k, from.r, from.zip, from.scales_at_end)?;

        let mut reference_panel = Tensor::zero_dt(to.dt, &[k, from.r])?;
        let mut tested_panel = Tensor::zero_dt(to.dt, &[k, from.r])?;

        for panel in 0..packed_block_quant.panels_count() {
            unsafe {
                from.bq.extract_packed_panel(
                    &packed_block_quant,
                    to,
                    panel,
                    reference_panel.as_bytes_mut().as_mut_ptr(),
                )?;

                let source =
                    packed_block_quant.packed.as_ptr().add(packed_block_quant.panel_bytes * panel);
                (extractor.kernel)(source, tested_panel.as_bytes_mut().as_mut_ptr(), k);
            }
            compare_panels(&tested_panel, &reference_panel, from.r, k);
        }
        Ok(())
    }

    fn compare_panels(tested_panel: &Tensor, reference_panel: &Tensor, r: usize, k: usize) {
        if tested_panel != reference_panel {
            if reference_panel.datum_type() == f32::datum_type() {
                crate::frame::mmm::tests::display_error(
                    tested_panel.try_as_plain().unwrap().as_slice::<f32>().unwrap(),
                    reference_panel.try_as_plain().unwrap().as_slice::<f32>().unwrap(),
                    r,
                    k,
                );
            } else {
                crate::frame::mmm::tests::display_error(
                    tested_panel.try_as_plain().unwrap().as_slice::<f16>().unwrap(),
                    reference_panel.try_as_plain().unwrap().as_slice::<f16>().unwrap(),
                    r,
                    k,
                );
            }
        }
        assert_eq!(tested_panel, reference_panel);
    }
}
