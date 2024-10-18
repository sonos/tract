use std::fmt::{Debug, Display};

use super::pack::PackedFormat;
use super::{EagerPackedInput, MMMInputFormat, MMMInputValue};

type Kernel = unsafe fn(input: *const u8, output: *mut u8, k: usize);

#[derive(Hash, Clone)]
pub struct PanelExtractor {
    pub name: String,
    pub from: Box<dyn MMMInputFormat>,
    pub to: PackedFormat,
    pub kernel: Kernel,
    pub supported_predicate: fn() -> bool,
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

impl PanelExtractor {
    #[allow(unused_variables)]
    pub fn is_supported_here(&self) -> bool {
        true
    }
}

#[derive(Clone, Hash)]
pub struct PanelExtractInput {
    pub format: PanelExtractor,
    pub data: EagerPackedInput,
}

impl MMMInputValue for PanelExtractInput {
    fn scratch_panel_buffer_layout(&self) -> Option<std::alloc::Layout> {
        Some(self.format.to.single_panel_layout(self.data.k(), self.format.to.dt.size_of()))
    }
    fn panel_bytes(&self, i: usize, buffer: Option<*mut u8>) -> tract_data::TractResult<*const u8> {
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

#[macro_export]
macro_rules! panel_extractor {
    ( $func:path as $id:ident($from:expr, $to: expr)
            $(where($where:expr))?
     ) => {
        paste! {
            lazy_static::lazy_static! {
                pub static ref $id: $crate::frame::mmm::panel_extract::PanelExtractor = {
                    let mut it = $crate::frame::mmm::panel_extract::PanelExtractor {
                        name: stringify!($id).to_string(),
                        from: $from,
                        to: $to,
                        kernel: $func,
                        supported_predicate: || true
                    };
                    $(
                        it.supported_predicate = $where;
                    )?
                    it
                };
            }

            #[cfg(test)]
            mod [<test_$id>] {
                use super::$id;
                use $crate::frame::block_quant::*;
                #[test]
                fn repack_1block_1panel() {
                    let bq = $id.from.downcast_ref::<PackedBlockQuantFormat>().unwrap();
                    $crate::frame::mmm::panel_extract::test::test_packing(&$id, bq.bq.block_len(), bq.r).unwrap();
                }

                #[test]
                fn repack_2block_1panel() {
                    let bq = $id.from.downcast_ref::<PackedBlockQuantFormat>().unwrap();
                    $crate::frame::mmm::panel_extract::test::test_packing(&$id, bq.bq.block_len(), bq.r).unwrap();
                }
            }
        }
    };
}

#[cfg(test)]
pub mod test {
    use crate::frame::block_quant::PackedBlockQuantFormat;
    use tract_data::internal::*;
    use tract_ndarray::Array2;

    use super::*;

    pub fn test_packing(extractor: &PanelExtractor, k: usize, m: usize) -> TractResult<()> {
        if !extractor.is_supported_here() {
            return Ok(())
        }
        assert!(extractor.from.r() == extractor.to.r());
        assert!(m % extractor.from.r() == 0);
        let from = extractor.from.downcast_ref::<PackedBlockQuantFormat>().unwrap();
        let to = &extractor.to;
        let weights_orig =
            Array2::from_shape_fn((m, k), |(m, k)| ((m * 31 + k * 17) % 20) as f32 - 10.)
                .into_tensor()
                .cast_to_dt(to.dt)?
                .into_owned();
        let weights = if to.dt == f32::datum_type() {
            from.bq
                .dequant_f32(&from.bq.quant_f32(weights_orig.as_slice::<f32>()?)?)?
                .into_shape(&[m, k])?
        } else {
            from.bq
                .dequant_f16(&from.bq.quant_f16(weights_orig.as_slice::<f16>()?)?)?
                .into_shape(&[m, k])?
        };
        let block_quant = if to.dt == f32::datum_type() {
            from.bq.quant_f32(weights.as_slice::<f32>()?)?
        } else {
            from.bq.quant_f16(weights.as_slice::<f16>()?)?
        };
        let packed_block_quant =
            from.bq.pack(&block_quant, k, from.r, from.zip, from.scales_at_end)?;

        for panel in 0..packed_block_quant.panels_count() {
            unsafe {
                let mut reference_extracted = Tensor::zero_dt(to.dt, &[k, from.r])?;
                from.bq.extract_panel(
                    &packed_block_quant,
                    to,
                    panel,
                    reference_extracted.as_bytes_mut().as_mut_ptr(),
                )?;

                let mut tested_extracted = Tensor::zero_dt(to.dt, &[k, from.r])?;
                let source =
                    packed_block_quant.packed.as_ptr().add(packed_block_quant.panel_bytes * panel);
                (extractor.kernel)(source, tested_extracted.as_bytes_mut().as_mut_ptr(), k);
                if tested_extracted != reference_extracted {
                    if to.dt == f32::datum_type() {
                        crate::frame::mmm::tests::display_error(
                            tested_extracted.as_slice::<f32>().unwrap(),
                            reference_extracted.as_slice::<f32>().unwrap(),
                            from.r,
                            k,
                        );
                    } else {
                        crate::frame::mmm::tests::display_error(
                            tested_extracted.as_slice::<f16>().unwrap(),
                            reference_extracted.as_slice::<f16>().unwrap(),
                            from.r,
                            k,
                        );
                    }
                }
                assert_eq!(tested_extracted, reference_extracted);
            }
        }
        Ok(())
    }
}
