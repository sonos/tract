use std::fmt::Debug;

use tract_data::prelude::DatumType;

use crate::block_quant::{BlockQuant, PackedBlockQuantFormat};

use crate::mmm::{MMMInputFormat, MatMatMul, PanelExtractor};
use crate::pack::PackedFormat;

#[derive(Clone)]
pub enum WeightType {
    Plain(DatumType),
    BlockQuant(Box<dyn BlockQuant>),
}

impl From<DatumType> for WeightType {
    fn from(value: DatumType) -> Self {
        match value {
            DatumType::F16 => WeightType::Plain(DatumType::F16),
            DatumType::F32 => WeightType::Plain(DatumType::F32),
            DatumType::I32 => WeightType::Plain(DatumType::I32),
            _ => panic!(),
        }
    }
}

impl From<Box<dyn MMMInputFormat>> for WeightType {
    fn from(value: Box<dyn MMMInputFormat>) -> Self {
        (&*value).into()
    }
}

impl From<&dyn MMMInputFormat> for WeightType {
    fn from(value: &dyn MMMInputFormat) -> Self {
        if let Some(pf) = value.downcast_ref::<PackedFormat>() {
            WeightType::Plain(pf.dt)
        } else if let Some(pbqf) = value.downcast_ref::<PackedBlockQuantFormat>() {
            WeightType::BlockQuant(dyn_clone::clone_box(&*pbqf.bq))
        } else {
            todo!()
        }
    }
}

impl PartialEq for WeightType {
    fn eq(&self, other: &Self) -> bool {
        use WeightType::*;
        match (self, other) {
            (Plain(a), Plain(b)) => a == b,
            (BlockQuant(a), BlockQuant(b)) => a.same_as(&**b),
            _ => false,
        }
    }
}

impl<BQ: BlockQuant> From<BQ> for WeightType {
    fn from(value: BQ) -> Self {
        WeightType::BlockQuant(dyn_clone::clone_box(&value))
    }
}

impl Debug for WeightType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Plain(p) => write!(f, "{:?}", p),
            Self::BlockQuant(bq) => write!(f, "{:?}", bq),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum KitDatumType {
    F16,
    F32,
    I32,
}

impl From<DatumType> for KitDatumType {
    fn from(value: DatumType) -> Self {
        match value {
            DatumType::F16 => KitDatumType::F16,
            DatumType::F32 => KitDatumType::F32,
            DatumType::I32 => KitDatumType::I32,
            _ => panic!(),
        }
    }
}

impl From<&dyn MMMInputFormat> for KitDatumType {
    fn from(value: &dyn MMMInputFormat) -> Self {
        if let Some(pf) = value.downcast_ref::<PackedFormat>() {
            pf.dt.into()
        } else {
            todo!()
        }
    }
}

impl From<Box<dyn MMMInputFormat>> for KitDatumType {
    fn from(value: Box<dyn MMMInputFormat>) -> Self {
        (&*value).into()
    }
}

pub enum KitRequirement {
    MMM { accumulator: KitDatumType, activation: KitDatumType },
}

#[derive(Debug)]
pub struct Kit {
    pub weight: WeightType,
    pub static_packer: Box<dyn MMMInputFormat>,
    pub mmms: Vec<MMMKitItem>,
    pub generic_fallback: bool,
}

#[derive(Debug)]
pub struct MMMKitItem {
    pub mmm: Box<dyn MatMatMul>,
    pub packing: usize,
    pub weight_panel_extractor: Option<PanelExtractor>,
}

impl Kit {
    pub(crate) fn new(weight: impl Into<WeightType>, static_packer: &dyn MMMInputFormat) -> Kit {
        let weight = weight.into();
        let kit = Kit {
            weight,
            generic_fallback: false,
            static_packer: dyn_clone::clone_box(static_packer),
            mmms: vec![],
        };
        match &kit.weight {
            WeightType::Plain(p) => {
                debug_assert!(
                    kit.static_packer.downcast_ref::<PackedFormat>().is_some_and(|pf| pf.dt == *p),
                    "Static packer not compatible with weight format {kit:?}"
                )
            }
            WeightType::BlockQuant(bq) => debug_assert!(
                kit.static_packer
                    .downcast_ref::<PackedBlockQuantFormat>()
                    .is_some_and(|pbqf| pbqf.bq.same_as(&**bq)),
                "Static packer not compatible with weight format {kit:?}"
            ),
        };
        kit
    }

    fn add_item(
        mut self,
        mmm: Box<dyn MatMatMul>,
        packing: usize,
        weight_panel_extractor: Option<PanelExtractor>,
    ) -> Self {
        self.mmms.push(MMMKitItem { mmm, packing, weight_panel_extractor });
        self
    }

    pub(crate) fn with_native(self, mmm: Box<dyn MatMatMul>, packing: usize) -> Self {
        debug_assert!(
            mmm.packings()[packing].0.same_as(&*self.static_packer),
            "Weight packing mismatch {self:?} {mmm:?}/{packing} {:?}",
            mmm.packings()[packing].0
        );
        self.add_item(mmm, packing, None)
    }

    #[allow(dead_code)]
    pub(crate) fn with_extracting(
        self,
        mmm: Box<dyn MatMatMul>,
        packing: usize,
        weight_panel_extractor: PanelExtractor,
    ) -> Self {
        debug_assert!(
            self.static_packer.same_as(&*weight_panel_extractor.from),
            "Static weight packing/extractor mismatch {self:?} {mmm:?}/{packing} {:?} {weight_panel_extractor:?}",
            mmm.packings()[packing].0
        );
        debug_assert!(
            weight_panel_extractor.to.same_as(&*mmm.packings()[packing].0),
            "Extractor/kernel packing mismatch {self:?} {mmm:?}/{packing} {:?} {weight_panel_extractor:?}",
            mmm.packings()[packing].0
        );
        self.add_item(mmm, packing, Some(weight_panel_extractor))
    }

    pub(crate) fn with_generic_fallback(self, generic_fallback: bool) -> Self {
        Self { generic_fallback, ..self }
    }

    pub fn name(&self) -> String {
        self.static_packer.to_string()
    }

    pub fn can_do_mmm(&self, accumulator: KitDatumType, activation: KitDatumType) -> bool {
        self.mmms.iter().any(|mmm| {
            mmm.accumulator() == accumulator
                && mmm
                    .b_packing()
                    .downcast_ref::<PackedFormat>()
                    .is_some_and(|pf| KitDatumType::from(pf.dt) == activation)
        })
    }

    pub fn item_for_mv(&self) -> &MMMKitItem {
        self.mmms.iter().min_by_key(|item| item.n()).unwrap()
    }

    pub fn item_for_squarish(&self) -> &MMMKitItem {
        self.mmms.iter().max_by_key(|item| item.n()).unwrap()
    }
}

impl MMMKitItem {
    pub fn n(&self) -> usize {
        self.mmm.packings()[self.packing].1.r()
    }

    pub fn accumulator(&self) -> KitDatumType {
        self.mmm.internal_type().into()
    }

    pub fn b_packing(&self) -> &dyn MMMInputFormat {
        &*self.mmm.packings()[self.packing].1
    }
}
