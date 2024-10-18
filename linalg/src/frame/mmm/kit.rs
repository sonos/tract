use std::fmt::Debug;

use tract_data::prelude::DatumType;

use crate::frame::block_quant::BlockQuant;

use super::panel_extract::PanelExtractor;
use super::{MMMInputFormat, MatMatMul};

// final hypothesis
// * A is const weight. either a DT, or a blockquant
// * m, k are constant, n is an undetermined TDim
//
// for now (?) acc.dt == B.dt == C.dt

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

#[derive(Debug)]
pub struct MMMKit {
    pub weight: WeightType,
    pub accumulator: KitDatumType,
    pub activation: KitDatumType,
    pub static_packer: Box<dyn MMMInputFormat>,
    pub items: Vec<MMMKitItem>,
    pub generic_fallback: bool,
}

#[derive(Debug)]
pub struct MMMKitItem {
    pub mmm: Box<dyn MatMatMul>,
    pub packing: usize,
    pub weight_panel_extractor: Option<PanelExtractor>,
    pub activation_panel_extractor: Option<PanelExtractor>,
}

impl MMMKit {
    pub(crate) fn new(
        weight: impl Into<WeightType>,
        accumulator: impl Into<KitDatumType>,
        activation: impl Into<KitDatumType>,
        static_packer: &dyn MMMInputFormat,
    ) -> MMMKit {
        MMMKit {
            weight: weight.into(),
            accumulator: accumulator.into(),
            activation: activation.into(),
            static_packer: dyn_clone::clone_box(static_packer),
            items: vec![],
            generic_fallback: false,
        }
    }

    pub(crate) fn with_native(mut self, mmm: Box<dyn MatMatMul>, packing: usize) -> Self {
        assert!(mmm.packings()[packing].0.same_as(&*self.static_packer));
        assert!(self.accumulator == mmm.internal_type().into());
        self.items.push(MMMKitItem {
            mmm,
            packing,
            weight_panel_extractor: None,
            activation_panel_extractor: None,
        });
        self
    }

    pub(crate) fn with_generic_fallback(self, generic_fallback: bool) -> Self {
        Self { generic_fallback, ..self }
    }

    pub fn name(&self) -> &str {
        self.items[0].mmm.name()
    }
}
