use std::fmt::Debug;
use tract_data::prelude::DatumType;

use crate::block_quant::{BlockQuant, PackedBlockQuantFormat};

use crate::mmm::MMMInputFormat;
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
            DatumType::F64 => WeightType::Plain(DatumType::F64),
            DatumType::I32 => WeightType::Plain(DatumType::I32),
            DatumType::I8 | DatumType::QI8(_) => WeightType::Plain(DatumType::I8),
            DatumType::U8 | DatumType::QU8(_) => WeightType::Plain(DatumType::U8),
            _ => panic!("Can't build a WeightType from {value:?}"),
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
            Self::Plain(p) => write!(f, "{p:?}"),
            Self::BlockQuant(bq) => write!(f, "{bq:?}"),
        }
    }
}

impl WeightType {
    pub fn as_dt(&self) -> Option<DatumType> {
        match self {
            WeightType::Plain(dt) => Some(*dt),
            _ => None,
        }
    }
}
