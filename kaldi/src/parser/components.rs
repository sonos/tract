use tract_core::prelude::*;

use maplit::hashmap;
use std::collections::HashMap;

use super::multispaced;

use nom::{
    branch::*,
    bytes::complete::*,
    combinator::*,
    multi::many_m_n,
    number::complete::{le_f32, le_f64},
    sequence::*,
    IResult,
};

pub enum KaldiAttributeKind {
    Bool,
    Int,
    Float,
    FloatVector,
    FloatMatrix,
}

impl KaldiAttributeKind {
    pub fn parse_bin<'a>(&self, i: &'a [u8]) -> IResult<&'a [u8], Tensor> {
        match self {
            Bool => alt((
                map(tag("F"), |_| Tensor::from(false)),
                map(tag("T"), |_| Tensor::from(true)),
            ))(i),
            Int => map(super::integer(true), Tensor::from)(i),
            Float => map(Self::parse_float_value, Tensor::from)(i),
            FloatVector => preceded(multispaced(tag("FV")), Self::parse_float_vector)(i),
            FloatMatrix => preceded(multispaced(tag("FM")), Self::parse_float_matrix)(i),
        }
    }

    fn parse_float_value<'a>(i: &'a [u8]) -> IResult<&'a [u8], f32> {
        alt((preceded(tag([4]), le_f32), map(preceded(tag([8]), le_f64), |f| f as f32)))(i)
    }

    fn parse_float_vector<'a>(i: &'a [u8]) -> IResult<&'a [u8], Tensor> {
        let (i, len) = super::integer(true)(i)?;
        // FIXME pending merge of https://github.com/Geal/nom/pull/995
        if len == 0 {
            Ok((i, tensor1(&[0.0f32])))
        } else {
            map(many_m_n(len as usize, len as usize, le_f32), |data| tensor1(&*data))(i)
        }
    }

    fn parse_float_matrix<'a>(i: &'a [u8]) -> IResult<&'a [u8], Tensor> {
        let (i, rows) = super::integer(true)(i)?;
        let (i, cols) = super::integer(true)(i)?;
        let len = (rows * cols) as usize;
        // FIXME pending merge of https://github.com/Geal/nom/pull/995
        if len == 0 {
            Ok((i, tensor2(&[[0.0f32; 0]; 0])))
        } else {
            map(
                map_res(many_m_n(len, len, le_f32), move |buf| {
                    tract_core::ndarray::Array2::from_shape_vec((rows as usize, cols as usize), buf)
                }),
                Tensor::from,
            )(i)
        }
    }
}

use KaldiAttributeKind::*;

lazy_static::lazy_static! {
    pub static ref COMPONENTS: HashMap<&'static str, HashMap<&'static str, KaldiAttributeKind>> = hashmap! {
        "FixedAffineComponent" => hashmap! {
            "LinearParams" => FloatMatrix,
            "BiasParams" => FloatVector,
        },
        "NaturalGradientAffineComponent" => hashmap! {
            "LearningRateFactor" => Float,
            "MaxChange" => Float,
            "LearningRate" => Float,
            "LinearParams" => FloatMatrix,
            "BiasParams" => FloatVector,
            "RankIn" => Int,
            "RankOut" => Int,
            "UpdatePeriod" => Int,
            "NumSamplesHistory" => Float,
            "Alpha" => Float,
            "IsGradient" => Bool,
        },
        "NormalizeComponent" => hashmap!{
            "InputDim" => Int,
            "TargetRms" => Float,
            "AddLogStddev" => Bool,
        },
        "FakeQuantizationComponent" => hashmap!{
            "Activated" => Bool,
            "Dim" => Int,
            "MaxValue" => Float,
            "MinValue" => Float,
        },
        "LstmNonlinearityComponent" => hashmap!{
            "MaxChange" => Float,
            "LearningRate" => Float,
            "Params" => FloatMatrix,
            "ValueAvg" => FloatMatrix,
            "DerivAvg" => FloatMatrix,
            "SelfRepairConfig" => FloatVector,
            "SelfRepairProb" => FloatVector,
            "Count" => Float,
        },
        "BackpropTruncationComponent" => hashmap!{
            "Dim" => Int,
            "Scale" => Float,
            "ClippingThreshold" => Float,
            "ZeroingThreshold" => Float,
            "ZeroingInterval" => Int,
            "RecurrenceInterval" => Int,
            "NumElementsClipped" => Float,
            "NumElementsZeroed" => Float,
            "NumElementsProcessed" => Float,
            "NumZeroingBoundaries" => Float,
        },
        "LogSoftmaxComponent" => hashmap!{
            "Dim" => Int,
            "ValueAvg" => FloatVector,
            "DerivAvg" => FloatVector,
            "Count" => Int,
            "NumDimsSelfRepaired" => Int,
            "NumDimsProcessed" => Int,
        },
        "RectifiedLinearComponent" => hashmap!{
            "Dim" => Int,
            "ValueAvg" => FloatVector,
            "DerivAvg" => FloatVector,
            "Count" => Float,
            "NumDimsSelfRepaired" => Float,
            "NumDimsProcessed" => Float,
            "SelfRepairScale" => Float,
        }
    };
}
