use crate::GpuStream;
use crate::tensor::{DeviceTensor, DeviceTensorExt};
use std::fmt;
use tract_core::internal::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp {
    Mul,
    Add,
    Div,
    Sub,
    Pow,
    Min,
    Max,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Equals,
    NotEquals,
    And,
    Or,
    BitAnd,
    BitOr,
    BitXor,
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl BinOp {
    pub fn is_logic(&self) -> bool {
        matches!(
            self,
            Self::Less
                | Self::LessEqual
                | Self::Greater
                | Self::GreaterEqual
                | Self::Equals
                | Self::NotEquals
                | Self::And
                | Self::Or
        )
    }

    pub fn is_comparison(&self) -> bool {
        matches!(
            self,
            Self::Less
                | Self::LessEqual
                | Self::Greater
                | Self::GreaterEqual
                | Self::Equals
                | Self::NotEquals
        )
    }

    pub fn is_bitwise(&self) -> bool {
        matches!(self, Self::BitAnd | Self::BitOr | Self::BitXor)
    }

    pub fn is_arithmetic(&self) -> bool {
        matches!(
            self,
            Self::Mul | Self::Add | Self::Div | Self::Sub | Self::Pow | Self::Min | Self::Max
        )
    }

    pub fn is_supported_dt(&self, dt: DatumType) -> bool {
        if self.is_arithmetic() || self.is_comparison() {
            dt.is_number()
        } else if self.is_logic() {
            dt.is::<bool>()
        } else if self.is_bitwise() {
            dt.is_signed() || dt.is_unsigned()
        } else {
            false
        }
    }

    fn output_datum_type(&self, a: DatumType, b: DatumType) -> TractResult<DatumType> {
        ensure!(a == b);
        if self.is_logic() { Ok(DatumType::Bool) } else { Ok(a) }
    }

    fn output_shape<D: DimLike>(&self, a: &[D], b: &[D]) -> TractResult<TVec<D>> {
        tract_core::broadcast::multi_broadcast(&[a, b])
            .with_context(|| format!("Error while broadcasting {:?} {:?}", a, b))
    }
}

pub type DispatchBinOpFn =
    fn(&dyn GpuStream, BinOp, &DeviceTensor, &DeviceTensor, &DeviceTensor) -> TractResult<()>;

#[derive(Clone, Debug)]
pub struct GpuBinOp {
    pub backend_name: &'static str,
    pub op: BinOp,
    pub dispatch: DispatchBinOpFn,
}

impl PartialEq for GpuBinOp {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name && self.op == other.op
    }
}

impl Eq for GpuBinOp {}

impl std::hash::Hash for GpuBinOp {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
        self.op.hash(state);
    }
}

impl GpuBinOp {
    fn resolve_output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let (a, b) = (inputs[0], inputs[1]);
        if a.rank() != b.rank() {
            bail!(
                "Typed ops require rank match. Invalid inputs for {}: {{a: {:?}, b: {:?}}}",
                self.name(),
                a.shape,
                b.shape
            );
        }
        let out_shape = self.op.output_shape(&a.shape, &b.shape)?;
        let out_dt = self.op.output_datum_type(a.datum_type, b.datum_type)?;
        Ok(tvec!(out_dt.fact(out_shape)))
    }
}

impl Op for GpuBinOp {
    fn name(&self) -> StaticName {
        format!("{}{}", self.backend_name, self.op).into()
    }

    op_as_typed_op!();
}

impl EvalOp for GpuBinOp {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let (a_val, b_val) = args_2!(inputs);
        let a = a_val.to_device_tensor()?;
        let b = b_val.to_device_tensor()?;
        let out_shape = self.op.output_shape(a.shape(), b.shape())?;
        let out_dt = self.op.output_datum_type(a.datum_type(), b.datum_type())?;
        let output =
            crate::session_handler::make_tensor_for_node(session, node_id, out_dt, &out_shape)?;
        if a.len() > 0 && b.len() > 0 {
            crate::with_stream(|stream| {
                (self.dispatch)(stream, self.op, a, b, &output)
                    .with_context(|| format!("Error while dispatching eval for {}", self.name()))
            })?;
        }
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for GpuBinOp {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| self.resolve_output_facts(facts))
            .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}
