use ops::prelude::*;

#[derive(Debug, Clone)]
pub enum StepValue {
    Const(Value),
    Stream(Stream),
}

#[derive(Debug, Clone)]
pub struct Stream {
    pub info: StreamInfo,
    pub offset: u64,
    pub chunk: Option<Value>,
}

#[derive(Debug, Copy, Clone, Default)]
pub struct StreamInfo {
    pub axis: usize,
    pub len: TDim,
}

impl StepValue {
    pub fn as_value(&self) -> Option<&Value> {
        match self {
            StepValue::Const(v) => Some(v),
            StepValue::Stream(s) => s.chunk.as_ref(),
        }
    }

    pub fn into_value(self) -> Option<Value> {
        match self {
            StepValue::Const(v) => Some(v),
            StepValue::Stream(s) => s.chunk,
        }
    }

    pub fn as_const(&self) -> Option<&Value> {
        match self {
            StepValue::Const(v) => Some(v),
            _ => None,
        }
    }

    pub fn into_const(self) -> Option<Value> {
        match self {
            StepValue::Const(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_stream(&self) -> Option<&Stream> {
        match self {
            StepValue::Stream(s) => Some(s),
            _ => None,
        }
    }

    pub fn into_stream(self) -> Option<Stream> {
        match self {
            StepValue::Stream(s) => Some(s),
            _ => None,
        }
    }

    pub fn stream_info(&self) -> Option<StreamInfo> {
        self.as_stream().map(|s| s.info)
    }

    pub fn is_const(&self) -> bool {
        match self {
            StepValue::Const(_) => true,
            StepValue::Stream(_) => false,
        }
    }
}
