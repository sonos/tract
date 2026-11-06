use crate::tensor::{DeviceTensor, DeviceTensorExt};
use std::fmt;
use tract_core::internal::*;
use tract_core::ops::nn as core_ops_nn;
use tract_itertools::Itertools;

pub type DispatchReduceFn = fn(&Reducer, &DeviceTensor, usize, &DeviceTensor) -> TractResult<()>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reducer {
    Mean,
    MeanOfSquares,
    Sum,
    Prod,
    Min,
    Max,
    All,
    Any,
    /// Index of the largest element, the last one among ties when the flag is
    /// set; NaN never wins. Writes i64, matching core's `Reducer::ArgMax`.
    ArgMax(bool),
    /// Index of the smallest element, the last one among ties when the flag is
    /// set; NaN never wins. Writes i64, matching core's `Reducer::ArgMin`.
    ArgMin(bool),
}

impl fmt::Display for Reducer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Mean => write!(f, "mean"),
            Self::MeanOfSquares => write!(f, "mean_of_squares"),
            Self::Sum => write!(f, "sum"),
            Self::Prod => write!(f, "prod"),
            Self::Min => write!(f, "min"),
            Self::Max => write!(f, "max"),
            Self::All => write!(f, "all"),
            Self::Any => write!(f, "any"),
            Self::ArgMax(false) => write!(f, "argmax"),
            Self::ArgMax(true) => write!(f, "argmax_last"),
            Self::ArgMin(false) => write!(f, "argmin"),
            Self::ArgMin(true) => write!(f, "argmin_last"),
        }
    }
}

impl Reducer {
    pub const ALL: [Reducer; 8] = [
        Self::Mean,
        Self::MeanOfSquares,
        Self::Sum,
        Self::Prod,
        Self::Min,
        Self::Max,
        Self::All,
        Self::Any,
    ];

    /// The index reducers, kept out of `ALL`: a backend opts into them with its
    /// own dtype check once it has their kernels.
    pub const ARG: [Reducer; 4] =
        [Self::ArgMax(false), Self::ArgMax(true), Self::ArgMin(false), Self::ArgMin(true)];

    pub fn is_logic(&self) -> bool {
        *self == Reducer::All || *self == Reducer::Any
    }

    pub fn is_arg(&self) -> bool {
        matches!(self, Reducer::ArgMax(_) | Reducer::ArgMin(_))
    }

    /// Value reducers on the backend-shared dtypes; false for the index
    /// reducers, which each backend enables separately.
    pub fn is_supported_dt(&self, dt: DatumType) -> bool {
        if self.is_arg() {
            false
        } else if self.is_logic() {
            dt.is::<bool>()
        } else {
            dt.is::<f32>() || dt.is::<f16>()
        }
    }

    pub fn output_dt(&self, input_dt: DatumType) -> DatumType {
        if self.is_arg() { DatumType::I64 } else { input_dt }
    }

    pub fn from_tract_core(reducer: &core_ops_nn::Reducer) -> TractResult<Self> {
        match reducer {
            core_ops_nn::Reducer::Sum => Ok(Reducer::Sum),
            core_ops_nn::Reducer::Mean => Ok(Reducer::Mean),
            core_ops_nn::Reducer::MeanOfSquares => Ok(Reducer::MeanOfSquares),
            core_ops_nn::Reducer::Prod => Ok(Reducer::Prod),
            core_ops_nn::Reducer::Min => Ok(Reducer::Min),
            core_ops_nn::Reducer::Max => Ok(Reducer::Max),
            core_ops_nn::Reducer::All => Ok(Reducer::All),
            core_ops_nn::Reducer::Any => Ok(Reducer::Any),
            core_ops_nn::Reducer::ArgMax(last) => Ok(Reducer::ArgMax(*last)),
            core_ops_nn::Reducer::ArgMin(last) => Ok(Reducer::ArgMin(*last)),
        }
    }
}

#[derive(Clone, Debug)]
pub struct GpuReduce {
    pub axes: TVec<usize>,
    pub reducer: Reducer,
    pub backend_name: &'static str,
    pub dispatch: DispatchReduceFn,
}

impl PartialEq for GpuReduce {
    fn eq(&self, other: &Self) -> bool {
        self.axes == other.axes
            && self.reducer == other.reducer
            && self.backend_name == other.backend_name
    }
}

impl Eq for GpuReduce {}

impl std::hash::Hash for GpuReduce {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.axes.hash(state);
        self.reducer.hash(state);
        self.backend_name.hash(state);
    }
}

impl GpuReduce {
    pub fn new(
        axes: TVec<usize>,
        reducer: Reducer,
        backend_name: &'static str,
        dispatch: DispatchReduceFn,
    ) -> TractResult<Self> {
        ensure!(axes.len() == 1, "Only one axis of reduce is supported by {backend_name}Reduce");
        Ok(Self { axes, reducer, backend_name, dispatch })
    }

    pub fn from_tract_core(
        core_reduce: &core_ops_nn::Reduce,
        backend_name: &'static str,
        dispatch: DispatchReduceFn,
    ) -> TractResult<Self> {
        let reducer = Reducer::from_tract_core(&core_reduce.reducer)?;
        Self::new(core_reduce.axes.clone(), reducer, backend_name, dispatch)
    }
}

impl Op for GpuReduce {
    fn name(&self) -> StaticName {
        format!("{}Reduce<{:?}>", self.backend_name, self.reducer).into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axes: {:?}", self.axes)])
    }
    op_as_typed_op!();
}

impl EvalOp for GpuReduce {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input_value = args_1!(inputs);
        let input = input_value.to_device_tensor()?;
        let mut output_shape = input.shape().to_vec();
        output_shape[self.axes[0]] = 1;
        let output = crate::turn_handler::make_tensor_for_node(
            ctx,
            self.reducer.output_dt(input.datum_type()),
            &output_shape,
        )?;
        (self.dispatch)(&self.reducer, input, self.axes[0], &output)?;
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for GpuReduce {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(self.axes.iter().tuple_windows().all(|(a, b)| a < b));
        crate::utils::facts_to_device_facts(inputs, |facts| {
            let mut shape: TVec<_> = facts[0].shape.to_tvec();
            for &ax in &self.axes {
                shape[ax] = 1.to_dim();
            }
            let dt = self.reducer.output_dt(facts[0].datum_type);
            Ok(tvec!(dt.fact(shape)))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}
