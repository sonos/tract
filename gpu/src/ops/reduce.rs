use std::fmt;
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use tract_core::internal::*;
use tract_core::ops::nn as core_ops_nn;
use tract_itertools::Itertools;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reducer {
    MeanOfSquares,
    Sum,
    Prod,
    Min,
    Max,
    All,
    Any,
}

impl fmt::Display for Reducer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::MeanOfSquares => write!(f, "mean_of_squares"),
            Self::Sum => write!(f, "sum"),
            Self::Prod => write!(f, "prod"),
            Self::Min => write!(f, "min"),
            Self::Max => write!(f, "max"),
            Self::All => write!(f, "all"),
            Self::Any => write!(f, "any"),
        }
    }
}

impl Reducer {
    pub const ALL: [Reducer; 7] =
        [Self::MeanOfSquares, Self::Sum, Self::Prod, Self::Min, Self::Max, Self::All, Self::Any];

    pub fn is_logic(&self) -> bool {
        *self == Reducer::All || *self == Reducer::Any
    }

    pub fn is_supported_dt(&self, dt: DatumType) -> bool {
        if self.is_logic() { dt.is::<bool>() } else { dt.is::<f32>() || dt.is::<f16>() }
    }

    pub fn from_tract_core(reducer: &core_ops_nn::Reducer) -> TractResult<Self> {
        match reducer {
            core_ops_nn::Reducer::Sum => Ok(Reducer::Sum),
            core_ops_nn::Reducer::MeanOfSquares => Ok(Reducer::MeanOfSquares),
            core_ops_nn::Reducer::Prod => Ok(Reducer::Prod),
            core_ops_nn::Reducer::Min => Ok(Reducer::Min),
            core_ops_nn::Reducer::Max => Ok(Reducer::Max),
            core_ops_nn::Reducer::All => Ok(Reducer::All),
            core_ops_nn::Reducer::Any => Ok(Reducer::Any),
            _ => bail!("Unsupported reducer {:?} on GPU", reducer),
        }
    }
}

pub trait GpuReduceBackend: Clone + Debug + Hash + PartialEq + Eq + Send + Sync + 'static {
    fn name() -> &'static str;
    fn eval(
        reducer: &Reducer,
        axes: &[usize],
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>>;
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct GpuReduce<B: GpuReduceBackend> {
    pub axes: TVec<usize>,
    pub reducer: Reducer,
    _backend: PhantomData<B>,
}

impl<B: GpuReduceBackend> GpuReduce<B> {
    pub fn new(axes: TVec<usize>, reducer: Reducer) -> TractResult<Self> {
        ensure!(axes.len() == 1, "Only one axis of reduce is supported by {}Reduce", B::name());
        Ok(Self { axes, reducer, _backend: PhantomData })
    }

    pub fn from_tract_core(core_reduce: &core_ops_nn::Reduce) -> TractResult<Self> {
        let reducer = Reducer::from_tract_core(&core_reduce.reducer)?;
        Self::new(core_reduce.axes.clone(), reducer)
    }
}

impl<B: GpuReduceBackend> Op for GpuReduce<B> {
    fn name(&self) -> StaticName {
        format!("{}Reduce<{:?}>", B::name(), self.reducer).into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axes: {:?}", self.axes)])
    }
    op_as_typed_op!();
}

impl<B: GpuReduceBackend> EvalOp for GpuReduce<B> {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        B::eval(&self.reducer, &self.axes, node_id, session, inputs)
    }
}

impl<B: GpuReduceBackend> TypedOp for GpuReduce<B> {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(self.axes.iter().tuple_windows().all(|(a, b)| a < b));
        crate::utils::facts_to_device_facts(inputs, |facts| {
            let mut shape: TVec<_> = facts[0].shape.to_tvec();
            for &ax in &self.axes {
                shape[ax] = 1.to_dim();
            }
            let dt = facts[0].datum_type;
            Ok(tvec!(dt.fact(shape)))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}
