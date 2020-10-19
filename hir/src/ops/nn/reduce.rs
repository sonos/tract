use crate::internal::*;

use tract_core::ops::nn::Reduce as TReduce;
use tract_core::ops::nn::Reducer as TReducer;

#[derive(Clone, Copy, Debug, Hash, PartialEq)]
pub enum Reducer {
    ArgMax(bool), // take last
    ArgMin(bool),
    L1,
    L2,
    LogSum,
    LogSumExp,
    Max,
    Mean,
    Min,
    Prod,
    Sum,
    SumSquare,
}

impl Reducer {
    pub fn wire(
        &self,
        axes: TVec<usize>,
        name: &str,
        target: &mut TypedModel,
        mut wire: OutletId,
    ) -> TractResult<OutletId> {
        use tract_core::ops::math;
        use Reducer::*;
        match self {
            ArgMax(last) => {
                wire =
                    target.wire_node(name, TReduce::new(axes, TReducer::ArgMax(*last)), &[wire])?[0]
            }
            ArgMin(last) => {
                wire =
                    target.wire_node(name, TReduce::new(axes, TReducer::ArgMin(*last)), &[wire])?[0]
            }
            Max => wire = target.wire_node(name, TReduce::new(axes, TReducer::Max), &[wire])?[0],
            Min => wire = target.wire_node(name, TReduce::new(axes, TReducer::Min), &[wire])?[0],
            Sum => wire = target.wire_node(name, TReduce::new(axes, TReducer::Sum), &[wire])?[0],
            Prod => wire = target.wire_node(name, TReduce::new(axes, TReducer::Prod), &[wire])?[0],

            L1 => {
                wire = target.wire_node(format!("{}.abs", name), math::abs(), &[wire])?[0];
                wire = target.wire_node(
                    format!("{}.sum", name),
                    TReduce::new(axes, TReducer::Sum),
                    &[wire],
                )?[0];
            }
            L2 => {
                wire = target.wire_node(format!("{}.sq", name), math::square(), &[wire])?[0];
                wire = target.wire_node(
                    format!("{}.sum", name),
                    TReduce::new(axes, TReducer::Sum),
                    &[wire],
                )?[0];
                wire = target.wire_node(format!("{}.sqrt", name), math::sqrt(), &[wire])?[0];
            }
            LogSum => {
                wire = target.wire_node(
                    format!("{}.sum", name),
                    TReduce::new(axes, TReducer::Sum),
                    &[wire],
                )?[0];
                wire = target.wire_node(format!("{}.ln", name), math::ln(), &[wire])?[0];
            }
            LogSumExp => {
                wire = target.wire_node(format!("{}.exp", name), math::exp(), &[wire])?[0];
                wire = target.wire_node(
                    format!("{}.sum", name),
                    TReduce::new(axes, TReducer::Sum),
                    &[wire],
                )?[0];
                wire = target.wire_node(format!("{}.ln", name), math::ln(), &[wire])?[0];
            }
            SumSquare => {
                wire = target.wire_node(format!("{}.sq", name), math::square(), &[wire])?[0];
                wire = target.wire_node(
                    name.to_string() + ".sum",
                    TReduce::new(axes, TReducer::Sum),
                    &[wire],
                )?[0]
            }
            Mean => {
                let fact = target.outlet_fact(wire)?.clone();
                wire = target.wire_node(
                    name.to_string() + ".sum",
                    TReduce::new(axes.clone(), TReducer::Sum),
                    &[wire],
                )?[0];
                let size:TDim = axes.iter().map(|ax| &fact.shape[*ax]).maybe_product()?;
                let size = size.to_i64()?;
                let size = tensor0(size)
                    .cast_to_dt(fact.datum_type)?
                    .into_owned()
                    .into_shape(&*tvec!(1; fact.rank()))?;
                let size = target.add_const(name.to_string() + ".divisor", size)?;
                wire = target.wire_node(
                    name.to_string() + ".norm",
                    math::div::bin_typed(),
                    &[wire, size],
                )?[0];
            }
        };
        Ok(wire)
    }
}

#[derive(Clone, Debug, new, Hash)]
pub struct Reduce {
    axes: Option<Vec<i64>>,
    keep_dims: bool,
    reducer: Reducer,
}

tract_data::impl_dyn_hash!(Reduce);

impl Reduce {
    pub fn must_reduce(&self, ax: usize, rank: usize) -> bool {
        let resolved_axes: Option<Vec<usize>> = match &self.axes {
            None => None,
            Some(original_axes) => {
                let mut ans: Vec<usize> = vec![];
                for or_ax in original_axes.iter() {
                    ans.push(Self::resolve_axis(*or_ax, rank).unwrap());
                }
                Some(ans)
            }
        };

        resolved_axes.as_ref().map(|axes| axes.contains(&ax)).unwrap_or(true)
    }

    fn output_shape(&self, shape: &[TDim]) -> TVec<TDim> {
        shape
            .iter()
            .enumerate()
            .filter_map(|(ix, d)| {
                if self.must_reduce(ix, shape.len()) {
                    if self.keep_dims {
                        Some(1.to_dim())
                    } else {
                        None
                    }
                } else {
                    Some(d.clone())
                }
            })
            .collect()
    }

    fn resolve_axis(axis: i64, rank: usize) -> TractResult<usize> {
        if 0 <= axis && axis as usize <= rank - 1 {
            Ok(axis as usize)
        } else if -(rank as i64) <= axis && axis < 0 {
            Ok((axis + rank as i64) as usize)
        } else {
            bail!("Illegal combination of values for rank and axis: {} and {}", rank, axis)
        }
    }

    fn resolve_axes(&self, input_rank: usize) -> TractResult<TVec<usize>> {
        let mut axes: TVec<usize> = match self.axes.as_ref() {
            None => Ok((0..input_rank).collect()),
            Some(axis) => axis.iter().map(|&a| Self::resolve_axis(a, input_rank)).collect(),
        }?;
        axes.sort();
        Ok(axes)
    }
}

impl Expansion for Reduce {
    fn name(&self) -> Cow<str> {
        format!("Reduce<{:?}>", self.reducer).into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axes: {:?} keep_dims: {}", self.axes, self.keep_dims)])
    }
    op_hir!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        if let Reducer::ArgMax(_) | Reducer::ArgMin(_) = self.reducer {
            s.equals(&outputs[0].datum_type, DatumType::I64)?;
        } else {
            s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        }
        if self.keep_dims {
            s.equals(&inputs[0].rank, &outputs[0].rank)?;
        } else if let Some(axes) = self.axes.as_ref() {
            s.equals(inputs[0].rank.bex() - axes.len() as i64, &outputs[0].rank)?;
        } else {
            s.equals(&outputs[0].rank, 0)?;
        }
        s.given(&inputs[0].shape, move |s, shape| {
            let out_shape = self.output_shape(&*shape);
            s.equals(&outputs[0].shape, out_shape)
        })
    }

    fn wire(
        &self,
        name: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let input = inputs[0];
        let fact = target.outlet_fact(input)?;
        let mut axes = self.resolve_axes(fact.rank())?;
        axes.sort();
        let mut wire = self.reducer.wire(axes.clone(), name, target, input)?;
        if !self.keep_dims {
            for axis in axes.into_iter().rev() {
                wire = target.wire_node(
                    format!("{}-dispose-dims-{}", name, axis),
                    AxisOp::Rm(axis),
                    &[wire],
                )?[0];
            }
        }
        Ok(tvec!(wire))
    }
}
