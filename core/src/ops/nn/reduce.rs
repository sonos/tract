use crate::internal::*;
use ndarray::prelude::*;
use num_traits::cast::AsPrimitive;

macro_rules! reduce_numbers {
    ($($path:ident)::* ($dt:expr) ($($args:expr),*)) => {
        match $dt {
            DatumType::U8   => $($path)::*::<u8,_>($($args),*),
            DatumType::U16  => $($path)::*::<u16,_>($($args),*),
            DatumType::I8   => $($path)::*::<i8,_>($($args),*),
            DatumType::I16  => $($path)::*::<i16,_>($($args),*),
            DatumType::I32  => $($path)::*::<i32,_>($($args),*),
            DatumType::I64  => $($path)::*::<i64,_>($($args),*),
            DatumType::F32  => $($path)::*::<f32,_>($($args),*),
            DatumType::F64  => $($path)::*::<f64,_>($($args),*),
            _ => bail!("{:?} is not a number", $dt)
        }
    }
}

macro_rules! reduce_floatlike {
    ($($path:ident)::* ($dt:expr) ($($args:expr),*)) => {
        match $dt {
            DatumType::F32  => $($path)::*::<f32,_>($($args),*),
            DatumType::F64  => $($path)::*::<f64,_>($($args),*),
            _ => bail!("{:?} is not float like", $dt)
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Reducer {
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
    fn reduce(&self, reduce: &Reduce, input: Arc<Tensor>) -> TractResult<Arc<Tensor>> {
        let dt = input.datum_type();
        match self {
            Reducer::L1 => match dt {
                DatumType::U8 => self.reduce_t::<u8, _>(reduce, input, l1u_t),
                DatumType::U16 => self.reduce_t::<u16, _>(reduce, input, l1u_t),
                DatumType::I8 => self.reduce_t::<i8, _>(reduce, input, l1s_t),
                DatumType::I16 => self.reduce_t::<i16, _>(reduce, input, l1s_t),
                DatumType::I32 => self.reduce_t::<i32, _>(reduce, input, l1s_t),
                DatumType::I64 => self.reduce_t::<i64, _>(reduce, input, l1s_t),
                DatumType::F32 => self.reduce_t::<f32, _>(reduce, input, l1s_t),
                DatumType::F64 => self.reduce_t::<f64, _>(reduce, input, l1s_t),
                _ => bail!("{:?} is not a number valid for L1 norm", dt),
            },
            Reducer::L2 => reduce_numbers!(Self::reduce_t(dt)(self, reduce, input, l2_t)),
            Reducer::LogSum => {
                reduce_floatlike!(Self::reduce_t(dt)(self, reduce, input, log_sum_t))
            }
            Reducer::LogSumExp => {
                reduce_floatlike!(Self::reduce_t(dt)(self, reduce, input, log_sum_exp_t))
            }
            Reducer::Mean => reduce_numbers!(Self::reduce_t(dt)(self, reduce, input, mean_t)),
            Reducer::Min => reduce_numbers!(Self::reduce_t(dt)(self, reduce, input, min_t)),
            Reducer::Max => reduce_numbers!(Self::reduce_t(dt)(self, reduce, input, max_t)),
            Reducer::Prod => reduce_numbers!(Self::reduce_t(dt)(self, reduce, input, prod_t)),
            Reducer::Sum => reduce_numbers!(Self::reduce_t(dt)(self, reduce, input, sum_t)),
            Reducer::SumSquare => {
                reduce_numbers!(Self::reduce_t(dt)(self, reduce, input, sum_square_t))
            }
        }
    }

    fn reduce_t<T, F>(&self, reduce: &Reduce, input: Arc<Tensor>, f: F) -> TractResult<Arc<Tensor>>
    where
        F: for<'a> Fn(ArrayViewD<'a, T>) -> T,
        T: Copy + Datum,
    {
        use ndarray::*;
        let rank = input.shape().len();
        let input = input.to_array_view::<T>()?;
        let full_output_shape: Vec<usize> = input
            .shape()
            .iter()
            .enumerate()
            .map(|(ax, &d)| if reduce.must_reduce(ax, rank) { 1 } else { d })
            .collect();
        let mut result = Array::from_shape_fn(&*full_output_shape, |coords| {
            let slice_spec: Vec<SliceOrIndex> = coords
                .slice()
                .iter()
                .enumerate()
                .map(|(ax, &d)| if reduce.must_reduce(ax, rank) { (..).into() } else { d.into() })
                .collect();
            let slice_info = SliceInfo::new(&slice_spec).unwrap();
            let slice = input.slice(slice_info.as_ref());
            f(slice)
        });
        if !reduce.keep_dims {
            for ax in (0..full_output_shape.len()).rev() {
                if reduce.must_reduce(ax, rank) {
                    result = result.index_axis_move(Axis(ax), 0);
                }
            }
        }
        Ok(result.into_arc_tensor())
    }
}

fn l1s_t<'a, T>(v: ArrayViewD<'a, T>) -> T
where
    T: Copy + Datum + num_traits::Signed + num_traits::Zero,
{
    v.fold(T::zero(), |acc, &v| acc + v.abs())
}

fn l1u_t<'a, T>(v: ArrayViewD<'a, T>) -> T
where
    T: Copy + Datum + num_traits::Unsigned + num_traits::Zero,
{
    v.fold(T::zero(), |acc, &v| acc + v)
}

fn l2_t<'a, T>(v: ArrayViewD<'a, T>) -> T
where
    T: Copy + Datum + AsPrimitive<f64>,
    f64: AsPrimitive<T>,
{
    v.fold(0.0f64, |acc, &v| acc + (v.as_()).powi(2)).sqrt().as_()
}

fn log_sum_t<'a, T>(v: ArrayViewD<'a, T>) -> T
where
    T: Copy + Datum + num_traits::Zero + num_traits::Float,
{
    v.scalar_sum().ln()
}

fn log_sum_exp_t<'a, T>(v: ArrayViewD<'a, T>) -> T
where
    T: Copy + Datum + num_traits::Zero + num_traits::Float,
{
    let max = v.fold(T::min_value(), |acc, &v| if acc > v { acc } else { v });
    max + v.fold(T::zero(), |acc, &v| acc + (v - max).exp()).ln()
}

fn max_t<'a, T>(v: ArrayViewD<'a, T>) -> T
where
    T: Copy + Datum + num_traits::Bounded + ::std::cmp::PartialOrd,
{
    v.fold(T::min_value(), |acc, &v| if acc > v { acc } else { v })
}

fn mean_t<'a, T>(v: ArrayViewD<'a, T>) -> T
where
    T: Copy + Datum + num_traits::Zero + ::std::ops::Div<Output = T>,
    usize: AsPrimitive<T>,
{
    let (sum, count) = v.fold((T::zero(), 0), |acc, &v| (acc.0 + v, acc.1 + 1));
    sum / count.as_()
}

fn min_t<'a, T>(v: ArrayViewD<'a, T>) -> T
where
    T: Copy + Datum + num_traits::Bounded + ::std::cmp::PartialOrd,
{
    v.fold(T::max_value(), |acc, &v| if acc < v { acc } else { v })
}

fn prod_t<'a, T>(v: ArrayViewD<'a, T>) -> T
where
    T: Copy + Datum + num_traits::One,
{
    v.fold(T::one(), |acc, &v| acc * v)
}

fn sum_t<'a, T>(v: ArrayViewD<'a, T>) -> T
where
    T: Copy + Datum + num_traits::Zero,
{
    v.scalar_sum()
}

fn sum_square_t<'a, T>(v: ArrayViewD<'a, T>) -> T
where
    T: Copy + Datum + num_traits::Zero + ::std::ops::Mul<T, Output = T>,
{
    v.fold(T::zero(), |acc, &v| acc + v * v)
}

#[derive(Clone, Debug, new)]
pub struct Reduce {
    axes: Option<Vec<i64>>,
    keep_dims: bool,
    reducer: Reducer,
}

impl Reduce {
    pub fn must_reduce(&self, ax: usize, rank: usize) -> bool {
        let resolved_axes: Option<Vec<usize>> = match &self.axes {
            None => None,
            Some(original_axes) => {
                let mut ans: Vec<usize> = vec![];
                for or_ax in original_axes.iter() {
                    ans.push(Self::resolve_axis(*or_ax, rank as i64).unwrap());
                }
                Some(ans)
            }
        };

        resolved_axes.as_ref().map(|axes| axes.contains(&ax)).unwrap_or(true)
    }

    fn resolve_axis(axis: i64, rank: i64) -> TractResult<usize> {
        if 0 <= axis && axis <= rank - 1 {
            Ok(axis as usize)
        } else if -rank <= axis && axis < 0 {
            Ok((axis + rank) as usize)
        } else {
            bail!("Illegal combination of values for rank and axis: {} and {}", rank, axis)
        }
    }
}

impl Op for Reduce {
    fn name(&self) -> Cow<str> {
        format!("Reduce<{:?}>", self.reducer).into()
    }

    fn pulsify(
        &self,
        _source: &NormalizedModel,
        node: &NormalizedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let input = mapping[&node.inputs[0]];
        let fact = target.outlet_fact(input)?.clone();
        let id = target.chain_after(input, &*node.name, self.clone(), tvec!(fact))?;
        Ok(tvec!(OutletId::new(id, 0)))
    }

    to_typed!();
}

impl StatelessOp for Reduce {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        Ok(tvec!(self.reducer.reduce(&self, args_1!(inputs))?))
    }
}

impl InferenceRulesOp for Reduce {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        if self.keep_dims {
            s.equals(&inputs[0].rank, &outputs[0].rank)?;
        } else if let Some(axes) = self.axes.as_ref() {
            s.equals((&inputs[0].rank).bex() - axes.len() as i32, &outputs[0].rank)?;
        } else {
            s.equals(&outputs[0].rank, 0)?;
        }
        s.given(&inputs[0].shape, move |s, shape| {
            let out_shape: TVec<TDim> = shape
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
                .collect();
            s.equals(&outputs[0].shape, out_shape)
        })
    }

    inference_op_as_op!();
}

impl TypedOp for Reduce {
    typed_op_as_op!();
}
