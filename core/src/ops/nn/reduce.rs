use ndarray::prelude::*;
use num;
use num::cast::AsPrimitive;
use ops::prelude::*;

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
    fn reduce(&self, reduce: &Reduce, input: Tensor) -> TractResult<Tensor> {
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

    fn reduce_t<T, F>(&self, reduce: &Reduce, input: Tensor, f: F) -> TractResult<Tensor>
    where
        F: for<'a> Fn(ArrayViewD<'a, T>) -> T,
        T: Datum,
    {
        use ndarray::*;
        let input = input.to_array::<T>()?;
        let full_output_shape: Vec<usize> = input
            .shape()
            .iter()
            .enumerate()
            .map(|(ax, &d)| if reduce.must_reduce(ax) { 1 } else { d })
            .collect();
        let mut result = Array::from_shape_fn(&*full_output_shape, |coords| {
            let slice_spec: Vec<SliceOrIndex> = coords
                .slice()
                .iter()
                .enumerate()
                .map(|(ax, &d)| {
                    if reduce.must_reduce(ax) {
                        (..).into()
                    } else {
                        d.into()
                    }
                }).collect();
            let slice_info = SliceInfo::new(&slice_spec).unwrap();
            let slice = input.slice(slice_info.as_ref());
            f(slice)
        });
        if !reduce.keep_dims {
            for ax in (0..full_output_shape.len()).rev() {
                if reduce.must_reduce(ax) {
                    result = result.remove_axis(Axis(ax));
                }
            }
        }
        Ok(result.into())
    }
}

fn l1s_t<'a, T>(v: ArrayViewD<'a, T>) -> T
where
    T: Datum + num::Signed + num::Zero,
{
    v.fold(T::zero(), |acc, &v| acc + v.abs())
}

fn l1u_t<'a, T>(v: ArrayViewD<'a, T>) -> T
where
    T: Datum + num::Unsigned + num::Zero,
{
    v.fold(T::zero(), |acc, &v| acc + v)
}

fn l2_t<'a, T>(v: ArrayViewD<'a, T>) -> T
where
    T: Datum + AsPrimitive<f64>,
    f64: AsPrimitive<T>,
{
    v.fold(0.0f64, |acc, &v| acc + (v.as_()).powi(2))
        .sqrt()
        .as_()
}

fn log_sum_t<'a, T>(v: ArrayViewD<'a, T>) -> T
where
    T: Datum + num::Zero + num::Float,
{
    v.scalar_sum().ln()
}

fn log_sum_exp_t<'a, T>(v: ArrayViewD<'a, T>) -> T
where
    T: Datum + num::Zero + num::Float,
{
    let max = v.fold(T::min_value(), |acc, &v| if acc > v { acc } else { v });
    max + v.fold(T::zero(), |acc, &v| acc + (v - max).exp()).ln()
}

fn max_t<'a, T>(v: ArrayViewD<'a, T>) -> T
where
    T: Datum + num::Bounded + ::std::cmp::PartialOrd,
{
    v.fold(T::min_value(), |acc, &v| if acc > v { acc } else { v })
}

fn mean_t<'a, T>(v: ArrayViewD<'a, T>) -> T
where
    T: Datum + num::Zero + ::std::ops::Div<Output = T>,
    usize: AsPrimitive<T>,
{
    let (sum, count) = v.fold((T::zero(), 0), |acc, &v| (acc.0 + v, acc.1 + 1));
    sum / count.as_()
}

fn min_t<'a, T>(v: ArrayViewD<'a, T>) -> T
where
    T: Datum + num::Bounded + ::std::cmp::PartialOrd,
{
    v.fold(T::max_value(), |acc, &v| if acc < v { acc } else { v })
}

fn prod_t<'a, T>(v: ArrayViewD<'a, T>) -> T
where
    T: Datum + num::One,
{
    v.fold(T::one(), |acc, &v| acc * v)
}

fn sum_t<'a, T>(v: ArrayViewD<'a, T>) -> T
where
    T: Datum + num::Zero,
{
    v.scalar_sum()
}

fn sum_square_t<'a, T>(v: ArrayViewD<'a, T>) -> T
where
    T: Datum + num::Zero + ::std::ops::Mul<T, Output = T>,
{
    v.fold(T::zero(), |acc, &v| acc + v * v)
}

#[derive(Clone, Debug, new)]
pub struct Reduce {
    axes: Option<Vec<usize>>,
    keep_dims: bool,
    reducer: Reducer,
}

impl Reduce {
    pub fn must_reduce(&self, ax: usize) -> bool {
        self.axes
            .as_ref()
            .map(|axes| axes.contains(&ax))
            .unwrap_or(true)
    }
}

impl Op for Reduce {
    fn name(&self) -> &str {
        "Reduce"
    }
}

impl StatelessOp for Reduce {
    fn eval(&self, mut inputs: TVec<Tensor>) -> TractResult<TVec<Tensor>> {
        Ok(tvec!(self.reducer.reduce(&self, args_1!(inputs))?))
    }
}

impl InferenceRulesOp for Reduce {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 1)?;
        s.equals(&outputs.len, 1)?;
        if self.keep_dims {
            s.equals(&inputs[0].rank, &outputs[0].rank)?;
        } else if let Some(axes) = self.axes.as_ref() {
            s.equals(
                (&inputs[0].rank).bex() - axes.len() as i32,
                &outputs[0].rank,
            )?;
        } else {
            s.equals(&outputs[0].rank, 0)?;
        }
        s.given(&inputs[0].shape, move |s, shape| {
            let out_shape: TVec<TDim> = shape
                .iter()
                .enumerate()
                .filter_map(|(ix, &d)| {
                    if self.must_reduce(ix) {
                        if self.keep_dims {
                            Some(1.to_dim())
                        } else {
                            None
                        }
                    } else {
                        Some(d)
                    }
                }).collect();
            s.equals(&outputs[0].shape, out_shape)
        })
    }
}
