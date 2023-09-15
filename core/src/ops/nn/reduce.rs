use crate::internal::Axis;
use crate::internal::*;
use std::convert::TryFrom;
use tract_data::internal::ClampCast;
use tract_data::itertools::Itertools;
use tract_ndarray::prelude::*;
use tract_num_traits::Bounded;

macro_rules! r {
    ($($path:ident)::* ($dt:expr) ($($args:expr),*)) => {
        match $dt {
            DatumType::U8   => $($path)::*::<u8,_,_,_>($($args),*),
            DatumType::I8   => $($path)::*::<i8,_,_,_>($($args),*),
            DatumType::U16  => $($path)::*::<u16,_,_,_>($($args),*),
            DatumType::I16  => $($path)::*::<i16,_,_,_>($($args),*),
            DatumType::I32  => $($path)::*::<i32,_,_,_>($($args),*),
            DatumType::I64  => $($path)::*::<i64,_,_,_>($($args),*),
            DatumType::F32  => $($path)::*::<f32,_,_,_>($($args),*),
            DatumType::F64  => $($path)::*::<f64,_,_,_>($($args),*),
            DatumType::QI8(_)  => $($path)::*::<i8,_,_,_>($($args),*),
            DatumType::QU8(_)  => $($path)::*::<u8,_,_,_>($($args),*),
            _ => bail!("{:?} is not a number", $dt)
        }
    };
    ($($path:ident)::* ($dt:expr) ($($args:expr),*); $($q_path:ident)::* ($($q_args:expr),*)) => {
        match $dt {
            DatumType::U8   => $($path)::*::<u8,_,_,_>($($args),*),
            DatumType::I8   => $($path)::*::<i8,_,_,_>($($args),*),
            DatumType::U16  => $($path)::*::<u16,_,_,_>($($args),*),
            DatumType::I16  => $($path)::*::<i16,_,_,_>($($args),*),
            DatumType::I32  => $($path)::*::<i32,_,_,_>($($args),*),
            DatumType::I64  => $($path)::*::<i64,_,_,_>($($args),*),
            DatumType::F32  => $($path)::*::<f32,_,_,_>($($args),*),
            DatumType::F64  => $($path)::*::<f64,_,_,_>($($args),*),
            DatumType::QI8(_)  => $($q_path)::*::<i8,_,_,_>($($q_args),*),
            DatumType::QU8(_)  => $($q_path)::*::<u8,_,_,_>($($q_args),*),
            _ => bail!("{:?} is not a number", $dt)
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Reducer {
    ArgMax(bool), // take last
    ArgMin(bool),
    Max,
    Min,
    Prod,
    Sum,
}

impl Reducer {
    pub fn reduce(&self, axes: &[usize], input: &Tensor) -> TractResult<Tensor> {
        use Reducer::*;
        let dt = input.datum_type();
        let output_shape: Vec<usize> = input
            .shape()
            .iter()
            .enumerate()
            .map(|(ax, &d)| if axes.contains(&ax) { 1 } else { d })
            .collect();
        let (zp, scale) = input.datum_type().zp_scale();
        unsafe {
            let mut t = match self {
                ArgMax(last) => {
                    r!(Self::reduce_t(dt)(self, axes, &output_shape, input, argmax_t, *last))
                }
                ArgMin(last) => {
                    r!(Self::reduce_t(dt)(self, axes, &output_shape, input, argmin_t, *last))
                }
                Min => r!(Self::reduce_t(dt)(self, axes, &output_shape, input, min_t, ())),
                Max => r!(Self::reduce_t(dt)(self, axes, &output_shape, input, max_t, ())),
                Prod => {
                    r!(Self::reduce_t(dt)(self, axes, &output_shape, input, prod_t, ()); Self::reduce_t(self, axes, &output_shape, input, q_prod_t, (zp, scale)))
                }
                Sum => {
                    if dt.is_float() {
                        dispatch_floatlike!(Self::sum(dt)(self, axes, input))
                    } else {
                        r!(Self::reduce_t(dt)(
                            self,
                            axes,
                            &output_shape,
                            input,
                            q_sum_t,
                            (zp, scale)
                        ))
                    }
                }
            };
            if input.datum_type().is_quantized()
                && input.datum_type().unquantized() == t.datum_type().unquantized()
            {
                t.set_datum_type(input.datum_type());
            }
            Ok(t)
        }
    }

    unsafe fn reduce_t<T, TO, F, A>(
        &self,
        axes: &[usize],
        output_shape: &[usize],
        input_tensor: &Tensor,
        f: F,
        args: A,
    ) -> Tensor
    where
        F: for<'a> Fn(ArrayViewD<'a, T>, A) -> TO,
        T: Copy + Datum,
        TO: Copy + Datum,
        A: Copy,
    {
        use ndarray::*;
        let input = input_tensor.to_array_view_unchecked::<T>();
        let result = Array::from_shape_fn(output_shape, |coords| {
            let slice_spec: Vec<SliceInfoElem> = coords
                .slice()
                .iter()
                .enumerate()
                .map(|(ax, &d)| if axes.contains(&ax) { (..).into() } else { d.into() })
                .collect();
            let slice_info = SliceInfo::<_, IxDyn, IxDyn>::try_from(slice_spec).unwrap();
            let slice = input.slice(&slice_info);
            f(slice, args)
        });
        result.into_tensor()
    }

    // sum is a special citizen: enough activity that it gets "special"
    // treatment. we could use the same "algo" for min, max and prod, to the
    // price of more code in the library. argmax and argmin are more
    // tricky (not associative)
    unsafe fn sum<T>(&self, axes: &[usize], input: &Tensor) -> Tensor
    where
        T: Copy + Datum + num_traits::Zero,
    {
        if axes.len() == 0 {
            return input.to_owned();
        }
        let mut output: Option<ArrayD<T>> = None;
        for axis in axes.iter() {
            let current_input = output
                .as_ref()
                .map(|o| o.view())
                .unwrap_or_else(|| input.to_array_view_unchecked::<T>());
            let mut new_shape = current_input.shape().to_vec();
            let reduced_dim = current_input.shape()[*axis];
            new_shape[*axis] = 1;
            let input_stride = current_input.strides()[*axis] as usize;
            let current_output = if current_input.shape().iter().take(*axis).all(|d| *d == 1) {
                // we are actually summing _reduced_dim_ contiguous vector term to term
                let mut output = ArrayD::<T>::zeros(new_shape);
                let first = current_input.as_ptr();
                let output_ptr = output.as_mut_ptr();
                for i in 0..reduced_dim as isize {
                    let slice = first.offset(i * input_stride as isize);
                    for j in 0..input_stride as isize {
                        *output_ptr.offset(j) = *output_ptr.offset(j) + *slice.offset(j);
                    }
                }
                output
            } else {
                ArrayD::from_shape_fn(new_shape, |coords| {
                    let first: *const T = &current_input[coords];
                    let mut sum = T::zero();
                    for i in 0..reduced_dim {
                        sum = sum + *(first.add(i * input_stride));
                    }
                    sum
                })
            };
            output = Some(current_output);
        }
        output.unwrap().into_tensor()
    }
}

fn argmax_t<T>(v: ArrayViewD<T>, last: bool) -> i64
where
    T: Copy + Datum + num_traits::Bounded + ::std::cmp::PartialOrd,
{
    v.iter()
        .copied()
        .enumerate()
        .fold(
            (0usize, T::min_value()),
            |acc, v| if v.1 > acc.1 || (last && acc.1 == v.1) { v } else { acc },
        )
        .0 as i64
}

fn argmin_t<T>(v: ArrayViewD<T>, last: bool) -> i64
where
    T: Copy + Datum + num_traits::Bounded + ::std::cmp::PartialOrd,
{
    v.iter()
        .copied()
        .enumerate()
        .fold(
            (0usize, T::max_value()),
            |acc, v| if v.1 < acc.1 || (last && acc.1 == v.1) { v } else { acc },
        )
        .0 as i64
}

fn max_t<T>(v: ArrayViewD<T>, _: ()) -> T
where
    T: Copy + Datum + num_traits::Bounded + ::std::cmp::PartialOrd,
{
    v.fold(T::min_value(), |acc, &v| if acc > v { acc } else { v })
}

fn min_t<T>(v: ArrayViewD<T>, _: ()) -> T
where
    T: Copy + Datum + num_traits::Bounded + ::std::cmp::PartialOrd,
{
    v.fold(T::max_value(), |acc, &v| if acc < v { acc } else { v })
}

fn prod_t<T>(v: ArrayViewD<T>, _: ()) -> T
where
    T: Copy + Datum + num_traits::One,
{
    v.fold(T::one(), |acc, &v| acc * v)
}

fn q_prod_t<T>(v: ArrayViewD<T>, zp_scale: (i32, f32)) -> T
where
    T: Copy + num_traits::AsPrimitive<f32> + Bounded + Datum,
    f32: num_traits::AsPrimitive<T>,
{
    let (zp, scale) = zp_scale;
    (v.fold(1f32, |acc, &v| acc * (v.as_() - zp as f32)) * scale.powi(v.len() as i32 - 1)
        + zp as f32)
        .clamp_cast()
}

fn q_sum_t<T>(v: ArrayViewD<T>, zp_scale: (i32, f32)) -> T
where
    T: Copy + Bounded + num_traits::AsPrimitive<i32> + Datum,
    i32: num_traits::AsPrimitive<T>,
{
    let (zp, _) = zp_scale;
    (v.fold(0i32, |acc, &v| acc + v.as_()) - zp * (v.len() as i32 - 1)).clamp_cast()
}

#[derive(Clone, Debug, new, Hash)]
pub struct Reduce {
    pub axes: TVec<usize>,
    pub reducer: Reducer,
}

impl Op for Reduce {
    fn name(&self) -> Cow<str> {
        format!("Reduce<{:?}>", self.reducer).into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axes: {:?}", self.axes)])
    }
    op_as_typed_op!();
}

impl EvalOp for Reduce {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        Ok(tvec!(self.reducer.reduce(&self.axes, &inputs[0])?.into()))
    }
}

impl TypedOp for Reduce {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(self.axes.iter().tuple_windows().all(|(a, b)| a < b));
        if inputs[0].datum_type == TDim::datum_type() {
            bail!("Reduce input must be cast from TDim to i64 beforehand")
        }
        let mut shape: TVec<_> = inputs[0].shape.to_tvec();
        for &ax in &self.axes {
            shape[ax] = 1.to_dim();
        }
        let dt = if let Reducer::ArgMax(_) | Reducer::ArgMin(_) = self.reducer {
            DatumType::I64
        } else {
            inputs[0].datum_type
        };
        Ok(tvec!(dt.fact(shape)))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        let mut letters = 'a'..;
        let axes = (0..inputs[0].rank()).flat_map(|ix| {
            if self.axes.contains(&ix) {
                tvec!(
                    Axis::new(letters.next().unwrap(), inputs.len(), outputs.len()).input(0, ix),
                    Axis::new(letters.next().unwrap(), inputs.len(), outputs.len()).output(0, ix),
                )
            } else {
                tvec!(Axis::new(letters.next().unwrap(), inputs.len(), outputs.len())
                    .input(0, ix)
                    .output(0, ix))
            }
            .into_iter()
        }).collect_vec();
        AxesMapping::new(1, 1, axes)
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        let mut axes = tvec!();
        for reduced in &self.axes {
            if let Some(axis) = change.transform_axis(*reduced) {
                axes.push(axis);
            } else {
                return Ok(None);
            }
        }
        axes.sort();
        let op = Some(Box::new(Self { axes, ..self.clone() }) as _);
        Ok(Some(AxisChangeConsequence::new(model, node, op, change)))
    }

    as_op!();
}
