use crate::internal::Axis;
use crate::internal::*;
use crate::ops::binary::TypedBinOp;
use crate::ops::cast::cast;
use crate::ops::change_axes::wire_with_rank_broadcast;
use crate::ops::element_wise::ElementWiseOp;
use crate::ops::math::{div, square, Mul, Square};
use std::convert::TryFrom;
use std::iter::Sum;
use std::mem::transmute;
use tract_data::internal::ClampCast;
use tract_data::itertools::Itertools;
use tract_ndarray::prelude::*;
use tract_num_traits::{AsPrimitive, Bounded};

macro_rules! r {
    ($($path:ident)::* ($dt:expr) ($($args:expr),*)) => {
        match $dt {
            DatumType::U8   => $($path)::*::<u8,_,_,_>($($args),*),
            DatumType::I8   => $($path)::*::<i8,_,_,_>($($args),*),
            DatumType::U16  => $($path)::*::<u16,_,_,_>($($args),*),
            DatumType::I16  => $($path)::*::<i16,_,_,_>($($args),*),
            DatumType::I32  => $($path)::*::<i32,_,_,_>($($args),*),
            DatumType::I64  => $($path)::*::<i64,_,_,_>($($args),*),
            DatumType::F16  => $($path)::*::<f16,_,_,_>($($args),*),
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
            DatumType::F16  => $($path)::*::<f16,_,_,_>($($args),*),
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
    MeanOfSquares,
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
                MeanOfSquares => self.mean_of_squares(axes, input)?,
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
        T: Copy + Datum + num_traits::Zero + Sum,
        f16: AsPrimitive<T>,
        f32: AsPrimitive<T>,
    {
        if axes.len() == 0 {
            return input.to_owned();
        }

        // use tract-optimized path only when single reuction axis and is at end
        if axes.len() > 1 || axes[0] != input.rank() - 1 {
            let mut operative_axes = vec![];
            let mut operative_shape: Vec<usize> = vec![];
            for (ix, dim) in input.shape().iter().enumerate() {
                // axis is reduced, but is not the first of a series of reduced axes
                if ix > 0 && axes.contains(&ix) && axes.contains(&(ix - 1)) {
                    *operative_shape.last_mut().unwrap() *= *dim;
                } else if axes.contains(&ix) {
                    operative_axes.push(operative_shape.len());
                    operative_shape.push(*dim);
                } else {
                    operative_shape.push(*dim);
                }
            }
            let mut output = input
                .to_array_view_unchecked::<T>()
                .into_shape_with_order(operative_shape)
                .unwrap()
                .sum_axis(Axis(*operative_axes.iter().max().unwrap()));

            for axis in operative_axes.iter().rev().skip(1) {
                output = output.sum_axis(Axis(*axis));
            }

            let mut output = output.into_tensor();

            for &axis in axes {
                output.insert_axis(axis).unwrap();
            }

            output
        } else {
            let mut output: Option<ArrayD<T>> = None;
            for axis in axes.iter().copied() {
                let input_view = output
                    .as_ref()
                    .map(|o| o.view())
                    .unwrap_or_else(|| input.to_array_view_unchecked::<T>());

                // Create array that will contain intermidiate result
                let reduced_dim = input_view.shape()[axis];
                let input_stride = input_view.strides()[axis] as usize;
                let output_shape = input_view
                    .shape()
                    .iter()
                    .enumerate()
                    .map(|(idx, dim)| if idx != axis { *dim } else { 1 })
                    .collect_vec();

                output = Some(ArrayD::from_shape_fn(output_shape.clone(), |coords| {
                    let mut view = input_view.view();
                    for ix in 0..output_shape.len() {
                        if ix != axis {
                            view.collapse_axis(Axis(ix), coords[ix]);
                        }
                    }

                    if let Some(slice) = view.as_slice() {
                        if T::datum_type() == f16::datum_type() {
                            let slice: &[f16] = unsafe { std::mem::transmute(slice) };
                            (tract_linalg::ops().sum_f16)()
                                .run_with_params(slice, ())
                                .unwrap()
                                .as_()
                        } else if T::datum_type() == f32::datum_type() {
                            let slice: &[f32] = unsafe { std::mem::transmute(slice) };
                            (tract_linalg::ops().sum_f32)()
                                .run_with_params(slice, ())
                                .unwrap()
                                .as_()
                        } else {
                            slice.iter().cloned().sum::<T>()
                        }
                    } else {
                        let first: *const T = &input_view[coords];
                        let mut sum = T::zero();
                        for i in 0..reduced_dim {
                            sum = sum + *(first.add(i * input_stride));
                        }
                        sum
                    }
                }));
            }
            output.unwrap().into_tensor()
        }
    }

    fn mean_of_squares(&self, axis: &[usize], input: &Tensor) -> TractResult<Tensor> {
        let dt = input.datum_type();
        let mut input = input.cast_to::<f32>()?.into_owned();
        input.as_slice_mut::<f32>()?.iter_mut().for_each(|x| *x = *x * *x);
        let mut output = unsafe { self.sum::<f32>(axis, &input) };
        let norm = output.len() as f32 / input.len() as f32;
        output.as_slice_mut::<f32>()?.iter_mut().for_each(|x| *x *= norm);
        Ok(output.cast_to_dt(dt)?.into_owned())
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
            |acc, v| {
                if v.1 > acc.1 || (last && acc.1 == v.1) {
                    v
                } else {
                    acc
                }
            },
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
            |acc, v| {
                if v.1 < acc.1 || (last && acc.1 == v.1) {
                    v
                } else {
                    acc
                }
            },
        )
        .0 as i64
}

fn max_t<T>(v: ArrayViewD<T>, _: ()) -> T
where
    T: Copy + Datum + num_traits::Bounded + ::std::cmp::PartialOrd,
{
    if T::datum_type() == f32::datum_type() {
        if let Some(slice) = v.as_slice() {
            let slice = unsafe { transmute::<&[T], &[f32]>(slice) };
            (tract_linalg::ops().max_f32)().run(slice).unwrap();
        }
    }
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

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if let Some(patch) = self.declutter_mean_of_square(model, node)? {
            return Ok(Some(patch));
        }
        if let Some(patch) = self.declutter_scalar_mul_then_sum(model, node)? {
            return Ok(Some(patch));
        }
        if let Some(patch) = self.declutter_reduce_reduce(model, node)? {
            return Ok(Some(patch));
        }
        Ok(None)
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        let mut letters = 'a'..;
        let axes = (0..inputs[0].rank())
            .flat_map(|ix| {
                if self.axes.contains(&ix) {
                    tvec!(
                        Axis::new(letters.next().unwrap(), inputs.len(), outputs.len())
                            .input(0, ix),
                        Axis::new(letters.next().unwrap(), inputs.len(), outputs.len())
                            .output(0, ix),
                    )
                } else {
                    tvec!(Axis::new(letters.next().unwrap(), inputs.len(), outputs.len())
                        .input(0, ix)
                        .output(0, ix))
                }
                .into_iter()
            })
            .collect_vec();
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

    fn slice(
        &self,
        patch: &mut TypedModelPatch,
        _model: &TypedModel,
        node: &TypedNode,
        _prefix: &str,
        inputs: &[OutletId],
        output_axis: usize,
        _start: &TDim,
        _end: &TDim,
    ) -> TractResult<Option<TVec<OutletId>>> {
        if self.axes.contains(&output_axis) {
            return Ok(None);
        }
        patch.wire_node(&node.name, &node.op, inputs).map(Some)
    }

    as_op!();
}

impl Reduce {
    fn declutter_reduce_reduce(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let Some(prec) = model.single_prec(node.id)? else {
            return Ok(None);
        };
        let Some(prec_reduce) = prec.op_as::<Self>() else {
            return Ok(None);
        };
        use Reducer::*;
        if prec_reduce.reducer != self.reducer || ![Sum, Prod, Min, Max].contains(&self.reducer) {
            return Ok(None);
        }
        let mut patch = TypedModelPatch::default();
        let wire = patch.tap_model(model, prec.inputs[0])?;
        let wire = patch.wire_node(
            &node.name,
            Self {
                reducer: self.reducer,
                axes: prec_reduce
                    .axes
                    .iter()
                    .chain(self.axes.iter())
                    .copied()
                    .sorted()
                    .dedup()
                    .collect(),
            },
            &[wire],
        )?;
        patch.shunt_outside(model, node.id.into(), wire[0])?;
        Ok(Some(patch))
    }

    fn declutter_scalar_mul_then_sum(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if self.reducer == Reducer::Sum {
            let Some(prec) = model.single_prec(node.id)? else {
                return Ok(None);
            };
            let Some(prec_bin) = prec.op_as::<TypedBinOp>() else {
                return Ok(None);
            };
            if !prec_bin.0.is::<Mul>() {
                return Ok(None);
            }
            let mul_input_fact = model.node_input_facts(prec.id)?;
            let Some(scalar_slot) = mul_input_fact
                .iter()
                .position(|f| f.konst.as_ref().is_some_and(|k| k.volume() == 1))
            else {
                return Ok(None);
            };
            let mut patch = TypedModelPatch::default();
            let scalar = patch.tap_model(model, prec.inputs[scalar_slot])?;
            let wire = patch.tap_model(model, prec.inputs[1 - scalar_slot])?;
            let wire = patch.wire_node(&node.name, self.clone(), &[wire])?[0];
            let wire = patch.wire_node(&prec.name, prec_bin.clone(), &[wire, scalar])?[0];
            patch.shunt_outside(model, node.id.into(), wire)?;
            return Ok(Some(patch));
        }
        Ok(None)
    }

    fn declutter_mean_of_square(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if self.reducer == Reducer::Sum {
            let Some(prec) = model.single_prec(node.id)? else {
                return Ok(None);
            };
            let Some(prec_ew) = prec.op_as::<ElementWiseOp>() else {
                return Ok(None);
            };
            if !prec_ew.0.is::<Square>() {
                return Ok(None);
            }
            if node.outputs.len() != 1 || node.outputs[0].successors.len() != 1 {
                return Ok(None);
            }
            let our_inlet = node.outputs[0].successors[0];
            let succ = model.node(our_inlet.node);
            let Some(succ_bin) = succ.op_as::<TypedBinOp>() else {
                return Ok(None);
            };
            if !succ_bin.0.is::<Mul>() {
                return Ok(None);
            }
            let other = succ.inputs[1 - our_inlet.slot];
            let Some(other_konst) = model.outlet_fact(other)?.uniform.as_ref() else {
                return Ok(None);
            };
            let norm: TDim = self.axes.iter().map(|&ax| &prec.outputs[0].fact.shape[ax]).product();
            let Some(norm) = norm.as_i64() else {
                return Ok(None);
            };
            if norm == 0 {
                return Ok(None);
            }
            let norm = tensor0((norm as f32).recip());
            if other_konst.close_enough(&norm, Approximation::Close).is_ok() {
                let mut patch = TypedModelPatch::default();
                let wire = patch.tap_model(model, prec.inputs[0])?;
                let wire = patch.wire_node(
                    &node.name,
                    Reduce::new(self.axes.clone(), Reducer::MeanOfSquares),
                    &[wire],
                )?[0];
                patch.shunt_outside(model, succ.id.into(), wire)?;
                return Ok(Some(patch));
            }
        }
        Ok(None)
    }
}

pub fn expand_mean_of_squares(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    op: &Reduce,
) -> TractResult<Option<TypedModelPatch>> {
    if op.reducer == Reducer::MeanOfSquares {
        let mut patch = TypedModelPatch::default();
        let mut wire = tvec!(patch.tap_model(model, node.inputs[0])?);
        let input_fact = model.outlet_fact(node.inputs[0])?;
        let dt = input_fact.datum_type;
        if dt != f32::datum_type() {
            wire = patch.wire_node(format!("{name}.to_f32"), cast(f32::datum_type()), &wire)?;
        }
        wire = patch.wire_node(format!("{name}.sqr"), square(), &wire)?;
        wire = patch.wire_node(
            format!("{name}.sum"),
            Reduce::new(op.axes.clone(), Reducer::Sum),
            &wire,
        )?;
        let card = input_fact
            .shape
            .iter()
            .enumerate()
            .filter(|(ix, _dim)| op.axes.contains(ix))
            .map(|(_ix, dim)| dim)
            .product::<TDim>();
        let card = patch.add_const(format!("{name}.card"), tensor0(card))?;
        let card =
            patch.wire_node(format!("{name}.card_to_f32"), cast(f32::datum_type()), &[card])?;

        wire = wire_with_rank_broadcast(
            format!("{name}.norm"),
            &mut patch,
            div(),
            &[wire[0], card[0]],
        )?;
        if dt != f32::datum_type() {
            wire = patch.wire_node(format!("{name}.from_f32"), cast(dt), &wire)?;
        }
        patch.shunt_outside(model, node.id.into(), wire[0])?;
        Ok(Some(patch))
    } else {
        Ok(None)
    }
}
