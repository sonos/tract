#![allow(clippy::unnecessary_cast)]

use crate::internal::*;
use crate::ops::element_wise::{ElementWiseMiniOp, ElementWiseOp};
use crate::ops::math::QScale;
use num_traits::AsPrimitive;
use tract_linalg::Scaler;
use tract_linalg::lut::Lut;
use tract_linalg::mmm::RoundingPolicy;

use super::binary::TypedBinOp;
use super::math::round_ties_to_even;

/// Byte-wise lookup table over an 8-bit tensor.
///
/// It rewrites each element through `table`, so it accepts any single-byte type
/// (`i8`/`u8` and their quantized forms) and preserves the input type by
/// default. Wrapped in an `ElementWiseOp` carrying an output-type override, a
/// requantization can fuse into it and change the result's quantization
/// parameters (see `Cast::codegen`).
#[derive(Debug, Clone)]
pub struct LookupTable {
    pub table: Box<dyn Lut>,
}

impl PartialEq for LookupTable {
    fn eq(&self, other: &Self) -> bool {
        *self.table == *other.table
    }
}

impl Eq for LookupTable {}

impl ElementWiseMiniOp for LookupTable {
    fn name(&self) -> String {
        format!("{}LookupTable", self.prefix())
    }

    fn output_type(&self, input_type: DatumType) -> Option<DatumType> {
        (input_type.size_of() == 1).then_some(input_type)
    }

    fn eval_out_of_place(&self, t: &Tensor, out_dt: Option<DatumType>) -> TractResult<Tensor> {
        ensure!(
            t.datum_type().size_of() == 1,
            "LookupTable expects a byte tensor, got {:?}",
            t.datum_type()
        );
        let mut dst =
            unsafe { Tensor::uninitialized_dt(out_dt.unwrap_or(t.datum_type()), t.shape())? };
        dst.as_bytes_mut().copy_from_slice(t.as_bytes());
        self.table.run(dst.as_bytes_mut());
        Ok(dst)
    }
}

pub fn lookup_table(table: Box<dyn Lut>) -> ElementWiseOp {
    ElementWiseOp(Box::new(LookupTable { table }), None)
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Scale;

impl crate::ops::binary::BinMiniOp for Scale {
    fn name(&self) -> &'static str {
        "Scale"
    }
    fn result_datum_type(&self, a: DatumType, b: DatumType) -> TractResult<DatumType> {
        if !a.is_float() {
            bail!("Scale left operand must be float, got {:?}", a);
        }
        Ok(b)
    }

    fn operating_datum_type(&self, a: DatumType, b: DatumType) -> TractResult<DatumType> {
        if !a.is_float() {
            bail!("Scale left operand must be float, got {:?}", a);
        }
        Ok(b)
    }

    fn eval_out_of_place(&self, c: &mut Tensor, a: &Tensor, b: &Tensor) -> TractResult<()> {
        let a = a.cast_to::<f32>()?;
        let a = a.to_plain_array_view::<f32>()?;
        unsafe fn eval_out_of_place_t<T: Datum + AsPrimitive<f32>>(
            c: &mut Tensor,
            a: &ndarray::ArrayViewD<f32>,
            b: &Tensor,
        ) where
            f32: AsPrimitive<T>,
        {
            let b = unsafe { b.to_array_view_unchecked::<T>() };
            let mut c = unsafe { c.to_array_view_mut_unchecked::<T>() };
            ndarray::Zip::from(&mut c)
                .and_broadcast(a)
                .and_broadcast(b)
                .for_each(|c, a, b| *c = scale_by(*b, *a))
        }
        unsafe { dispatch_numbers!(eval_out_of_place_t(b.datum_type())(c, &a, b)) }
        Ok(())
    }

    fn eval_in_a(&self, a: &mut Tensor, b: &Tensor) -> TractResult<()> {
        let mut a_plain = a.try_as_plain_mut()?;
        let a = a_plain.to_array_view_mut::<f32>()?;
        let b = b.to_plain_array_view::<f32>()?;
        ndarray::Zip::from(a).and_broadcast(b).for_each(|a, b| *a = scale_by(*b, *a));
        Ok(())
    }

    fn is_commutative(&self) -> bool {
        false
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let a = model.outlet_fact(node.inputs[0])?;
        if let Some(a) = &a.uniform {
            if a.cast_to_scalar::<f32>()? == 1. {
                return Ok(Some(TypedModelPatch::rewire(
                    model,
                    &node.inputs[1..2],
                    &[node.id.into()],
                    &|_p, x| Ok(x.into()),
                )?));
            } else if node.outputs[0].fact.datum_type == DatumType::I32 {
                let factor = a.cast_to_scalar::<f32>()?;
                let scaler = Scaler::new(factor, RoundingPolicy::Even);

                let op = ElementWiseOp(Box::new(QScale { scaler }), None);
                let patch =
                    TypedModelPatch::replace_single_op(model, node, &node.inputs[1..2], op)?;

                return Ok(Some(patch));
            }
        }
        Ok(None)
    }
}

#[inline]
pub(crate) fn scale_by<T: Datum + AsPrimitive<f32>>(b: T, a: f32) -> T
where
    f32: AsPrimitive<T>,
{
    let b = b.as_();
    (round_ties_to_even(b.abs() * a) * b.signum()).as_()
}

pub fn scale() -> TypedBinOp {
    TypedBinOp(Box::new(Scale), None)
}

/// Offsets i8 integers as u8 integers.
pub(crate) fn offset_i8_as_u8_elementwise(x: i8) -> u8 {
    (x as u8).wrapping_add(128)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OffsetI8asU8;
impl ElementWiseMiniOp for OffsetI8asU8 {
    fn name(&self) -> String {
        format!("{}{}", self.prefix(), stringify!(OffsetI8asU8))
    }
    fn output_type(&self, input_type: DatumType) -> Option<DatumType> {
        Some(if let DatumType::QI8(qp) = input_type {
            let (zp, scale) = qp.zp_scale();
            DatumType::QU8(QParams::ZpScale { zero_point: zp + 128, scale })
        } else if input_type == DatumType::I8 {
            DatumType::U8
        } else {
            input_type
        })
    }
    fn eval_out_of_place(&self, t: &Tensor, out_dt: Option<DatumType>) -> TractResult<Tensor> {
        let output_type = out_dt.unwrap_or(self.output_type(t.datum_type()).unwrap());
        let mut dst = unsafe { Tensor::uninitialized_dt(output_type, t.shape())? };
        if t.datum_type().unquantized() == i8::datum_type() {
            t.try_as_plain()?
                .as_slice::<i8>()?
                .iter()
                .zip(dst.try_as_plain_mut()?.as_slice_mut::<u8>()?.iter_mut())
                .for_each(|(x, y)| *y = offset_i8_as_u8_elementwise(*x));
            return Ok(dst);
        }

        bail!("{} does not support {:?}", self.name(), t.datum_type());
    }
}

pub fn offset_i8_as_u8() -> ElementWiseOp {
    ElementWiseOp(Box::new(OffsetI8asU8 {}), None)
}

/// Offsets u8 integers as i8 integers.
pub(crate) fn offset_u8_as_i8_elementwise(x: u8) -> i8 {
    x.wrapping_sub(128) as i8
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OffsetU8asI8;
impl ElementWiseMiniOp for OffsetU8asI8 {
    fn name(&self) -> String {
        format!("{}{}", self.prefix(), stringify!(OffsetU8asI8))
    }
    fn output_type(&self, input_type: DatumType) -> Option<DatumType> {
        Some(if let DatumType::QU8(qp) = input_type {
            let (zp, scale) = qp.zp_scale();
            DatumType::QI8(QParams::ZpScale { zero_point: zp - 128, scale })
        } else if input_type == DatumType::U8 {
            DatumType::I8
        } else {
            input_type
        })
    }
    fn eval_out_of_place(&self, t: &Tensor, out_dt: Option<DatumType>) -> TractResult<Tensor> {
        let output_type = out_dt.unwrap_or(self.output_type(t.datum_type()).unwrap());
        let mut dst = unsafe { Tensor::uninitialized_dt(output_type, t.shape())? };
        if t.datum_type().unquantized() == u8::datum_type() {
            t.try_as_plain()?
                .as_slice::<u8>()?
                .iter()
                .zip(dst.try_as_plain_mut()?.as_slice_mut::<i8>()?.iter_mut())
                .for_each(|(x, y)| *y = offset_u8_as_i8_elementwise(*x));
            return Ok(dst);
        }

        bail!("{} does not support {:?}", self.name(), t.datum_type());
    }
}
pub fn offset_u8_as_i8() -> ElementWiseOp {
    ElementWiseOp(Box::new(OffsetU8asI8 {}), None)
}

#[cfg(test)]
pub mod scale {
    use crate::internal::*;
    use crate::ops::einsum::EinSum;
    use crate::ops::math::round_ties_to_even;
    use proptest::prelude::*;

    fn test_scale(a: i8, b: i8, scale: f32) {
        let expected = (((a as i32) * (b as i32)) as f32) / scale;
        let expected = round_ties_to_even(expected.abs()) * expected.signum();
        let expected = (expected as i32).clamp(-128, 127);
        let expected = tensor2(&[[expected as i8]]);

        let input = tvec!(tensor2(&[[b]]).into_tvalue());
        let mut model = TypedModel::default();
        let a = model.add_const("a", tensor2(&[[a]])).unwrap();
        let b = model.add_source("b", i8::fact([1, 1])).unwrap();
        let bias = model.add_const("bias", tensor0(0i32)).unwrap();
        let a0 = model.add_const("a0", tensor0(0i8)).unwrap();
        let a_scale = model.add_const("a_scale", tensor0(1f32)).unwrap();
        let b0 = model.add_const("b0", tensor0(0i8)).unwrap();
        let b_scale = model.add_const("b_scale", tensor0(1f32)).unwrap();
        let c0 = model.add_const("c0", tensor0(0i8)).unwrap();
        let c_scale = model.add_const("c_scale", tensor0(scale)).unwrap();
        let op = EinSum {
            axes: "mk,kn,,,,,,,->mn".parse().unwrap(),
            operating_dt: i32::datum_type(),
            q_params: Some(i8::datum_type()),
        };
        let output = model
            .wire_node("mmm", op, &[a, b, bias, a0, a_scale, b0, b_scale, c0, c_scale])
            .unwrap();
        model.select_output_outlets(&output).unwrap();

        let plain = model.clone().into_runnable().unwrap().run(input.clone()).unwrap();
        assert_eq!(*plain[0], expected);

        let optim = model.into_optimized().unwrap().into_runnable().unwrap().run(input).unwrap();
        assert_eq!(*optim[0], expected);
    }

    proptest! {
        #[test]
        fn prop(a in any::<i8>(), b in any::<i8>(), scale in 0.00001f32..1000.) {
            test_scale(a, b, scale);
        }
    }

    #[test]
    fn t1() {
        test_scale(-117, 15, 37.753822);
    }

    #[test]
    fn t2() {
        test_scale(-4, -60, 475.21674);
    }
}
