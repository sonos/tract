use std::ops::Mul;

use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::NodeProto;
use rustfft::num_traits::{Float, FromPrimitive};
use rustfft::{FftDirection, FftNum};
use tract_hir::internal::*;
use tract_hir::ops::array::Pad;
use tract_hir::ops::cast::cast;
use tract_hir::ops::math::div;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("DFT", dft)
}

pub fn dft(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(1);
    let inverse = node.get_attr_opt("inverse")?.unwrap_or(0i64) != 0;
    let onesided = node.get_attr_opt("onesided")?.unwrap_or(0) != 0;
    if onesided {
        bail!("onesided is not supported yet")
    }
    Ok((
        expand(OnnxDft { axis, inverse, onesided, has_length_input: node.input.len() == 2 }),
        vec![],
    ))
}

#[derive(Clone, Debug, Hash)]
struct OnnxDft {
    axis: usize,
    inverse: bool,
    onesided: bool,
    has_length_input: bool,
}

impl_dyn_hash!(OnnxDft);

impl Expansion for OnnxDft {
    fn name(&self) -> Cow<str> {
        "DFT".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 1 + self.has_length_input as usize)?;
        check_output_arity(outputs, 1)?;

        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        if self.has_length_input {
            s.equals(&inputs[1].rank, 0)?;
        }
        s.given(&inputs[0].rank, |s, rank| {
            for ax in 0..rank as usize - 1 {
                if ax != self.axis {
                    s.equals(&inputs[0].shape[ax], &outputs[0].shape[ax])?;
                }
            }
            s.equals(&outputs[0].shape[rank as usize - 1], 2.to_dim())?;
            Ok(())
        })?;
        if self.has_length_input {
            s.given(&inputs[1].value[0], |s, len| {
                s.equals(len.to_dim(), &outputs[0].shape[self.axis])
            })?;
        } else {
            s.equals(&inputs[0].shape[self.axis], &outputs[0].shape[self.axis])?;
        }
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let fact = model.outlet_fact(inputs[0])?.clone();
        let mut wire: TVec<OutletId> = inputs.into();
        if fact.shape.last() == Some(&1.to_dim()) {
            let mut pads = vec![(0, 0); fact.rank() - 1];
            pads.push((0, 1));
            wire = model.wire_node(
                format!("{}.add_imaginary", prefix),
                Pad { mode: tract_hir::ops::array::PadMode::Constant(rctensor0(0f32)), pads },
                &wire,
            )?;
        };
        wire = model.wire_node(
            format!("{}.pair_to_cplx", prefix),
            tract_core::ops::math::InnerDimToComplex,
            &wire,
        )?;
        wire = model.wire_node(
            format!("{}.fft", prefix),
            Fft { axis: self.axis, inverse: self.inverse },
            &wire,
        )?;
        wire = model.wire_node(
            format!("{}.to_pair", prefix),
            tract_core::ops::math::ComplexToInnerDim,
            &wire,
        )?;
        if self.inverse {
            let len = model.add_const(
                format!("{}.len", prefix),
                tensor0(fact.shape[self.axis].clone()).broadcast_into_rank(fact.rank())?,
            )?;
            let casted =
                model.wire_node(format!("{}.cast", prefix), cast(fact.datum_type), &[len])?;
            wire = model.wire_node(format!("{}.norm", prefix), div(), &[wire[0], casted[0]])?;
        }
        Ok(wire)
    }
}

#[derive(Clone, Debug, Hash)]
struct Fft {
    axis: usize,
    inverse: bool,
}

impl_dyn_hash!(Fft);

impl Fft {
    fn eval_t<T: Datum + FftNum + FromPrimitive + Float>(
        &self,
        tensor: &mut Tensor,
    ) -> TractResult<()>
    where
        Complex<T>: Datum + Mul<Complex<T>, Output = Complex<T>>,
    {
        let mut iterator_shape: TVec<usize> = tensor.shape().into();
        iterator_shape[self.axis] = 1;
        let len = tensor.shape()[self.axis];
        let direction = if self.inverse { FftDirection::Inverse } else { FftDirection::Forward };
        let fft = rustfft::FftPlanner::new().plan_fft(len, direction);
        let mut array = tensor.to_array_view_mut::<Complex<T>>()?;
        let mut v = Vec::with_capacity(len);
        for coords in tract_ndarray::indices(&*iterator_shape) {
            v.clear();
            let mut slice = array.slice_each_axis_mut(|ax| {
                if ax.axis.index() == self.axis {
                    (..).into()
                } else {
                    let c = coords[ax.axis.index()] as isize;
                    (c..=c).into()
                }
            });
            v.extend(slice.iter().copied());
            fft.process(&mut *v);
            slice.iter_mut().zip(v.iter()).for_each(|(s, v)| *s = *v);
        }
        Ok(())
    }
}

impl Op for Fft {
    fn name(&self) -> Cow<str> {
        "Fft".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![if self.inverse { "inverse" } else { "forward" }.into()])
    }

    op_as_typed_op!();
}

impl EvalOp for Fft {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let mut tensor = args_1!(inputs).into_tensor();
        match tensor.datum_type() {
            DatumType::ComplexF32 => self.eval_t::<f32>(&mut tensor)?,
            DatumType::ComplexF64 => self.eval_t::<f64>(&mut tensor)?,
            _ => bail!("FFT not implemented for type {:?}", tensor.datum_type()),
        }
        Ok(tvec!(tensor.into_tvalue()))
    }
}

impl TypedOp for Fft {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        if !inputs[0].datum_type.is_complex() {
            bail!("Fft operators expect input in complex form");
        }
        Ok(tvec!(inputs[0].without_value()))
    }

    as_op!();
}
