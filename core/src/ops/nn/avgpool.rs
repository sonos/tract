use crate::ops::prelude::*;
use ndarray::prelude::*;
use num_traits::{AsPrimitive, Float};

use super::{DataFormat, PaddingSpec, Patch};

#[derive(Debug, Clone, new, Default)]
pub struct AvgPool {
    data_fmt: DataFormat,
    kernel_shape: TVec<usize>,
    padding: PaddingSpec,
    strides: Option<TVec<usize>>,
    count_include_pad: bool,
}

impl AvgPool {
    fn patch(&self, input_full_shape: &[usize]) -> Patch {
        let hw_rank = self.data_fmt.shape(input_full_shape).hw_rank();
        Patch::new(
            self.data_fmt,
            tvec![1; hw_rank],
            self.kernel_shape.clone(),
            &self.padding,
            self.strides.clone().unwrap_or_else(|| tvec![1; hw_rank]),
            input_full_shape.into(),
        )
    }

    fn eval_t<T>(&self, inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>>
    where
        T: Datum + Float,
        usize: AsPrimitive<T>,
    {
        let patch = self.patch(inputs[0].shape());
        FixedAvgPool::new(patch, self.count_include_pad).eval(inputs)
    }
}

impl Op for AvgPool {
    fn name(&self) -> Cow<str> {
        "AvgPool".into()
    }

    fn reduce(
        &self,
        inputs: TVec<&TensorFact>,
        _outputs: TVec<&TensorFact>,
        phase: ReductionPhase,
    ) -> TractResult<Option<ReducedOpRewire>> {
        if phase == ReductionPhase::Normalize {
            return Ok(None);
        }
        if let (Some(shape), Some(dt)) = (inputs[0].shape.as_concrete_finite()?, inputs[0].datum_type.concretize()) {
            let patch = self.patch(&*shape);
            fn fixed<T>(patch: Patch, count_include_pad: bool) -> Box<Op> 
            where
                T: Datum + Float,
                usize: AsPrimitive<T>,
            {
                Box::new(FixedAvgPool::new(patch, count_include_pad))
            }
            let op = dispatch_floatlike!(fixed(dt)(patch, self.count_include_pad));
            return Ok(Some(ReducedOpRewire::unary(op)));
        }
        Ok(None)
    }
}

impl StatelessOp for AvgPool {
    fn eval(&self, inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        dispatch_floatlike!(Self::eval_t(inputs[0].datum_type())(self, inputs))
    }
}

impl InferenceRulesOp for AvgPool {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&outputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&outputs[0].rank, &inputs[0].rank)?;
        s.given(&inputs[0].shape, move |s, ishape| {
            let ishape = self.data_fmt.shape(ishape);
            let ones = tvec![1; ishape.hw_rank()];
            let computed = self.padding.compute(
                ishape.hw_dims(),
                &*self.kernel_shape,
                &ones,
                self.strides.as_ref().unwrap_or(&ones),
            );
            for (ix, &d) in computed.output.iter().enumerate() {
                s.equals(&outputs[0].shape[ix + ishape.h_axis()], d)?;
            }
            s.equals(&outputs[0].shape[ishape.n_axis()], ishape.n_dim())?;
            s.equals(&outputs[0].shape[ishape.c_axis()], ishape.c_dim())?;
            Ok(())
        })
    }
}

#[derive(Debug, Clone, new)]
pub struct FixedAvgPool<T>
where
    T: Datum + Float,
    usize: AsPrimitive<T>,
{
    patch: Patch,
    count_include_pad: bool,
    _phantom: PhantomData<T>,
}

impl<T: Datum + Float> Op for FixedAvgPool<T>
where
    T: Datum + Float,
    usize: AsPrimitive<T>,
{
    fn name(&self) -> Cow<str> {
        format!("FixedAvgPool<{:?}>", T::datum_type()).into()
    }
}

impl<T> StatelessOp for FixedAvgPool<T>
where
    T: Datum + Float,
    usize: AsPrimitive<T>,
{
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let input = args_1!(inputs);
        let input = input.to_array_view::<T>()?;
        let visitor = self.patch.wrap(&input);
        let shape: TVec<usize> = self.patch.output_full_shape(self.patch.input_shape.c_dim());

        let output = ArrayD::from_shape_fn(&*shape, |coords| -> T {
            let pair = visitor
                .at(&coords.slice())
                .map(|ov| ov.map(|v| (v, true)).unwrap_or((T::zero(), false)))
                .filter(|pair| pair.1 || self.count_include_pad)
                .fold((T::zero(), 0), |acc, pair| (acc.0 + pair.0, acc.1 + 1));
            pair.0 / (pair.1.as_())
        });

        Ok(tvec!(output.into()))
    }
}

impl<T> InferenceRulesOp for FixedAvgPool<T>
where
    T: Datum + Float,
    usize: AsPrimitive<T>,
{
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&outputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, T::datum_type())?;
        s.equals(&outputs[0].datum_type, T::datum_type())?;
        s.equals(&outputs[0].rank, &inputs[0].rank)?;
        s.equals(&inputs[0].shape, ShapeFact::from(&*self.patch.input_shape.shape))?;
        let shape: TVec<usize> = self.patch.output_full_shape(self.patch.input_shape.c_dim());
        s.equals(&outputs[0].shape, ShapeFact::from(shape))?;
        Ok(())
    }
}
