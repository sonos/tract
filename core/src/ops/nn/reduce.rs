use crate::internal::*;
use ndarray::prelude::*;

#[derive(Clone, Copy, Debug)]
pub enum Reducer {
    Max,
    Min,
    Prod,
    Sum,
}

impl Reducer {
    pub fn reduce(&self, axes: &[usize], input: &Tensor) -> TractResult<Tensor> {
        let dt = input.datum_type();
        let output_shape: Vec<usize> = input
            .shape()
            .iter()
            .enumerate()
            .map(|(ax, &d)| if axes.contains(&ax) { 1 } else { d })
            .collect();
        unsafe {
            let mut output = Tensor::uninitialized_dt(dt, &*output_shape)?;
            let ostrides = output.strides();
            let istrides = input.strides();
            dispatch_numbers!(Reducer::init_t(dt)(self, &mut output))?;
            let reduced_shape = input
                .shape()
                .iter()
                .enumerate()
                .map(|(ix, d)| if axes.contains(&ix) { *d } else { 1 })
                .collect::<TVec<_>>();
            ndarray::indices(&*output_shape).into_iter().for_each(|output_indices| {
                let o_offset =
                    izip!(&*ostrides, output_indices.slice()).map(|(a, b)| a * b).sum::<usize>();
                ndarray::indices(&*reduced_shape).into_iter().for_each(|reduced_indices| {
                    let i_offset =
                        izip!(&*istrides, output_indices.slice(), reduced_indices.slice())
                            .map(|(s, o, r)| s * (o + r))
                            .sum::<usize>();
                    dispatch_numbers!(Reducer::fold_t(dt)(
                        self,
                        &mut output,
                        input,
                        o_offset,
                        i_offset
                    ))
                    .unwrap();
                });
            });
            Ok(output)
        }
    }

    unsafe fn init_t<T>(&self, output: &mut Tensor) -> TractResult<()>
    where
        T: Copy + Datum + num_traits::Bounded + num_traits::One + num_traits::Zero,
    {
        use Reducer::*;
        let v = match self {
            Min => T::max_value(),
            Max => T::min_value(),
            Prod => T::one(),
            Sum => T::zero(),
        };
        output.as_slice_mut_unchecked::<T>().iter_mut().for_each(|s| *s = v);
        Ok(())
    }

    unsafe fn fold_t<T>(
        &self,
        output: &mut Tensor,
        input: &Tensor,
        o: usize,
        i: usize,
    ) -> TractResult<()>
    where
        T: Copy
            + Datum
            + num_traits::Bounded
            + num_traits::One
            + num_traits::Zero
            + std::cmp::PartialOrd,
    {
        use Reducer::*;
        let i = &input.as_slice_unchecked::<T>()[i];
        let o = &mut output.as_slice_mut_unchecked::<T>()[o];
        match self {
            Min => {
                if *i < *o {
                    *o = *i
                }
            }
            Max => {
                if *i > *o {
                    *o = *i
                }
            }
            Sum => *o = *o + *i,
            Prod => *o = *o * *i,
        }
        Ok(())
    }
}

#[derive(Clone, Debug, new)]
pub struct Reduce {
    axes: TVec<usize>,
    reducer: Reducer,
}

impl Op for Reduce {
    fn name(&self) -> Cow<str> {
        format!("Reduce<{:?}>", self.reducer).into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axes: {:?}", self.axes)])
    }
    canonic!();
    op_as_typed_op!();
    op_as_pulsed_op!();
}

impl StatelessOp for Reduce {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        Ok(tvec!(self.reducer.reduce(&*self.axes, inputs[0].as_ref())?.into_arc_tensor()))
    }
}

impl TypedOp for Reduce {
    as_op!();
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut shape: TVec<_> = inputs[0].shape.to_tvec();
        for &ax in &self.axes {
            shape[ax] = 1.to_dim();
        }
        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, &*shape)?))
    }

    #[allow(unused_variables)]
    fn invariants(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Invariants> {
        let input = model.outlet_fact(node.inputs[0])?;
        let axes = (0..input.rank())
            .filter(|axis| !self.axes.contains(axis))
            .map(|axis| AxisInfo::simple(axis))
            .collect::<TVec<_>>();
        Ok(axes.into())
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
        let op = Some(Box::new(Self { axes, ..self.clone() }) as _);
        Ok(Some(AxisChangeConsequence::new(model, node, op, change)))
    }

    fn pulsify(
        &self,
        _source: &NormalizedModel,
        node: &NormalizedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
        _pulse: usize,
    ) -> TractResult<TVec<OutletId>> {
        let input = mapping[&node.inputs[0]];
        let axis = target.outlet_fact(input)?.axis;
        if self.axes.contains(&axis) {
            bail!("Can not reduce over streaming axis");
        }
        target.wire_node(&*node.name, self.clone(), &[input])
    }
}

impl PulsedOp for Reduce {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        for &ax in &self.axes {
            fact.shape[ax] = 1;
        }
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
