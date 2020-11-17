use crate::internal::*;
use tract_ndarray::prelude::*;

#[derive(Debug, PartialEq, Clone, Hash)]
pub struct OneHot {
    pub axis: usize,
    pub dim: usize,
    pub off: Arc<Tensor>,
    pub on: Arc<Tensor>,
}

impl_dyn_hash!(OneHot);

impl Op for OneHot {
    fn name(&self) -> Cow<str> {
        "Onehot".into()
    }

    op_core!();
    op_as_typed_op!();
}

impl TypedOp for OneHot {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut shape = inputs[0].shape.to_tvec();
        shape.insert(self.axis, self.dim.to_dim());
        Ok(tvec!(TypedFact::dt_shape(self.off.datum_type(), &*shape)))
    }

    fn invariants(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Invariants> {
        let mut axes = vec![];
        for i in 0..model.outlet_fact(node.inputs[0])?.rank() {
            axes.push(AxisInfo {
                inputs: tvec!(Some(i)),
                outputs: tvec!(Some(i + (i >= self.axis) as usize)),
                period: 1,
                disposable: true,
            });
        }
        Ok(axes.into_iter().collect())
    }

    as_op!();
}

impl EvalOp for OneHot {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let mut shape: TVec<usize> = input.shape().into();
        shape.insert(self.axis, self.dim);
        unsafe {
            let mut output = self.off.broadcast_scalar_to_shape(&mut shape)?;
            dispatch_datum_by_size!(Self::eval_t(self.off.datum_type())(
                self,
                &input,
                &mut output
            ))?;
            Ok(tvec!(output.into_arc_tensor()))
        }
    }
}

impl OneHot {
    unsafe fn eval_t<T: Datum + Clone>(
        &self,
        input: &Tensor,
        output: &mut Tensor,
    ) -> TractResult<()> {
        let on = self.on.to_scalar_unchecked::<T>();
        let mut shape: TVec<usize> = input.shape().into();
        shape.insert(self.axis, self.dim);
        let mut array = output.to_array_view_mut_unchecked::<T>();
        let input = input.cast_to::<i32>()?;
        let input = input.to_array_view::<i32>()?;
        for icoord in tract_ndarray::indices_of(&input) {
            let mut ocoord: Vec<usize> = icoord.slice().into();
            let coord = input[&icoord];
            let coord = if coord < 0 { coord + self.dim as i32 } else { coord } as usize;
            ocoord.insert(self.axis, coord);
            array[&*ocoord] = on.clone();
        }
        Ok(())
    }
}
