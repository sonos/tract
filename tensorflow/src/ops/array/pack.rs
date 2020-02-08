use tract_core::infer::*;
use tract_core::internal::*;
use tract_core::ndarray;

use crate::model::ParsingContext;
use crate::tfpb::tensorflow::NodeDef;

pub fn pack(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let dtype = pb.get_attr_datum_type("T")?;
    let n = pb.input.len();
    let axis = pb.get_attr_int("axis")?;

    Ok(Box::new(Pack::new(dtype, n, axis)))
}

//TODO: incorporate as Concat

#[derive(Debug, Clone, new)]
pub struct Pack {
    t: DatumType,
    n: usize, // The number of inputs
    axis: usize,
}

impl Pack {
    /// Evaluates the operation given the input tensors.
    fn eval_t<T: Datum>(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        use tract_core::ndarray::Axis;
        let arrays =
            inputs.iter().map(|m| Ok(m.cast_to::<T>()?)).collect::<TractResult<Vec<_>>>()?;
        let views: Vec<_> = arrays
            .iter()
            .map(|v| v.to_array_view::<T>().unwrap().insert_axis(Axis(self.axis)))
            .collect();
        let mut shape = views[0].shape().to_vec();
        shape[self.axis] = views.iter().map(|v| v.shape()[self.axis]).sum();
        let mut array = ndarray::Array::<T, _>::default(&*shape);
        let mut offset = 0;
        for v in views {
            let len = v.shape()[self.axis];
            array.slice_axis_mut(Axis(self.axis), (offset..(offset + len)).into()).assign(&v);
            offset += len;
        }
        Ok(tvec![array.into_arc_tensor()])
    }
}

impl Op for Pack {
    fn name(&self) -> Cow<str> {
        "tf.Pack".into()
    }

    not_a_typed_op!();
}

impl StatelessOp for Pack {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let dt = DatumType::super_type_for(inputs.iter().map(|dt| dt.datum_type()))
            .ok_or("Could not find a supertype")?;
        match dt {
            DatumType::TDim => self.eval_t::<TDim>(inputs),
            DatumType::I32 => self.eval_t::<i32>(inputs),
            DatumType::F32 => self.eval_t::<f32>(inputs),
            _ => panic!("unsupported type"),
        }
    }
}

impl InferenceRulesOp for Pack {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        let axis = self.axis;
        check_input_arity(&inputs, self.n)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].rank, inputs[0].rank.bex() + 1)?;
        s.equals_all((0..self.n).map(|i| inputs[i].rank.bex()).collect())?;
        s.given_all((0..self.n).map(move |i| &inputs[i].datum_type), move |s, dts| {
            if let Some(dt) = DatumType::super_type_for(dts) {
                s.equals(&outputs[0].datum_type, dt)?;
            }
            Ok(())
        })?;
        s.given(&inputs[0].rank, move |s, r| {
            for d in 0..r as usize {
                s.equals_all((0..self.n).map(|i| inputs[i].shape[d].bex()).collect())?;
            }
            Ok(())
        })?;
        s.given(&inputs[0].rank, move |s, r| {
            for d in 0..axis {
                s.equals(&outputs[0].shape[d], &inputs[0].shape[d])?;
            }
            if r > 0 {
                for d in axis..r as usize {
                    s.equals(&outputs[0].shape[d + 1], &inputs[0].shape[d])?
                }
            }
            Ok(())
        })?;
        s.equals(&outputs[0].shape[axis], self.n.to_dim())
    }

    as_op!();

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let inputs: TVec<OutletId> = node
            .inputs
            .iter()
            .enumerate()
            .map(|(ix, &o)| {
                Ok(target.wire_node(
                    format!("{}-add_dims-{}", node.name, ix),
                    AxisOp::Add(self.axis),
                    [mapping[&o]].as_ref(),
                )?[0])
            })
            .collect::<TractResult<TVec<OutletId>>>()?;
        target.wire_node(
            &*node.name,
            tract_core::ops::array::Concat::new(
                self.axis as usize,
                tvec!(tract_core::ops::array::ConcatSlice::Var; node.inputs.len()),
            ),
            &*inputs,
        )
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use super::*;
    use num_traits::Zero;

    #[test]
    fn pack_0() {
        let inputs = tvec![rctensor1(&[1, 4]), rctensor1(&[2, 5]), rctensor1(&[3, 6]),];
        assert_eq!(
            Pack::new(DatumType::I32, 3, 0).eval(inputs.clone()).unwrap().remove(0),
            rctensor2(&[[1, 4], [2, 5], [3, 6]])
        );
        assert_eq!(
            Pack::new(DatumType::I32, 3, 1).eval(inputs.clone()).unwrap().remove(0),
            rctensor2(&[[1, 2, 3], [4, 5, 6]])
        );
    }

    #[test]
    fn pack_1() {
        let pack = Pack::new(DatumType::I32, 3, 0);
        let input = rctensor1::<i32>(&[]);
        let exp = rctensor2::<i32, _>(&[[]]);
        let found = pack.eval(tvec![input]).unwrap();

        if let Err(e) = exp.close_enough(&found[0], false) {
            panic!("expected: {:?} found: {:?}. {:?}", exp, found[0], e)
        }
    }

    #[test]
    fn inference_1() {
        let mut pack = Pack::new(DatumType::I32, 2, 0);
        let a = InferenceFact::from(Tensor::from(0i32));
        let b = InferenceFact::from(Tensor::from(TDim::zero()));
        let any = InferenceFact::default();
        let (_, output_facts, _) = pack.infer_facts(tvec![&a, &b], tvec![&any], tvec!()).unwrap();
        let exp: TVec<InferenceFact> =
            tvec!(InferenceFact::dt_shape(DatumType::TDim, vec![2usize]));
        assert_eq!(output_facts, exp)
    }

    #[test]
    fn inference_2() {
        let mut pack = Pack::new(DatumType::I32, 2, 0);
        let a = InferenceFact::from(rctensor0(0i32));
        let b = InferenceFact::from(rctensor0(TDim::zero()));
        let any = InferenceFact::default();
        let (_, output_facts, _) = pack.infer(tvec![&a, &b], tvec![&any], tvec!()).unwrap();
        let exp: TVec<InferenceFact> = tvec!(tensor1(&[TDim::zero(), TDim::zero()]).into());
        assert_eq!(output_facts, exp);
    }
}
