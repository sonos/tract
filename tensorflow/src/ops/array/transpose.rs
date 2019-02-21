use tract_core::ops::prelude::*;
use crate::tfpb::node_def::NodeDef;

#[derive(Debug, Clone, new)]
pub struct Transpose {
    t: DatumType,
    t_perm: DatumType,
}

pub fn transpose(pb: &NodeDef) -> TractResult<Box<Op>> {
    let t = pb.get_attr_datum_type("T")?;
    let t_perm = pb.get_attr_datum_type("Tperm")?;
    Ok(Box::new(Transpose::new(t, t_perm)))
}

impl Transpose {
    fn compute_shape<D: DimLike>(shape: &[D], perm: &[i32]) -> TVec<D> {
        let mut new_shape = tvec![D::zero(); shape.len()];
        for (ix, &d) in perm.iter().enumerate() {
            new_shape[ix] = shape[d as usize];
        }
        new_shape
    }

    fn eval_t<T: Datum>(&self, input: SharedTensor, perm: &[usize]) -> TractResult<TVec<SharedTensor>> {
        Ok(tvec![input.to_array::<T>()?.permuted_axes(perm).into()])
    }
}

impl Op for Transpose {
    fn name(&self) -> Cow<str> {
        "tf.Transpose".into()
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let inputs = model.node_input_facts(node.id)?;
        if let Some(ref perm) = inputs[1].konst {
            let perm:Vec<usize> = perm.cast_to::<i32>()?.as_slice::<i32>()?.iter().map(|&x| x as usize).collect();
            let op = ::tract_core::ops::array::PermuteAxes::new(Some(perm));
            return Ok(Some(TypedModelPatch::single_unary_op(&model, &node, op)?))
        }
        Ok(None)
    }
}

impl StatelessOp for Transpose {
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let (data, perm) = args_2!(inputs);
        let perm:TVec<usize> = perm.cast_to::<i32>()?.as_slice::<i32>()?.iter().map(|&x| x as usize).collect();
        dispatch_datum!(Self::eval_t(data.datum_type())(self, data, &*perm))
    }
}

impl InferenceRulesOp for Transpose {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_output_arity(&inputs, 2)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, self.t)?;
        s.equals(&inputs[1].datum_type, self.t_perm)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&outputs[0].rank, &inputs[0].rank)?;
        s.equals(&inputs[1].rank, 1)?;
        s.equals(&inputs[1].shape[0], inputs[0].rank.bex().to_dim())?;
        s.given_2(&inputs[0].shape, &inputs[1].value, move |s, shape, perm| {
            let perm = perm.cast_to::<i32>()?;
            let output_shape = Self::compute_shape(&shape, perm.as_slice::<i32>()?);
            s.equals(&outputs[0].shape, output_shape)
        })
    }
}
