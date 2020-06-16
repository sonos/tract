use crate::infer::*;
use crate::internal::*;

use tract_core::ops::array::TypedReshape;

#[derive(Debug, Clone, new, Default, Hash)]
pub struct Reshape {}

tract_linalg::impl_dyn_hash!(Reshape);

impl Reshape {
    fn compute_shape(&self, input: &[TDim], shape: &Tensor) -> TractResult<TVec<TDim>> {
        if let (Ok(ishape), Ok(shape)) = (
            input
                .iter()
                .map(|d| d.to_integer().map(|d| d as usize))
                .collect::<TractResult<TVec<usize>>>(),
            shape
                .cast_to::<i32>()
                .map(|t| t.as_slice::<i32>().unwrap().to_vec())
                .map(|d| d.iter().map(|d| *d as isize).collect::<TVec<_>>()),
        ) {
            self.compute_shape_ints(&ishape, &shape).map(|d| d.iter().map(|d| d.to_dim()).collect())
        } else if shape
            .cast_to::<TDim>()?
            .as_slice::<TDim>()?
            .iter()
            .any(|d| d.to_integer().map(|d| d > 0).unwrap_or(false))
        {
            Ok(shape.cast_to::<TDim>()?.as_slice::<TDim>()?.into())
        } else {
            bail!("Can not compute shape with streaming dimension and -1 placeholder")
        }
    }

    fn compute_shape_ints(&self, input: &[usize], shape: &[isize]) -> TractResult<TVec<usize>> {
        let mut shape = shape.to_vec();
        for i in 0..shape.len() {
            if shape[i] == 0 {
                shape[i] = *input.get(i).ok_or("Can not use -1 in Reshape shape for axis beyond data rank: input shape: {:?} reshaping to: {:?}")? as isize;
            }
        }
        if shape.iter().all(|d| *d > 0) {
            return Ok(shape.iter().map(|&d| d as usize).collect());
        }
        let volume = input.iter().product::<usize>();
        let partial_volume = shape.iter().copied().filter(|&d| d > 0).product::<isize>() as usize;
        Ok(shape
            .iter()
            .copied()
            .map(|d| if d > 0 { d as usize } else { volume / partial_volume })
            .collect())
    }
}

impl Op for Reshape {
    fn name(&self) -> Cow<str> {
        "Reshape".into()
    }

    op_hir!();
    not_a_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for Reshape {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (input, shape) = args_2!(inputs);
        let shape: Vec<isize> =
            shape.cast_to::<i64>()?.to_array_view::<i64>()?.iter().map(|&i| i as isize).collect();
        let oshape = self.compute_shape_ints(input.shape(), &shape)?;
        unsafe { Ok(tvec![input.into_tensor().into_shape(&*oshape)?.into_arc_tensor()]) }
    }
}

impl InferenceRulesOp for Reshape {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.given_2(&inputs[0].shape, &inputs[1].value, move |s, ishape, shape| {
            let shape = self.compute_shape(&ishape, &shape)?;
            s.equals(&outputs[0].shape, ShapeFactoid::from(shape))
        })
    }

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        if let Some(ref shape) = target.outlet_fact(mapping[&node.inputs[1]])?.konst {
            let input_shape: TVec<TDim> =
                target.outlet_fact(mapping[&node.inputs[0]])?.shape.to_tvec();
            let shape = self.compute_shape(&input_shape, &shape)?;
            let op = TypedReshape::new(shape);
            return target.wire_node(&*node.name, op, [mapping[&node.inputs[0]]].as_ref());
        }
        bail!("shape input is variable")
    }

    as_op!();
}
