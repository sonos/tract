use crate::internal::*;
use crate::infer::*;
use itertools::Itertools;

// FIXME: try to recanonicalize as flatten (maybe extended) / add_dims / rm_dims ?

#[derive(Debug, Clone, new, Default)]
pub struct Reshape {}

impl Reshape {
    fn compute_shape<D: DimLike>(&self, input: &[D], shape: &[isize]) -> TractResult<TVec<D>> {
        if shape.iter().all(|d| *d > 0) {
            return Ok(shape.iter().map(|&d| D::from(d as usize)).collect());
        }
        let mut result: TVec<D> = shape
            .iter()
            .zip(input.iter().chain(std::iter::repeat(&D::from(1))))
            .map(|(&shape, input)| if shape > 0 { D::from(shape as usize) } else { input.clone() })
            .collect();
        if let Some(minus_one) = shape.iter().position(|d| *d == -1) {
            let prod_input: usize =
                input.iter().try_fold(1, |acc, dim| dim.to_integer().map(|a| a as usize * acc))?;
            let prod_shape: usize = result
                .iter()
                .enumerate()
                .filter(|(ix, _)| *ix != minus_one)
                .try_fold(1, |acc, (_, dim)| dim.to_integer().map(|a| a as usize * acc))?;
            result[minus_one] = D::from(prod_input / prod_shape);
        }
        Ok(result)
    }
}

impl Op for Reshape {
    fn name(&self) -> Cow<str> {
        "Reshape".into()
    }

    not_a_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for Reshape {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (input, shape) = args_2!(inputs);
        let shape: Vec<isize> =
            shape.cast_to::<i64>()?.to_array_view::<i64>()?.iter().map(|&i| i as isize).collect();
        let oshape = self.compute_shape(input.shape(), &shape)?;
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
            let shape: Vec<isize> = shape
                .cast_to::<i64>()?
                .to_array_view::<i64>()?
                .iter()
                .map(|&i| i as isize)
                .collect();
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
            let shape_spec: TVec<isize> =
                shape.cast_to::<i64>()?.as_slice::<i64>()?.iter().map(|&i| i as isize).collect();
            let shape = self.compute_shape(&input_shape, &shape_spec)?;
            let op = TypedReshape::new(shape);
            return target.wire_node(&*node.name, op, [mapping[&node.inputs[0]]].as_ref());
        }
        bail!("shape input is variable")
    }

    as_op!();
}

#[derive(Debug, Clone, new, Default)]
pub struct TypedReshape {
    shape: TVec<TDim>,
}

impl Op for TypedReshape {
    fn name(&self) -> Cow<str> {
        "TypedReshape".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("to shape: {}", self.shape.iter().map(|d| format!("{:?}", d)).join("x"))])
    }

    op_as_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for TypedReshape {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let shape: TVec<usize> =
            self.shape.iter().map(|d| Ok(d.to_integer()? as usize)).collect::<TractResult<_>>()?;
        let o = unsafe { input.into_tensor().into_shape(&*shape)?.into_arc_tensor() };
        Ok(tvec!(o))
    }
}

impl TypedOp for TypedReshape {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, &*self.shape)?))
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let input_fact = model.outlet_fact(node.inputs[0])?;
        if input_fact.shape.to_tvec() == self.shape {
            return Ok(Some(TypedModelPatch::shunt_one_op(model, node)?));
        } else if let Ok(shape) =
            self.shape.iter().map(|d| Ok(d.to_integer()? as usize)).collect::<TractResult<_>>()
        {
            return Ok(Some(TypedModelPatch::single_unary_op(
                model,
                node,
                FiniteReshape::new(shape),
            )?));
        }
        Ok(None)
    }
}

#[derive(Debug, Clone, new, Default)]
pub struct FiniteReshape {
    pub shape: TVec<usize>,
}

impl Op for FiniteReshape {
    fn name(&self) -> Cow<str> {
        "FiniteReshape".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("to shape: {}", self.shape.iter().join("x"))])
    }

    op_as_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for FiniteReshape {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let o = unsafe { input.into_tensor().into_shape(&*self.shape)?.into_arc_tensor() };
        Ok(tvec!(o))
    }
}

impl TypedOp for FiniteReshape {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, &*self.shape)?))
    }

    as_op!();
}
