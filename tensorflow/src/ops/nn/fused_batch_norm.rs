use tract_hir::internal::*;
use tract_hir::tract_core::itertools::izip;

use crate::model::ParsingContext;
use crate::tfpb::tensorflow::NodeDef;

pub fn fused_batch_norm(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let epsilon = pb.get_attr_float::<f32>("epsilon")?;
    Ok(expand(FusedBatchNorm::new(epsilon)))
}

#[derive(Debug, Clone, new, Educe)]
#[educe(Hash)]
struct FusedBatchNorm {
    #[educe(Hash(method = "hash_f32"))]
    epsilon: f32,
}

tract_data::impl_dyn_hash!(FusedBatchNorm);

impl Expansion for FusedBatchNorm {
    fn name(&self) -> Cow<str> {
        "FusedBatchNorm".into()
    }

    fn validation(&self) -> Validation {
        Validation::Rounding
    }

    op_tf!();

    /// Registers the inference rules of the operator.
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 5)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, f32::datum_type())?;
        s.equals(&inputs[1].datum_type, f32::datum_type())?;
        s.equals(&inputs[2].datum_type, f32::datum_type())?;
        s.equals(&inputs[3].datum_type, f32::datum_type())?;
        s.equals(&inputs[4].datum_type, f32::datum_type())?;
        s.equals(&outputs[0].datum_type, f32::datum_type())?;
        s.equals(&outputs[0].shape, &inputs[0].shape)?;
        s.equals(&inputs[0].rank, 4)?;
        s.equals(&inputs[1].rank, 1)?;
        s.equals(&inputs[2].rank, 1)?;
        s.equals(&inputs[3].rank, 1)?;
        s.equals(&inputs[4].rank, 1)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        s.equals(&inputs[1].shape[0], &inputs[0].shape[3])?;
        s.equals(&inputs[2].shape[0], &inputs[0].shape[3])?;
        s.equals(&inputs[3].shape[0], &inputs[0].shape[3])?;
        s.equals(&inputs[4].shape[0], &inputs[0].shape[3])?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let scale = target.outlet_fact(inputs[1])?;
        let offset = target.outlet_fact(inputs[2])?;
        let mean = target.outlet_fact(inputs[3])?;
        let variance = target.outlet_fact(inputs[4])?;
        if let (Some(scale), Some(offset), Some(mean), Some(variance)) =
            (&scale.konst, &offset.konst, &mean.konst, &variance.konst)
        {
            let scale = scale.as_slice::<f32>()?;
            let offset = offset.as_slice::<f32>()?;
            let mean = mean.as_slice::<f32>()?;
            let variance = variance.as_slice::<f32>()?;
            let slope: Vec<f32> =
                izip!(variance, scale).map(|(v, s)| s / (v + self.epsilon).sqrt()).collect();
            let inter: Vec<f32> = izip!(offset, mean, &slope).map(|(o, m, s)| o - m * s).collect();
            let shape = tvec!(1, 1, 1, scale.len());
            let slope = tensor1(&slope).into_shape(&shape)?;
            let inter = tensor1(&inter).into_shape(&shape)?;
            let wire = target.wire_node(
                format!("{}.mul", prefix),
                tract_hir::ops::math::mul::unary(slope.into_arc_tensor()),
                &[inputs[0]],
            )?;
            return target.wire_node(
                format!("{}.add", prefix),
                tract_hir::ops::math::add::unary(inter.into_arc_tensor()),
                &wire,
            );
        };
        bail!("Batch norm parameters expected to be known")
    }
}
