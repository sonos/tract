use crate::model::ParsingContext;
use crate::pb::NodeProto;
use tract_hir::internal::*;
use tract_hir::ops::nn::DataFormat;

pub fn conv_transpose(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    Ok((expand(ConvTranspose::default()), vec!()))
}

#[derive(Debug, Clone, new, Default, Hash)]
pub struct ConvTranspose {}

impl_dyn_hash!(ConvTranspose);

impl Expansion for ConvTranspose {
    fn name(&self) -> Cow<str> {
        "ConvTranspose".into()
    }

    op_onnx!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 2)?;
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].datum_type, &inputs[1].datum_type)?;
        s.equals(&inputs[0].rank, 4)?;
        s.equals(&inputs[1].rank, 4)?;
        s.equals(&outputs[0].rank, 4)?;
        s.equals(&outputs[0].shape[0], &inputs[0].shape[0])?; // N
        s.equals(&inputs[0].shape[1], &inputs[1].shape[0])?; // O
        s.equals(&outputs[0].shape[1], &inputs[1].shape[1])?; // I
        s.equals(&outputs[0].shape[2], inputs[0].shape[2].bex() + &inputs[1].shape[2] - 1.to_dim())?; // H
        s.equals(&outputs[0].shape[3], inputs[0].shape[3].bex() + &inputs[1].shape[3] - 1.to_dim())?; // W
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        if let Some(k) = &target.outlet_fact(inputs[1])?.konst {
            target.wire_node(
                prefix,
                tract_core::ops::cnn::DeconvUnary::new(k.clone()),
                &[inputs[0]],
            )
        } else {
            bail!("Kernel values are expected to be constant.")
        }
    }
}
