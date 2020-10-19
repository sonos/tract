use crate::model::ParsingContext;
use crate::tfpb::tensorflow::NodeDef;
use tract_hir::internal::*;

pub fn gather_v2(_ctx: &ParsingContext, _pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    Ok(expand(GatherV2))
}

#[derive(Debug, Clone, new, Hash)]
pub struct GatherV2;

tract_data::impl_dyn_hash!(GatherV2);

impl Expansion for GatherV2 {
    fn name(&self) -> Cow<str> {
        "GatherV2".into()
    }

    op_tf!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 3)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[1].datum_type, i32::datum_type())?;
        s.equals(&inputs[2].datum_type, i32::datum_type())?;
        s.equals(&inputs[2].rank, 0)?;
        s.given_3(
            &inputs[0].shape,
            &inputs[1].shape,
            &inputs[2].value,
            move |s, input_shape, indices_shape, axis| {
                let mut axis = *axis.to_scalar::<i32>()?;
                if axis < 0 {
                    axis = input_shape.len() as i32 + axis;
                }
                let op = tract_hir::ops::array::Gather::new(axis as usize);
                let output_shape = op.compute_output_shape(&input_shape, &indices_shape)?;
                s.equals(&outputs[0].shape, output_shape)
            },
        )
    }

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        if let Some(axis) = target.outlet_fact(inputs[2])?.konst.as_ref() {
            let mut axis = *axis.to_scalar::<i32>()?;
            if axis < 0 {
                axis = target.outlet_fact(inputs[0])?.rank() as i32 + axis;
            }
            let op = tract_hir::ops::array::Gather::new(axis as usize);
            target.wire_node(&*prefix, op, &inputs[0..2])
        } else {
            bail!("Need to know axis to type GatherV2")
        }
    }
}
