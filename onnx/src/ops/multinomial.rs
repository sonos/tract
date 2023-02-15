use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;

pub fn multinomial(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let dtype = match node.get_attr_opt("dtype")?.unwrap_or(6) {
        6 => DatumType::I32,
        7 => DatumType::I64,
        i => bail!("Unsupported datum type {} for ONNX Multinomial", i),
    };
    let sample_size = node.get_attr_opt("sample_size")?.unwrap_or(1);
    let seed = node.get_attr::<f32>("seed").ok();

    Ok((expand(Multinomial { dtype, sample_size, seed }), vec![]))
}

#[derive(Clone, Debug)]
pub struct Multinomial {
    dtype: DatumType,
    sample_size: i32,
    pub seed: Option<f32>,
}

impl Expansion for Multinomial {
    fn name(&self) -> Cow<str> {
        "Multinomial".into()
    }


    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_output_arity(outputs, 1)?;
        check_input_arity(inputs, 1)?;

        // inputs[0]: tensor(float16), tensor(float), tensor(double) ; [batch_size, class_size]
        // outputs[0]: tensor(int32), tensor(int64) {depending on self.datum_type} ; [batch_size, sample_size]

        s.equals(&inputs[0].rank, 2)?;
        s.equals(&outputs[0].rank, 2)?;
        s.equals(&outputs[0].datum_type, self.dtype)?;
        s.equals(&inputs[0].shape[0], &outputs[0].shape[0])?; // batch_size
        s.equals(&outputs[0].shape[1], self.sample_size.to_dim())?; // sample_size

        Ok(())
    }

    fn wire(
        &self,
        name: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        model.wire_node(
            name,
            tract_onnx_opl::multinomial::Multinomial {
                dtype: self.dtype,
                sample_size: self.sample_size,
                seed: self.seed,
            },
            &[inputs[0]],
        )
    }
}
