use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;
use tract_onnx_opl::grid_sample::{InterpolationMode, PaddingMode};

pub fn grid_sample(
    ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let mode = match node.get_attr_opt("mode")?.unwrap_or("linear") {
        "bilinear" | "linear" => InterpolationMode::Bilinear,
        "nearest" => InterpolationMode::Nearest,
        "bicubic" | "cubic" => InterpolationMode::Bicubic,
        s => bail!("Unsupported GridSample mode: {}", s),
    };
    let padding_mode = match node.get_attr_opt("padding_mode")?.unwrap_or("zeros") {
        "zeros" => PaddingMode::Zeros,
        "border" => PaddingMode::Border,
        "reflection" => PaddingMode::Reflection,
        s => bail!("Unsupported GridSample padding_mode: {}", s),
    };
    let align_corners = node.get_attr_opt::<i64>("align_corners")?.unwrap_or(0) != 0;

    match ctx.onnx_operator_set_version {
        16.. => {}
        v => bail!("Unsupported operator set for GridSample operator ({v})"),
    }

    Ok((expand(GridSampleInference { mode, padding_mode, align_corners }), vec![]))
}

#[derive(Clone, Debug)]
struct GridSampleInference {
    mode: InterpolationMode,
    padding_mode: PaddingMode,
    align_corners: bool,
}

impl Expansion for GridSampleInference {
    fn name(&self) -> StaticName {
        "GridSample".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 2)?;
        check_output_arity(outputs, 1)?;

        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[1].datum_type, f32::datum_type())?;

        s.equals(&inputs[0].rank, &inputs[1].rank)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;

        s.equals(&inputs[0].shape[0], &inputs[1].shape[0])?;
        s.equals(&inputs[0].shape[0], &outputs[0].shape[0])?;

        s.equals(&inputs[0].shape[1], &outputs[0].shape[1])?;

        s.given(&inputs[0].rank, move |s, rank| {
            let rank = rank as usize;
            let spatial_rank = rank - 2;
            for d in 0..spatial_rank {
                s.equals(&outputs[0].shape[2 + d], &inputs[1].shape[1 + d])?;
            }
            s.equals(&inputs[1].shape[rank - 1], (spatial_rank as i64).to_dim())?;
            Ok(())
        })
    }

    fn wire(
        &self,
        name: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        model.wire_node(
            name,
            tract_onnx_opl::grid_sample::GridSample {
                mode: self.mode.clone(),
                padding_mode: self.padding_mode.clone(),
                align_corners: self.align_corners,
            },
            inputs,
        )
    }
}
