use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::NodeProto;
use tract_hir::internal::*;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("DepthToSpace", depth_to_space);
}

pub fn depth_to_space(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let blocksize = node.get_attr_opt("blocksize")?.unwrap_or(2);
    let mode = depth_to_space_mode(node)?;
    Ok((expand(DepthToSpace { blocksize, mode }), vec![]))
}

pub fn depth_to_space_mode(node: &NodeProto) -> TractResult<DepthToSpaceMode> {
    let mode = match node.get_attr_opt("mode")? {
        None => None,
        Some(mode) => node.check_value(
            "mode",
            match mode {
                "DCR" => Ok(Some(DepthToSpaceMode::DCR)),
                "CRD" => Ok(Some(DepthToSpaceMode::CRD)),
                _ => Err(mode),
            },
        )?,
    }
    .unwrap_or(DepthToSpaceMode::DCR);
    Ok(mode)
}

#[derive(Debug, Clone, Hash)]
pub enum DepthToSpaceMode {
    DCR,
    CRD,
}

#[derive(Debug, Clone, Hash)]
struct DepthToSpace {
    blocksize: usize,
    mode: DepthToSpaceMode,
}

impl_dyn_hash!(DepthToSpace);

impl DepthToSpace {
    pub fn compute_shape(&self, shape: &[TDim]) -> TractResult<TVec<TDim>> {
        let oshape = tvec!(
            shape[0].clone(),
            shape[1].clone() / (self.blocksize * self.blocksize),
            shape[2].clone() * self.blocksize,
            shape[3].clone() * self.blocksize,
        );
        Ok(oshape)
    }

    pub fn to_axis_ops(
        &self,
        input_depth: usize,
        input_height: usize,
        input_width: usize,
    ) -> TractResult<TVec<AxisOp>> {
        let mut stack: TVec<AxisOp> = tvec!();

        let ishape_from = tvec!(input_depth.to_dim());
        let mut ishape_to = tvec!(
            self.blocksize.to_dim(),
            self.blocksize.to_dim(),
            (input_depth / (self.blocksize * self.blocksize)).to_dim()
        );

        let oshape_from = tvec!(
            input_height.to_dim(),
            self.blocksize.to_dim(),
            input_width.to_dim(),
            self.blocksize.to_dim()
        );
        let oshape_to = tvec!(
            (input_height * self.blocksize).to_dim(),
            (input_width * self.blocksize).to_dim()
        );

        match self.mode {
            DepthToSpaceMode::DCR => {
                stack.push(AxisOp::Reshape(1, ishape_from, ishape_to));
                stack.push(AxisOp::Move(2, 5));
                stack.push(AxisOp::Move(1, 3));
                stack.push(AxisOp::Reshape(2, oshape_from, oshape_to));
            }
            DepthToSpaceMode::CRD => {
                ishape_to.reverse();
                stack.push(AxisOp::Reshape(1, ishape_from, ishape_to));
                stack.push(AxisOp::Move(3, 5));
                stack.push(AxisOp::Move(2, 3));
                stack.push(AxisOp::Reshape(2, oshape_from, oshape_to));
            }
        };

        Ok(stack)
    }
}

impl Expansion for DepthToSpace {
    fn name(&self) -> Cow<str> {
        "DepthToSpace".into()
    }

    op_onnx!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].rank, 4)?;
        s.equals(&outputs[0].rank, 4)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.given(&inputs[0].shape, move |s, ishape| {
            let oshape = self.compute_shape(&*ishape)?;
            s.equals(&outputs[0].shape, ShapeFactoid::from(oshape))
        })
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let ishape = model.outlet_fact(inputs[0])?.shape.to_tvec();
        let idepth = ishape[1].to_usize()?;
        let iheight = ishape[2].to_usize()?;
        let iwidth = ishape[3].to_usize()?;

        if idepth % (self.blocksize * self.blocksize) != 0 {
            bail!("DepthToSpace requires input depth to be a multiple of (blocksize * bloksize)")
        }
        let mut wire = tvec!(inputs[0]);
        for (ix, op) in self.to_axis_ops(idepth, iheight, iwidth)?.into_iter().enumerate() {
            wire = model.wire_node(format!("{}.{}", prefix, ix), op, &wire)?;
        }
        Ok(wire)
    }
}
