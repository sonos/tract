use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::NodeProto;
use tract_hir::internal::*;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("SpaceToDepth", space_to_depth);
}

pub fn space_to_depth(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let blocksize = node.get_attr_opt("blocksize")?.unwrap_or(2);
    Ok((expand(SpaceToDepth { blocksize }), vec![]))
}

#[derive(Debug, Clone, Hash)]
struct SpaceToDepth {
    blocksize: usize,
}

impl SpaceToDepth {
    pub fn compute_shape(&self, shape: &[TDim]) -> TVec<TDim> {
        tvec!(
            shape[0].clone(),
            shape[1].clone() * self.blocksize * self.blocksize,
            shape[2].clone() / self.blocksize,
            shape[3].clone() / self.blocksize,
        )
    }

    pub fn to_axis_ops(&self, shape: &[TDim]) -> TractResult<TVec<AxisOp>> {
        let mut stack: TVec<AxisOp> = tvec!();

        let ishape_from = tvec!(shape[2].clone(), shape[3].clone());
        let ishape_to = tvec!(
            shape[2].clone() / self.blocksize,
            self.blocksize.into(),
            shape[3].clone() / self.blocksize,
            self.blocksize.into(),
        );

        let oshape_from = tvec!(self.blocksize.into(), self.blocksize.into(), shape[1].clone());
        let oshape_to = tvec!(shape[1].clone() * self.blocksize * self.blocksize);

        stack.push(AxisOp::Reshape(2, ishape_from, ishape_to));
        stack.push(AxisOp::Move(3, 1));
        stack.push(AxisOp::Move(5, 2));
        stack.push(AxisOp::Reshape(1, oshape_from, oshape_to));

        Ok(stack)
    }
}



impl Expansion for SpaceToDepth {
    fn name(&self) -> Cow<str> {
        "SpaceToDepth".into()
    }


    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 1)?;
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[0].rank, 4)?;
        s.equals(&outputs[0].rank, 4)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.given(&inputs[0].shape, move |s, ishape| {
            let oshape = self.compute_shape(&ishape);
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
        let iheight = ishape[2].to_usize()?;
        let iwidth = ishape[3].to_usize()?;

        if iheight % self.blocksize != 0 {
            bail!("SpaceToDepth requires input height to be a multiple of blocksize")
        }
        if iwidth % self.blocksize != 0 {
            bail!("SpaceToDepth requires input width to be a multiple of blocksize")
        }

        let mut wire = tvec!(inputs[0]);
        for (ix, op) in self.to_axis_ops(&ishape)?.into_iter().enumerate() {
            wire = model.wire_node(format!("{prefix}.{ix}"), op, &wire)?;
        }
        Ok(wire)
    }
}
