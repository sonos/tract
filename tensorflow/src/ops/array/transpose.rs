use tract_hir::internal::*;

use crate::model::ParsingContext;
use crate::tfpb::tensorflow::NodeDef;

#[derive(Debug, Clone, new, Hash)]
pub struct Transpose {
    t: DatumType,
    t_perm: DatumType,
}

tract_data::impl_dyn_hash!(Transpose);

pub fn transpose(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let t = pb.get_attr_datum_type("T")?;
    let t_perm = pb.get_attr_datum_type("Tperm")?;
    Ok(expand(Transpose::new(t, t_perm)))
}

impl Transpose {
    fn compute_shape<D: DimLike>(shape: &[D], perm: &[i32]) -> TVec<D> {
        let mut new_shape = tvec![D::zero(); shape.len()];
        for (ix, &d) in perm.iter().enumerate() {
            new_shape[ix] = shape[d as usize].clone();
        }
        new_shape
    }
}

impl Expansion for Transpose {
    fn name(&self) -> Cow<str> {
        "Transpose".into()
    }

    op_tf!();

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

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        if let Some(axes) = &target.outlet_fact(inputs[1])?.konst {
            let axes: TVec<usize> =
                axes.cast_to::<i64>()?.as_slice::<i64>()?.iter().map(|i| *i as usize).collect();
            let mut wire = tvec!(inputs[0]);
            for pair in tract_hir::tract_core::ops::change_axes::perm_to_ops(&axes) {
                wire = target.wire_node(format!("{}.{:?}", prefix, pair), pair, &wire)?;
            }
            Ok(wire)
        } else {
            bail!("Expect permutation input to be const")
        }
    }
}
