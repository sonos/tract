use crate::infer::*;
use crate::internal::*;

pub use tract_core::ops::matmul::MatMul;
pub use tract_core::ops::quant::QParams;

#[derive(Debug, Clone, Default, Hash)]
pub struct MatMulInference {
    pub a_trans: bool,
    pub b_trans: bool,
    pub c_trans: bool,
    pub q_params: Option<QParams>,
}

tract_data::impl_dyn_hash!(MatMulInference);

impl MatMulInference {
    pub fn with_a_trans(self, a_trans: bool) -> MatMulInference {
        MatMulInference { a_trans, ..self }
    }

    pub fn with_b_trans(self, b_trans: bool) -> MatMulInference {
        MatMulInference { b_trans, ..self }
    }

    pub fn with_c_trans(self, c_trans: bool) -> MatMulInference {
        MatMulInference { c_trans, ..self }
    }

    pub fn with_q_params(self, q_params: QParams) -> MatMulInference {
        MatMulInference { q_params: Some(q_params), ..self }
    }
}

impl Expansion for MatMulInference {
    fn name(&self) -> Cow<str> {
        "MatMulInference".into()
    }

    op_hir!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 2)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &inputs[1].datum_type)?;
        if let Some(qp) = &self.q_params {
            s.equals(&outputs[0].datum_type, &qp.c_datum_type)?;
        } else {
            s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        }
        s.given_2(&inputs[0].shape, &inputs[1].shape, move |s, ashape, bshape| {
            let (_, _, _, cshape) =
                compute_shapes(ashape, bshape, self.a_trans, self.b_trans, self.c_trans)?;
            s.equals(&outputs[0].shape, cshape)
        })?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let inputs = crate::ops::binary::wire_rank_broadcast(prefix, target, inputs)?;
        target.wire_node(
            prefix,
            tract_core::ops::matmul::MatMul {
                a_trans: self.a_trans,
                b_trans: self.b_trans,
                c_trans: self.c_trans,
                q_params: self.q_params.clone(),
            },
            &inputs,
        )
    }
}

pub fn compute_shapes<D: DimLike>(
    ashape_orig: TVec<D>,
    bshape_orig: TVec<D>,
    a_trans: bool,
    b_trans: bool,
    c_trans: bool,
) -> TractResult<(TVec<D>, TVec<D>, TVec<D>, TVec<D>)> {
    let mut ashape = ashape_orig.clone();
    let mut bshape = bshape_orig.clone();
    let mut implicit_m = false;
    let mut implicit_n = false;
    if ashape.len() < 2 {
        implicit_m = true;
        ashape.insert(a_trans as usize, D::one());
    }
    if bshape.len() < 2 {
        implicit_n = true;
        bshape.insert(!b_trans as usize, D::one());
    }
    while ashape.len() < bshape.len() {
        ashape.insert(0, D::one());
    }
    while bshape.len() < ashape.len() {
        bshape.insert(0, D::one());
    }
    let c_bc_shape_prefix = tract_core::broadcast::multi_broadcast(&[
        &ashape[..(ashape.len() - 2)],
        &bshape[..(bshape.len() - 2)],
    ])
    .ok_or_else(|| format_err!("Could not broadcast"))?;
    let mut c_bc_shape: TVec<D> = c_bc_shape_prefix.clone();
    let (mut m, mut ka) = (ashape[ashape.len() - 2].clone(), ashape[ashape.len() - 1].clone());
    let (mut kb, mut n) = (bshape[bshape.len() - 2].clone(), bshape[bshape.len() - 1].clone());
    if a_trans {
        std::mem::swap(&mut m, &mut ka);
    }
    if b_trans {
        std::mem::swap(&mut kb, &mut n);
    }
    if ka != kb {
        bail!(
            "Inconsistent matmul: a: {:?} b: {:?}, a_trans: {} b_trans: {} c_trans: {}",
            ashape,
            bshape,
            a_trans,
            b_trans,
            c_trans
        );
    }
    let mut c_shape_final = c_bc_shape.clone();
    if c_trans {
        c_bc_shape.push(n.clone());
        c_bc_shape.push(m.clone());
        if !implicit_n {
            c_shape_final.push(n.clone());
        }
        if !implicit_m {
            c_shape_final.push(m.clone());
        }
    } else {
        c_bc_shape.push(m.clone());
        c_bc_shape.push(n.clone());
        if !implicit_m {
            c_shape_final.push(m.clone());
        }
        if !implicit_n {
            c_shape_final.push(n.clone());
        }
    }
    Ok((ashape, bshape, c_bc_shape, c_shape_final))
}
