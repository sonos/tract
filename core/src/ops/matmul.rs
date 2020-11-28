pub mod lir_unary;
pub mod mir;
pub mod mir_unary;
pub mod mir_quant;
pub mod pack;

use crate::internal::*;
use tract_itertools::Itertools;
use tract_ndarray::prelude::*;

pub use self::mir_quant::QMatMul;
pub use self::mir::MatMul;
pub use self::mir_unary::MatMulUnary;
use self::pack::MatMatMulPack;
use crate::ops::quant::QParams;

pub fn compute_shape<D: DimLike>(
    ashape: &[D],
    bshape: &[D],
    a_trans: bool,
    b_trans: bool,
    c_trans: bool,
) -> TractResult<(D, D, D, TVec<D>)> {
    let mut c_shape = crate::broadcast::multi_broadcast(&[
        &ashape[..(ashape.len() - 2)],
        &bshape[..(bshape.len() - 2)],
    ])
    .ok_or_else(|| format_err!("Could not broadcast"))?;
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
            "Inconsistent matmul: a: {} b: {}, a_trans: {} b_trans: {} c_trans: {}",
            ashape.iter().join(","),
            bshape.iter().join(","),
            a_trans,
            b_trans,
            c_trans
        );
    }
    if c_trans {
        c_shape.push(n.clone());
        c_shape.push(m.clone());
    } else {
        c_shape.push(m.clone());
        c_shape.push(n.clone());
    }
    Ok((m, ka, n, c_shape))
}

pub(super) fn eval(
    a: &Tensor,
    b: &Tensor,
    a_trans: bool,
    b_trans: bool,
    c_trans: bool,
) -> TractResult<Tensor> {
    unsafe {
        let rank = a.rank();
        let (m, k, n, c_shape) = compute_shape(a.shape(), b.shape(), a_trans, b_trans, c_trans)?;
        let dt = a.datum_type();
        let mut mm = tract_linalg::ops()
            .mmm(a.datum_type(), b.datum_type(), dt, m, k, n)
            .with_context(|| {
                format!(
                    "No matrix multiplier for {:?}x{:?} to {:?}",
                    a.datum_type(),
                    b.datum_type(),
                    dt
                )
            })?;
        let c_storage = mm.c_from_data_and_strides(
            if c_trans { 1 } else { c_shape[rank - 1] as isize },
            if !c_trans { 1 } else { c_shape[rank - 1] as isize },
        );

        let mut c = Tensor::uninitialized_dt(dt, &c_shape)?;

        let a_pack = mm.a_pack();
        let b_pack = mm.b_pack();

        let mut packed_a =
            Tensor::uninitialized_aligned_dt(a.datum_type(), &[a_pack.len(m)], a_pack.alignment())?;
        let mut packed_b =
            Tensor::uninitialized_aligned_dt(b.datum_type(), &[b_pack.len(n)], b_pack.alignment())?;

        for prefix in tract_ndarray::indices(&c_shape[..rank - 2]).into_iter() {
            let mut a_prefix = tvec!();
            let mut b_prefix = tvec!();
            for (axis, &dim) in prefix.slice().iter().enumerate() {
                a_prefix.push(dim.min(a.shape()[axis] - 1));
                b_prefix.push(dim.min(b.shape()[axis] - 1));
            }
            a_pack.pack(
                packed_a.view_mut(),
                &a.view_at_prefix(&a_prefix)?,
                !a_trans as usize,
                a_trans as usize,
            );
            b_pack.pack(
                packed_b.view_mut(),
                &b.view_at_prefix(&b_prefix)?,
                b_trans as usize,
                !b_trans as usize,
            );
            mm.run(
                &mm.a_packed().wrap(&packed_a.view()),
                &mm.b_packed().wrap(&packed_b.view()),
                &mut c_storage.wrap(&c.view_at_prefix_mut(prefix.slice())?),
                &[],
            )?;
        }
        Ok(c)
    }
}
