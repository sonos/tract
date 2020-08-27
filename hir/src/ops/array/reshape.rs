use crate::infer::*;
use crate::internal::*;

#[derive(Debug, Clone, new, Default, Hash)]
pub struct Reshape {}

tract_linalg::impl_dyn_hash!(Reshape);

impl Expansion for Reshape {
    fn name(&self) -> Cow<str> {
        "Reshape".into()
    }

    op_hir!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.given_2(&inputs[0].shape, &inputs[1].value, move |s, ishape, shape| {
            let shape = shape.cast_to::<TDim>()?;
            let shape = shape.as_slice::<TDim>()?;
            let oshape = compute_shape(&ishape, &shape)?;
            s.equals(&outputs[0].shape, ShapeFactoid::from(oshape))
        })
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        if let Some(ref shape) = model.outlet_fact(inputs[1])?.konst {
            let input_shape: TVec<TDim> = model.outlet_fact(inputs[0])?.shape.to_tvec();
            let shape = shape.cast_to::<TDim>()?;
            let shape = shape.as_slice::<TDim>()?;
            let mut wire = tvec!(inputs[0]);
            for (ix, op) in to_axis_ops(&input_shape, shape)?.into_iter().enumerate() {
                wire = model.wire_node(format!("{}.{}", prefix, ix), op, &wire)?;
            }
            return Ok(wire);
        }
        bail!("shape input is variable")
    }
}

fn compute_shape(input: &[TDim], shape_spec: &[TDim]) -> TractResult<TVec<TDim>> {
    let mut shape: TVec<TDim> = shape_spec.into();

    // deal with zeros, stop if we see a -1
    fn deal_with_zero<'a>(
        mut input_dims: std::iter::Peekable<impl Iterator<Item = &'a TDim>>,
        shape: &mut [TDim],
    ) -> TractResult<()> {
        let mut remaining_dim_input = 1.to_dim();
        for slot in shape.iter_mut() {
            if *slot == (-1).into() {
                break;
            }
            if *slot == 0.into() {
                if remaining_dim_input != TDim::one() {
                    bail!("Invalid");
                }
                *slot = input_dims.peek().ok_or("Invalid")?.clone().clone();
            }
            loop {
                let quotient = remaining_dim_input.maybe_div(slot);
                if quotient.is_err() || quotient.as_ref().unwrap().1 != 1 {
                    remaining_dim_input =
                        remaining_dim_input.maybe_mul(input_dims.next().ok_or("Invalid")?)?;
                } else {
                    break;
                }
            }
            remaining_dim_input = remaining_dim_input.maybe_div(&slot)?.0;
        }
        Ok(())
    }

    deal_with_zero(input.iter().peekable(), &mut shape)?;
    shape.reverse();
    deal_with_zero(input.iter().rev().peekable(), &mut shape)?;
    shape.reverse();

    if let Some(pos) = shape.iter().position(|d| *d == (-1).into()) {
        let input_vol = input.iter().try_fold(1.to_dim(), |a, b| a.maybe_mul(b))?;
        let shape_vol = shape
            .iter()
            .filter(|d| **d != (-1).into())
            .try_fold(1.to_dim(), |a, b| a.maybe_mul(b))?;
        let div = input_vol.maybe_div(&shape_vol)?;
        if div.1 != 1 {
            bail!("invalid")
        }
        shape[pos] = div.0;
    }
    Ok(shape)
}

pub fn to_axis_ops(input_orig: &[TDim], output_spec: &[TDim]) -> TractResult<TVec<AxisOp>> {
    let final_output = compute_shape(input_orig, output_spec)?;
    let mut stack: TVec<AxisOp> = tvec!();
    'top: loop {
        let current_input = stack.iter().fold(TVec::from(input_orig), |shape, op| {
            let mut shape = shape.into();
            op.change_shape_array(&mut shape);
            shape
        });
        if &current_input == &final_output {
            return Ok(stack);
        }
        if let Some(common) =
            current_input.iter().zip(final_output.iter()).position(|(a, b)| a != b)
        {
            if current_input[common].is_one() {
                stack.push(AxisOp::Rm(common));
            } else if final_output[common].is_one() {
                stack.push(AxisOp::Add(common));
            } else {
                // actual regrouping. search for a match. this is quadratic, but
                // rank is expected to be somewhat reasonable
                for i in common..current_input.len() {
                    let i_group = &current_input[common..i + 1];
                    let i_volume: TDim =
                        if let Ok(v) = i_group.iter().maybe_product() { v } else { break };
                    for o in common..final_output.len() {
                        let o_group = &final_output[common..o + 1];
                        let o_volume: TDim =
                            if let Ok(v) = o_group.iter().maybe_product() { v } else { break };
                        if i_volume == o_volume {
                            stack.push(AxisOp::Reshape(common, i_group.into(), o_group.into()));
                            continue 'top;
                        }
                    }
                }
                todo!()
            }
        } else {
            if final_output.len() > current_input.len() {
                stack.push(AxisOp::Add(current_input.len()));
            } else {
                stack.push(AxisOp::Rm(current_input.len() - 1));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use AxisOp::*;

    use tract_core::pulse::stream_dim as stream;

    macro_rules! s {
        ($($a:expr),*) => {&[ $($a.into()),* ]}
    }

    macro_rules! r {
        ($at: expr ; $($from:expr),* => $($to:expr),*) => {
            AxisOp::Reshape($at, tvec!($($from.into()),*),  tvec!($($to.into()),*))
        }
    }

    #[test]
    fn compute_invalid() {
        assert!(compute_shape(s![3, 4, 5], s!(100)).is_err());
    }

    #[test]
    fn compute_with_leading_zero() {
        assert_eq!(&*compute_shape(s![3, 4, 5], s!(0, 0, 5)).unwrap(), s![3, 4, 5])
    }

    #[test]
    fn compute_with_leading_zero_with_flatten() {
        assert_eq!(&*compute_shape(s![2, 3, 5, 7], s!(2, 0, 35)).unwrap(), s![2, 3, 35])
    }

    #[test]
    fn compute_with_trailing_zero() {
        assert_eq!(&*compute_shape(s![3, 4, 5], s!(3, -1, 0)).unwrap(), s![3, 4, 5])
    }

    #[test]
    fn compute_bug_1() {
        assert_eq!(
            &*compute_shape(s![stream(), 1, 2, 128], s!(0, 0, -1)).unwrap(),
            s![stream(), 1, 256]
        )
    }

    #[test]
    fn axis_op_rm_begin() {
        assert_eq!(&*to_axis_ops(s![1, 2, 3], s!(2, 3)).unwrap(), &[Rm(0)])
    }

    #[test]
    fn axis_op_rm_end() {
        assert_eq!(&*to_axis_ops(s![2, 3, 1], s!(2, 3)).unwrap(), &[Rm(2)])
    }

    #[test]
    fn axis_op_insert_begin() {
        assert_eq!(&*to_axis_ops(s![2, 3], s!(1, 2, 3)).unwrap(), &[Add(0)])
    }

    #[test]
    fn axis_op_insert_end() {
        assert_eq!(&*to_axis_ops(s![2, 3], s!(2, 3, 1)).unwrap(), &[Add(2)])
    }

    #[test]
    fn axis_op_merge() {
        assert_eq!(&*to_axis_ops(s![2, 3, 5, 7], s!(2, 0, 35)).unwrap(), &[r!(2 ; 5,7 => 35 )])
    }

    #[test]
    fn axis_op_complex() {
        assert_eq!(
            &*to_axis_ops(s![1, 2, 3, 5, 7], s!(2, 1, 3, 35, 1)).unwrap(),
            &[Rm(0), Add(1), r!(3 ; 5,7 => 35 ), Add(4)]
        )
    }
}
