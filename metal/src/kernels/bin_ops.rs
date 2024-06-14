use crate::MetalTensor;
use crate::{LibraryName, MetalContext};
use anyhow::bail;
use anyhow::{ensure, Result};
use metal::{MTLSize, NSUInteger};
use std::fmt;
use tract_core::internal::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BinOps {
    Mul,
    Add,
    Div,
    Sub,
    Pow,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Equals,
    NotEquals,
    And,
    Or,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOpBroadcastKind {
    Unicast,
    ByScalarLeft,
    ByScalarRight,
    Nd2,
    Nd3,
    Nd4,
}

impl fmt::Display for BinOps {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl BinOps {
    pub fn name(&self) -> Cow<str> {
        format!("{}", self).into()
    }

    pub fn validation(&self) -> Validation {
        Validation::Accurate
    }

    pub fn output_datum_type(&self, a: DatumType, b: DatumType) -> Result<DatumType> {
        ensure!(a == b);
        if self.is_logic() {
            Ok(DatumType::Bool)
        } else {
            Ok(a)
        }
    }

    pub fn is_logic(&self) -> bool {
        matches!(
            self,
            Self::Less
                | Self::LessEqual
                | Self::Greater
                | Self::GreaterEqual
                | Self::Equals
                | Self::NotEquals
                | Self::And
                | Self::Or
        )
    }

    pub fn kernel_name(&self, dt: DatumType, broadcast_kind: BinOpBroadcastKind) -> Result<String> {
        let tname = match dt {
            DatumType::F32 => "f32",
            DatumType::F16 => "f16",
            DatumType::U8 => "u8",
            DatumType::U16 => "u16",
            DatumType::U32 => "u32",
            DatumType::U64 => "u64",
            DatumType::I8 => "i8",
            DatumType::I16 => "i16",
            DatumType::I32 => "i32",
            DatumType::I64 => "i64",
            DatumType::Bool => "bool",
            _ => bail!("Unsupport dt for metal binary ops: {:?}", self),
        };

        let kname = match self {
            Self::Mul => "mul",
            Self::Add => "add",
            Self::Div => "div",
            Self::Sub => "sub",
            Self::Pow => "pow",
            Self::Greater => "greater",
            Self::GreaterEqual => "greater_equal",
            Self::Equals => "equals",
            Self::NotEquals => "not_equals",
            Self::Less => "less",
            Self::LessEqual => "less_equal",
            Self::And => "and",
            Self::Or => "or",
        };

        let kbroadcast_name = match broadcast_kind {
            BinOpBroadcastKind::Unicast => "unicast",
            BinOpBroadcastKind::ByScalarLeft => "by_scalar_lhs",
            BinOpBroadcastKind::ByScalarRight => "by_scalar_rhs",
            BinOpBroadcastKind::Nd2 => "nd2",
            BinOpBroadcastKind::Nd3 => "nd3",
            BinOpBroadcastKind::Nd4 => "nd4",
        };

        Ok(format!("bin_ops::{kname}_{kbroadcast_name}_{tname}"))
    }

    pub fn eval(
        &self,
        context: &MetalContext,
        lhs: &MetalTensor,
        rhs: &MetalTensor,
    ) -> Result<MetalTensor> {
        let out_shape = tract_core::broadcast::multi_broadcast(&[lhs.shape(), rhs.shape()])?;
        let out_dt = self.output_datum_type(lhs.datum_type(), rhs.datum_type())?;

        let output = unsafe { MetalTensor::uninitialized_dt(out_dt, &out_shape)? };

        let broadcast = if lhs.len() == 1 {
            BinOpBroadcastKind::ByScalarLeft
        } else if rhs.len() == 1 {
            BinOpBroadcastKind::ByScalarRight
        } else if lhs.shape() == rhs.shape() {
            BinOpBroadcastKind::Unicast
        } else if output.rank() == 2 {
            BinOpBroadcastKind::Nd2
        } else if output.rank() == 3 {
            BinOpBroadcastKind::Nd3
        } else if output.rank() == 4 {
            BinOpBroadcastKind::Nd4
        } else {
            bail!("Unsupport broadcast for bin op: {:?}", self);
        };

        let kernel_name = self.kernel_name(lhs.datum_type(), broadcast)?;

        match broadcast {
            BinOpBroadcastKind::ByScalarLeft
            | BinOpBroadcastKind::ByScalarRight
            | BinOpBroadcastKind::Unicast => {
                let lhs_buffer = lhs.metal();
                let rhs_buffer = rhs.metal();
                let output_buffer = output.metal();
                let pipeline =
                    context.shared_context().load_pipeline(LibraryName::BinOps, &kernel_name)?;
                let command_buffer = context.command_buffer()?;
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_buffer(0, Some(lhs_buffer), 0);
                encoder.set_buffer(1, Some(rhs_buffer), 0);
                encoder.set_buffer(2, Some(output.metal()), 0);

                let grid_size = MTLSize { width: output.len() as NSUInteger, height: 1, depth: 1 };
                let group_size = MTLSize { width: 1, height: 1, depth: 1 };
                encoder.use_resource(lhs_buffer, metal::MTLResourceUsage::Read);
                encoder.use_resource(rhs_buffer, metal::MTLResourceUsage::Read);
                encoder.use_resource(output_buffer, metal::MTLResourceUsage::Write);
                encoder.dispatch_thread_groups(grid_size, group_size);
                encoder.end_encoding();
            }
            BinOpBroadcastKind::Nd2 | BinOpBroadcastKind::Nd3 | BinOpBroadcastKind::Nd4 => {
                ensure!(lhs.rank() == rhs.rank());
                let lhs_buffer = lhs.metal();
                let rhs_buffer = rhs.metal();
                let lhs_strides = lhs
                    .strides()
                    .into_iter()
                    .zip(lhs.shape())
                    .map(|(s, dim)| if *dim == 1 { 0 } else { *s as u32 })
                    .collect::<Vec<_>>();

                let rhs_strides = rhs
                    .strides()
                    .into_iter()
                    .zip(rhs.shape())
                    .map(|(s, dim)| if *dim == 1 { 0 } else { *s as u32 })
                    .collect::<Vec<_>>();
                let output_shape = output.shape().iter().map(|d| *d as u32).collect::<Vec<_>>();

                let output_buffer = output.metal();
                let pipeline =
                    context.shared_context().load_pipeline(LibraryName::BinOps, &kernel_name)?;
                let command_buffer = context.command_buffer()?;
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_buffer(0, Some(lhs_buffer), 0);
                encoder.set_bytes(
                    1,
                    (lhs_strides.len() * std::mem::size_of::<u32>()) as NSUInteger,
                    lhs_strides.as_ptr() as *const _,
                );
                encoder.set_buffer(2, Some(rhs_buffer), 0);
                encoder.set_bytes(
                    3,
                    (rhs_strides.len() * std::mem::size_of::<u32>()) as NSUInteger,
                    rhs_strides.as_ptr() as *const _,
                );
                encoder.set_buffer(4, Some(output.metal()), 0);
                encoder.set_bytes(
                    5,
                    (output_shape.len() * std::mem::size_of::<u32>()) as NSUInteger,
                    output_shape.as_ptr() as *const _,
                );

                let grid_size = MTLSize {
                    width: out_shape[out_shape.len() - 1] as NSUInteger,
                    height: out_shape[out_shape.len() - 2] as NSUInteger,
                    depth: (out_shape[..out_shape.len() - 2].iter().product::<usize>())
                        as NSUInteger,
                };

                let group_size = MTLSize { width: 1, height: 1, depth: 1 };
                encoder.use_resource(lhs_buffer, metal::MTLResourceUsage::Read);
                encoder.use_resource(rhs_buffer, metal::MTLResourceUsage::Read);
                encoder.use_resource(output_buffer, metal::MTLResourceUsage::Write);
                encoder.dispatch_thread_groups(grid_size, group_size);
                encoder.end_encoding();
            }
        }

        context.wait_until_completed()?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IntoMetal;
    use derive_new::new;
    use num_traits::AsPrimitive;
    use num_traits::Zero;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use tract_core::internal::Tensor;

    fn reference<FI: Datum, FO: Datum>(
        a: &Tensor,
        b: &Tensor,
        cab: impl Fn(&mut FO, &FI, &FI) -> (),
    ) -> Result<Tensor> {
        let out_shape = tract_core::broadcast::multi_broadcast(&[a.shape(), b.shape()])?;
        let mut out = unsafe { Tensor::uninitialized_dt(FO::datum_type(), &out_shape)? };
        let a_view = a.to_array_view::<FI>()?;
        let b_view = b.to_array_view::<FI>()?;
        let mut c = out.to_array_view_mut::<FO>()?;
        tract_core::ndarray::Zip::from(&mut c)
            .and_broadcast(a_view)
            .and_broadcast(b_view)
            .for_each(|c, a, b| (cab)(c, a, b));
        Ok(out)
    }

    fn run_test_case_logic<F: Datum + Zero>(
        op: BinOps,
        a_shape: &[usize],
        b_shape: &[usize],
        cab: impl Fn(&mut bool, &F, &F) -> (),
    ) -> Result<()> {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let a_len = a_shape.iter().product::<usize>();
                let b_len = b_shape.iter().product::<usize>();

                let a =
                    Tensor::from_shape(a_shape, &(0..a_len).map(|f| f as f32).collect::<Vec<_>>())?
                        .into_metal()?;
                let b = Tensor::from_shape(
                    b_shape,
                    &(0..b_len).rev().map(|f| f as f32).collect::<Vec<_>>(),
                )?
                .into_metal()?;
                let output = op.eval(context, &a, &b)?;
                let ref_output = reference::<F, bool>(a.tensor(), b.tensor(), cab)?;
                assert_eq!(&ref_output, output.tensor());
                Ok(())
            })
        })
    }

    fn run_test_case<F: Datum + Zero>(
        op: BinOps,
        a_shape: &[usize],
        b_shape: &[usize],
        cab: impl Fn(&mut F, &F, &F) -> (),
    ) -> Result<()> {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let a_len = a_shape.iter().product::<usize>();
                let b_len = b_shape.iter().product::<usize>();

                let a =
                    Tensor::from_shape(a_shape, &(0..a_len).map(|f| f as f32).collect::<Vec<_>>())?
                        .into_metal()?;
                let b = Tensor::from_shape(
                    b_shape,
                    &(0..b_len).rev().map(|f| f as f32).collect::<Vec<_>>(),
                )?
                .into_metal()?;
                let output = op.eval(context, &a, &b)?;
                let ref_output = reference::<F, F>(a.tensor(), b.tensor(), cab)?;
                assert_eq!(&ref_output, output.tensor());
                Ok(())
            })
        })
    }

    #[test]
    fn test_bin_ops_mul_unicast() -> Result<()> {
        run_test_case::<f32>(BinOps::Mul, &[4, 4], &[4, 4], |c, a, b| *c = *a * *b)?;
        run_test_case::<f32>(BinOps::Mul, &[2, 16], &[2, 16], |c, a, b| *c = *a * *b)?;
        Ok(())
    }

    #[test]
    fn test_bin_ops_mul_with_broadcast_nd2() -> Result<()> {
        run_test_case::<f32>(BinOps::Mul, &[4, 1], &[1, 20], |c, a, b| *c = *a * *b)?;
        run_test_case::<f32>(BinOps::Mul, &[1, 20], &[10, 20], |c, a, b| *c = *a * *b)?;
        run_test_case::<f32>(BinOps::Add, &[4, 1], &[4, 20], |c, a, b| *c = *a + *b)?;
        run_test_case::<f32>(BinOps::Sub, &[1, 20], &[10, 20], |c, a, b| *c = *a - *b)?;
        run_test_case_logic::<f32>(BinOps::Less, &[1, 20], &[10, 20], |c, a, b| *c = *a < *b)?;
        run_test_case_logic::<f32>(BinOps::Greater, &[1, 20], &[10, 20], |c, a, b| *c = *a > *b)?;
        run_test_case_logic::<f32>(BinOps::Equals, &[1, 20], &[10, 20], |c, a, b| *c = *a == *b)?;
        Ok(())
    }

    #[test]
    fn test_bin_ops_mul_with_broadcast_nd3() -> Result<()> {
        run_test_case::<f32>(BinOps::Mul, &[4, 1, 10], &[1, 20, 1], |c, a, b| *c = *a * *b)?;
        run_test_case::<f32>(BinOps::Mul, &[1, 20, 1], &[10, 20, 10], |c, a, b| *c = *a * *b)?;
        run_test_case::<f32>(BinOps::Add, &[4, 1, 10], &[1, 20, 1], |c, a, b| *c = *a + *b)?;
        run_test_case::<f32>(BinOps::Sub, &[1, 20, 1], &[10, 20, 10], |c, a, b| *c = *a - *b)?;
        Ok(())
    }

    #[test]
    fn test_bin_ops_mul_with_broadcast_nd4() -> Result<()> {
        run_test_case::<f32>(BinOps::Mul, &[4, 1, 10, 1], &[1, 20, 1, 5], |c, a, b| *c = *a * *b)?;
        run_test_case::<f32>(BinOps::Mul, &[1, 20, 1, 5], &[5, 20, 10, 5], |c, a, b| *c = *a * *b)?;
        run_test_case::<f32>(BinOps::Add, &[4, 1, 10, 1], &[1, 20, 1, 5], |c, a, b| *c = *a + *b)?;
        run_test_case::<f32>(BinOps::Sub, &[1, 20, 1, 5], &[5, 20, 10, 5], |c, a, b| *c = *a - *b)?;
        Ok(())
    }

    #[test]
    fn test_bin_ops_mul_by_scalar() -> Result<()> {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let a =
                    Tensor::from_shape(&[4, 4], &(0..4 * 4).map(|f| f as f32).collect::<Vec<_>>())?
                        .into_metal()?;
                let b = Tensor::from_shape(&[1], &[2f32])?.into_metal()?;
                dbg!(BinOps::Add.eval(context, &a, &b)?);

                Ok(())
            })
        })
    }

    proptest::proptest! {
        #[test]
        fn bin_ops_prop_f32(pb in any::<BinaryOpProblem<f32>>()) {
            prop_assert_eq!(pb.run().unwrap(), pb.reference().unwrap())
        }

        #[test]
        fn bin_ops_prop_f16(pb in any::<BinaryOpProblem<f16>>()) {
            prop_assert_eq!(pb.run().unwrap(), pb.reference().unwrap())
        }
    }

    #[derive(Debug, new)]
    pub struct BinaryOpProblem<F: Datum>
    where
        F: Datum + Copy,
        usize: AsPrimitive<F>,
    {
        pub b: usize,
        pub m: usize,
        pub n: usize,
        pub lhs: Tensor,
        pub rhs: Tensor,
        _phantom: std::marker::PhantomData<F>,
    }

    impl<F> Arbitrary for BinaryOpProblem<F>
    where
        F: Datum + Copy,
        usize: AsPrimitive<F>,
    {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_: ()) -> Self::Strategy {
            (1usize..2, 1usize..20, 1usize..20)
                .prop_flat_map(|(b, m, n)| {
                    let lhs_len = b * m * n;
                    let rhs_len = b * m * n;
                    let lhs = (0usize..10).prop_map(|x| x.as_());
                    let rhs = (0usize..10).prop_map(|x| x.as_());
                    (
                        Just(b),
                        Just(m),
                        Just(n),
                        vec(lhs, lhs_len..=lhs_len),
                        vec(rhs, rhs_len..=rhs_len),
                    )
                })
                .prop_map(|(b, m, n, lhs, rhs)| Self {
                    b,
                    m,
                    n,
                    lhs: Tensor::from_shape(&[b, m, n], &lhs).unwrap(),
                    rhs: Tensor::from_shape(&[b, m, n], &rhs).unwrap(),
                    _phantom: std::marker::PhantomData,
                })
                .boxed()
        }
    }

    impl<F> BinaryOpProblem<F>
    where
        F: Datum + Zero + Copy + std::ops::AddAssign + std::ops::Mul<Output = F>,
        usize: AsPrimitive<F>,
    {
        pub fn reference(&self) -> Result<Tensor> {
            let out_shape =
                tract_core::broadcast::multi_broadcast(&[self.lhs.shape(), self.rhs.shape()])?;
            let mut out = Tensor::zero_dt(F::datum_type(), &out_shape)?;
            let a = self.lhs.to_array_view::<F>()?;
            let b = self.rhs.to_array_view::<F>()?;
            let mut c = out.to_array_view_mut::<F>()?;
            tract_core::ndarray::Zip::from(&mut c)
                .and_broadcast(a)
                .and_broadcast(b)
                .for_each(|c, a, b| *c = *a * *b);
            Ok(out)
        }

        pub fn run(&self) -> Result<Tensor> {
            objc::rc::autoreleasepool(|| {
                crate::METAL_CONTEXT.with_borrow(|context| {
                    let lhs = self.lhs.clone().into_metal()?;
                    let rhs = self.rhs.clone().into_metal()?;
                    let c = BinOps::Mul.eval(context, &lhs, &rhs)?;
                    Ok(c.into_tensor())
                })
            })
        }
    }
}
