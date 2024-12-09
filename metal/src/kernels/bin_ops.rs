use super::BroadcastKind;
use crate::encoder::EncoderExt;
use crate::MetalTensor;
use crate::{LibraryName, MetalContext};
use anyhow::bail;
use anyhow::{ensure, Result};
use metal::{MTLSize, NSUInteger};
use std::fmt;
use tract_core::internal::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
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

impl fmt::Display for BinOps {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl BinOps {
    pub const ALL: [BinOps; 13] = [
        Self::Mul,
        Self::Add,
        Self::Div,
        Self::Sub,
        Self::Pow,
        Self::Less,
        Self::LessEqual,
        Self::Greater,
        Self::GreaterEqual,
        Self::Equals,
        Self::NotEquals,
        Self::And,
        Self::Or,
    ];

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

    pub fn output_shape<D: DimLike>(&self, a: &[D], b: &[D]) -> Result<TVec<D>> {
        tract_core::broadcast::multi_broadcast(&[a, b])
            .with_context(|| anyhow!("Error while broadcasting {:?} {:?}", a, b))
    }

    pub fn all_functions() -> Vec<String> {
        Self::ALL
            .into_iter()
            .flat_map(|op| MetalTensor::SUPPORTED_DT.into_iter().map(move |dt| (op, dt)))
            .flat_map(|(op, dt)| BroadcastKind::ALL.into_iter().map(move |b| (op, dt, b)))
            .flat_map(|(op, dt, b)| op.kernel_name(dt, b).into_iter())
            .collect()
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

    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(
            dt,
            DatumType::F32
                | DatumType::F16
                | DatumType::U8
                | DatumType::U16
                | DatumType::U32
                | DatumType::U64
                | DatumType::I8
                | DatumType::I16
                | DatumType::I32
                | DatumType::I64
                | DatumType::Bool
        )
    }

    pub fn kernel_name(&self, dt: DatumType, broadcast_kind: BroadcastKind) -> Result<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupport dt {:?} for metal binary ops", dt);

        let tname = MetalTensor::tname(dt)?;

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

        let kbroadcast_name = broadcast_kind.to_func_part();

        Ok(format!("bin_ops::{kname}_{kbroadcast_name}_{tname}"))
    }

    pub fn eval(
        &self,
        context: &MetalContext,
        lhs: &MetalTensor,
        rhs: &MetalTensor,
    ) -> TractResult<MetalTensor> {
        let out_shape = self.output_shape(lhs.shape(), rhs.shape())?;
        let out_dt = self.output_datum_type(lhs.datum_type(), rhs.datum_type())?;
        let output = unsafe { MetalTensor::uninitialized_dt(out_dt, &out_shape)? };
        self.dispatch_eval(context, lhs, rhs, &output)?;
        context.wait_until_completed()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        context: &MetalContext,
        lhs: &MetalTensor,
        rhs: &MetalTensor,
        output: &MetalTensor,
    ) -> Result<()> {
        lhs.retain_until_completion();
        rhs.retain_until_completion();
        output.retained_until_completion();

        let out_shape = output.shape();

        let broadcast_kind = if lhs.len() == 1 {
            BroadcastKind::ByScalarLeft
        } else if rhs.len() == 1 {
            BroadcastKind::ByScalarRight
        } else if lhs.shape() == rhs.shape() {
            BroadcastKind::Unicast
        } else if output.rank() == 2 {
            BroadcastKind::Nd2
        } else if output.rank() == 3 {
            BroadcastKind::Nd3
        } else if output.rank() == 4 {
            BroadcastKind::Nd4
        } else if output.rank() == 5 {
            BroadcastKind::Nd5
        } else {
            bail!(
                "Unsupported broadcast for bin op: {:?}: (a: {:?}, b: {:?}, c: {:?})",
                self,
                lhs.shape(),
                rhs.shape(),
                out_shape
            );
        };

        let kernel_name = self.kernel_name(lhs.datum_type(), broadcast_kind)?;

        match broadcast_kind {
            BroadcastKind::ByScalarLeft | BroadcastKind::ByScalarRight | BroadcastKind::Unicast => {
                let pipeline =
                    context.shared_context().load_pipeline(LibraryName::BinOps, &kernel_name)?;
                let command_buffer = context.command_buffer();
                command_buffer.encode(|encoder| {
                    encoder.set_compute_pipeline_state(&pipeline);
                    encoder.set_metal_tensor(0, lhs, metal::MTLResourceUsage::Read);
                    encoder.set_metal_tensor(1, rhs, metal::MTLResourceUsage::Read);
                    encoder.set_metal_tensor(2, output, metal::MTLResourceUsage::Write);

                    let grid_size =
                        MTLSize { width: output.len() as NSUInteger, height: 1, depth: 1 };
                    let group_size = MTLSize { width: 1, height: 1, depth: 1 };
                    encoder.dispatch_thread_groups(grid_size, group_size);
                    encoder.end_encoding();
                });
            }
            BroadcastKind::Nd1 | BroadcastKind::Nd6 => {
                bail!("Unsupported broadcast kind {:?} for bin ops: {:?}", broadcast_kind, self)
            }
            BroadcastKind::Nd2 | BroadcastKind::Nd3 | BroadcastKind::Nd4 | BroadcastKind::Nd5 => {
                ensure!(lhs.rank() == rhs.rank());

                let lhs_strides =
                    crate::utils::compute_broadcast_strides::<usize>(lhs.shape(), lhs.strides())?;

                let rhs_strides =
                    crate::utils::compute_broadcast_strides::<usize>(rhs.shape(), rhs.strides())?;

                let output_shape = output.shape();

                let pipeline =
                    context.shared_context().load_pipeline(LibraryName::BinOps, &kernel_name)?;
                let command_buffer = context.command_buffer();
                command_buffer.encode(|encoder| {
                    encoder.set_compute_pipeline_state(&pipeline);
                    encoder.set_metal_tensor(0, lhs, metal::MTLResourceUsage::Read);
                    encoder.set_slice(1, &lhs_strides);
                    encoder.set_metal_tensor(2, rhs, metal::MTLResourceUsage::Read);
                    encoder.set_slice(3, &rhs_strides);
                    encoder.set_metal_tensor(4, output, metal::MTLResourceUsage::Write);
                    encoder.set_slice(5, output_shape);

                    let grid_size = MTLSize {
                        width: out_shape[out_shape.len() - 1] as NSUInteger,
                        height: out_shape[out_shape.len() - 2] as NSUInteger,
                        depth: (out_shape[..out_shape.len() - 2].iter().product::<usize>())
                            as NSUInteger,
                    };

                    let group_size = MTLSize { width: 1, height: 1, depth: 1 };
                    encoder.dispatch_thread_groups(grid_size, group_size);
                    encoder.end_encoding();
                });
            }
        }
        Ok(())
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

    fn reference<FI: Datum, FO: Datum>(
        a: &Tensor,
        b: &Tensor,
        cab: impl Fn(&mut FO, &FI, &FI),
    ) -> Result<Tensor> {
        let out_shape = tract_core::broadcast::multi_broadcast(&[a.shape(), b.shape()])?;
        let mut out = unsafe { Tensor::uninitialized_dt(FO::datum_type(), &out_shape)? };
        let a_view = a.to_array_view::<FI>()?;
        let b_view = b.to_array_view::<FI>()?;
        let mut c = out.to_array_view_mut::<FO>()?;
        tract_core::ndarray::Zip::from(&mut c)
            .and_broadcast(a_view)
            .and_broadcast(b_view)
            .for_each(cab);
        Ok(out)
    }

    fn run_test_case_logic<F: Datum + Zero>(
        op: BinOps,
        a_shape: &[usize],
        b_shape: &[usize],
        cab: impl Fn(&mut bool, &F, &F),
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
                let ref_output = reference::<F, bool>(&a.to_cpu()?, &b.to_cpu()?, cab)?;
                assert_eq!(ref_output, output.to_cpu()?);
                Ok(())
            })
        })
    }

    fn run_test_case<F: Datum + Zero>(
        op: BinOps,
        a_shape: &[usize],
        b_shape: &[usize],
        cab: impl Fn(&mut F, &F, &F),
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
                let ref_output = reference::<F, F>(&a.to_cpu()?, &b.to_cpu()?, cab)?;
                assert_eq!(ref_output, output.to_cpu()?);
                Ok(())
            })
        })
    }

    #[test]
    fn test_bin_ops_unicast() -> Result<()> {
        run_test_case::<f32>(BinOps::Mul, &[4, 4], &[4, 4], |c, a, b| *c = *a * *b)?;
        run_test_case::<f32>(BinOps::Mul, &[2, 16], &[2, 16], |c, a, b| *c = *a * *b)?;
        Ok(())
    }

    #[test]
    fn test_bin_ops_with_broadcast_nd2() -> Result<()> {
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
    fn test_bin_ops_with_broadcast_nd3() -> Result<()> {
        run_test_case::<f32>(BinOps::Mul, &[4, 1, 10], &[1, 20, 1], |c, a, b| *c = *a * *b)?;
        run_test_case::<f32>(BinOps::Mul, &[1, 20, 1], &[10, 20, 10], |c, a, b| *c = *a * *b)?;
        run_test_case::<f32>(BinOps::Add, &[4, 1, 10], &[1, 20, 1], |c, a, b| *c = *a + *b)?;
        run_test_case::<f32>(BinOps::Sub, &[1, 20, 1], &[10, 20, 10], |c, a, b| *c = *a - *b)?;
        Ok(())
    }

    #[test]
    fn test_bin_ops_with_broadcast_nd4() -> Result<()> {
        run_test_case::<f32>(BinOps::Mul, &[4, 1, 10, 1], &[1, 20, 1, 5], |c, a, b| *c = *a * *b)?;
        run_test_case::<f32>(BinOps::Mul, &[1, 20, 1, 5], &[5, 20, 10, 5], |c, a, b| *c = *a * *b)?;
        run_test_case::<f32>(BinOps::Add, &[4, 1, 10, 1], &[1, 20, 1, 5], |c, a, b| *c = *a + *b)?;
        run_test_case::<f32>(BinOps::Sub, &[1, 20, 1, 5], &[5, 20, 10, 5], |c, a, b| *c = *a - *b)?;
        Ok(())
    }

    #[test]
    fn test_bin_ops_mul_by_scalar() -> Result<()> {
        run_test_case::<f32>(BinOps::Add, &[4, 4], &[1], |c, a, b| *c = *a + *b)?;
        run_test_case::<f32>(BinOps::Mul, &[4, 4], &[1], |c, a, b| *c = *a * *b)?;
        Ok(())
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
                    c.to_cpu()
                })
            })
        }
    }
}
