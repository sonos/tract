use super::BroadcastKind;
use crate::encoder::EncoderExt;
use crate::kernels::utils::compute_broadcast_strides;
use crate::{LibraryName, MetalStream};
use anyhow::{bail, ensure};
use metal::{MTLSize, NSUInteger};
use std::fmt;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

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

    pub fn output_datum_type(&self, a: DatumType, b: DatumType) -> TractResult<DatumType> {
        ensure!(a == b);
        if self.is_logic() {
            Ok(DatumType::Bool)
        } else {
            Ok(a)
        }
    }

    pub fn output_shape<D: DimLike>(&self, a: &[D], b: &[D]) -> TractResult<TVec<D>> {
        tract_core::broadcast::multi_broadcast(&[a, b])
            .with_context(|| format!("Error while broadcasting {:?} {:?}", a, b))
    }

    pub fn all_functions() -> Vec<String> {
        Self::ALL
            .into_iter()
            .flat_map(|op| DeviceTensor::SUPPORTED_DT.into_iter().map(move |dt| (op, dt)))
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

    pub fn kernel_name(&self, dt: DatumType, broadcast_kind: BroadcastKind) -> TractResult<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupport dt {:?} for metal binary ops", dt);

        let tname = DeviceTensor::tname(dt)?;

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

        let kbroadcast_name = broadcast_kind.name();

        Ok(format!("bin_ops::{kname}_{kbroadcast_name}_{tname}"))
    }

    pub fn eval(
        &self,
        stream: &MetalStream,
        lhs: &DeviceTensor,
        rhs: &DeviceTensor,
    ) -> TractResult<DeviceTensor> {
        let out_shape = self.output_shape(lhs.shape(), rhs.shape())?;
        let out_dt = self.output_datum_type(lhs.datum_type(), rhs.datum_type())?;
        let output = unsafe { DeviceTensor::uninitialized_dt(out_dt, &out_shape)? };

        self.dispatch_eval(stream, lhs, rhs, &output)?;

        stream.wait_until_completed()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        stream: &MetalStream,
        lhs: &DeviceTensor,
        rhs: &DeviceTensor,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        stream.retain_tensor(lhs);
        stream.retain_tensor(rhs);
        stream.retain_tensor(output);

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
                let pipeline = stream.load_pipeline(LibraryName::BinOps, &kernel_name)?;

                let command_buffer = stream.command_buffer();
                command_buffer.encode(|encoder| {
                    encoder.set_compute_pipeline_state(&pipeline);
                    encoder.set_metal_tensor(0, lhs, metal::MTLResourceUsage::Read);
                    encoder.set_metal_tensor(1, rhs, metal::MTLResourceUsage::Read);
                    encoder.set_metal_tensor(2, output, metal::MTLResourceUsage::Write);

                    let grid_size =
                        MTLSize { width: output.len() as NSUInteger, height: 1, depth: 1 };
                    let group_size = MTLSize { width: 1, height: 1, depth: 1 };
                    encoder.dispatch_thread_groups(grid_size, group_size);
                });
            }
            BroadcastKind::Nd1 | BroadcastKind::Nd6 => {
                bail!("Unsupported broadcast kind {:?} for bin ops: {:?}", broadcast_kind, self)
            }
            BroadcastKind::Nd2 | BroadcastKind::Nd3 | BroadcastKind::Nd4 | BroadcastKind::Nd5 => {
                ensure!(lhs.rank() == rhs.rank());

                let lhs_strides = compute_broadcast_strides::<usize>(lhs.shape(), lhs.strides())?;

                let rhs_strides = compute_broadcast_strides::<usize>(rhs.shape(), rhs.strides())?;

                let output_shape = output.shape();

                let pipeline = stream.load_pipeline(LibraryName::BinOps, &kernel_name)?;
                let command_buffer = stream.command_buffer();
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
                });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::with_borrowed_metal_stream;

    use super::*;
    use derive_new::new;
    use num_traits::AsPrimitive;
    use num_traits::Zero;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use tract_gpu::tensor::IntoDevice;

    fn reference<FI: Datum, FO: Datum>(
        a: &Tensor,
        b: &Tensor,
        cab: impl Fn(&mut FO, &FI, &FI),
    ) -> TractResult<Tensor> {
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
    ) -> TractResult<()> {
        with_borrowed_metal_stream(|stream| {
            let a_len = a_shape.iter().product::<usize>();
            let b_len = b_shape.iter().product::<usize>();

            let a = Tensor::from_shape(a_shape, &(0..a_len).map(|f| f as f32).collect::<Vec<_>>())?
                .into_device()?;
            let b = Tensor::from_shape(
                b_shape,
                &(0..b_len).rev().map(|f| f as f32).collect::<Vec<_>>(),
            )?
            .into_device()?;
            let output = op.eval(stream, &a, &b)?;
            let ref_output = reference::<F, bool>(
                &a.to_host()?.into_tensor(),
                &b.to_host()?.into_tensor(),
                cab,
            )?;
            assert_eq!(ref_output, output.to_host()?.into_tensor());
            Ok(())
        })
    }

    fn run_test_case<F: Datum + Zero>(
        op: BinOps,
        a_shape: &[usize],
        b_shape: &[usize],
        cab: impl Fn(&mut F, &F, &F),
    ) -> TractResult<()> {
        with_borrowed_metal_stream(|stream| {
            let a_len = a_shape.iter().product::<usize>();
            let b_len = b_shape.iter().product::<usize>();

            let a = Tensor::from_shape(a_shape, &(0..a_len).map(|f| f as f32).collect::<Vec<_>>())?
                .into_device()?;
            let b = Tensor::from_shape(
                b_shape,
                &(0..b_len).rev().map(|f| f as f32).collect::<Vec<_>>(),
            )?
            .into_device()?;
            let output = op.eval(stream, &a, &b)?;

            let ref_output =
                reference::<F, F>(&a.to_host()?.into_tensor(), &b.to_host()?.into_tensor(), cab)?;
            assert_eq!(ref_output, output.to_host()?.into_tensor());
            Ok(())
        })
    }

    #[test]
    fn test_bin_ops_unicast() -> TractResult<()> {
        run_test_case::<f32>(BinOps::Mul, &[4, 4], &[4, 4], |c, a, b| *c = *a * *b)?;
        run_test_case::<f32>(BinOps::Mul, &[2, 16], &[2, 16], |c, a, b| *c = *a * *b)?;
        Ok(())
    }

    #[test]
    fn test_bin_ops_with_broadcast_nd2() -> TractResult<()> {
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
    fn test_bin_ops_with_broadcast_nd3() -> TractResult<()> {
        run_test_case::<f32>(BinOps::Mul, &[4, 1, 10], &[1, 20, 1], |c, a, b| *c = *a * *b)?;
        run_test_case::<f32>(BinOps::Mul, &[1, 20, 1], &[10, 20, 10], |c, a, b| *c = *a * *b)?;
        run_test_case::<f32>(BinOps::Add, &[4, 1, 10], &[1, 20, 1], |c, a, b| *c = *a + *b)?;
        run_test_case::<f32>(BinOps::Sub, &[1, 20, 1], &[10, 20, 10], |c, a, b| *c = *a - *b)?;
        Ok(())
    }

    #[test]
    fn test_bin_ops_with_broadcast_nd4() -> TractResult<()> {
        run_test_case::<f32>(BinOps::Mul, &[4, 1, 10, 1], &[1, 20, 1, 5], |c, a, b| *c = *a * *b)?;
        run_test_case::<f32>(BinOps::Mul, &[1, 20, 1, 5], &[5, 20, 10, 5], |c, a, b| *c = *a * *b)?;
        run_test_case::<f32>(BinOps::Add, &[4, 1, 10, 1], &[1, 20, 1, 5], |c, a, b| *c = *a + *b)?;
        run_test_case::<f32>(BinOps::Sub, &[1, 20, 1, 5], &[5, 20, 10, 5], |c, a, b| *c = *a - *b)?;
        Ok(())
    }

    #[test]
    fn test_bin_ops_mul_by_scalar() -> TractResult<()> {
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
        pub fn reference(&self) -> TractResult<Tensor> {
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

        pub fn run(&self) -> TractResult<Tensor> {
            with_borrowed_metal_stream(|stream| {
                let lhs = self.lhs.clone().into_device()?;
                let rhs = self.rhs.clone().into_device()?;
                let c = BinOps::Mul.eval(stream, &lhs, &rhs)?;
                Ok(c.to_host()?.into_tensor())
            })
        }
    }
}
