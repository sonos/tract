use crate::kernels::matmul::{GemmImpl, GemmKernel};
use crate::ops::MetalEvalOp;

use crate::{MetalContext, MetalTensorExt};
use anyhow::{bail, ensure};
use tract_core::internal::*;

#[derive(Debug, Default, Clone)]
pub struct MetalGemm<K: GemmKernel> {
    pub kernel: GemmImpl<K>,
}

impl<K: GemmKernel> MetalGemm<K> {
    pub fn new(transpose_a: bool, transpose_b: bool) -> Self {
        Self { kernel: GemmImpl::<K>::new(transpose_a, transpose_b) }
    }
}

impl<K: GemmKernel + 'static> Op for MetalGemm<K> {
    fn name(&self) -> Cow<str> {
        format!("Metal{}", self.kernel).into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![
            format!("transpose_a: {} transpose_b: {}", self.transpose_a(), self.transpose_b(),),
        ])
    }

    op_as_typed_op!();
}

impl<K: GemmKernel> MetalGemm<K> {
    fn transpose_a(&self) -> bool {
        self.kernel.transpose_a
    }

    fn transpose_b(&self) -> bool {
        self.kernel.transpose_b
    }

    fn resolve_output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let [a, b] = inputs else {
            bail!("Expects 2 inputs");
        };
        ensure!(a.rank() == b.rank());
        ensure!(a.rank() >= 2);
        ensure!(
            a.shape[a.rank() - 2 + !self.transpose_a() as usize]
                == b.shape[b.rank() - 2 + self.transpose_b() as usize]
        );

        if a.datum_type == f16::datum_type() {
            ensure!(b.datum_type == f16::datum_type());
            Ok(tvec!(f16::fact(self.kernel.output_shape(&a.shape, &b.shape))))
        } else {
            ensure!(a.datum_type == f32::datum_type());
            ensure!(b.datum_type == f32::datum_type());
            Ok(tvec!(f32::fact(self.kernel.output_shape(&a.shape, &b.shape))))
        }
    }
}

impl<K: GemmKernel + 'static> MetalEvalOp for MetalGemm<K> {
    fn metal_eval(
        &self,
        context: &MetalContext,
        node_id: usize,
        session: &mut SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let (a_opaque, b_opaque) = args_2!(inputs);
        let a = a_opaque
            .to_metal_tensor()
            .with_context(|| anyhow!("A tensor is not a metal tensor: {:?}", a_opaque))?;
        let b = b_opaque
            .to_metal_tensor()
            .with_context(|| anyhow!("B tensor is not a metal tensor {:?}", b_opaque))?;
        let c_dt = a.datum_type();
        let c_shape = self.kernel.output_shape(a.shape(), b.shape());
        let c = crate::ops::make_tensor_for_node(session, node_id, c_dt, &c_shape)?;
        self.kernel.dispatch_eval(context, a, b, &c)?;
        Ok(tvec![c.into_opaque_tensor().into_tvalue()])
    }
}

impl<K: GemmKernel + 'static> EvalOp for MetalGemm<K> {
    fn is_stateless(&self) -> bool {
        false
    }

    #[allow(unused_variables)]
    fn state(
        &self,
        session: &mut tract_core::internal::SessionState,
        node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(crate::ops::MetalOpState::new(node_id, self.clone()))))
    }
}

impl<K: GemmKernel + 'static> TypedOp for MetalGemm<K> {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::metal_facts_from_gpu(inputs, |input_facts| {
            self.resolve_output_facts(input_facts)
        })
        .with_context(|| anyhow::anyhow!("Error while computing output facts for {}", self.name()))
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        crate::utils::metal_facts(inputs, |input_facts| {
            let fma = self
                .kernel
                .output_shape(&input_facts[0].shape, &input_facts[1].shape)
                .iter()
                .product::<TDim>()
                * input_facts[0].shape.last().unwrap();
            if input_facts[0].datum_type == f16::datum_type() {
                Ok(tvec!((Cost::FMA(f16::datum_type()), fma)))
            } else {
                Ok(tvec!((Cost::FMA(f32::datum_type()), fma)))
            }
        })
        .with_context(|| anyhow::anyhow!("Error while computing cost for {:?}", self.name()))
    }

    as_op!();
}
