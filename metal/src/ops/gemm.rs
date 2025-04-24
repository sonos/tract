use crate::kernels::matmul::{GemmImpl, GemmKernel};
use crate::ops::MetalEvalOp;

use crate::MetalStream;
use anyhow::{bail, ensure};
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensorExt;
use tract_gpu::utils::{as_q40_fact, as_q40_tensor};

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

        if a.datum_type.is_number() && b.datum_type.is_number() {
            ensure!(a.rank() == b.rank());
            ensure!(a.rank() >= 2);
            ensure!(
                a.shape[a.rank() - 2 + !self.transpose_a() as usize]
                    == b.shape[b.rank() - 2 + self.transpose_b() as usize]
            );
            let out_shape = self.kernel.output_shape(&a.shape, &b.shape);
            Ok(self.kernel.output_facts(&out_shape, a.datum_type, b.datum_type)?)
        } else if let Some(opf) = as_q40_fact(inputs[0]) {
            let a_shape: ShapeFact =
                a.shape.iter().cloned().chain(opf.shape().iter().map(|d| d.to_dim())).collect();

            let out_shape = self.kernel.output_shape(&a_shape, &b.shape);
            Ok(self.kernel.output_facts(&out_shape, a.datum_type, b.datum_type)?)
        } else if let Some(opf) = as_q40_fact(inputs[1]) {
            let b_shape: ShapeFact =
                b.shape.iter().cloned().chain(opf.shape().iter().map(|d| d.to_dim())).collect();
            let out_shape = self.kernel.output_shape(&a.shape, &b_shape);
            Ok(self.kernel.output_facts(&out_shape, a.datum_type, b.datum_type)?)
        } else {
            todo!()
        }
    }
}
impl<K: GemmKernel + 'static> MetalEvalOp for MetalGemm<K> {
    fn metal_eval(
        &self,
        stream: &MetalStream,
        node_id: usize,
        session: &mut SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let (a_opaque, b_opaque) = args_2!(inputs);
        let a = a_opaque
            .to_device_tensor()
            .with_context(|| format!("A tensor is not a metal tensor: {:?}", a_opaque))?;
        let b = b_opaque
            .to_device_tensor()
            .with_context(|| format!("B tensor is not a metal tensor {:?}", b_opaque))?;

        let b_shape = as_q40_tensor(b.view().tensor)
            .map(|bqv| b.shape().iter().cloned().chain(bqv.fact.shape().iter().copied()).collect())
            .unwrap_or(b.shape().to_vec());

        let c_dt = self.kernel.matmul.output_dt(a.datum_type(), b.datum_type())?;
        let c_shape = self.kernel.output_shape(a.shape(), &b_shape);
        let c = crate::ops::make_tensor_for_node(session, node_id, c_dt, &c_shape)?;
        self.kernel.dispatch_eval(stream, a, b, &c)?;
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
        tract_gpu::utils::facts_to_device_facts(inputs, |input_facts| {
            self.resolve_output_facts(input_facts)
        })
        .with_context(|| format!("Error while computing output facts for {}", self.name()))
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        tract_gpu::utils::get_device_facts(inputs, |input_facts| {
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
        .with_context(|| format!("Error while computing cost for {:?}", self.name()))
    }

    as_op!();
}
