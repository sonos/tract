use crate::internal::*;
use crate::ops::cnn::{KernelFormat, PaddingSpec};
use crate::ops::nn::DataFormat;

// NCHW OIHW rank=4 valid, no-stride, no-dil, no-bias, no-group, f32

#[derive(Clone, Debug, new, Hash)]
pub struct DeconvUnary {
    pub data_format: DataFormat,
    pub kernel_format: KernelFormat,
    pub padding: PaddingSpec,
    pub kernel: Arc<Tensor>,
}

impl DeconvUnary {
    fn output_shape<D: DimLike>(&self, x_shape: &[D]) -> TractResult<TVec<D>> {
        super::output_shape(
            &self.data_format,
            &self.kernel_format,
            &self.padding,
            &self.kernel.shape(),
            x_shape,
        )
    }

    fn wire_with_deconv_sum(
        &self,
        name: &str,
        target: &mut TypedModel,
        input: OutletId,
    ) -> TractResult<TVec<OutletId>> {
        use std::iter::once;
        let input_shape = target.outlet_fact(input)?.shape.clone();
        assert_eq!(self.kernel_format, KernelFormat::OIHW);
        assert_eq!(self.data_format, DataFormat::NCHW);
        let shape = self.data_format.shape(input_shape.to_tvec())?;
        let geo_dim = shape.hw_dims().iter().maybe_product()?;
        let reshaped = target.wire_node(
            format!("{}.reshaped_input", name),
            AxisOp::Reshape(shape.h_axis(), shape.hw_dims().into(), tvec!(geo_dim)),
            &[input],
        )?;

        // kernel from OIHW to [1, IHW, O]
        let kernel_spatial_shape = self.kernel_format.spatial_shape(self.kernel.shape());
        let kernel_spatial_len: usize = kernel_spatial_shape.iter().product();

        let permutation_to_ihw_o: TVec<usize> = match self.kernel_format {
            KernelFormat::OIHW => (1..self.kernel.rank()).chain(once(0)).collect(),
            KernelFormat::HWIO => once(self.kernel.rank() - 2)
                .chain(0..self.kernel.rank() - 2)
                .chain(once(self.kernel.rank() - 1))
                .collect(),
        };
        let kernel_t = self.kernel.clone().into_tensor().permute_axes(&permutation_to_ihw_o)?;
        let kernel_shape = &[
            1,
            self.kernel_format.i(self.kernel.shape()) * kernel_spatial_len,
            kernel_t.shape()[kernel_t.rank() - 1],
        ];
        let kernel_t = kernel_t.into_shape(kernel_shape)?;
        let gemm = target.wire_node(
            format!("{}.gemm", name),
            crate::ops::matmul::MatMulUnary::new(kernel_t.into_arc_tensor(), false, false, false),
            &reshaped,
        )?;
        let deconv_sum = target.wire_node(
            format!("{}.deconv_sum", name),
            super::deconv_sum::DeconvSum::new(
                self.data_format.clone(),
                self.kernel_format.clone(),
                self.padding.clone(),
                self.kernel.shape().into(),
                input_shape.to_tvec(),
            ),
            &gemm,
        )?;
        Ok(deconv_sum)
    }
}

impl_dyn_hash!(DeconvUnary);

impl Op for DeconvUnary {
    fn name(&self) -> Cow<str> {
        "DeconvUnary".into()
    }
    op_core_mir!();
    op_as_typed_op!();
}

impl EvalOp for DeconvUnary {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let mut model = TypedModel::default();
        let source =
            model.add_source("source", TypedFact::dt_shape(input.datum_type(), input.shape()))?;
        let output = self.wire_with_deconv_sum("adhoc", &mut model, source)?;
        model.set_output_outlets(&*output)?;
        Ok(tvec!(model
            .into_runnable()?
            .run(tvec!(input.into_tensor()))?
            .remove(0)
            .into_arc_tensor()))
    }
}

impl TypedOp for DeconvUnary {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let x_fact = inputs[0];
        let output_shape = self.output_shape(&*x_fact.shape)?;
        Ok(tvec!(TypedFact::dt_shape(x_fact.datum_type, &output_shape)))
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let mut patch = TypedModelPatch::default();
        let input = patch.tap_model(model, node.inputs[0])?;
        let output = self.wire_with_deconv_sum(&node.name, &mut patch, input)?;
        patch.shunt_outside(model, (node.id, 0).into(), output[0])?;
        Ok(Some(patch))
    }

    as_op!();
}
