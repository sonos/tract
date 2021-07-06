use crate::internal::*;
use crate::ops::cnn::KernelFormat;
use crate::ops::cnn::PoolSpec;

// no-bias, no-group, f32

#[derive(Clone, Debug, new, Hash)]
pub struct DeconvUnary {
    pub pool_spec: PoolSpec,
    pub kernel_format: KernelFormat,
    pub kernel: Arc<Tensor>,
    pub bias: Option<Arc<Tensor>>,

    pub adjustments: TVec<usize>,
    pub group: usize,
}

impl DeconvUnary {
    fn wire_with_deconv_sum(
        &self,
        name: &str,
        target: &mut TypedModel,
        input: OutletId,
    ) -> TractResult<TVec<OutletId>> {
        use std::iter::once;
        let input_shape = target.outlet_fact(input)?.shape.clone();
        let shape = self.pool_spec.data_format.shape(input_shape.to_tvec())?;
        let geo_dim = shape.hw_dims().iter().product();

        // collapse input as (N) I HW or (N) HW I
        let mut input = target.wire_node(
            format!("{}.reshaped_input", name),
            AxisOp::Reshape(shape.h_axis(), shape.hw_dims().into(), tvec!(geo_dim)),
            &[input],
        )?;

        // rework input to (N) (G) I/G HW or (N) (G) HW I/G
        if self.group != 1 {
            // input is (N) HW I or (N) I HW
            let i_axis = self.pool_spec.data_format.has_n() as usize
                + self.pool_spec.data_format.c_is_last() as usize;
            let i_dim = target.outlet_fact(input[0])?.shape[i_axis].clone();
            input = target.wire_node(
                format!("{}.reshaped_input_for_group", name),
                AxisOp::Reshape(
                    i_axis,
                    tvec![i_dim.clone()],
                    tvec!(self.group.to_dim(), i_dim / self.group),
                ),
                &input,
            )?;
            if self.pool_spec.data_format.c_is_last() {
                input = target.wire_node(
                    format!("{}.group_axis_left", name),
                    AxisOp::Move(
                        self.pool_spec.data_format.has_n() as usize + 1,
                        self.pool_spec.data_format.has_n() as usize,
                    ),
                    &input,
                )?;
            }
        }

        let kernel_spatial_shape = self.kernel_format.spatial_shape(self.kernel.shape());

        // kernel: insert G: before O in OIHW, before I in HWIO
        let kernel_shape_with_g: TVec<usize> = match self.kernel_format {
            KernelFormat::OIHW => once(self.group)
                .chain(once(self.kernel.shape()[0] / self.group))
                .chain(self.kernel.shape()[1..].iter().cloned())
                .collect(),
            KernelFormat::HWIO => kernel_spatial_shape
                .iter()
                .cloned()
                .chain(once(self.group))
                .chain(once(self.kernel.shape()[self.kernel.rank() - 2] / self.group))
                .chain(once(self.kernel.shape()[self.kernel.rank() - 1]))
                .collect(),
        };
        let kernel_with_group =
            self.kernel.clone().into_tensor().into_shape(&kernel_shape_with_g)?;

        // gemm: m=OHkWk, k=I, n=HW
        // kernel from OIHW to [(batch=1), (group maybe), OHW, I]

        let permutation_to_g_o_h_w_i: TVec<usize> = match self.kernel_format {
            // kernel_with_group is in G O I H W
            KernelFormat::OIHW => {
                once(0).chain(once(1)).chain(3..kernel_with_group.rank()).chain(once(2)).collect()
            }
            // kernel_with_group is in H W G I O
            KernelFormat::HWIO => once(kernel_with_group.rank() - 3)
                .chain(once(kernel_with_group.rank() - 1))
                .chain(0..kernel_with_group.rank() - 3)
                .chain(once(kernel_with_group.rank() - 2))
                .collect(),
        };
        let kernel_as_g_o_h_w_i = kernel_with_group.permute_axes(&permutation_to_g_o_h_w_i)?;
        let mut shape_g_ohw_i = tvec!(
            kernel_as_g_o_h_w_i.shape()[1..kernel_as_g_o_h_w_i.rank() - 1].iter().product(),
            kernel_as_g_o_h_w_i.shape()[kernel_as_g_o_h_w_i.rank() - 1],
        );
        if self.group != 1 {
            shape_g_ohw_i.insert(0, self.group);
        }
        if self.pool_spec.data_format.has_n() {
            shape_g_ohw_i.insert(0, 1);
        }
        let kernel_as_g_ohw_i = kernel_as_g_o_h_w_i.into_shape(&*&shape_g_ohw_i)?;
        let trans_data = self.pool_spec.data_format.c_is_last();
        let gemm = target.wire_node(
            format!("{}.gemm", name),
            crate::ops::matmul::MatMulUnary::new(
                kernel_as_g_ohw_i.into_arc_tensor(),
                false,
                trans_data,
                false,
            ),
            &input,
        )?;
        // gemm must be (N_)CHkWk_HW
        let deconv_sum = target.wire_node(
            format!("{}.deconv_sum", name),
            super::deconv_sum::DeconvSum::new(
                self.pool_spec.clone(),
                self.kernel_format.clone(),
                input_shape.to_tvec(),
                self.adjustments.clone(),
                self.bias.clone(),
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
        let output_shape = super::output_shape(&self.pool_spec, &*x_fact.shape, &self.adjustments)?;
        Ok(tvec!(TypedFact::dt_shape(x_fact.datum_type, &output_shape)))
    }

    fn invariants(&self, _inputs: &[&TypedFact], _outputs: &[&TypedFact]) -> TractResult<Invariants> {
        let mut invariants = Invariants::default();
        if self.pool_spec.data_format.has_n() {
            invariants.axes.push(AxisInfo::simple(0))
        }
        for geo_axis in 0..self.pool_spec.kernel_shape.len() {
            let kernel_len = self.pool_spec.kernel_shape[geo_axis];
            if kernel_len == 1 {
            invariants.axes.push(AxisInfo::simple(geo_axis + self.pool_spec.data_format.h_axis()))
            }
        }
        Ok(invariants)
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
