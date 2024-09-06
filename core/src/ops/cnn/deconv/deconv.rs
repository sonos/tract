use crate::internal::*;
use crate::ops::array::MultiBroadcastTo;
use crate::ops::cnn::wire_reshape_bias_for_bin;
use crate::ops::cnn::KernelFormat;
use crate::ops::cnn::PoolSpec;
use crate::ops::einsum::EinSum;

#[derive(Clone, Debug, new, Hash)]
pub struct Deconv {
    pub pool_spec: PoolSpec,
    pub kernel_format: KernelFormat,
    pub adjustments: TVec<usize>,
    pub group: usize,
}

impl Deconv {
    fn wire_with_deconv_sum(
        &self,
        name: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let input_shape = target.outlet_fact(inputs[0])?.shape.clone();
        let shape = self.pool_spec.data_format.shape(input_shape.to_tvec())?;
        let geo_dim = shape.hw_dims().iter().product();

        // collapse H and W together in input: (N) I HW or (N) HW I
        let mut input = target.wire_node(
            format!("{name}.reshaped_input"),
            AxisOp::Reshape(shape.h_axis(), shape.hw_dims().into(), tvec!(geo_dim)),
            &[inputs[0]],
        )?;

        // rework input to (N) (G) I/G HW or (N) (G) HW I/G
        if self.group != 1 {
            // input is (N) HW I or (N) I HW
            let i_axis = self.pool_spec.data_format.has_n() as usize
                + self.pool_spec.data_format.c_is_last() as usize;
            let i_dim = target.outlet_fact(input[0])?.shape[i_axis].clone();
            input = target.wire_node(
                format!("{name}.reshaped_input_for_group"),
                AxisOp::Reshape(
                    i_axis,
                    tvec![i_dim.clone()],
                    tvec!(self.group.to_dim(), i_dim / self.group),
                ),
                &input,
            )?;
            if self.pool_spec.data_format.c_is_last() {
                input = target.wire_node(
                    format!("{name}.group_axis_left"),
                    AxisOp::Move(
                        self.pool_spec.data_format.has_n() as usize + 1,
                        self.pool_spec.data_format.has_n() as usize,
                    ),
                    &input,
                )?;
            }
        }

        let mut kernel = tvec!(inputs[1]);
        let kernel_fact = target.outlet_fact(kernel[0])?.clone();
        for (ix, op) in self
            .kernel_format
            .kernel_as_group_o_i_hw_ops(&kernel_fact.shape, self.group)
            .into_iter()
            .enumerate()
        {
            kernel = target.wire_node(format!("{name}.kernel.{ix}"), op, &kernel)?;
        }

        kernel = target.wire_node(format!("{name}.kernel.mv_i"), AxisOp::Move(2, 3), &kernel)?;
        kernel =
            AxisOp::wire_collapse_axis(target, format!("{name}.kernel.col_ohw"), kernel[0], 1)?;
        if self.group == 1 {
            kernel = target.wire_node(format!("{name}.kernel.rm_g"), AxisOp::Rm(0), &kernel)?;
        }
        let mut expr = if self.pool_spec.data_format.c_is_last() {
            "gmk,Ngnk->Ngmn".to_string()
        } else {
            "gmk,Ngkn->Ngmn".to_string()
        };
        if !self.pool_spec.data_format.has_n() {
            expr = expr.replace('N', "");
        }
        if self.group == 1 {
            expr = expr.replace('g', "");
        }
        let einsum = target.wire_node(
            format!("{name}.einsum"),
            EinSum { axes: expr.parse()?, operating_dt: kernel_fact.datum_type, q_params: None },
            &[kernel[0], input[0]],
        )?;

        let mut bias = wire_reshape_bias_for_bin(
            target,
            format!("{name}.reshape_bias"),
            inputs[2],
            shape.rank(),
            shape.c_axis(),
            self.pool_spec.output_channels,
        )?[0];
        let output_shape = super::output_shape(&self.pool_spec, &shape.shape, &self.adjustments)?;
        bias = target.wire_node(
            format!("{name}.broadcast_bias"),
            MultiBroadcastTo { shape: output_shape.into() },
            &[bias],
        )?[0];

        // einsum must be (N_)CHkWk_HW
        let deconv_sum = target.wire_node(
            format!("{name}.deconv_sum"),
            super::deconv_sum::DeconvSum::new(
                self.pool_spec.clone(),
                self.kernel_format,
                input_shape,
                self.adjustments.clone(),
                self.group,
            ),
            &[einsum[0], bias],
        )?;
        Ok(deconv_sum)
    }
}

impl Op for Deconv {
    fn name(&self) -> Cow<str> {
        "Deconv".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("{:?}", self.pool_spec)])
    }

    op_as_typed_op!();
}

impl EvalOp for Deconv {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == 3);
        let mut model = TypedModel::default();
        let inputs = inputs
            .into_iter()
            .enumerate()
            .map(|(ix, input)| model.add_const(format!("s{ix}"), input.into_tensor()))
            .collect::<TractResult<TVec<OutletId>>>()?;
        let output = self.wire_with_deconv_sum("adhoc", &mut model, &inputs)?;
        model.set_output_outlets(&output)?;
        model.into_runnable()?.run(tvec![]).context("In adhoc deconvolution eval")
    }
}

impl TypedOp for Deconv {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 3);
        let x_fact = inputs[0];
        let k_fact = inputs[1];
        ensure!(
            &self.pool_spec.input_channels.to_dim()
                == self.pool_spec.data_format.shape(&inputs[0].shape)?.c()
        );
        ensure!(
            self.pool_spec.input_channels.to_dim()
                == *self.kernel_format.input_channels(&k_fact.shape, self.group)
        );
        let output_shape = super::output_shape(&self.pool_spec, &x_fact.shape, &self.adjustments)?;
        Ok(tvec!(x_fact.datum_type.fact(&output_shape)))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        let fact = &inputs[0];
        let k_fact = &inputs[1];
        let shape = self.pool_spec.data_format.shape(&fact.shape)?;
        let mut axes = AxesMapping::disconnected(inputs, outputs)?
            .renaming((InOut::In(0), shape.c_axis()), 'I')?
            .renaming((InOut::Out(0), shape.c_axis()), 'O')?;
        if let Some(n_axis) = shape.n_axis() {
            axes = axes
                .renaming((InOut::In(0), n_axis), 'N')?
                .linking('N', (InOut::Out(0), n_axis))?;
        }
        let h_axis = shape.h_axis();
        let geo = "HWXYZ".chars().chain('a'..);
        let kernel_spatial_shape = self.kernel_format.spatial_shape(&k_fact.shape);
        for ((ix, dim), repr) in kernel_spatial_shape.iter().enumerate().zip(geo) {
            if dim.is_one()
                && self.pool_spec.stride(ix) == 1
                && self.pool_spec.padding.valid_dim(ix, true)
                && self.adjustments[ix] == 0
            {
                axes = axes
                    .renaming((InOut::In(0), ix + h_axis), repr)?
                    .linking((InOut::In(0), ix + h_axis), (InOut::Out(0), ix + h_axis))?;
            }
        }
        Ok(axes)
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let mut patch = TypedModelPatch::default();
        let inputs = patch.taps(model, &node.inputs)?;
        let output = self
            .wire_with_deconv_sum(&node.name, &mut patch, &inputs)
            .context("In wire_with_deconv_sum")?;
        patch.shunt_outside(model, node.id.into(), output[0])?;
        Ok(Some(patch))
    }

    as_op!();
}
