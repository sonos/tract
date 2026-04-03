use cudarc::driver::{LaunchConfig, PushKernelArg};
use tract_core::internal::*;
use tract_core::ops::element_wise::ElementWiseMiniOp;
use tract_gpu::tensor::DeviceTensor;

use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::launch_args::TractLaunchArgs;
use crate::kernels::*;

const ALL_OP_NAMES: &[&str] = &[
    "neg",
    "abs",
    "square",
    "sqrt",
    "rsqrt",
    "recip",
    "ceil",
    "floor",
    "round",
    "roundhalftoeven",
    "exp",
    "sigmoid",
    "sin",
    "sinh",
    "asin",
    "asinh",
    "cos",
    "cosh",
    "acos",
    "acosh",
    "tan",
    "tanh",
    "atan",
    "atanh",
    "erf",
    "ln",
    "silu",
    "sign",
    "hardswish",
    "bitnot",
];

pub fn all_functions() -> Vec<String> {
    ALL_OP_NAMES
        .iter()
        .flat_map(|kname| {
            DeviceTensor::SUPPORTED_DT.into_iter().flat_map(move |dt| {
                let tname = DeviceTensor::tname(dt).ok()?;
                Some(format!("element_wise_{kname}_{tname}"))
            })
        })
        .collect()
}

pub fn is_supported(mini_op: &dyn ElementWiseMiniOp, dt: DatumType) -> bool {
    let name = mini_op.name().to_lowercase();
    ALL_OP_NAMES.contains(&name.as_str())
        && if name == "bitnot" {
            dt.is_integer()
        } else {
            matches!(dt, DatumType::F32 | DatumType::F16)
        }
}

pub fn dispatch_eval(
    stream: &TractCudaStream,
    mini_op: &dyn ElementWiseMiniOp,
    input: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    ensure!(output.shape() == input.shape());
    ensure!(output.datum_type() == input.datum_type());

    let op_name = mini_op.name().to_lowercase();
    let tname = DeviceTensor::tname(input.datum_type())?;
    let kname = format!("element_wise_{op_name}_{tname}");

    let func = cuda_context().load_pipeline(LibraryName::ElementWise, kname)?;

    let len = input.len();

    let i_view = get_cuda_view(input);
    let o_view = get_cuda_view(output);

    let cfg = LaunchConfig::for_num_elems(len as _);
    let mut launch_args = TractLaunchArgs::new(stream, &func);
    launch_args.push_view(&i_view);
    launch_args.push_view(&o_view);
    launch_args.push_i32(len);

    launch_args.launch(cfg)
}

pub fn cuda_element_wise_dispatch(
    mini_op: &dyn ElementWiseMiniOp,
    input: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    crate::with_cuda_stream(|stream| dispatch_eval(stream, mini_op, input, output))
}

pub fn cuda_element_wise_op(
    mini_op: Box<dyn ElementWiseMiniOp>,
) -> tract_gpu::ops::element_wise::GpuElementWise {
    tract_gpu::ops::element_wise::GpuElementWise {
        backend_name: "Cuda",
        mini_op,
        dispatch: cuda_element_wise_dispatch,
    }
}

// Generic element-wise fallback — checked after LeakyRelu, GeluApproximate.
crate::register_cuda_op!(tract_core::ops::element_wise::ElementWiseOp, |source, node, op| {
    rule_if!(is_supported(&*op.0, source.node_input_facts(node.id)?[0].datum_type));
    Ok(Some(Box::new(cuda_element_wise_op(op.0.clone()))))
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::with_cuda_stream;
    use num_traits::AsPrimitive;
    use num_traits::Float;
    use tract_gpu::tensor::IntoDevice;

    fn test_case<F>(mini_op: &dyn ElementWiseMiniOp, shape: &[usize]) -> TractResult<()>
    where
        F: Float + Datum,
        usize: AsPrimitive<f32>,
        f32: AsPrimitive<F>,
    {
        with_cuda_stream(|stream| {
            let len = shape.iter().product::<usize>();
            let input = Tensor::from_shape(
                shape,
                &(0..len)
                    .map(|f| -> F {
                        let v: f32 = f.as_();
                        (v / len as f32).as_()
                    })
                    .collect::<Vec<_>>(),
            )?
            .into_device()?;

            let output =
                unsafe { DeviceTensor::uninitialized_dt(input.datum_type(), input.shape())? };
            dispatch_eval(stream, mini_op, &input, &output)?;
            stream.synchronize()?;

            let out = output.to_host()?.into_tensor();
            assert_eq!(out.shape(), shape);
            Ok(())
        })
    }

    use tract_core::ops::math;
    use tract_core::ops::nn;

    #[test]
    fn test_element_wise_exp() -> TractResult<()> {
        test_case::<f32>(&math::Exp {}, &[4, 4])?;
        test_case::<f16>(&math::Exp {}, &[4, 4])?;
        Ok(())
    }

    #[test]
    fn test_element_wise_sigmoid() -> TractResult<()> {
        test_case::<f32>(&nn::Sigmoid {}, &[4, 4])?;
        test_case::<f16>(&nn::Sigmoid {}, &[4, 4])?;
        Ok(())
    }

    #[test]
    fn test_element_wise_abs() -> TractResult<()> {
        test_case::<f32>(&math::Abs {}, &[4, 4])?;
        Ok(())
    }
}
