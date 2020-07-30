use crate::ast::*;
use std::collections::HashMap;
use tract_core::internal::*;

use crate::model::{AugmentedInvocation, ModelBuilder};

pub type Primitives = HashMap<
    String,
    Arc<dyn Fn(&mut ModelBuilder, &AugmentedInvocation) -> TractResult<TVec<OutletId>>>,
>;

pub fn primitives() -> Primitives {
    use tract_core::ops::{logic, math};
    let mut primitives: Primitives = Default::default();
    primitives.insert("external".to_string(), Arc::new(external));
    primitives.insert("variable".to_string(), Arc::new(variable));
    primitives.insert("conv".to_string(), Arc::new(conv));

    macro_rules! mew {
        ($nnef: ident, $tract: expr) => {
            primitives.insert(
                stringify!($nnef).to_string(),
                Arc::new(|b, i| multiary_elementwise(b, i, Box::new($tract))),
            );
        };
    };

    mew!(add, math::add::bin_typed());
    mew!(sub, math::sub::bin_typed());
    mew!(mul, math::mul::bin_typed());
    mew!(div, math::div::bin_typed());
    mew!(pow, math::pow::bin_typed());

    mew!(lt, logic::lesser::bin_typed());
    mew!(gt, logic::greater::bin_typed());
    mew!(le, logic::lesser_equal::bin_typed());
    mew!(ge, logic::greater_equal::bin_typed());
    mew!(eq, logic::equals::bin_typed());
    mew!(new, logic::not_equals::bin_typed());

    mew!(select, logic::Iff);

    primitives.insert("sum_reduce".to_string(), Arc::new(reduce));
    primitives.insert("max_reduce".to_string(), Arc::new(reduce));
    primitives.insert("min_reduce".to_string(), Arc::new(reduce));

    primitives.insert("matmul".to_string(), Arc::new(matmul));

    primitives
}

fn external(
    builder: &mut ModelBuilder,
    invocation: &AugmentedInvocation,
) -> TractResult<TVec<OutletId>> {
    let type_name = invocation.invocation.generic_type_name.unwrap_or(TypeName::Scalar);
    let dt = if type_name == TypeName::Scalar { f32::datum_type() } else { todo!() };
    let shape = invocation.named_arg("shape")?.to_shape_fact(builder)?;
    Ok(tvec!(builder.model.add_source("", TypedFact::dt_shape(dt, shape)?)?))
}

fn variable(
    builder: &mut ModelBuilder,
    invocation: &AugmentedInvocation,
) -> TractResult<TVec<OutletId>> {
    let type_name = invocation.invocation.generic_type_name.unwrap_or(TypeName::Scalar);
    let dt = if type_name == TypeName::Scalar { f32::datum_type() } else { todo!() };
    let shape = invocation.named_arg("shape")?.to_shape_fact(builder)?;
    let shape = shape.as_finite().unwrap();
    Ok(tvec!(builder.model.add_const("", Tensor::zero_dt(dt, &shape)?.into_arc_tensor())?))
}

/*
fragment conv( input: tensor<scalar>, filter: tensor<scalar>,
bias: tensor<scalar> = 0.0, border: string = 'constant',
padding: (integer,integer)[] = [], stride: integer[] = [],
dilation: integer[] = [], groups: integer = 1 )
-> ( output: tensor<scalar> );
*/

fn conv(
    builder: &mut ModelBuilder,
    invocation: &AugmentedInvocation,
) -> TractResult<TVec<OutletId>> {
    use tract_core::ops::cnn::{ConvUnary, KernelFormat, PaddingSpec, PoolSpec};
    use tract_core::ops::nn::DataFormat;
    let input = invocation.pos_arg(0)?.to_wire(builder)?;
    let kernel = invocation.pos_arg(1)?.to_tensor(builder)?;
    let bias = invocation.pos_arg(2)?.to_tensor(builder)?;
    let dilation = invocation.named_arg("dilation")?.to_shape_fact(builder)?;
    let stride = invocation.named_arg("stride")?.to_shape_fact(builder)?;
    let padding = invocation.named_arg("padding")?.to_tensor(builder)?;
    let border = invocation.named_arg("border")?.to_tensor(builder)?;
    assert_eq!(border, rctensor0("constant".to_string()));
    let group = invocation.named_arg("groups")?.to_tensor(builder)?.cast_to_scalar::<i64>()?;
    let padding = if padding.len() == 0 {
        PaddingSpec::Valid
    } else {
        let padding: tract_ndarray::ArrayView2<i64> =
            padding.to_array_view::<i64>()?.into_dimensionality()?;
        PaddingSpec::Explicit(
            padding.row(0).iter().map(|x| *x as usize).collect(),
            padding.row(1).iter().map(|x| *x as usize).collect(),
            false,
        )
    };
    let pool_spec = PoolSpec::new(
        DataFormat::NCHW,
        kernel.shape()[2..].into(),
        padding,
        Some(dilation.as_finite().unwrap().into()),
        Some(stride.as_finite().unwrap().into()),
        Some(kernel.shape()[0]),
    );
    let op = ConvUnary::new(
        pool_spec,
        KernelFormat::OIHW,
        kernel.clone(),
        group as usize,
        Some(bias.clone()),
        None,
    );
    builder.model.wire_node("", op, &[input])
}

/*
 *   fragment sum_reduce( input: tensor<scalar>, axes: integer[], normalize: logical = false ) -> ( output: tensor<scalar> );
 *   fragment max_reduce( input: tensor<scalar>, axes: integer[] ) -> ( output: tensor<scalar> );
 *   + max, min, argmax, armmin, any, all
 */

fn reduce(
    builder: &mut ModelBuilder,
    invocation: &AugmentedInvocation,
) -> TractResult<TVec<OutletId>> {
    use tract_core::ops::nn::{Reduce, Reducer};
    let input = invocation.pos_arg(0)?.to_wire(builder)?;
    let axes = invocation.pos_arg(1)?.to_tensor(builder)?;
    let axes = axes.cast_to::<i64>()?;
    let axes = axes.as_slice::<i64>()?.iter().map(|&i| i as usize).collect::<TVec<_>>();
    let reducer = match invocation.invocation.id.split("_").next().unwrap() {
        "sum" => Reducer::Sum,
        "min" => Reducer::Min,
        "max" => Reducer::Max,
        _ => bail!("unsupported reducer: {}", invocation.invocation.id),
    };
    let mut wire = builder.model.wire_node("", Reduce::new(axes.clone(), reducer), &[input])?;
    if let Some(norm_arg) = invocation.get_pos_arg(2) {
        let tensor = norm_arg.to_tensor(builder)?;
        if tensor.cast_to_scalar::<bool>()? {
            let input_shape = &builder.model.outlet_fact(input)?.shape;
            let cardinality = axes.iter().map(|ax| input_shape.dim(*ax)).maybe_product()?;
            let cardinality = tensor0(cardinality).broadcast_into_rank(input_shape.rank())?;
            wire = builder.model.wire_node(
                "",
                tract_core::ops::math::div::unary(cardinality.into_arc_tensor()),
                &[input],
            )?;
        }
    }
    Ok(wire)
}

/*
 * fragment matmul( A: tensor<scalar>, B: tensor<scalar>, transposeA: logical = false, transposeB: logical = false ) -> ( C: tensor<scalar> );
 */
fn matmul(
    builder: &mut ModelBuilder,
    invocation: &AugmentedInvocation,
) -> TractResult<TVec<OutletId>> {
    dbg!(&invocation);
    let a = invocation.named_arg("A")?.to_wire(builder)?;
    let b = invocation.named_arg("B")?.to_wire(builder)?;
    let a_trans =
        invocation.named_arg("transposeA")?.to_tensor(builder)?.cast_to_scalar::<bool>()?;
    let b_trans =
        invocation.named_arg("transposeB")?.to_tensor(builder)?.cast_to_scalar::<bool>()?;
    builder.model.wire_node(
        "",
        tract_core::ops::matmul::MatMul { a_trans, b_trans, c_trans: false, q_params: None },
        &[a, b],
    )
}

fn multiary_elementwise(
    builder: &mut ModelBuilder,
    invocation: &AugmentedInvocation,
    op: Box<dyn TypedOp>,
) -> TractResult<TVec<OutletId>> {
    let inputs = invocation
        .invocation
        .arguments
        .iter()
        .map(|arg| Ok(arg.rvalue.to_wire(builder)?))
        .collect::<TractResult<TVec<_>>>()?;
    let inputs = multicast(builder, &inputs)?;
    builder.model.wire_node("", op, &inputs)
}

fn multicast(builder: &mut ModelBuilder, inputs: &[OutletId]) -> TractResult<TVec<OutletId>> {
    let ranks = inputs
        .iter()
        .map(|&i| Ok(builder.model.outlet_fact(i)?.rank()))
        .collect::<TractResult<Vec<usize>>>()?;
    let max_rank = ranks.iter().copied().max().unwrap();
    (inputs.iter())
        .zip(ranks.iter())
        .map(|(&i, &r)| {
            (r..max_rank)
                .try_fold(i, |w, n| Ok(builder.model.wire_node("", AxisOp::Add(n), &[w])?[0]))
        })
        .collect()
}
