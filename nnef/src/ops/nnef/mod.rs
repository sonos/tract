use tract_core::ops;

use crate::internal::*;
use crate::ops::tract_core;

pub mod deser;
pub mod ser;

pub fn tract_nnef() -> Registry {
    let mut registry = Registry::new("tract_nnef");
    let mut stdlib = crate::framework::stdlib();

    let mut primitive = |registry: &mut Registry, id: &str, func: ToTract| {
        let pos = stdlib.iter().position(|f| f.decl.id.0 == id).unwrap();
        let decl = stdlib.remove(pos).decl;
        registry.register_primitive(id, &decl.parameters, &decl.results, func);
    };

    registry.register_dumper(pin_const);

    primitive(&mut registry, "external", deser::external);
    registry.register_dumper(ser::source);
    primitive(&mut registry, "variable", deser::variable);
    registry.register_dumper(ser::konst);

    primitive(&mut registry, "reshape", deser::reshape);
    primitive(&mut registry, "transpose", deser::transpose);

    primitive(&mut registry, "concat", deser::concat);
    registry.register_dumper(ser::concat);
    primitive(&mut registry, "slice", deser::slice);
    registry.register_dumper(ser::slice);

    primitive(&mut registry, "squeeze", deser::squeeze);
    primitive(&mut registry, "unsqueeze", deser::unsqueeze);
    registry.register_dumper(ser::axis_op);

    primitive(&mut registry, "tile", deser::tile);
    registry.register_dumper(ser::tile);
    registry.register_dumper(ser::dyn_tile);

    primitive(&mut registry, "pad", deser::pad);
    registry.register_dumper(ser::pad);

    primitive(&mut registry, "stack", deser::stack);
    primitive(&mut registry, "unstack", deser::unstack);

    registry.register_binary("add", &ops::math::Add {});
    registry.register_binary("sub", &ops::math::Sub {});
    registry.register_binary("mul", &ops::math::Mul {});
    registry.register_binary("div", &ops::math::Div {});
    registry.register_binary("pow", &ops::math::Pow {});

    registry.register_unit_element_wise("exp", &ops::math::Exp {});
    registry.register_unit_element_wise("log", &ops::math::Ln {});
    registry.register_unit_element_wise("sin", &ops::math::Sin {});
    registry.register_unit_element_wise("cos", &ops::math::Cos {});
    registry.register_unit_element_wise("abs", &ops::math::Abs {});
    registry.register_unit_element_wise("neg", &ops::math::Neg {});
    registry.register_unit_element_wise("sign", &ops::math::Sign {});
    registry.register_unit_element_wise("recip", &ops::math::Recip {});

    registry.register_unit_element_wise("tan", &ops::math::Tan {});
    registry.register_unit_element_wise("acos", &ops::math::Acos {});
    registry.register_unit_element_wise("asin", &ops::math::Asin {});
    registry.register_unit_element_wise("atan", &ops::math::Atan {});
    registry.register_unit_element_wise("cosh", &ops::math::Cosh {});
    registry.register_unit_element_wise("sinh", &ops::math::Sinh {});
    registry.register_unit_element_wise("acosh", &ops::math::Acosh {});
    registry.register_unit_element_wise("asinh", &ops::math::Asinh {});
    registry.register_unit_element_wise("atanh", &ops::math::Atanh {});

    registry.register_unit_element_wise("floor", &ops::math::Floor {});
    registry.register_unit_element_wise("ceil", &ops::math::Ceil {});
    registry.register_unit_element_wise("round", &ops::math::Round {});

    registry.register_unit_element_wise("square", &ops::math::Square {});
    registry.register_unit_element_wise("sqrt", &ops::math::Sqrt {});
    registry.register_unit_element_wise("rsqrt", &ops::math::Rsqrt {});

    registry.register_unit_element_wise("tanh", &ops::math::Tanh {});
    registry.register_unit_element_wise("sigmoid", &ops::nn::Sigmoid {});

    registry.register_unit_element_wise("not", &ops::logic::Not {});

    registry.register_unit_element_wise("neg", &ops::math::Neg {});

    registry.register_element_wise(
        "leaky_relu",
        TypeId::of::<ops::nn::LeakyRelu>(),
        Box::new(ser::leaky_relu),
        vec![TypeName::Scalar.tensor().named("x"), TypeName::Scalar.named("alpha")],
        deser::leaky_relu,
    );

    registry.register_dumper(ser::comp);
    for c in ["eq", "ne", "ge", "gt", "le", "lt"] {
        primitive(&mut registry, c, deser::comp);
    }

    registry.register_binary("and", &ops::logic::And {});
    registry.register_binary("or", &ops::logic::Or {});

    registry.register_binary("select", &ops::logic::Or {});
    registry.register_dumper(ser::select);
    primitive(&mut registry, "select", deser::select);

    registry.register_binary("min", &ops::math::Min {});
    registry.register_binary("max", &ops::math::Max {});

    primitive(&mut registry, "matmul", deser::matmul);
    //    registry.register_dumper(ser::matmul);

    primitive(&mut registry, "conv", deser::conv);
    registry.register_dumper(ser::conv);
    primitive(&mut registry, "deconv", deser::deconv);
    registry.register_dumper(ser::deconv);

    primitive(&mut registry, "sum_reduce", deser::reduce);
    primitive(&mut registry, "max_reduce", deser::reduce);
    primitive(&mut registry, "min_reduce", deser::reduce);
    primitive(&mut registry, "argmax_reduce", deser::reduce);
    primitive(&mut registry, "argmin_reduce", deser::reduce);
    registry.register_dumper(ser::reduce);

    primitive(&mut registry, "softmax", deser::softmax);
    registry.register_dumper(ser::softmax);

    primitive(&mut registry, "max_pool_with_index", deser::max_pool_with_index);
    registry.register_dumper(ser::max_pool);
    primitive(&mut registry, "box", deser::sum_pool);
    registry.register_dumper(ser::sum_pool);

    registry.register_dumper(ser::basic_matmul);

    for frag in stdlib {
        if frag.body.is_some() {
            registry.register_fragment(frag);
        }
    }
    registry
}

pub fn pin_const(
    ast: &mut IntoAst,
    node: &TypedNode,
    _op: &ops::identity::PinConst,
) -> TractResult<Option<Arc<RValue>>> {
    Ok(Some(ast.mapping[&node.inputs[0]].clone()))
}
