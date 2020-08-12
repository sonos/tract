use crate::internal::*;

mod deser;
mod ser;

pub fn tract_nnef() -> Registry {
    use tract_core::ops;
    let mut registry = Registry::new("tract_nnef");
    macro_rules! dumper {
        ($op:ty, $path: path) => {
            registry.register_dumper(TypeId::of::<$op>(), |ast, node| {
                $path(ast, node, node.op().downcast_ref::<$op>().unwrap())
            })
        };
    };
    let mut stdlib = crate::framework::stdlib();

    let mut primitive = |registry: &mut Registry, id: &str, func: ToTract| {
        let pos = stdlib.iter().position(|f| f.decl.id == id).unwrap();
        let decl = stdlib.remove(pos).decl;
        registry.register_primitive(id, &decl.parameters, func)
    };

    primitive(&mut registry, "external", deser::external);
    primitive(&mut registry, "variable", deser::variable);
    dumper!(ops::konst::Const, ser::konst);

    primitive(&mut registry, "reshape", deser::reshape);
    primitive(&mut registry, "transpose", deser::transpose);

    primitive(&mut registry, "concat", deser::concat);
    dumper!(ops::array::TypedConcat, ser::concat);
    primitive(&mut registry, "slice", deser::slice);
    dumper!(ops::array::Slice<TDim>, ser::slice<TDim>);
    dumper!(ops::array::Slice<usize>, ser::slice<usize>);

    primitive(&mut registry, "squeeze", deser::squeeze);
    primitive(&mut registry, "unsqueeze", deser::unsqueeze);
    dumper!(ops::change_axes::AxisOp, ser::axis_op);

    primitive(&mut registry, "tile", deser::tile);
    dumper!(ops::array::Tile, ser::tile);

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

    registry.register_binary("lt", &ops::logic::Lesser {});
    registry.register_binary("gt", &ops::logic::Greater {});
    registry.register_binary("le", &ops::logic::LesserEqual {});
    registry.register_binary("ge", &ops::logic::GreaterEqual {});
    registry.register_binary("eq", &ops::logic::Equals {});
    registry.register_binary("ne", &ops::logic::NotEquals {});

    registry.register_binary("and", &ops::logic::And {});
    registry.register_binary("or", &ops::logic::Or {});

    registry.register_binary("select", &ops::logic::Or {});
    dumper!(ops::logic::Iff, ser::select);
    primitive(&mut registry, "select", deser::select);

    registry.register_binary("min", &ops::math::Min {});
    registry.register_binary("max", &ops::math::Max {});

    primitive(&mut registry, "matmul", deser::matmul);
    dumper!(ops::matmul::MatMulUnary, ser::matmul_unary);
    dumper!(ops::matmul::MatMul, ser::matmul);

    primitive(&mut registry, "conv", deser::conv);
    dumper!(ops::cnn::ConvUnary, ser::conv);

    primitive(&mut registry, "sum_reduce", deser::reduce);
    primitive(&mut registry, "max_reduce", deser::reduce);
    primitive(&mut registry, "min_reduce", deser::reduce);
    dumper!(ops::nn::Reduce, ser::reduce);

    primitive(&mut registry, "max_pool_with_index", deser::max_pool_with_index);
    dumper!(ops::cnn::MaxPool, ser::max_pool);
    primitive(&mut registry, "box", deser::sum_pool);
    dumper!(ops::cnn::SumPool, ser::sum_pool);

    for frag in stdlib {
        if frag.body.is_some() {
            registry.register_fragment(frag);
        }
    }
    registry
}
