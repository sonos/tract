use tract_nnef::internal::*;

element_wise_oop!(is_nan, IsNan,
    [f16, f32] => bool |_, xs, ys| {
        xs.iter().zip(ys.iter_mut()).for_each(|(x,y)| *y = x.is_nan());
        Ok(())
    };
    prefix: "extra."
);

pub fn register(registry: &mut Registry) {
    registry.register_unit_element_wise("tract_extra_is_nan", &IsNan {});
}
