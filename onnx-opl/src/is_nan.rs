use tract_nnef::internal::*;

element_wise_oop!(is_nan, IsNan,
    [f32] => bool |_, xs, ys| {
        xs.iter().zip(ys.iter_mut()).for_each(|(x,y)| *y = x.is_nan());
        Ok(())
    },
    [f16] => bool |_, xs, ys| {
        xs.iter().zip(ys.iter_mut()).for_each(|(x,y)| *y = x.is_nan());
        Ok(())
    };
    prefix: "onnx."
);
