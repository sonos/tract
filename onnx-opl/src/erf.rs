use tract_nnef::internal::*;

tract_core::element_wise!(erf, Erf,
    [f32] => |_, xs| {
        xs.iter_mut().for_each(|x| *x = erf_f32(*x));
        Ok(())
    };
    prefix: "onnx."
);

#[allow(non_upper_case_globals)]
fn erf_f32(x: f32) -> f32 {
    const a1: f32 = 0.0705230784;
    const a2: f32 = 0.0422820123;
    const a3: f32 = 0.0092705272;
    const a4: f32 = 0.0001520143;
    const a5: f32 = 0.0002765672;
    const a6: f32 = 0.0000430638;

    let signum = x.signum();
    let x = x.abs();
    let y = a6 * x;
    let y = (a5 + y) * x;
    let y = (a4 + y) * x;
    let y = (a3 + y) * x;
    let y = (a2 + y) * x;
    let y = (a1 + y) * x;
    let y = 1.0 - (y + 1.0).powi(16).recip();

    y.copysign(signum)
}
