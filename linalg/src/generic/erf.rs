use crate::element_wise::ElementWiseKer;

#[allow(non_upper_case_globals)]
#[allow(clippy::excessive_precision)]
fn serf(x: &mut f32) {
    const a1: f32 = 0.0705230784;
    const a2: f32 = 0.0422820123;
    const a3: f32 = 0.0092705272;
    const a4: f32 = 0.0001520143;
    const a5: f32 = 0.0002765672;
    const a6: f32 = 0.0000430638;

    let signum = x.signum();
    let abs = x.abs();
    let y = a6 * abs;
    let y = (a5 + y) * abs;
    let y = (a4 + y) * abs;
    let y = (a3 + y) * abs;
    let y = (a2 + y) * abs;
    let y = (a1 + y) * abs;
    let y = 1.0 - (y + 1.0).powi(16).recip();

    *x = y.copysign(signum)
}

#[derive(Clone, Debug)]
pub struct SErf4;

impl ElementWiseKer<f32> for SErf4 {
    fn name() -> &'static str {
        "generic"
    }

    fn alignment_items() -> usize {
        16
    }

    fn alignment_bytes() -> usize {
        16
    }

    fn nr() -> usize {
        4
    }

    fn run(x: &mut [f32], _: ()) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(serf)
    }
}
