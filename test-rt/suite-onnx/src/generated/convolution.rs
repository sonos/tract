use super::*;
use infra::Test;
use proptest::collection::vec;
use proptest::prelude::*;

#[derive(Clone, Debug)]
struct Convolution {
    ci: usize,
    co: usize,
    group: usize,
    width: usize,
    kernel: usize,
    signed: [bool; 3],
    zeros: [i32; 3],
    present: [bool; 2],
    integer: bool,
    x: Vec<i32>,
    w: Vec<i32>,
    bias: Vec<i32>,
    channel_zeros: bool,
}
impl Arbitrary for Convolution {
    type Parameters = bool;
    type Strategy = BoxedStrategy<Self>;
    fn arbitrary_with(integer: bool) -> Self::Strategy {
        (
            1usize..4,
            1usize..4,
            1usize..3,
            2usize..9,
            1usize..3,
            any::<[bool; 3]>(),
            any::<[bool; 2]>(),
            any::<bool>(),
        )
            .prop_flat_map(move |(ci, co, group, width, kernel, signed, present, channel_zeros)| {
                (
                    vec(-8i32..9, ci * group * width),
                    vec(-7i32..8, co * group * ci * kernel),
                    vec(-5i32..6, co * group),
                )
                    .prop_map(move |(x, w, bias)| Self {
                        ci: ci * group,
                        co: co * group,
                        group,
                        width,
                        kernel,
                        signed,
                        present,
                        integer,
                        x,
                        w,
                        bias,
                        channel_zeros,
                        zeros: signed.map(|s| if s { -3 } else { 120 }),
                    })
            })
            .boxed()
    }
}
impl Test for Convolution {
    fn run_with_approx(&self, _: &str, runtime: &dyn Runtime, _: Approximation) -> TractResult<()> {
        let Self {
            ci,
            co,
            group,
            width,
            kernel,
            signed,
            zeros,
            present,
            integer,
            channel_zeros,
            ..
        } = *self;
        let x: Vec<_> = self.x.iter().map(|x| x + zeros[0]).collect();
        let wz: Vec<_> =
            (0..co).map(|o| zeros[1] + if channel_zeros { o as i32 % 3 } else { 0 }).collect();
        let w: Vec<_> =
            self.w.iter().enumerate().map(|(ix, w)| w + wz[ix / (ci / group * kernel)]).collect();
        let out_width = width - kernel + 1;
        let mut reference = vec![0i32; co * out_width];
        for o in 0..co {
            for p in 0..out_width {
                let mut sum = if integer { 0 } else { self.bias[o] };
                for c in 0..ci / group {
                    for k in 0..kernel {
                        let a = x[(o / (co / group) * (ci / group) + c) * width + p + k]
                            - if !integer || present[0] { zeros[0] } else { 0 };
                        let b = w[(o * (ci / group) + c) * kernel + k]
                            - if !integer || present[1] { wz[o] } else { 0 };
                        sum += a * b;
                    }
                }
                reference[o * out_width + p] = if integer {
                    sum
                } else {
                    (sum + zeros[2])
                        .clamp(if signed[2] { -128 } else { 0 }, if signed[2] { 127 } else { 255 })
                };
            }
        }
        let x = byte_tensor(&[1, ci, width], &x, signed[0]);
        let w = byte_tensor(&[co, ci / group, kernel], &w, signed[1]);
        let mut constants = vec![("w", w)];
        let mut args = vec!["x", "w"];
        let expected = if integer {
            if present[0] || present[1] {
                args.push(if present[0] { "xz" } else { "" });
            }
            if present[1] {
                args.push("wz");
            }
            Tensor::from_shape(&[1, co, out_width], &reference)?
        } else {
            args = vec!["x", "xs", "xz", "w", "ws", "wz", "ys", "yz", "bias"];
            constants.extend([
                ("xs", tensor0(0.25f32)),
                ("ws", tensor0(0.5f32)),
                ("ys", tensor0(0.125f32)),
                ("yz", byte_tensor(&[], &[zeros[2]], signed[2])),
                ("bias", tensor1(&self.bias)),
            ]);
            byte_tensor(&[1, co, out_width], &reference, signed[2])
        };
        let wz_shape = if channel_zeros { vec![co] } else { vec![] };
        constants.extend([
            ("xz", byte_tensor(&[], &[zeros[0]], signed[0])),
            ("wz", byte_tensor(&wz_shape, if channel_zeros { &wz } else { &wz[..1] }, signed[1])),
        ]);
        check(
            NodeProto {
                op_type: if integer { "ConvInteger" } else { "QLinearConv" }.into(),
                input: args.iter().map(|s| (*s).into()).collect(),
                output: vec!["y".into()],
                attribute: vec![int_attr("group", group as i64)],
                ..Default::default()
            },
            vec![("x", x)],
            constants,
            vec![expected],
            runtime,
            Approximation::Exact,
        )
    }
}
pub fn suite() -> TestSuite {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<Convolution>("qlinear", false);
    suite.add_arbitrary::<Convolution>("integer", true);
    suite
}
