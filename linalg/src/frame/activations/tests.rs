macro_rules! prop_activation {
    ($cond:expr, $ti: ty, $ker: ty, $name: ident ( $($param:ident),* )) => {
        proptest::proptest! {
            #[test]
            fn $name(x in proptest::prelude::any::<$ti>(), repeat in 1usize..4, $($param in proptest::prelude::any::<$ti>()),*) {
                if $cond {
                    let mut input = tract_data::prelude::Tensor::zero_aligned::<$ti>(&[<$ker>::nr() * repeat], <$ker>::alignment_bytes()).unwrap();
                    input.fill_t::<$ti>(x).unwrap();
                    let prog = crate::frame::activations::definitions::$name($($param),*);
                    <$ker>::run(&prog.ops, &prog.csts, &mut input.as_slice_mut::<$ti>().unwrap());
                    let expected = crate::frame::activations::reference::$name(x, $($param),*);
                    let mut output = tract_data::prelude::Tensor::zero_aligned::<$ti>(&[<$ker>::nr() * repeat], <$ker>::alignment_bytes()).unwrap();
                    output.fill_t::<$ti>(expected).unwrap();
                    output.close_enough(&input, true).unwrap();
                }
            }
        }
    }
}

#[macro_export]
macro_rules! act_frame_tests {
    ($cond:expr, $ker:ty, $ti:ty) => {
        prop_activation!($cond, $ti, $ker, relu());
        prop_activation!($cond, $ti, $ker, affine(alpha, beta));
        prop_activation!($cond, $ti, $ker, leaky_relu(alpha));
        prop_activation!($cond, $ti, $ker, threshold_relu(alpha));
        prop_activation!($cond, $ti, $ker, softsign());
        prop_activation!($cond, $ti, $ker, hardswish());
        /*
        prop_activation!($cond, $ti, $ker, sigmoid());
        prop_activation!($cond, $ti, $ker, exp2f());
        */
    };
}
