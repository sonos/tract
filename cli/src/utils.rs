#[allow(unused_imports)]
use CliResult;
use tfdeploy::{DatumType, Tensor};

/// Compares the outputs of a node in tfdeploy and tensorflow.
#[cfg(feature = "tensorflow")]
pub fn compare_outputs<Tensor1, Tensor2>(rtf: &[Tensor1], rtfd: &[Tensor2]) -> CliResult<()>
where
    Tensor1: ::std::borrow::Borrow<Tensor>,
    Tensor2: ::std::borrow::Borrow<Tensor>,
{
    if rtf.len() != rtfd.len() {
        bail!(
            "Number of output differ: tf={}, tfd={}",
            rtf.len(),
            rtfd.len()
        )
    }

    for (ix, (mtf, mtfd)) in rtf.iter().zip(rtfd.iter()).enumerate() {
        if mtf.borrow().shape().len() != 0 && mtf.borrow().shape() != mtfd.borrow().shape() {
            bail!(
                "Shape mismatch for output {}: tf={:?}, tfd={:?}",
                ix,
                mtf.borrow().shape(),
                mtfd.borrow().shape()
            )
        } else {
            if !mtf.borrow().close_enough(mtfd.borrow(), true) {
                bail!(
                    "Data mismatch: tf={:?}, tfd={:?}",
                    mtf.borrow(),
                    mtfd.borrow()
                )
            }
        }
    }

    Ok(())
}

/// Generates a random tensor of a given size and type.
pub fn random_tensor(sizes: Vec<usize>, datum_type: DatumType) -> Tensor {
    use std::iter::repeat_with;
    use rand;
    let len = sizes.iter().product();
    macro_rules! r {($t:ty) =>
        { repeat_with(|| rand::random::<$t>()).take(len).collect::<Vec<_>>() }
    }

    match datum_type {
        DatumType::F64 => Tensor::f64s(&*sizes, &*r!(f64)),
        DatumType::F32 => Tensor::f32s(&*sizes, &*r!(f32)),
        DatumType::I32 => Tensor::i32s(&*sizes, &*r!(i32)),
        DatumType::I8 => Tensor::i8s(&*sizes, &*r!(i8)),
        DatumType::U8 => Tensor::u8s(&*sizes, &*r!(u8)),
        _ => unimplemented!("missing type"),
    }.unwrap()
}
