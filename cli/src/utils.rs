use ndarray;
use rand;
use rand::Rng;
use tfdeploy::tfpb::types::DataType;
use tfdeploy::Tensor;

/// Compares the outputs of a node in tfdeploy and tensorflow.
#[cfg(feature = "tensorflow")]
pub fn compare_outputs<Tensor1, Tensor2>(rtf: &[Tensor1], rtfd: &[Tensor2]) -> Result<()>
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
            if !mtf.borrow().close_enough(mtfd.borrow()) {
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
pub fn random_tensor(sizes: Vec<usize>, datatype: DataType) -> Tensor {
    macro_rules! for_type {
        ($t:ty) => {
            ndarray::Array::from_shape_fn(sizes, |_| rand::thread_rng().gen())
                as ndarray::ArrayD<$t>
        };
    }

    match datatype {
        DataType::DT_DOUBLE => for_type!(f64).into(),
        DataType::DT_FLOAT => for_type!(f32).into(),
        DataType::DT_INT32 => for_type!(i32).into(),
        DataType::DT_INT8 => for_type!(i8).into(),
        DataType::DT_UINT8 => for_type!(u8).into(),
        _ => unimplemented!(),
    }
}
