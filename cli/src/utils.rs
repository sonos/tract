extern crate ndarray;
extern crate rand;
extern crate tfdeploy;

use rand::Rng;
use tfdeploy::tfpb::types::DataType;
use tfdeploy::Matrix;

use errors::*;

/// Tries to autodetect the names of the input nodes.
pub fn detect_inputs(model: &tfdeploy::Model) -> Result<Option<Vec<String>>> {
    let mut inputs = Vec::new();

    for node in model.nodes() {
        if node.op_name == "Placeholder" {
            inputs.push(node.name.clone());
        }
    }

    if inputs.len() > 0 {
        info!("Autodetecting input nodes: {:?}.", inputs);
        Ok(Some(inputs))
    } else {
        Ok(None)
    }
}

/// Tries to autodetect the name of the output node.
pub fn detect_output(model: &tfdeploy::Model) -> Result<Option<String>> {
    // We search for the only node in the graph with no successor.
    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); model.nodes().len()];

    for node in model.nodes() {
        for &link in &node.inputs {
            succs[link.0].push(node.id);
        }
    }

    for (i, s) in succs.iter().enumerate() {
        if s.len() == 0 {
            let output = model.get_node_by_id(i)?.name.clone();
            info!("Autodetecting output node: {:?}.", output);

            return Ok(Some(output));
        }
    }

    Ok(None)
}

/// Compares the outputs of a node in tfdeploy and tensorflow.
#[cfg(feature = "tensorflow")]
pub fn compare_outputs<Matrix1, Matrix2>(rtf: &[Matrix1], rtfd: &[Matrix2]) -> Result<()>
where
    Matrix1: ::std::borrow::Borrow<Matrix>,
    Matrix2: ::std::borrow::Borrow<Matrix>,
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

/// Generates a random matrix of a given size and type.
pub fn random_matrix(sizes: Vec<usize>, datatype: DataType) -> Matrix {
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
