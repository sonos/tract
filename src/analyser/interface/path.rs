use Result;
use Tensor;
use tfpb::types::DataType;
use analyser::types::{TensorFact, ShapeFact, ValueFact};
use analyser::interface::expressions::{Datum, Wrapped};

use num_traits::cast::ToPrimitive;

/// A symbolic path for a value.
pub type Path = Vec<isize>;

/// Extracts the value at a given path from a given set of TensorFacts.
///
/// This function returns None if the value is not known (e.g. in the case
/// of TypeFact::Any or DimFact::Any), and Some(wrapped) otherwise.
pub fn get_value_at_path(
    path: &Path,
    facts: &(Vec<TensorFact>, Vec<TensorFact>)
) -> Result<Option<Wrapped>> {
    macro_rules! return_wrapped {
        ($ty:ident, $expr:expr) => (Ok(Some($ty::into_wrapped($expr as $ty))))
    };

    macro_rules! return_wrapped_option {
        ($ty:ident, $expr:expr) => (Ok($expr.concretize().map(|v| $ty::into_wrapped(v as $ty))))
    };

    // Choosing which property of a TensorsFact to use.
    fn get_tensorsfact_property_at_path(path: &[isize], facts: &Vec<TensorFact>) -> Result<Option<Wrapped>> {
        match path[0] {
            -1 => return_wrapped!(isize, facts.len()),

            k if k >= 0 && (k.to_usize().unwrap()) < facts.len() =>
                get_tensorfact_property_at_path(&path[1..], &facts[k.to_usize().unwrap()]),

            k => bail!("Invalid TensorFact index: {:?}.", k),
        }
    }

    // Choosing which property of a TensorFact to use.
    fn get_tensorfact_property_at_path(path: &[isize], fact: &TensorFact) -> Result<Option<Wrapped>> {
        match path[0] {
            // Getting the TensorFact's type.
            0 => return_wrapped_option!(DataType, fact.datatype),

            // Getting the TensorFact's rank.
            1 => if fact.shape.open {
                Ok(None)
            } else {
                return_wrapped!(isize, fact.shape.dims.len())
            },

            2 => get_dimension_at_path(&path[1..], &fact.shape),
            3 => get_element_at_path(&path[1..], &fact.value),
            k => bail!("Invalid TensorFact property index: {:?}.", k),
        }
    }

    // Choosing which dimension of a shape to use.
    fn get_dimension_at_path(path: &[isize], shape: &ShapeFact) -> Result<Option<Wrapped>> {
        let index = path[0].to_usize().unwrap();

        if index < shape.dims.len() {
            return_wrapped_option!(isize, shape.dims[index])
        } else if shape.open {
            Ok(None)
        } else {
            bail!("There is no dimension {:?} in shape {:?}.", index, shape);
        }
    }

    // Choosing which element of the value of a shape to use.
    fn get_element_at_path(path: &[isize], value: &ValueFact) -> Result<Option<Wrapped>> {
        let path: Vec<_> = path.iter().map(|i| i.to_usize().unwrap()).collect();

        macro_rules! get_element_inner {
            ($array:expr) => ({
                match $array.get(path.as_slice()) {
                    Some(&v) => return_wrapped!(isize, v),
                    None => bail!("There is no index {:?} in value {:?}.", path, $array),
                }
            })
        };

        match value.concretize() {
            None => Ok(None),
            Some(tensor) => match tensor {
                Tensor::I32(array) => get_element_inner!(array),
                Tensor::I8(array) => get_element_inner!(array),
                Tensor::U8(array) => get_element_inner!(array),
                _ => bail!("Found value {:?}, but the solver only supports integer values.", tensor),
            },
        }
    }

    // Choosing which TensorsFact to use.
    match path[0] {
        0 => get_tensorsfact_property_at_path(&path[1..], &facts.0),
        1 => get_tensorsfact_property_at_path(&path[1..], &facts.1),
        k => bail!("Invalid TensorsFact index: {:?} (should be 0 or 1).", k),
    }
}