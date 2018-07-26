//! TODO(liautaud):
//! Right now most of the code in this module is duplicated to handle both
//! &T and &mut T, so I should find a way to abstract this.
use Result;
use Tensor;
use analyser::prelude::*;
use analyser::interface::solver::Context;
use analyser::interface::expressions::{Datum, Wrapped};

use num_traits::cast::ToPrimitive;

macro_rules! to_isize {
    ($expr:expr) => (
        $expr
            .to_isize()
            .ok_or(format!("Cannot convert {:?} to isize.", stringify!($expr)))?)
}

macro_rules! wrap_isize {
    ($expr:expr) => (Ok(Some(isize::into_wrapped(to_isize!($expr)))))
}

macro_rules! wrap_isize_option {
    ($expr:expr) => ({
        if let Some(v) = $expr.concretize() {
            Ok(Some(isize::into_wrapped(to_isize!(v))))
        } else {
            Ok(None)
        }
    })
}

macro_rules! wrap_type_option {
    ($expr:expr) => ({
        if let Some(v) = $expr.concretize() {
            Ok(Some(DataType::into_wrapped(v)))
        } else {
            Ok(None)
        }
    })
}

/// A symbolic path for a value.
pub type Path = Vec<isize>;

/// Returns the value at the given path (starting from a context).
pub fn get_path(context: &Context, path: &[isize]) -> Result<Option<Wrapped>> {
    match path[0] {
        0 => get_tensorfacts_path(&context.inputs, &path[1..]),
        1 => get_tensorfacts_path(&context.outputs, &path[1..]),
        _ => bail!("The first component of path {:?} should be 0 (for the `inputs` \
                    set of facts) or 1 (for the `outputs` set of facts).", path)
    }
}

/// Sets the value at the given path (starting from a context).
pub fn set_path(context: &mut Context, path: &[isize], value: Wrapped) -> Result<()> {
    match path[0] {
        0 => set_tensorfacts_path(&mut context.inputs, &path[1..], value),
        1 => set_tensorfacts_path(&mut context.outputs, &path[1..], value),
        _ => bail!("The first component of path {:?} should be 0 (for the `inputs` \
                    set of facts) or 1 (for the `outputs` set of facts).", path)
    }
}

/// Returns the value at the given path (starting from a set of TensorFacts).
fn get_tensorfacts_path(facts: &Vec<TensorFact>, path: &[isize]) -> Result<Option<Wrapped>> {
    match path {
        // Get the number of facts in the set.
        [-1] => wrap_isize!(facts.len()),

        slice if slice[0] >= 0 => {
            let k = slice[0].to_usize().unwrap(); // checked

            if k < facts.len() {
                get_tensorfact_path(&facts[k], &slice[1..])
            } else {
                bail!("There are only {:?} facts in the given set, so the index \
                       {:?} is not valid.", facts.len(), k)
            }
        },

        _ => bail!("The first component of subpath {:?} should either be -1 (for \
                    the number of facts in the set) or a valid fact index.", path)
    }
}

/// Sets the value at the given path (starting from a set of TensorFacts).
fn set_tensorfacts_path(facts: &mut Vec<TensorFact>, path: &[isize], value: Wrapped) -> Result<()> {
    match path {
        // Set the number of facts in the set.
        [-1] => {
            let value = isize::from_wrapped(value).to_usize().unwrap();

            if value != facts.len() {
                bail!("Can't set the length of the given set of facts to {:?} \
                       because it already has length {:?}.", value, facts.len());
            } else {
                Ok(())
            }
        },

        slice if slice[0] >= 0 => {
            let k = slice[0].to_usize().unwrap(); // checked

            if k < facts.len() {
                set_tensorfact_path(&mut facts[k], &path[1..], value)
            } else {
                bail!("There are only {:?} facts in the given set, so the index \
                       {:?} is not valid.", facts.len(), k)
            }
        },

        _ => bail!("The first component of subpath {:?} should either be -1 (for \
                    the number of facts in the set) or a valid fact index.", path)
    }
}

/// Returns the value at the given path (starting from a TensorFact).
fn get_tensorfact_path(fact: &TensorFact, path: &[isize]) -> Result<Option<Wrapped>> {
    match path {
        // Get the type of the TensorFact.
        [0] => wrap_type_option!(fact.datatype),

        // Get the rank of the TensorFact.
        [1] => if fact.shape.open {
            Ok(None)
        } else {
            wrap_isize!(fact.shape.dims.len())
        },

        slice if slice[0] == 2 => get_shape_path(&fact.shape, &slice[1..]),
        slice if slice[0] == 3 => get_value_path(&fact.value, &slice[1..]),

        _ => bail!("The subpath {:?} should start with 0, 1, 2 or 3 (for the type, \
                    rank, dimension or value of the fact respectively).", path)
    }
}

/// Sets the value at the given path (starting from a TensorFact).
fn set_tensorfact_path(fact: &mut TensorFact, path: &[isize], value: Wrapped) -> Result<()> {
    match path {
        // Set the type of the TensorFact.
        [0] => {
            let value = DataType::from_wrapped(value);

            fact.datatype = unify_datatype(
                &fact.datatype,
                &typefact!(value)
            )?;

            Ok(())
        },

        // Set the rank of the TensorFact.
        [1] => {
            let k = isize::from_wrapped(value).to_usize().unwrap();

            fact.shape = unify_shape(
                &fact.shape,
                &ShapeFact::closed(vec![dimfact!(_); k])
            )?;

            Ok(())
        },

        // Set the whole shape of the TensorFact.
        [2] => {
            let shape = ShapeFact::from_wrapped(value);
            fact.shape = unify_shape(&fact.shape, &shape)?;

            Ok(())
        },

        // Set a precise dimension of the TensorFact.
        [2, k] => {
            let k = k.to_usize().unwrap();
            let d = isize::from_wrapped(value).to_usize().unwrap();

            let mut dims = vec![dimfact!(_); k];
            dims.push(dimfact!(d));

            fact.shape = unify_shape(
                &fact.shape,
                &ShapeFact::open(dims)
            )?;

            Ok(())
        },

        // Set a value of the TensorFact.
        // FIXME(liautaud): This should do something.
        slice if slice[0] == 3 => Ok(()),

        _ => bail!("The subpath {:?} should start with 0, 1, 2 or 3 (for the type, \
                    rank, dimension or value of the fact respectively).", path)
    }
}

/// Returns the shape or dimension at the given path (starting from a ShapeFact).
fn get_shape_path(shape: &ShapeFact, path: &[isize]) -> Result<Option<Wrapped>> {
    match path {
        // Get the whole shape.
        [] => {
            if *shape == shapefact![..] {
                Ok(None)
            } else {
                Ok(Some(ShapeFact::into_wrapped(shape.clone())))
            }
        },

        // Get a precise dimension.
        [k] => {
            let k = k.to_usize().unwrap();

            if k < shape.dims.len() {
                // FIXME(liautaud): Return a DimFact directly to handle streaming.
                wrap_isize_option!(shape.dims[k])
            } else if shape.open {
                Ok(None)
            } else {
                bail!("The closed shape {:?} has no {:?}-th dimension.", shape.dims, k);
            }
        },

        _ => bail!("The subpath {:?} for the shape should either be [] (for the \
                    entire shape) or [k] with k the index of a dimension.", path)
    }
}

/// Returns the value at the given path (starting from a ValueFact).
fn get_value_path(value: &ValueFact, path: &[isize]) -> Result<Option<Wrapped>> {
    let path: Vec<_> = path.iter().map(|i| i.to_usize().unwrap()).collect();

    macro_rules! inner {
        ($array:expr) => ({
            match $array.get(path.as_slice()) {
                Some(&v) => wrap_isize!(v),
                None => bail!("There is no index {:?} in value {:?}.", path, $array),
            }
        })
    };

    match value.concretize() {
        None => Ok(None),
        Some(tensor) => match tensor {
            Tensor::I32(array) => inner!(array),
            Tensor::I8(array) => inner!(array),
            Tensor::U8(array) => inner!(array),
            _ => bail!("Found value {:?}, but the solver only supports \
                       integer values.", tensor),
        },
    }
}
