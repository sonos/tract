use std::fmt;

use tract_num_traits::ToPrimitive;

use crate::infer::*;

use self::super::super::factoid::*;
use self::super::expr::*;
use self::super::solver::Context;

/// A symbolic path for a value.
#[derive(PartialEq, Eq, Clone)]
pub struct Path(TVec<isize>);

impl From<TVec<isize>> for Path {
    fn from(v: TVec<isize>) -> Path {
        Path(v)
    }
}

impl From<Vec<isize>> for Path {
    fn from(v: Vec<isize>) -> Path {
        Path(v.into())
    }
}

impl ::std::ops::Deref for Path {
    type Target = [isize];
    fn deref(&self) -> &[isize] {
        &self.0
    }
}

impl fmt::Debug for Path {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        debug_path(self, formatter)
    }
}

/// Returns the value at the given path (starting from a context).
pub fn get_path(context: &Context, path: &[isize]) -> TractResult<Wrapped> {
    match path[0] {
        0 => get_tensorfacts_path(&context.inputs, &path[1..]),
        1 => get_tensorfacts_path(&context.outputs, &path[1..]),
        _ => bail!(
            "The first component of path {:?} should be 0 (for the `inputs` \
             set of facts) or 1 (for the `outputs` set of facts).",
            path
        ),
    }
}

/// Sets the value at the given path (starting from a context).
pub fn set_path(context: &mut Context, path: &[isize], value: Wrapped) -> TractResult<()> {
    match path[0] {
        0 => set_tensorfacts_path(&mut context.inputs, &path[1..], value),
        1 => set_tensorfacts_path(&mut context.outputs, &path[1..], value),
        _ => bail!(
            "The first component of path {:?} should be 0 (for the `inputs` \
             set of facts) or 1 (for the `outputs` set of facts).",
            path
        ),
    }
}

fn debug_path(path: &[isize], formatter: &mut fmt::Formatter) -> fmt::Result {
    write!(
        formatter,
        "{}",
        match path[0] {
            0 => "inputs",
            1 => "outputs",
            _ => "buggy_path",
        }
    )?;
    debug_tensorfacts_path(&path[1..], formatter)
}

/// Returns the value at the given path (starting from a set of InferenceFacts).
fn get_tensorfacts_path(facts: &TVec<InferenceFact>, path: &[isize]) -> TractResult<Wrapped> {
    match path {
        // Get the number of facts in the set.
        [-1] => Ok(facts.len().wrap()),

        slice if slice[0] >= 0 => {
            let k = slice[0].to_usize().unwrap(); // checked

            if k < facts.len() {
                get_tensorfact_path(&facts[k], &slice[1..])
            } else {
                bail!(
                    "There are only {:?} facts in the given set, so the index \
                     {:?} is not valid.",
                    facts.len(),
                    k
                )
            }
        }

        _ => bail!(
            "The first component of subpath {:?} should either be -1 (for \
             the number of facts in the set) or a valid fact index.",
            path
        ),
    }
}

/// Sets the value at the given path (starting from a set of InferenceFacts).
fn set_tensorfacts_path(
    facts: &mut TVec<InferenceFact>,
    path: &[isize],
    value: Wrapped,
) -> TractResult<()> {
    match path {
        // Set the number of facts in the set.
        [-1] => {
            // Conversion is checked.
            let value =
                IntFactoid::from_wrapped(value)?.concretize().map(|v| v.to_usize().unwrap());

            if value.is_some() && value.unwrap() != facts.len() {
                bail!(
                    "Can't set the length of the given set of facts to {:?} \
                     because it already has length {:?}.",
                    value,
                    facts.len()
                );
            }

            Ok(())
        }

        slice if slice[0] >= 0 => {
            // Conversion is checked.
            let k = slice[0].to_usize().unwrap();

            if k < facts.len() {
                set_tensorfact_path(&mut facts[k], &path[1..], value)
            } else {
                bail!(
                    "There are only {:?} facts in the given set, so the index \
                     {:?} is not valid.",
                    facts.len(),
                    k
                )
            }
        }

        _ => bail!(
            "The first component of subpath {:?} should either be -1 (for \
             the number of facts in the set) or a valid fact index.",
            path
        ),
    }
}

fn debug_tensorfacts_path(path: &[isize], formatter: &mut fmt::Formatter) -> fmt::Result {
    match path[0] {
        -1 => write!(formatter, ".len"),
        n => {
            write!(formatter, "[{n}]")?;
            debug_tensorfact_path(&path[1..], formatter)
        }
    }
}

/// Returns the value at the given path (starting from a InferenceFact).
fn get_tensorfact_path(fact: &InferenceFact, path: &[isize]) -> TractResult<Wrapped> {
    match path {
        // Get the type of the InferenceFact.
        [0] => Ok(fact.datum_type.wrap()),

        // Get the rank of the InferenceFact.
        [1] => Ok(fact.shape.rank().wrap()),

        slice if slice[0] == 2 => get_shape_path(&fact.shape, &slice[1..]),
        slice if slice[0] == 3 => get_value_path(&fact.value, &slice[1..]),

        _ => bail!(
            "The subpath {:?} should start with 0, 1, 2 or 3 (for the type, \
             rank, dimension or value of the fact respectively).",
            path
        ),
    }
}

/// Sets the value at the given path (starting from a InferenceFact).
fn set_tensorfact_path(
    fact: &mut InferenceFact,
    path: &[isize],
    value: Wrapped,
) -> TractResult<()> {
    match path {
        // Set the type of the InferenceFact.
        [0] => {
            let value = TypeFactoid::from_wrapped(value)?;
            fact.datum_type = value.unify(&fact.datum_type)?;
            Ok(())
        }

        // Set the rank of the InferenceFact.
        [1] => {
            if let Some(k) = IntFactoid::from_wrapped(value)?.concretize() {
                if k >= 0 {
                    let k = k.to_usize().unwrap();
                    fact.shape = fact.shape.unify(&ShapeFactoid::closed(tvec![dimfact!(_); k]))?;
                } else {
                    bail!("Infered a negative rank ({})", k)
                }
            }

            Ok(())
        }

        // Set the whole shape of the InferenceFact.
        [2] => {
            let shape = ShapeFactoid::from_wrapped(value)?;
            fact.shape = shape.unify(&fact.shape)?;

            Ok(())
        }

        // Set a precise dimension of the InferenceFact.
        [2, k] => {
            let k = k.to_usize().unwrap();
            let dim = DimFact::from_wrapped(value)?;

            let mut dims = tvec![dimfact!(_); k];
            dims.push(dim);

            fact.shape = fact.shape.unify(&ShapeFactoid::open(dims))?;

            Ok(())
        }

        // Set full InferenceFact value, also unifying type and shape.
        [3] => {
            let value = ValueFact::from_wrapped(value)?;
            fact.value = fact.value.unify(&value)?;
            if let Some(tensor) = fact.value.concretize() {
                fact.shape = fact.shape.unify(&ShapeFactoid::from(tensor.shape()))?;
                fact.datum_type = fact.datum_type.unify(&TypeFactoid::from(tensor.datum_type()))?;
            }
            Ok(())
        }

        slice if slice[0] == 3 => {
            debug!("FIXME Unimplemented set_value_path for individual value");
            Ok(())
        }

        _ => bail!(
            "The subpath {:?} should start with 0, 1, 2 or 3 (for the type, \
             rank, dimension or value of the fact respectively).",
            path
        ),
    }
}

fn debug_tensorfact_path(path: &[isize], formatter: &mut fmt::Formatter) -> fmt::Result {
    match path {
        [] => Ok(()),
        [0] => write!(formatter, ".datum_type"),
        [1] => write!(formatter, ".rank"),
        [2] => write!(formatter, ".shape"),
        [2, k] => write!(formatter, ".shape[{k}]"),
        slice if slice[0] == 3 => debug_value_path(&path[1..], formatter),
        _ => write!(formatter, ".invalid"),
    }
}

/// Returns the shape or dimension at the given path (starting from a ShapeFactoid).
fn get_shape_path(shape: &ShapeFactoid, path: &[isize]) -> TractResult<Wrapped> {
    match path {
        // Get the whole shape.
        [] => Ok(shape.clone().wrap()),

        // Get a precise dimension.
        [k] => {
            let k = k.to_usize().unwrap();
            if let Some(d) = shape.dims().nth(k) {
                Ok(d.clone().wrap())
            } else if shape.is_open() {
                Ok(dimfact!(_).wrap())
            } else {
                bail!("{:?} has no {:?}-th dimension.", shape, k);
            }
        }

        _ => bail!(
            "The subpath {:?} for the shape should either be [] (for the \
             entire shape) or [k] with k the index of a dimension.",
            path
        ),
    }
}

/// Returns the value at the given path (starting from a ValueFact).
fn get_value_path(value: &ValueFact, path: &[isize]) -> TractResult<Wrapped> {
    trace!("get_value_path path:{path:?} value:{value:?}");
    // Return the whole tensor.
    if path == [-1] || path.is_empty() {
        return Ok(value.clone().wrap());
    }

    let returns = match value.concretize() {
        None => Ok(IntFactoid::default().wrap()),
        Some(tensor) => {
            let path = path.iter().map(|i| *i as usize).collect::<TVec<usize>>();
            if tensor.rank() == 0 && path == tvec!(0) {
                Ok(tensor.cast_to_scalar::<i64>()?.wrap())
            } else {
                Ok(tensor.cast_to::<i64>()?.to_array_view::<i64>()?[&*path].wrap())
            }
        }
    };
    trace!("returns: {returns:?}");
    returns
}

fn debug_value_path(path: &[isize], formatter: &mut fmt::Formatter) -> fmt::Result {
    for p in path {
        write!(formatter, "[{p}]")?;
    }
    Ok(())
}
