use std::fmt::Debug;
use std::marker::PhantomData;

use num_traits::cast::ToPrimitive;
use num_traits::CheckedDiv;

use Result;
use tfpb::types::DataType;
use tensor::Tensor;
use analyser::interface::path::Path;
use analyser::interface::solver::Context;
use analyser::interface::proxies::ComparableProxy;
use analyser::types::{Fact, IntFact, TypeFact, ShapeFact, DimFact, ValueFact};
use analyser::types::SpecialKind;

/// A trait for values produced by expressions.
pub trait Output: Debug + Clone + PartialEq {
    /// Wraps self in the Wrapped type.
    fn wrap(self) -> Wrapped {
        Self::into_wrapped(self)
    }

    /// Wraps the fact in the Wrapped type.
    fn into_wrapped(source: Self) -> Wrapped;

    /// Retrieves the fact from the Wrapped type.
    /// Panics if wrapped doesn't have the right constructor.
    fn from_wrapped(wrapped: Wrapped) -> Result<Self>;
}

macro_rules! impl_output {
    ($type:ty, $constr:ident) => {
        impl Output for $type {
            fn into_wrapped(source: Self) -> Wrapped {
                Wrapped::$constr(source)
            }

            fn from_wrapped(wrapped: Wrapped) -> Result<$type> {
                if let Wrapped::$constr(v) = wrapped {
                    Ok(v)
                } else {
                    bail!("Tried to get a {} from {:?}.", stringify!($ty), wrapped);
                }
            }
        }
    }
}

impl_output!(IntFact, Int);
impl_output!(TypeFact, Type);
impl_output!(ShapeFact, Shape);
impl_output!(ValueFact, Value);

// Converts back and forth between Wrapped and DimFact.
impl Output for DimFact {
    fn into_wrapped(source: DimFact) -> Wrapped {
        IntFact::into_wrapped(source.into())
    }

    fn from_wrapped(wrapped: Wrapped) -> Result<DimFact> {
        match IntFact::from_wrapped(wrapped)? {
            IntFact::Any => Ok(DimFact::Any),

            IntFact::Only(i) =>
                i.to_usize()
                 .ok_or(format!("Tried to convert {:?} to a DimFact.", i).into())
                 .map(|d| DimFact::Only(d)),

            IntFact::Special(s) =>
                if s == SpecialKind::Streamed {
                    Ok(DimFact::Streamed)
                } else {
                    bail!("Tried to convert Special({:?}) to a DimFact, but the only\
                           special value supported by DimFact is Streamed.", s);
                }
        }
    }
}

// Converts back and forth between Wrapped and usize.
impl Output for usize {
    fn into_wrapped(source: usize) -> Wrapped {
        IntFact::into_wrapped(source.into())
    }

    fn from_wrapped(wrapped: Wrapped) -> Result<usize> {
        let message = format!("Tried to convert {:?} to a usize.", wrapped);

        IntFact::from_wrapped(wrapped)?
            .concretize()
            .and_then(|u| u.to_usize())
            .ok_or(message.into())
    }
}

// Converts back and forth between Wrapped and isize.
impl Output for isize {
    fn into_wrapped(source: isize) -> Wrapped {
        IntFact::into_wrapped(source.into())
    }

    fn from_wrapped(wrapped: Wrapped) -> Result<isize> {
        let message = format!("Tried to convert {:?} to a isize.", wrapped);

        IntFact::from_wrapped(wrapped)?
            .concretize()
            .ok_or(message.into())
    }
}

// Converts back and forth between Wrapped and Tensor.
impl Output for Tensor {
    fn into_wrapped(source: Tensor) -> Wrapped {
        ValueFact::into_wrapped(source.into())
    }

    fn from_wrapped(wrapped: Wrapped) -> Result<Tensor> {
        let message = format!("Tried to convert {:?} to a tensor.", wrapped);

        ValueFact::from_wrapped(wrapped)?
            .concretize()
            .ok_or(message.into())
    }
}

/// A wrapper for all the types of values that expressions can produce.
#[derive(Debug, Clone)]
pub enum Wrapped {
    Int(IntFact),
    Type(TypeFact),
    Shape(ShapeFact),
    Value(ValueFact),
}

/// An expression that can be compared by the solver.
pub trait Expression {
    type Output: Output;

    /// Returns the current value of the expression in the given context.
    fn get(&self, context: &Context) -> Result<Self::Output>;

    /// Tries to set the value of the expression in the given context.
    fn set(&self, context: &mut Context, value: Self::Output) -> Result<()>;

    /// Returns the paths that the expression depends on.
    fn get_paths(&self) -> Vec<&Path>;
}

/// A constant expression (e.g. `2` or `DataType::DT_INT32`).
pub struct ConstantExpression<T: Output>(T);

impl<T: Output> Expression for ConstantExpression<T> {
    type Output = T;

    /// Returns the current value of the expression in the given context.
    fn get(&self, _: &Context) -> Result<T> {
        Ok(self.0.clone())
    }

    /// Tries to set the value of the expression in the given context.
    fn set(&self, _: &mut Context, value: T) -> Result<()> {
        if self.0 == value {
            Ok(())
        } else {
            bail!("Cannot set the value of constant {:?} to {:?}.", self.0, value);
        }
    }

    /// Returns the paths that the expression depends on.
    fn get_paths(&self) -> Vec<&Path> {
        vec![]
    }
}


/// A reference to a variable.
///
/// For instance, `inputs[0].rank` is a reference to the rank of the first
/// input. Internally, a reference holds a Vec<usize> called a path (see
/// the documentation for `Proxy::get_path`).
pub struct VariableExpression<T: Output>(Path, PhantomData<T>);

impl<T: Output> Expression for VariableExpression<T> {
    type Output = T;

    /// Returns the current value of the expression in the given context.
    fn get(&self, context: &Context) -> Result<T> {
        context.get(&self.0)
    }

    /// Tries to set the value of the expression in the given context.
    fn set(&self, context: &mut Context, value: T) -> Result<()> {
        context.set(&self.0, value)
    }

    /// Returns the paths that the expression depends on.
    fn get_paths(&self) -> Vec<&Path> {
        vec![&self.0]
    }
}


/// A scalar product between a constant and another expression.
pub struct ProductExpression<E>(isize, E)
where
    E: Expression<Output = IntFact>;

impl<E> Expression for ProductExpression<E>
where
    E: Expression<Output = IntFact>
{
    type Output = IntFact;

    /// Returns the current value of the expression in the given context.
    fn get(&self, context: &Context) -> Result<IntFact> {
        Ok(self.1.get(context)? * self.0)
    }

    /// Tries to set the value of the expression in the given context.
    fn set(&self, context: &mut Context, value: IntFact) -> Result<()> {
        let k = &self.0;
        let m = value;

        if m == 0usize.into() && *k == 0 {
            // We want to set 0 * x <- 0, so we don't have to do anything.
            Ok(())
        } else if m == 0usize.into() {
            // We want to set k * x <- 0, where k != 0, so we have to set x <- 0.
            self.1.set(context, 0usize.into())
        } else {
            // We want to set k * x <- m, where k and m != 0, so we will try
            // to set x <- m / k using a checked division. This way, if m is
            // not divisible by k, we will return Err instead of panicking.
            let div = m
                .checked_div(&(*k).into())
                .ok_or(format!(
                    "Cannot set the value of ({:?}, _) to {:?} because \
                    {:?} is not divisible by {:?}.", k, m, m, k))?;

            self.1.set(context, div)
        }
    }

    /// Returns the paths that the expression depends on.
    fn get_paths(&self) -> Vec<&Path> {
        self.1.get_paths()
    }
}


/// A value that be converted into an expression.
///
/// I am aware that From<T> and Into<T> exist for this very purpose, but the
/// number of conflicting implementations of From<T> in the standard library
/// seems to confuse the compiler to the point where it is impossible to use
/// Into<T> in the signature of `equals`, `equals_all` and `equals_zero` w/o
/// having to specify the type parameters manually.
pub trait IntoExpression<T> {
    /// Converts the value to an Expression.
    fn into_expr(self) -> T;
}

/// Converts isize to ConstantExpression.
impl IntoExpression<ConstantExpression<IntFact>> for isize {
    fn into_expr(self) -> ConstantExpression<IntFact> {
        ConstantExpression(self.into())
    }
}

/// Converts &isize to ConstantExpression.
impl<'a> IntoExpression<ConstantExpression<IntFact>> for &'a isize {
    fn into_expr(self) -> ConstantExpression<IntFact> {
        ConstantExpression((*self).into())
    }
}

/// Converts DataType to ConstantExpression.
impl IntoExpression<ConstantExpression<TypeFact>> for DataType {
    fn into_expr(self) -> ConstantExpression<TypeFact> {
        ConstantExpression(self.into())
    }
}

/// Converts &DataType to ConstantExpression.
impl<'a> IntoExpression<ConstantExpression<TypeFact>> for &'a DataType {
    fn into_expr(self) -> ConstantExpression<TypeFact> {
        ConstantExpression((*self).into())
    }
}

/// Converts T: Fact + Output to ConstantExpression<T>.
impl<T> IntoExpression<ConstantExpression<T>> for T where T: Fact + Output {
    fn into_expr(self) -> ConstantExpression<T> {
        ConstantExpression(self)
    }
}

// Converts any comparable proxy to VariableExpression<Output>.
impl<T> IntoExpression<VariableExpression<T::Output>> for T where T: ComparableProxy {
    fn into_expr(self) -> VariableExpression<T::Output> {
        VariableExpression(self.get_path().to_vec(), PhantomData)
    }
}

/// Converts (isize, IntoExpression<Output = IntFact>) to ProductExpression.
impl<E, I> IntoExpression<ProductExpression<E>> for (isize, I)
where
    E: Expression<Output = IntFact>,
    I: IntoExpression<E>,
{
    fn into_expr(self) -> ProductExpression<E> {
        let (k, e) = self;
        ProductExpression(k, e.into_expr())
    }
}

/// Converts (i32, IntoExpression<Output = IntFact>) to ProductExpression.
impl<E, I> IntoExpression<ProductExpression<E>> for (i32, I)
where
    E: Expression<Output = IntFact>,
    I: IntoExpression<E>,
{
    fn into_expr(self) -> ProductExpression<E> {
        let (k, e) = self;
        ProductExpression(k as isize, e.into_expr())
    }
}
