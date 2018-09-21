use std::fmt;
use std::marker::PhantomData;
use std::ops::{Div, Mul};

use num::cast::ToPrimitive;
use num::Zero;

use analyser::prelude::*;
use analyser::rules::prelude::*;
use dim::TDim;
use {DatumType, TfdResult, Tensor};

/// A trait for values produced by expressions.
pub trait Output: fmt::Debug + Clone + PartialEq {
    /// Wraps self in the Wrapped type.
    fn wrap(self) -> Wrapped {
        Self::into_wrapped(self)
    }

    /// Wraps the fact in the Wrapped type.
    fn into_wrapped(source: Self) -> Wrapped;

    /// Retrieves the fact from the Wrapped type.
    /// Panics if wrapped doesn't have the right constructor.
    fn from_wrapped(wrapped: Wrapped) -> TfdResult<Self>;
}

macro_rules! impl_output {
    ($type:ty, $constr:ident) => {
        impl Output for $type {
            fn into_wrapped(source: Self) -> Wrapped {
                Wrapped::$constr(source)
            }

            fn from_wrapped(wrapped: Wrapped) -> TfdResult<$type> {
                if let Wrapped::$constr(v) = wrapped {
                    Ok(v)
                } else {
                    bail!("Tried to get a {} from {:?}.", stringify!($ty), wrapped);
                }
            }
        }
    };
}

impl_output!(IntFact, Int);
impl_output!(TypeFact, Type);
impl_output!(ShapeFact, Shape);
impl_output!(ValueFact, Value);
impl_output!(DimFact, Dim);

// Converts back and forth between Wrapped and usize.
impl Output for usize {
    fn into_wrapped(source: usize) -> Wrapped {
        IntFact::into_wrapped((source as isize).into())
    }

    fn from_wrapped(wrapped: Wrapped) -> TfdResult<usize> {
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

    fn from_wrapped(wrapped: Wrapped) -> TfdResult<isize> {
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

    fn from_wrapped(wrapped: Wrapped) -> TfdResult<Tensor> {
        let message = format!("Tried to convert {:?} to a tensor.", wrapped);

        ValueFact::from_wrapped(wrapped)?
            .concretize()
            .ok_or(message.into())
    }
}

// Converts back and forth between Wrapped and usize.
impl Output for TDim {
    fn into_wrapped(source: TDim) -> Wrapped {
        DimFact::into_wrapped(source.into())
    }

    fn from_wrapped(wrapped: Wrapped) -> TfdResult<TDim> {
        let message = format!("Tried to convert {:?} to a usize.", wrapped);

        DimFact::from_wrapped(wrapped)?
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
    Dim(DimFact),
}

/// An expression that can be compared by the solver.
pub trait Expression: fmt::Debug {
    type Output: Output;

    /// Returns the current value of the expression in the given context.
    fn get(&self, context: &Context) -> TfdResult<Self::Output>;

    /// Tries to set the value of the expression in the given context.
    fn set(&self, context: &mut Context, value: Self::Output) -> TfdResult<()>;

    /// Returns the paths that the expression depends on.
    fn get_paths(&self) -> Vec<&Path>;
}

/// A constant expression (e.g. `2` or `DatumType::DT_INT32`).
pub struct ConstantExpression<T: Output>(T);

impl<T: Output> Expression for ConstantExpression<T> {
    type Output = T;

    /// Returns the current value of the expression in the given context.
    fn get(&self, _: &Context) -> TfdResult<T> {
        Ok(self.0.clone())
    }

    /// Tries to set the value of the expression in the given context.
    fn set(&self, _: &mut Context, value: T) -> TfdResult<()> {
        if self.0 == value {
            Ok(())
        } else {
            bail!(
                "Cannot set the value of constant {:?} to {:?}.",
                self.0,
                value
            );
        }
    }

    /// Returns the paths that the expression depends on.
    fn get_paths(&self) -> Vec<&Path> {
        vec![]
    }
}

impl<T: Output> fmt::Debug for ConstantExpression<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{:?}", self.0)
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
    fn get(&self, context: &Context) -> TfdResult<T> {
        context.get(&self.0)
    }

    /// Tries to set the value of the expression in the given context.
    fn set(&self, context: &mut Context, value: T) -> TfdResult<()> {
        context.set(&self.0, value)
    }

    /// Returns the paths that the expression depends on.
    fn get_paths(&self) -> Vec<&Path> {
        vec![&self.0]
    }
}

impl<T: Output> fmt::Debug for VariableExpression<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{:?}", self.0)
    }
}

/// A scalar product between a constant and another expression.
pub struct ProductExpression<E, V>(isize, E)
where
    V: Zero + Mul<isize, Output = V> + Div<isize, Output = V> + Clone + Output,
    E: Expression<Output = V>;

impl<E, V> Expression for ProductExpression<E, V>
where
    V: Zero + Mul<isize, Output = V> + Div<isize, Output = V> + Clone + Output,
    E: Expression<Output = V>,
{
    type Output = V;

    /// Returns the current value of the expression in the given context.
    fn get(&self, context: &Context) -> TfdResult<V> {
        let v: V = self.1.get(context)?;
        Ok(v * self.0)
    }

    /// Tries to set the value of the expression in the given context.
    fn set(&self, context: &mut Context, value: V) -> TfdResult<()> {
        let k = &self.0;
        let m = value;

        if m.is_zero() && k.is_zero() {
            // We want to set 0 * x <- 0, so we don't have to do anything.
            Ok(())
        } else if m.is_zero() {
            // We want to set k * x <- 0, where k != 0, so we have to set x <- 0.
            self.1.set(context, V::zero())
        } else {
            /*
            // We want to set k * x <- m, where k and m != 0, so we will try
            // to set x <- m / k using a checked division. This way, if m is
            // not divisible by k, we will return Err instead of panicking.
            let div = m.div(&V::from(*k)).ok_or(format!(
                "Cannot set the value of ({:?}, _) to {:?} because \
                 {:?} is not divisible by {:?}.",
                k, m, m, k
            ))?;
            */

            let div = m.div(*k);
            self.1.set(context, div)
        }
    }

    /// Returns the paths that the expression depends on.
    fn get_paths(&self) -> Vec<&Path> {
        self.1.get_paths()
    }
}

impl<E, V> fmt::Debug for ProductExpression<E, V>
where
    V: Zero + Mul<isize, Output = V> + Div<isize, Output = V> + Clone + Output,
    E: Expression<Output = V>,
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{}*{{{:?}}}", self.0, self.1)
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

/*
impl IntoExpression<ConstantExpression<DimFact>> for isize {
    fn into_expr(self) -> ConstantExpression<DimFact> {
        ConstantExpression(self.into())
    }
}
*/

/// Converts &isize to ConstantExpression.
impl<'a> IntoExpression<ConstantExpression<IntFact>> for &'a isize {
    fn into_expr(self) -> ConstantExpression<IntFact> {
        ConstantExpression((*self).into())
    }
}

/// Converts DatumType to ConstantExpression.
impl IntoExpression<ConstantExpression<TypeFact>> for DatumType {
    fn into_expr(self) -> ConstantExpression<TypeFact> {
        ConstantExpression(self.into())
    }
}

/// Converts &DatumType to ConstantExpression.
impl<'a> IntoExpression<ConstantExpression<TypeFact>> for &'a DatumType {
    fn into_expr(self) -> ConstantExpression<TypeFact> {
        ConstantExpression((*self).into())
    }
}

/// Converts T: Fact + Output to ConstantExpression<T>.
impl<T> IntoExpression<ConstantExpression<T>> for T
where
    T: Fact + Output,
{
    fn into_expr(self) -> ConstantExpression<T> {
        ConstantExpression(self)
    }
}

/// Converts TDim to ConstantExpression.
impl IntoExpression<ConstantExpression<DimFact>> for TDim {
    fn into_expr(self) -> ConstantExpression<DimFact> {
        ConstantExpression(self.into())
    }
}

// Converts any comparable proxy to VariableExpression<Output>.
impl<T> IntoExpression<VariableExpression<T::Output>> for T
where
    T: ComparableProxy,
{
    fn into_expr(self) -> VariableExpression<T::Output> {
        VariableExpression(self.get_path().clone().into(), PhantomData)
    }
}

/// Converts (isize, IntoExpression<Output = IntFact>) to ProductExpression.
impl<E, V, I> IntoExpression<ProductExpression<E, V>> for (isize, I)
where
    V: Zero + Mul<isize, Output = V> + Div<isize, Output = V> + Clone + Output,
    E: Expression<Output = V>,
    I: IntoExpression<E>,
{
    fn into_expr(self) -> ProductExpression<E, V> {
        let (k, e) = self;
        ProductExpression(k, e.into_expr())
    }
}

/// Converts (i32, IntoExpression<Output = IntFact>) to ProductExpression.
impl<E, V, I> IntoExpression<ProductExpression<E, V>> for (i32, I)
where
    V: Zero + Mul<isize, Output = V> + Div<isize, Output = V> + Clone + Output,
    E: Expression<Output = V>,
    I: IntoExpression<E>,
{
    fn into_expr(self) -> ProductExpression<E, V> {
        let (k, e) = self;
        ProductExpression(k as isize, e.into_expr())
    }
}
