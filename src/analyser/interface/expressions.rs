use std::marker::PhantomData;
use std::fmt::Debug;

use num_traits::Num;
use num_traits::CheckedDiv;

use Result;
use tfpb::types::DataType;
use analyser::interface::path::Path;
use analyser::interface::solver::Context;
use analyser::interface::proxies::IntProxy;
use analyser::interface::proxies::TypeProxy;


/// The types of values that expressions can produce.
pub trait Datum: Debug + Clone + Copy + PartialEq {
    fn into_wrapped(source: Self) -> Wrapped;
    fn from_wrapped(wrapped: Wrapped) -> Self;
}

macro_rules! impl_datum {
    ($type:ty, $constr:ident) => {
        impl Datum for $type {
            fn into_wrapped(source: Self) -> Wrapped {
                Wrapped::$constr(source)
            }

            fn from_wrapped(wrapped: Wrapped) -> $type {
                if let Wrapped::$constr(v) = wrapped {
                    v
                } else {
                    panic!("Tried to get a {} from {:?}.", stringify!($ty), wrapped);
                }
            }
        }
    }
}

/// A wrapper for all the types of values that expressions can produce.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Wrapped {
    Int(isize),
    Type(DataType),
}

impl_datum!(isize, Int);
impl_datum!(DataType, Type);


/// An expression that can be compared by the solver.
pub trait Expression {
    type Output: Datum;

    /// Returns the current value of the expression in the given context.
    /// If the expression doesn't have a value, returns None.
    fn get(&self, context: &Context) -> Result<Option<Self::Output>>;

    /// Tries to set the value of the expression in the given context.
    fn set(&self, context: &mut Context, value: Self::Output) -> Result<()>;

    /// Returns the paths that the expression depends on.
    fn get_paths(&self) -> Vec<&Path>;
}


/// A constant expression (e.g. `2` or `DataType::DT_INT32`).
pub struct ConstantExpression<T: Datum>(T);

impl<T: Datum> Expression for ConstantExpression<T> {
    type Output = T;

    /// Returns the current value of the expression in the given context.
    fn get(&self, _: &Context) -> Result<Option<T>> {
        Ok(Some(self.0))
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
pub struct VariableExpression<T: Datum>(Path, PhantomData<T>);

impl<T: Datum> Expression for VariableExpression<T> {
    type Output = T;

    /// Returns the current value of the expression in the given context.
    fn get(&self, context: &Context) -> Result<Option<T>> {
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
pub struct ProductExpression<T, E>(T, E)
where
    T: Datum + Num + CheckedDiv,
    E: Expression<Output = T>;

impl<T, E> Expression for ProductExpression<T, E>
where
    T: Datum + Num + CheckedDiv,
    E: Expression<Output = T>
{
    type Output = T;

    /// Returns the current value of the expression in the given context.
    fn get(&self, context: &Context) -> Result<Option<T>> {
        let inner = self.1.get(context)?;

        Ok(inner.map(|v| v * self.0))
    }

    /// Tries to set the value of the expression in the given context.
    fn set(&self, context: &mut Context, value: T) -> Result<()> {
        let k = self.0;
        let m = value;

        if m == T::zero() && k == T::zero() {
            // We want to set 0 * x <- 0, so we don't have to do anything.
            Ok(())
        } else if m == T::zero() {
            // We want to set k * x <- 0, where k != 0, so we have to set x <- 0.
            self.1.set(context, T::zero())
        } else {
            // We want to set k * x <- m, where k and m != 0, so we will try
            // to set x <- m / k using a checked division. This way, if m is
            // not divisible by k, we will return Err instead of panicking.
            let div = m
                .checked_div(&k)
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

/// Converts &T to ConstantExpression<T>.
impl<'a, T> IntoExpression<ConstantExpression<T>> for &'a T where T: Datum {
    fn into_expr(self) -> ConstantExpression<T> {
        ConstantExpression(*self)
    }
}

/// Converts T to ConstantExpression<T>.
impl<T> IntoExpression<ConstantExpression<T>> for T where T: Datum {
    fn into_expr(self) -> ConstantExpression<T> {
        ConstantExpression(self)
    }
}

/// Converts IntProxy to VariableExpression<isize>.
impl<T> IntoExpression<VariableExpression<isize>> for T where T: IntProxy {
    fn into_expr(self) -> VariableExpression<isize> {
        VariableExpression(self.get_path().to_vec(), PhantomData)
    }
}

// FIXME(liautaud): Use a DimProxy to handle streaming.
// /// Converts DimProxy to VariableExpression<DimFact>.
// impl<T> IntoExpression<VariableExpression<DimFact>> for T where T: DimProxy {
//     fn into_expr(self) -> VariableExpression<DimFact> {
//         VariableExpression(self.get_path().to_vec(), PhantomData)
//     }
// }

/// Converts TypeProxy to VariableExpression<DataType>.
impl<T> IntoExpression<VariableExpression<DataType>> for T where T: TypeProxy {
    fn into_expr(self) -> VariableExpression<DataType> {
        VariableExpression(self.get_path().to_vec(), PhantomData)
    }
}

/// Converts (T, Into<Expression<Output = T>>) to ProductExpression<T>.
impl<T, E, I> IntoExpression<ProductExpression<T, E>> for (T, I)
where
    T: Datum + Num + CheckedDiv,
    E: Expression<Output = T>,
    I: IntoExpression<E>,
{
    fn into_expr(self) -> ProductExpression<T, E> {
        let (k, e) = self;
        ProductExpression(k, e.into_expr())
    }
}