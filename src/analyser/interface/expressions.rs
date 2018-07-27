use Result;
use tfpb::types::DataType;

use analyser::types::Fact;
use analyser::types::{IntFact, TypeFact, ShapeFact, DimFact, ValueFact};

use analyser::interface::path::Path;
use analyser::interface::solver::Context;
use analyser::interface::proxies::ComparableProxy;

/// A trait for values produced by expressions.
pub trait Output: Fact {
    /// Wraps self in the Wrapped type.
    fn wrap(self) -> Wrapped {
        Self::into_wrapped(self)
    }

    /// Wraps the fact in the Wrapped type.
    fn into_wrapped(source: Self) -> Wrapped;

    /// Retrieves the fact from the Wrapped type.
    /// Panics if wrapped doesn't have the right constructor.
    fn from_wrapped(wrapped: Wrapped) -> Self;
}

macro_rules! impl_output {
    ($type:ty, $constr:ident) => {
        impl Output for $type {
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

impl_output!(IntFact, Int);
impl_output!(DimFact, Dim);
impl_output!(TypeFact, Type);
impl_output!(ShapeFact, Shape);
impl_output!(ValueFact, Value);

/// A wrapper for all the types of values that expressions can produce.
#[derive(Debug, Clone)]
pub enum Wrapped {
    Int(IntFact),
    Dim(DimFact),
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
pub struct ProductExpression<T, E>(isize, E)
where
    T: Output + Mul<isize>,
    E: Expression<Output = T>;

impl<T, E> Expression for ProductExpression<T, E>
where
    T: Output + Mul<isize, Output=T>,
    E: Expression<Output = T>
{
    type Output = T;

    /// Returns the current value of the expression in the given context.
    fn get(&self, context: &Context) -> Result<T> {
        Ok(self.1.get(context)? * self.0)
    }

    /// Tries to set the value of the expression in the given context.
    fn set(&self, context: &mut Context, value: T) -> Result<()> {
        unimplemented!()
        // let k = &self.0;
        // let m = value;

        // if m == T::zero() && *k == T::zero() {
        //     // We want to set 0 * x <- 0, so we don't have to do anything.
        //     Ok(())
        // } else if m == T::zero() {
        //     // We want to set k * x <- 0, where k != 0, so we have to set x <- 0.
        //     self.1.set(context, T::zero())
        // } else {
        //     // We want to set k * x <- m, where k and m != 0, so we will try
        //     // to set x <- m / k using a checked division. This way, if m is
        //     // not divisible by k, we will return Err instead of panicking.
        //     let div = m
        //         .checked_div(&k)
        //         .ok_or(format!(
        //             "Cannot set the value of ({:?}, _) to {:?} because \
        //             {:?} is not divisible by {:?}.", k, m, m, k))?;

        //     self.1.set(context, div)
        // }
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
impl IntoExpression<ConstantExpression> for isize {
    fn into_expr(self) -> ConstantExpression {
        let fact: IntFact = self.into();
        ConstantExpression(fact.wrap())
    }
}

/// Converts &isize to ConstantExpression.
impl<'a> IntoExpression<ConstantExpression> for &'a isize {
    fn into_expr(self) -> ConstantExpression {
        let fact: IntFact = (*self).into();
        ConstantExpression(fact.wrap())
    }
}

/// Converts DataType to ConstantExpression.
impl IntoExpression<ConstantExpression> for DataType {
    fn into_expr(self) -> ConstantExpression {
        let fact: TypeFact = self.into();
        ConstantExpression(fact.wrap())
    }
}

/// Converts &DataType to ConstantExpression.
impl<'a> IntoExpression<ConstantExpression> for &'a DataType {
    fn into_expr(self) -> ConstantExpression {
        let fact: TypeFact = (*self).into();
        ConstantExpression(fact.wrap())
    }
}

/// Converts T: Output to ConstantExpression<T>.
impl<T> IntoExpression<ConstantExpression> for T where T: Output {
    fn into_expr(self) -> ConstantExpression {
        ConstantExpression(self.wrap())
    }
}

// Converts any comparable proxy to VariableExpression.
impl<T> IntoExpression<VariableExpression> for T where T: ComparableProxy {
    fn into_expr(self) -> VariableExpression {
        VariableExpression(self.get_path().to_vec())
    }
}

/// Converts &ShapeProxy to VariableExpression<ShapeFact>.
impl<'a> IntoExpression<VariableExpression<ShapeFact>> for &'a ShapeProxy {
    fn into_expr(self) -> VariableExpression<ShapeFact> {
        VariableExpression(self.get_path().to_vec(), PhantomData)
    }
}

// ---------------- Conversions from tuples ---------------- //

/// Converts (T, Into<Expression<Output = T>>) to ProductExpression<T>.
impl<T, E, I> IntoExpression<ProductExpression<T, E>> for (isize, I)
where
    T: Output + Mul<isize, Output=T>,
    E: Expression<Output = T>,
    I: IntoExpression<E>,
{
    fn into_expr(self) -> ProductExpression<T, E> {
        let (k, e) = self;
        ProductExpression(k, e.into_expr())
    }
}