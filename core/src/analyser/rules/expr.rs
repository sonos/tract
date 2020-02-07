use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};

use num_traits::ToPrimitive;
use num_traits::Zero;

use crate::internal::*;

use self::super::path::Path;
use self::super::proxies::*;
use self::super::solver::Context;

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
    fn from_wrapped(wrapped: Wrapped) -> TractResult<Self>;
}

macro_rules! impl_output {
    ($type:ty, $constr:ident, $name:expr) => {
        impl Output for $type {
            fn into_wrapped(source: Self) -> Wrapped {
                Wrapped::$constr(source)
            }

            fn from_wrapped(wrapped: Wrapped) -> TractResult<$type> {
                if let Wrapped::$constr(v) = wrapped {
                    Ok(v)
                } else {
                    bail!("Tried to get a {} from {:?}.", $name, wrapped);
                }
            }
        }
    };
}

impl_output!(IntFact, Int, "Int");
impl_output!(TypeFact, Type, "DatumType");
impl_output!(ShapeFactoid, Shape, "Shape");
impl_output!(ValueFact, Tensor, "Tensor");
impl_output!(DimFact, Dim, "TDim");

// Converts back and forth between Wrapped and usize.
impl Output for usize {
    fn into_wrapped(source: usize) -> Wrapped {
        IntFact::into_wrapped((source as i32).into())
    }

    fn from_wrapped(wrapped: Wrapped) -> TractResult<usize> {
        let message = format!("Tried to convert {:?} to a usize.", wrapped);

        IntFact::from_wrapped(wrapped)?
            .concretize()
            .and_then(|u| u.to_usize())
            .ok_or(message.into())
    }
}

// Converts back and forth between Wrapped and i32.
impl Output for i32 {
    fn into_wrapped(source: i32) -> Wrapped {
        IntFact::into_wrapped(source.into())
    }

    fn from_wrapped(wrapped: Wrapped) -> TractResult<i32> {
        let message = format!("Tried to convert {:?} to a i32.", wrapped);

        IntFact::from_wrapped(wrapped)?.concretize().ok_or(message.into())
    }
}

// Converts back and forth between Wrapped and Tensor.
impl Output for Arc<Tensor> {
    fn into_wrapped(source: Arc<Tensor>) -> Wrapped {
        ValueFact::into_wrapped(source.into())
    }

    fn from_wrapped(wrapped: Wrapped) -> TractResult<Arc<Tensor>> {
        let message = format!("Tried to convert {:?} to a tensor.", wrapped);

        ValueFact::from_wrapped(wrapped)?.concretize().ok_or(message.into())
    }
}

// Converts back and forth between Wrapped and usize.
impl Output for TDim {
    fn into_wrapped(source: TDim) -> Wrapped {
        DimFact::into_wrapped(source.into())
    }

    fn from_wrapped(wrapped: Wrapped) -> TractResult<TDim> {
        let message = format!("Tried to convert {:?} to a usize.", wrapped);

        DimFact::from_wrapped(wrapped)?.concretize().ok_or(message.into())
    }
}

/// A wrapper for all the types of values that expressions can produce.
#[derive(Debug, Clone)]
pub enum Wrapped {
    Int(IntFact),
    Type(TypeFact),
    Shape(ShapeFactoid),
    Tensor(ValueFact),
    Dim(DimFact),
}

/// An expression that can be compared by the solver.
pub trait TExp<T>: fmt::Debug {
    /// Returns the current value of the expression in the given context.
    fn get(&self, context: &Context) -> TractResult<T>;

    /// Tries to set the value of the expression in the given context.
    fn set(&self, context: &mut Context, value: T) -> TractResult<bool>;

    /// Returns the paths that the expression depends on.
    fn get_paths(&self) -> Vec<&Path>;
}

pub struct Exp<T>(Box<dyn TExp<T>>);
impl<T: Factoid + Output + Clone + fmt::Debug> TExp<T> for Exp<T> {
    /// Returns the current value of the expression in the given context.
    fn get(&self, context: &Context) -> TractResult<T> {
        self.0.get(context)
    }

    /// Tries to set the value of the expression in the given context.
    fn set(&self, context: &mut Context, value: T) -> TractResult<bool> {
        self.0.set(context, value)
    }

    /// Returns the paths that the expression depends on.
    fn get_paths(&self) -> Vec<&Path> {
        self.0.get_paths()
    }
}

impl<T> fmt::Debug for Exp<T>
where
    T: Factoid + Output + Clone + ::std::fmt::Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{:?}", self.0)
    }
}

pub trait IntoExp<T> {
    /// Converts the value to an Expression.
    fn bex(self) -> Exp<T>;
}

#[derive(new)]
pub struct SumExp<T>(Vec<Exp<T>>)
where
    T: Factoid + Output + Clone + ::std::fmt::Debug + 'static;

impl<T> TExp<T> for SumExp<T>
where
    T: Factoid + Output + Zero + Add<T> + Neg<Output = T> + Clone + ::std::fmt::Debug + 'static,
{
    /// Returns the current value of the expression in the given context.
    fn get(&self, context: &Context) -> TractResult<T> {
        self.0.iter().try_fold(T::zero(), |acc, it| Ok(acc + it.0.get(context)?))
    }

    /// Tries to set the value of the expression in the given context.
    fn set(&self, context: &mut Context, value: T) -> TractResult<bool> {
        let mut sum = T::zero();
        let mut misses = vec![];

        for item in &self.0 {
            let fact = item.get(context)?;
            if fact.is_concrete() {
                sum = sum + fact;
            } else {
                misses.push(item);
            }
        }

        if misses.len() > 1 {
            Ok(false)
        } else if misses.len() == 1 {
            misses[0].set(context, value + -sum)?;
            Ok(true)
        } else if sum == value {
            Ok(false)
        } else {
            bail!("{:?} set to {:?}, already is {:?}", self, value, sum)
        }
    }

    /// Returns the paths that the rule depends on.
    fn get_paths(&self) -> Vec<&Path> {
        self.0.iter().flat_map(|e| e.get_paths()).collect()
    }
}

impl<T> fmt::Debug for SumExp<T>
where
    T: Factoid + Output + Clone + ::std::fmt::Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        for (ix, t) in self.0.iter().enumerate() {
            if ix > 0 {
                write!(formatter, " + ")?;
            }
            t.fmt(formatter)?;
        }
        Ok(())
    }
}

/// A constant expression (e.g. `2` or `DatumType::DT_INT32`).
pub struct ConstantExp<T>(T)
where
    T: Factoid + Output + Clone + ::std::fmt::Debug;

impl<T> TExp<T> for ConstantExp<T>
where
    T: Factoid + Output + Clone + ::std::fmt::Debug,
{
    /// Returns the current value of the expression in the given context.
    fn get(&self, _: &Context) -> TractResult<T> {
        Ok(self.0.clone())
    }

    /// Tries to set the value of the expression in the given context.
    fn set(&self, _: &mut Context, value: T) -> TractResult<bool> {
        self.0.unify(&value)?;
        Ok(false)
    }

    /// Returns the paths that the expression depends on.
    fn get_paths(&self) -> Vec<&Path> {
        vec![]
    }
}

impl<T> fmt::Debug for ConstantExp<T>
where
    T: Factoid + Output + Clone + ::std::fmt::Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{:?}", self.0)
    }
}

/// A reference to a variable.
///
/// For instance, `inputs[0].rank` is a reference to the rank of the first
/// input. Internally, a reference holds a Vec<usize> called a path (see
/// the documentation for `Proxy::get_path`).
pub struct VariableExp<T>(Path, PhantomData<T>)
where
    T: Factoid + Output + Clone + ::std::fmt::Debug;

impl<T> TExp<T> for VariableExp<T>
where
    T: Factoid + Output + Clone + ::std::fmt::Debug,
{
    /// Returns the current value of the expression in the given context.
    fn get(&self, context: &Context) -> TractResult<T> {
        context.get(&self.0).map_err(|e| format!("while getting {:?}, {}", self.0, e).into())
    }

    /// Tries to set the value of the expression in the given context.
    fn set(&self, context: &mut Context, value: T) -> TractResult<bool> {
        let old = self.get(context)?;
        let new = old.unify(&value)?;
        let diff = old != new;
        context.set(&self.0, new).map_err(|e| format!("while setting {:?}, {}", self.0, e))?;
        Ok(diff)
    }

    /// Returns the paths that the expression depends on.
    fn get_paths(&self) -> Vec<&Path> {
        vec![&self.0]
    }
}

impl<T> fmt::Debug for VariableExp<T>
where
    T: Factoid + Output + Clone + ::std::fmt::Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{:?}", self.0)
    }
}

/// A scalar product between a constant and another expression.
pub struct ScaledExp<T>(i32, Exp<T>)
where
    T: Factoid + Output + Zero + Mul<i32, Output = T> + Div<i32, Output = T> + Clone;

impl<T> TExp<T> for ScaledExp<T>
where
    T: Factoid + Output + Zero + Mul<i32, Output = T> + Div<i32, Output = T> + Clone,
{
    /// Returns the current value of the expression in the given context.
    fn get(&self, context: &Context) -> TractResult<T> {
        let v: T = self.1.get(context)?;
        Ok(v * self.0)
    }

    /// Tries to set the value of the expression in the given context.
    fn set(&self, context: &mut Context, value: T) -> TractResult<bool> {
        let k = &self.0;
        let m = value;

        if m.is_zero() && k.is_zero() {
            // We want to set 0 * x <- 0, so we don't have to do anything.
            Ok(false)
        } else if m.is_zero() {
            // We want to set k * x <- 0, where k != 0, so we have to set x <- 0.
            self.1.set(context, T::zero())
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

impl<T> fmt::Debug for ScaledExp<T>
where
    T: Factoid + Output + Zero + Mul<i32, Output = T> + Div<i32, Output = T> + Clone,
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{}*{{{:?}}}", self.0, self.1)
    }
}

/// Cast an IntFact into a DimFact
pub struct IntoDimExp(Exp<IntFact>);

impl TExp<DimFact> for IntoDimExp {
    /// Returns the current value of the expression in the given context.
    fn get(&self, context: &Context) -> TractResult<DimFact> {
        let v: IntFact = self.0.get(context)?;
        match v {
            GenericFact::Only(i) => Ok(GenericFact::Only(i.to_dim())),
            GenericFact::Any => Ok(GenericFact::Any),
        }
    }

    /// Tries to set the value of the expression in the given context.
    fn set(&self, context: &mut Context, value: DimFact) -> TractResult<bool> {
        if let Some(concrete) = value.concretize() {
            if let Ok(int) = concrete.to_integer() {
                return self.0.set(context, GenericFact::Only(int));
            }
        }
        Ok(false)
    }

    /// Returns the paths that the expression depends on.
    fn get_paths(&self) -> Vec<&Path> {
        self.0.get_paths()
    }
}

impl fmt::Debug for IntoDimExp {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{{({:?}) as dim}}", self.0)
    }
}

// ops and cast on Exp

impl<T, E: TExp<T> + 'static> IntoExp<T> for E {
    fn bex(self) -> Exp<T> {
        Exp(Box::new(self))
    }
}

// Type

impl IntoExp<TypeFact> for TypeProxy {
    fn bex(self) -> Exp<TypeFact> {
        VariableExp(self.get_path().clone(), PhantomData).bex()
    }
}

impl<'a> IntoExp<TypeFact> for &'a TypeProxy {
    fn bex(self) -> Exp<TypeFact> {
        VariableExp(self.get_path().clone(), PhantomData).bex()
    }
}

impl IntoExp<TypeFact> for DatumType {
    fn bex(self) -> Exp<TypeFact> {
        ConstantExp(self.into()).bex()
    }
}

impl<'a> IntoExp<TypeFact> for &'a DatumType {
    fn bex(self) -> Exp<TypeFact> {
        ConstantExp((*self).into()).bex()
    }
}

// Int

impl<'a> IntoExp<IntFact> for &'a IntProxy {
    fn bex(self) -> Exp<IntFact> {
        VariableExp(self.get_path().clone(), PhantomData).bex()
    }
}

impl<'a> IntoExp<IntFact> for &'a ElementProxy {
    fn bex(self) -> Exp<IntFact> {
        VariableExp(self.get_path().clone(), PhantomData).bex()
    }
}

impl IntoExp<IntFact> for i32 {
    fn bex(self) -> Exp<IntFact> {
        ConstantExp(self.into()).bex()
    }
}

impl<IE: IntoExp<IntFact>> Add<IE> for Exp<IntFact> {
    type Output = Exp<IntFact>;
    fn add(self, other: IE) -> Exp<IntFact> {
        SumExp(vec![self.bex(), other.bex()]).bex()
    }
}

impl<IE: IntoExp<IntFact>> Sub<IE> for Exp<IntFact> {
    type Output = Exp<IntFact>;
    fn sub(self, other: IE) -> Exp<IntFact> {
        SumExp(vec![self.bex(), -1 * other.bex()]).bex()
    }
}

impl Mul<Exp<IntFact>> for i32 {
    type Output = Exp<IntFact>;
    fn mul(self, other: Exp<IntFact>) -> Exp<IntFact> {
        ScaledExp(self, other).bex()
    }
}

// Dim

impl<'a> IntoExp<DimFact> for &'a DimProxy {
    fn bex(self) -> Exp<DimFact> {
        VariableExp(self.get_path().clone(), PhantomData).bex()
    }
}

impl IntoExp<DimFact> for TDim {
    fn bex(self) -> Exp<DimFact> {
        ConstantExp(self.into()).bex()
    }
}

impl IntoExp<DimFact> for &TDim {
    fn bex(self) -> Exp<DimFact> {
        ConstantExp(self.clone().into()).bex()
    }
}

impl<IE: IntoExp<DimFact>> Add<IE> for Exp<DimFact> {
    type Output = Exp<DimFact>;
    fn add(self, other: IE) -> Exp<DimFact> {
        SumExp(vec![self.bex(), other.bex()]).bex()
    }
}

impl<IE: IntoExp<DimFact>> Sub<IE> for Exp<DimFact> {
    type Output = Exp<DimFact>;
    fn sub(self, other: IE) -> Exp<DimFact> {
        SumExp(vec![self.bex(), -1 * other.bex()]).bex()
    }
}

impl Mul<Exp<DimFact>> for i32 {
    type Output = Exp<DimFact>;
    fn mul(self, other: Exp<DimFact>) -> Exp<DimFact> {
        ScaledExp(self, other).bex()
    }
}

// Cast to dim

pub trait ToDimExp {
    fn to_dim(self) -> Exp<DimFact>;
}

impl ToDimExp for Exp<IntFact> {
    fn to_dim(self) -> Exp<DimFact> {
        IntoDimExp(self).bex()
    }
}

// Shape

impl IntoExp<ShapeFactoid> for ShapeFactoid {
    fn bex(self) -> Exp<ShapeFactoid> {
        ConstantExp(self).bex()
    }
}

impl IntoExp<ShapeFactoid> for ShapeProxy {
    fn bex(self) -> Exp<ShapeFactoid> {
        VariableExp(self.get_path().clone(), PhantomData).bex()
    }
}

impl<'a> IntoExp<ShapeFactoid> for &'a ShapeProxy {
    fn bex(self) -> Exp<ShapeFactoid> {
        VariableExp(self.get_path().clone(), PhantomData).bex()
    }
}

impl IntoExp<ShapeFactoid> for TVec<TDim> {
    fn bex(self) -> Exp<ShapeFactoid> {
        ConstantExp(self.into_iter().collect()).bex()
    }
}

// Arc<Tensor>

impl IntoExp<ValueFact> for ValueProxy {
    fn bex(self) -> Exp<ValueFact> {
        VariableExp(self.get_path().clone(), PhantomData).bex()
    }
}

impl<'a> IntoExp<ValueFact> for &'a ValueProxy {
    fn bex(self) -> Exp<ValueFact> {
        VariableExp(self.get_path().clone(), PhantomData).bex()
    }
}

impl IntoExp<ValueFact> for Arc<Tensor> {
    fn bex(self) -> Exp<ValueFact> {
        ConstantExp(self.into()).bex()
    }
}
