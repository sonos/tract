use std::fmt;
use std::marker::PhantomData;
use Result;

use num::Zero;
use std::ops::{ Add, Neg, Sub, Div, Mul };
use tensor::{ Tensor, DatumType };
use dim::TDim;
use analyser::types::*;
use analyser::interface::path::Path;
use analyser::interface::solver::Context;
use analyser::interface::expressions::Output;
use analyser::interface::proxies::*;

/// An expression that can be compared by the solver.
pub trait TExp<T>: fmt::Debug {
    /// Returns the current value of the expression in the given context.
    fn get(&self, context: &Context) -> Result<T>;

    /// Tries to set the value of the expression in the given context.
    fn set(&self, context: &mut Context, value: T) -> Result<bool>;

    /// Returns the paths that the expression depends on.
    fn get_paths(&self) -> Vec<&Path>;
}

pub struct Exp<T>(Box<TExp<T>>);
impl<T: Fact + Output + Clone + fmt::Debug> TExp<T> for Exp<T> {
    /// Returns the current value of the expression in the given context.
    fn get(&self, context: &Context) -> Result<T> {
        self.0.get(context)
    }

    /// Tries to set the value of the expression in the given context.
    fn set(&self, context: &mut Context, value: T) -> Result<bool> {
        self.0.set(context, value)
    }

    /// Returns the paths that the expression depends on.
    fn get_paths(&self) -> Vec<&Path> {
        self.0.get_paths()
    }
}

impl<T> fmt::Debug for Exp<T>
where T: Fact + Output + Clone + ::std::fmt::Debug
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{:?}", self.0)
    }
}

pub trait IntoExp<T> {
    /// Converts the value to an Expression.
    fn bex(self) -> Exp<T>;
}

pub struct SumExp<T>(Vec<Exp<T>>)
where T: Fact + Output + Clone + ::std::fmt::Debug;

impl<T> TExp<T> for SumExp<T>
where T: Fact + Output + Zero + Add<T> + Neg<Output=T> + Clone + ::std::fmt::Debug
{
    /// Returns the current value of the expression in the given context.
    fn get(&self, context: &Context) -> Result<T> {
        self.0.iter().try_fold(T::zero(), |acc, it| Ok(acc+it.0.get(context)?))
    }

    /// Tries to set the value of the expression in the given context.
    fn set(&self, context: &mut Context, value: T) -> Result<bool> {
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
where T: Fact + Output + Clone + ::std::fmt::Debug
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
where T: Fact + Output + Clone + ::std::fmt::Debug;

impl<T> TExp<T> for ConstantExp<T>
where T: Fact + Output + Clone + ::std::fmt::Debug
{
    /// Returns the current value of the expression in the given context.
    fn get(&self, _: &Context) -> Result<T> {
        Ok(self.0.clone())
    }

    /// Tries to set the value of the expression in the given context.
    fn set(&self, _: &mut Context, value: T) -> Result<bool> {
        if self.0 == value {
            Ok(false)
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

impl<T> fmt::Debug for ConstantExp<T>
where T: Fact + Output + Clone + ::std::fmt::Debug
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
where T: Fact + Output + Clone + ::std::fmt::Debug;

impl<T> TExp<T> for VariableExp<T>
where T: Fact + Output + Clone + ::std::fmt::Debug
{
    /// Returns the current value of the expression in the given context.
    fn get(&self, context: &Context) -> Result<T> {
        context.get(&self.0)
    }

    /// Tries to set the value of the expression in the given context.
    fn set(&self, context: &mut Context, value: T) -> Result<bool> {
        let old = self.get(context)?;
        let new = old.unify(&value)?;
        let diff = old != new;
        context.set(&self.0, new)?;
        Ok(diff)
    }

    /// Returns the paths that the expression depends on.
    fn get_paths(&self) -> Vec<&Path> {
        vec![&self.0]
    }
}

impl<T> fmt::Debug for VariableExp<T>
where T: Fact + Output + Clone + ::std::fmt::Debug
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{:?}", self.0)
    }
}

/// A scalar product between a constant and another expression.
pub struct ScaledExp<T>(isize, Exp<T>)
where
    T: Fact + Output + Zero + Mul<isize, Output=T> + Div<isize, Output=T> + Clone;

impl<T> TExp<T> for ScaledExp<T>
where
    T: Fact + Output + Zero + Mul<isize, Output=T> + Div<isize, Output=T> + Clone
{
    /// Returns the current value of the expression in the given context.
    fn get(&self, context: &Context) -> Result<T> {
        let v:T = self.1.get(context)?;
        Ok(v*self.0)
    }

    /// Tries to set the value of the expression in the given context.
    fn set(&self, context: &mut Context, value: T) -> Result<bool> {
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
    T: Fact + Output + Zero + Mul<isize, Output=T> + Div<isize, Output=T> + Clone
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{}*{{{:?}}}", self.0, self.1)
    }
}

/// Cast an IntFact into a DimFact
pub struct IntoDimExp(Exp<IntFact>);

impl TExp<DimFact> for IntoDimExp {
    /// Returns the current value of the expression in the given context.
    fn get(&self, context: &Context) -> Result<DimFact> {
        use dim::ToDim;
        let v:IntFact = self.0.get(context)?;
        match v {
            GenericFact::Only(i) => Ok(GenericFact::Only(i.to_dim())),
            GenericFact::Any => Ok(GenericFact::Any),
        }
    }

    /// Tries to set the value of the expression in the given context.
    fn set(&self, context: &mut Context, value: DimFact) -> Result<bool> {
        if let Some(concrete) = value.concretize() {
            if let Ok(int) = concrete.to_integer() {
                return self.0.set(context, GenericFact::Only(int))
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
        write!(formatter, "{{({:?}) as dim}}" , self.0)
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

impl IntoExp<IntFact> for isize {
    fn bex(self) -> Exp<IntFact> {
        ConstantExp(self.into()).bex()
    }
}

impl<IE:IntoExp<IntFact>> Add<IE> for Exp<IntFact> {
    type Output = Exp<IntFact>;
    fn add(self, other:IE) -> Exp<IntFact> {
        SumExp(vec!(self.bex(), other.bex())).bex()
    }
}

impl<IE:IntoExp<IntFact>> Sub<IE> for Exp<IntFact> {
    type Output = Exp<IntFact>;
    fn sub(self, other:IE) -> Exp<IntFact> {
        SumExp(vec!(self.bex(), -1 * other.bex())).bex()
    }
}

impl Mul<Exp<IntFact>> for isize {
    type Output = Exp<IntFact>;
    fn mul(self, other:Exp<IntFact>) -> Exp<IntFact> {
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

impl<IE:IntoExp<DimFact>> Add<IE> for Exp<DimFact> {
    type Output = Exp<DimFact>;
    fn add(self, other:IE) -> Exp<DimFact> {
        SumExp(vec!(self.bex(), other.bex())).bex()
    }
}

impl<IE:IntoExp<DimFact>> Sub<IE> for Exp<DimFact> {
    type Output = Exp<DimFact>;
    fn sub(self, other:IE) -> Exp<DimFact> {
        SumExp(vec!(self.bex(), -1 * other.bex())).bex()
    }
}

impl Mul<Exp<DimFact>> for isize {
    type Output = Exp<DimFact>;
    fn mul(self, other:Exp<DimFact>) -> Exp<DimFact> {
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

impl IntoExp<ShapeFact> for ShapeFact {
    fn bex(self) -> Exp<ShapeFact> {
        ConstantExp(self).bex()
    }
}

impl IntoExp<ShapeFact> for ShapeProxy {
    fn bex(self) -> Exp<ShapeFact> {
        VariableExp(self.get_path().clone(), PhantomData).bex()
    }
}

impl<'a> IntoExp<ShapeFact> for &'a ShapeProxy {
    fn bex(self) -> Exp<ShapeFact> {
        VariableExp(self.get_path().clone(), PhantomData).bex()
    }
}

impl IntoExp<ShapeFact> for Vec<TDim> {
    fn bex(self) -> Exp<ShapeFact> {
        ConstantExp(self.into_iter().collect()).bex()
    }
}

// Value

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

impl IntoExp<ValueFact> for Tensor {
    fn bex(self) -> Exp<ValueFact> {
        ConstantExp(self.into()).bex()
    }
}
