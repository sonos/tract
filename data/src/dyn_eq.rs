//! Equality between trait objects.
//!
//! A trait with [`DynEq`] as a supertrait can have `PartialEq` and `Eq`
//! implemented for its objects by [`eq_trait_object!`]. Objects of two
//! different concrete types are never equal; two of the same type compare with
//! that type's own `Eq`.
//!
//! Import [`DynEq`] in the modules that call `dyn_eq` and nowhere else — do not
//! re-export it from a prelude. The blanket impl also covers `Box<dyn Trait>`,
//! whose `dyn_eq` downcasts to the box and so answers false for every
//! comparison. Method resolution reaches that impl only where the trait is in
//! scope, so a wide re-export silently turns `boxed.dyn_eq(other)` from a
//! comparison of the pointees into a constant false.

use std::any::Any;

/// Type-erased equality, blanket-implemented for every `Eq + 'static` type.
///
/// Add it as a supertrait, then invoke [`eq_trait_object!`] on the trait — the
/// two go together, the trait alone gives objects nothing.
pub trait DynEq: Any {
    /// Upcast, so the comparison has something to downcast back from.
    #[doc(hidden)]
    fn as_any(&self) -> &dyn Any;

    /// True if `other` is the same concrete type as `self` and equal to it.
    #[doc(hidden)]
    fn dyn_eq(&self, other: &dyn Any) -> bool;
}

impl<T: Eq + 'static> DynEq for T {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dyn_eq(&self, other: &dyn Any) -> bool {
        other.downcast_ref::<T>().is_some_and(|other| self == other)
    }
}

pub use crate::eq_trait_object;

/// Implement `PartialEq` and `Eq` for the objects of a trait having [`DynEq`]
/// as a supertrait, in all four `Send`/`Sync` combinations.
///
/// Takes a bare trait name: a generic trait would need the argument list
/// threaded through, and tract has none needing this.
///
/// ```
/// use tract_data::dyn_eq::DynEq;
///
/// trait Weigh: DynEq {}
/// tract_data::eq_trait_object!(Weigh);
///
/// impl Weigh for u8 {}
/// impl Weigh for u16 {}
///
/// let five: &dyn Weigh = &5u8;
/// assert!(five == &5u8 as &dyn Weigh);
/// assert!(five != &6u8 as &dyn Weigh);
/// // same value, other type: not equal
/// assert!(five != &5u16 as &dyn Weigh);
/// ```
#[macro_export]
macro_rules! eq_trait_object {
    ($trait:ident) => {
        $crate::__eq_trait_object!($trait,);
        $crate::__eq_trait_object!($trait, + ::core::marker::Send);
        $crate::__eq_trait_object!($trait, + ::core::marker::Sync);
        $crate::__eq_trait_object!($trait, + ::core::marker::Send + ::core::marker::Sync);
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __eq_trait_object {
    ($trait:ident, $($auto:tt)*) => {
        impl<'eq> ::core::cmp::PartialEq for dyn $trait $($auto)* + 'eq {
            fn eq(&self, other: &Self) -> bool {
                $crate::dyn_eq::DynEq::dyn_eq(self, $crate::dyn_eq::DynEq::as_any(other))
            }
        }

        impl<'eq> ::core::cmp::Eq for dyn $trait $($auto)* + 'eq {}

        impl<'eq> ::core::cmp::PartialEq<&Self> for ::std::boxed::Box<dyn $trait $($auto)* + 'eq> {
            fn eq(&self, other: &&Self) -> bool {
                self == *other
            }
        }
    };
}
