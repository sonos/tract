//! Derive macros for `tract-core`.
//!
//! Currently provides `#[derive(SubstituteSymbols)]`, which emits an
//! `auto_subst_symbols(&self, subs: &HashMap<Symbol, TDim>) -> TractResult<Self>`
//! inherent method on the struct. The method clones every field; for
//! fields whose type is recognized as carrying symbolic dims it
//! applies `TDim::substitute_all` to descend into the value.
//!
//! Recognized "symbolic" types (substituted recursively):
//!
//! - `TDim`                              → `substitute_all`
//! - `Vec<TDim>` / `TVec<TDim>`          → element-wise `substitute_all`
//! - `Option<TDim>`                      → `.as_ref().map(...).transpose()`
//! - `ShapeFact`                         → has its own `substitute_all`
//!
//! Anything else is cloned. The detection is conservative: a field of
//! type `Custom<TDim>` would be cloned, not substituted -- if that
//! field also needs substitution, the contributor must either rename
//! the field's type to one of the recognized forms or write
//! `substitute_symbols` by hand.
//!
//! Pair this derive with the `substitute_symbols_default!` macro
//! exported by `tract-core` to provide the full `TypedOp::substitute_symbols`
//! impl in three lines.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{Data, DeriveInput, Fields, Type, TypePath, parse_macro_input};

/// `#[derive(SubstituteSymbols)]` -- generate an inherent
/// `auto_subst_symbols` method that returns `Self` with every TDim-
/// bearing field substituted via `substitute_all(subs)`.
#[proc_macro_derive(SubstituteSymbols)]
pub fn derive_substitute_symbols(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let field_exprs = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(fields) => fields
                .named
                .iter()
                .map(|f| {
                    let fname = f.ident.as_ref().expect("named field must have an ident");
                    let kind = classify(&f.ty);
                    field_expr(fname, kind)
                })
                .collect::<Vec<_>>(),
            Fields::Unit => Vec::new(),
            Fields::Unnamed(_) => {
                return syn::Error::new_spanned(
                    name,
                    "#[derive(SubstituteSymbols)] does not support tuple structs; use a struct \
                     with named fields",
                )
                .to_compile_error()
                .into();
            }
        },
        Data::Enum(_) | Data::Union(_) => {
            return syn::Error::new_spanned(
                name,
                "#[derive(SubstituteSymbols)] is only supported on structs with named fields",
            )
            .to_compile_error()
            .into();
        }
    };

    let expanded = quote! {
        impl #impl_generics #name #ty_generics #where_clause {
            #[allow(unused_variables)]
            pub fn auto_subst_symbols(
                &self,
                subs: &::std::collections::HashMap<crate::internal::Symbol, crate::internal::TDim>,
            ) -> crate::internal::TractResult<Self> {
                Ok(Self {
                    #(#field_exprs,)*
                })
            }
        }
    };

    expanded.into()
}

#[derive(Clone, Copy, Debug)]
enum FieldKind {
    /// `TDim`
    Tdim,
    /// `Vec<TDim>` or `TVec<TDim>` (or anything matching the simple
    /// shape `*<TDim>` whose head ident ends in `Vec`).
    VecTdim,
    /// `Option<TDim>`
    OptionTdim,
    /// `ShapeFact` -- has its own `substitute_all` method on the type.
    ShapeFact,
    /// Anything else -- plain clone.
    Other,
}

fn classify(ty: &Type) -> FieldKind {
    let Type::Path(TypePath { path, .. }) = ty else {
        return FieldKind::Other;
    };
    let Some(seg) = path.segments.last() else {
        return FieldKind::Other;
    };
    let head = seg.ident.to_string();
    match head.as_str() {
        "TDim" => return FieldKind::Tdim,
        "ShapeFact" => return FieldKind::ShapeFact,
        _ => {}
    }
    let is_vec_like = head == "Vec" || head == "TVec" || head.ends_with("Vec");
    let is_option = head == "Option";
    let inner_is_tdim = matches!(
        seg.arguments,
        syn::PathArguments::AngleBracketed(ref ab)
            if ab.args.iter().any(|a| matches!(
                a,
                syn::GenericArgument::Type(Type::Path(TypePath { path, .. }))
                    if path.segments.last().map(|s| s.ident == "TDim").unwrap_or(false)
            )),
    );
    match (is_vec_like, is_option, inner_is_tdim) {
        (true, false, true) => FieldKind::VecTdim,
        (false, true, true) => FieldKind::OptionTdim,
        _ => FieldKind::Other,
    }
}

fn field_expr(name: &syn::Ident, kind: FieldKind) -> TokenStream2 {
    match kind {
        FieldKind::Tdim => quote! {
            #name: self.#name.substitute_all(subs)?
        },
        FieldKind::VecTdim => quote! {
            #name: self.#name
                .iter()
                .map(|d| d.substitute_all(subs))
                .collect::<crate::internal::TractResult<_>>()?
        },
        FieldKind::OptionTdim => quote! {
            #name: self.#name
                .as_ref()
                .map(|d| d.substitute_all(subs))
                .transpose()?
        },
        FieldKind::ShapeFact => quote! {
            #name: self.#name.iter()
                .map(|d| d.substitute_all(subs))
                .collect::<crate::internal::TractResult<crate::internal::ShapeFact>>()?
        },
        FieldKind::Other => quote! {
            #name: self.#name.clone()
        },
    }
}
