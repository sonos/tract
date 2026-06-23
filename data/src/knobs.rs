//! Runtime configuration knobs.
//!
//! A knob is a named, typed runtime setting — a kernel-selection toggle or a
//! heuristic threshold. Its value is resolved, in priority order, from an
//! explicit programmatic override, then the process environment, then a
//! compiled-in default.
//!
//! Knobs are declared with [`declare_knob!`] and register themselves into a
//! stack-wide inventory (this crate is the lowest in the stack, so `linalg` and
//! everything above can declare). The inventory lets every knob be listed and
//! documented from one place, and set through the API on targets where
//! environment variables are unavailable (e.g. wasm).
//!
//! A knob is the lowest, deployer-facing layer of configuration; it is not a
//! substitute for settings that must travel with a model.

use std::collections::HashMap;
use std::str::FromStr;

use parking_lot::RwLock;

pub use inventory;

/// A type usable as a knob value: parseable from the string sources (env / API)
/// and renderable for listing.
pub trait KnobValue: Sized + Clone + Send + Sync + 'static {
    const TYPE_NAME: &'static str;
    /// Parse from a source string. `None` means "unset/invalid": the resolver
    /// falls through to the next source rather than overriding with garbage.
    fn parse_knob(s: &str) -> Option<Self>;
    fn render_knob(&self) -> String;
}

impl KnobValue for bool {
    const TYPE_NAME: &'static str = "bool";
    fn parse_knob(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "on" | "yes" => Some(true),
            "0" | "false" | "off" | "no" => Some(false),
            _ => None,
        }
    }
    fn render_knob(&self) -> String {
        if *self { "true" } else { "false" }.to_string()
    }
}

macro_rules! impl_knob_value_fromstr {
    ($ty:ty, $name:expr) => {
        impl KnobValue for $ty {
            const TYPE_NAME: &'static str = $name;
            fn parse_knob(s: &str) -> Option<Self> {
                <$ty>::from_str(s.trim()).ok()
            }
            fn render_knob(&self) -> String {
                self.to_string()
            }
        }
    };
}
impl_knob_value_fromstr!(i64, "i64");
impl_knob_value_fromstr!(usize, "usize");
impl_knob_value_fromstr!(f32, "f32");

impl KnobValue for String {
    const TYPE_NAME: &'static str = "string";
    fn parse_knob(s: &str) -> Option<Self> {
        Some(s.to_string())
    }
    fn render_knob(&self) -> String {
        self.clone()
    }
}

/// Optional knob: `None` is the unset/default state (e.g. "autodetect"), `Some`
/// is an explicit override. A source string that does not parse as `T` is
/// treated as unset, so the resolver falls through to the next source.
impl<T: KnobValue> KnobValue for Option<T> {
    const TYPE_NAME: &'static str = "option";
    fn parse_knob(s: &str) -> Option<Self> {
        T::parse_knob(s).map(Some)
    }
    fn render_knob(&self) -> String {
        match self {
            Some(v) => v.render_knob(),
            None => "(unset)".to_string(),
        }
    }
}

static OVERRIDES: RwLock<Option<HashMap<&'static str, String>>> = RwLock::new(None);

fn with_override<R>(name: &str, f: impl FnOnce(Option<&String>) -> R) -> R {
    let guard = OVERRIDES.read();
    f(guard.as_ref().and_then(|m| m.get(name)))
}

/// A declared knob. Construct via [`declare_knob!`] rather than directly so it
/// is registered for listing.
pub struct Knob<T: KnobValue> {
    pub name: &'static str,
    pub doc: &'static str,
    default: fn() -> T,
}

impl<T: KnobValue> Knob<T> {
    pub const fn new(name: &'static str, default: fn() -> T, doc: &'static str) -> Self {
        Knob { name, doc, default }
    }

    /// The compiled-in default, ignoring override and environment.
    pub fn default_value(&self) -> T {
        (self.default)()
    }

    /// Resolve the current value: programmatic override, then environment, then
    /// default.
    pub fn get(&self) -> T {
        if let Some(v) = with_override(self.name, |o| o.and_then(|s| T::parse_knob(s))) {
            return v;
        }
        if let Ok(s) = std::env::var(self.name)
            && let Some(v) = T::parse_knob(&s)
        {
            return v;
        }
        self.default_value()
    }

    /// Set a programmatic override (highest priority), effective for subsequent
    /// [`get`](Self::get) calls. Prefer threading config through build options
    /// where possible; this is global mutable state.
    pub fn set(&self, value: T) {
        OVERRIDES.write().get_or_insert_with(HashMap::new).insert(self.name, value.render_knob());
    }

    /// Drop the programmatic override, reverting to environment/default.
    pub fn clear(&self) {
        if let Some(m) = OVERRIDES.write().as_mut() {
            m.remove(self.name);
        }
    }
}

/// Type-erased description of a declared knob, collected stack-wide for listing
/// and documentation.
pub struct KnobInfo {
    pub name: &'static str,
    pub doc: &'static str,
    pub type_name: &'static str,
    /// The default value, rendered.
    pub default: fn() -> String,
    /// The currently-resolved value, rendered.
    pub current: fn() -> String,
}

inventory::collect!(KnobInfo);

/// All declared knobs, sorted by name.
pub fn all() -> Vec<&'static KnobInfo> {
    let mut v: Vec<&'static KnobInfo> = inventory::iter::<KnobInfo>.into_iter().collect();
    v.sort_by_key(|k| k.name);
    v
}

/// Set a knob by name from a raw string — the environment-style source, for
/// hosts where environment variables are unavailable (e.g. wasm). Returns
/// `false` if no knob by that name is declared.
pub fn set_str(name: &str, value: &str) -> bool {
    match all().into_iter().find(|k| k.name == name) {
        Some(info) => {
            OVERRIDES.write().get_or_insert_with(HashMap::new).insert(info.name, value.to_string());
            true
        }
        None => false,
    }
}

/// Drop a knob's programmatic override by name. Returns `false` if not declared.
pub fn clear_str(name: &str) -> bool {
    match all().into_iter().find(|k| k.name == name) {
        Some(info) => {
            if let Some(m) = OVERRIDES.write().as_mut() {
                m.remove(info.name);
            }
            true
        }
        None => false,
    }
}

/// Render all declared knobs as a human-readable table.
pub fn list() -> String {
    use std::fmt::Write;
    let mut out = String::new();
    for k in all() {
        let _ = writeln!(
            out,
            "{name} : {ty} = {current} (default {default})\n    {doc}",
            name = k.name,
            ty = k.type_name,
            current = (k.current)(),
            default = (k.default)(),
            doc = k.doc,
        );
    }
    out
}

/// Declare a knob: a typed runtime setting resolved from override / environment
/// / default, registered for listing.
///
/// ```ignore
/// declare_knob!(MY_FLAG, bool, false, "TRACT_MY_FLAG", "Enable the thing.");
/// // ... later ...
/// if MY_FLAG.get() { /* ... */ }
/// ```
#[macro_export]
macro_rules! declare_knob {
    ($ident:ident, $ty:ty, $default:expr, $name:literal, $doc:literal) => {
        #[doc = $doc]
        pub static $ident: $crate::knobs::Knob<$ty> =
            $crate::knobs::Knob::new($name, || $default, $doc);

        $crate::knobs::inventory::submit! {
            $crate::knobs::KnobInfo {
                name: $name,
                doc: $doc,
                type_name: <$ty as $crate::knobs::KnobValue>::TYPE_NAME,
                default: || $crate::knobs::KnobValue::render_knob(&$ident.default_value()),
                current: || $crate::knobs::KnobValue::render_knob(&$ident.get()),
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    declare_knob!(TEST_BOOL, bool, false, "TRACT_TEST_KNOB_BOOL", "Test bool knob.");
    declare_knob!(TEST_INT, usize, 7, "TRACT_TEST_KNOB_INT", "Test int knob.");

    #[test]
    fn default_then_override_then_clear() {
        assert!(!TEST_BOOL.get());
        assert_eq!(TEST_INT.get(), 7);

        TEST_BOOL.set(true);
        assert!(TEST_BOOL.get());
        TEST_BOOL.clear();
        assert!(!TEST_BOOL.get());
    }

    #[test]
    fn set_by_name() {
        assert!(set_str("TRACT_TEST_KNOB_INT", "42"));
        assert_eq!(TEST_INT.get(), 42);
        assert!(clear_str("TRACT_TEST_KNOB_INT"));
        assert_eq!(TEST_INT.get(), 7);
        assert!(!set_str("TRACT_TEST_KNOB_DOES_NOT_EXIST", "1"));
    }

    #[test]
    fn registered_in_inventory() {
        let names: Vec<_> = all().iter().map(|k| k.name).collect();
        assert!(names.contains(&"TRACT_TEST_KNOB_BOOL"));
        assert!(names.contains(&"TRACT_TEST_KNOB_INT"));
    }

    #[test]
    fn bool_parsing() {
        assert_eq!(bool::parse_knob("1"), Some(true));
        assert_eq!(bool::parse_knob("Off"), Some(false));
        assert_eq!(bool::parse_knob("maybe"), None);
    }
}
