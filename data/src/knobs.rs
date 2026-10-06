//! Runtime configuration knobs.
//!
//! A knob is a named, typed runtime setting — a kernel-selection toggle or a
//! heuristic threshold. Its value is resolved, in priority order, from an
//! explicit programmatic override, then the calling thread's [`KnobScope`],
//! then the process environment, then the global values a deployment config
//! installed ([`set_config_global`]), then a compiled-in default.
//!
//! A knob declared `scopable` is read while a model is transformed or
//! prepared, so a [`KnobScope`] entered around those calls tunes it for that
//! one model. Every other knob is read elsewhere (machine detection, eval) and
//! only takes a process-wide value.
//!
//! Knobs are declared with [`declare_knob!`] and register themselves into a
//! stack-wide inventory (this crate is the lowest in the stack, so `linalg` and
//! everything above can declare). The inventory lets every knob be listed and
//! documented from one place, and set through the API on targets where
//! environment variables are unavailable (e.g. wasm).
//!
//! A knob is the lowest, deployer-facing layer of configuration; it is not a
//! substitute for settings that must travel with a model.

use std::cell::RefCell;
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;

use anyhow::bail;
use parking_lot::RwLock;

use crate::TractResult;

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

static CONFIG_GLOBALS: RwLock<Option<HashMap<&'static str, String>>> = RwLock::new(None);

thread_local! {
    static SCOPE: RefCell<Option<KnobScope>> = const { RefCell::new(None) };
}

fn with_override<R>(name: &str, f: impl FnOnce(Option<&String>) -> R) -> R {
    let guard = OVERRIDES.read();
    f(guard.as_ref().and_then(|m| m.get(name)))
}

fn scoped_value<T: KnobValue>(name: &str) -> Option<T> {
    SCOPE.with_borrow(|s| s.as_ref().and_then(|s| s.0.get(name)).and_then(|v| T::parse_knob(v)))
}

fn config_global_value<T: KnobValue>(name: &str) -> Option<T> {
    CONFIG_GLOBALS.read().as_ref().and_then(|m| m.get(name)).and_then(|v| T::parse_knob(v))
}

fn find(name: &str) -> TractResult<&'static KnobInfo> {
    match all().into_iter().find(|k| k.name == name) {
        Some(info) => Ok(info),
        None => bail!("No knob named {name}"),
    }
}

fn checked(name: &str, value: &str) -> TractResult<&'static KnobInfo> {
    let info = find(name)?;
    if !(info.parses)(value) {
        bail!("{value:?} is not a valid {} for knob {name}", info.type_name);
    }
    Ok(info)
}

/// Knob values tuned for one model: entered around the calls that transform
/// and prepare it, they reach every [`scopable`](KnobInfo::scopable) knob read
/// on this thread meanwhile.
#[derive(Clone, Debug, Default)]
pub struct KnobScope(Arc<HashMap<&'static str, String>>);

impl KnobScope {
    /// Fails on a name no knob carries, a knob that is not scopable, or a value
    /// that does not parse as the knob's type.
    pub fn new<'a>(values: impl IntoIterator<Item = (&'a str, &'a str)>) -> TractResult<Self> {
        let mut map = HashMap::new();
        for (name, value) in values {
            let info = checked(name, value)?;
            if !info.scopable {
                bail!(
                    "Knob {name} is not read while a model is prepared: it only takes a global value"
                );
            }
            map.insert(info.name, value.to_string());
        }
        Ok(KnobScope(Arc::new(map)))
    }

    /// Run `f` with this scope active on the calling thread. The previously
    /// active scope, if any, is restored afterwards.
    pub fn enter<R>(&self, f: impl FnOnce() -> R) -> R {
        let previous = SCOPE.replace(Some(self.clone()));
        struct Restore(Option<KnobScope>);
        impl Drop for Restore {
            fn drop(&mut self) {
                SCOPE.set(self.0.take());
            }
        }
        let _restore = Restore(previous);
        f()
    }
}

/// Install a process-wide value below the environment, for a deployment
/// config. Setting a knob again to a different value fails: two configs in one
/// process must agree on what they both set globally.
pub fn set_config_global(name: &str, value: &str) -> TractResult<()> {
    let info = checked(name, value)?;
    let mut globals = CONFIG_GLOBALS.write();
    let globals = globals.get_or_insert_with(HashMap::new);
    if let Some(previous) = globals.get(info.name)
        && previous != value
    {
        bail!("Knob {name} already set to {previous:?} by another config, refusing {value:?}");
    }
    globals.insert(info.name, value.to_string());
    Ok(())
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

    /// Resolve the current value: programmatic override, then the thread's
    /// [`KnobScope`], then environment, then config globals, then default.
    pub fn get(&self) -> T {
        if let Some(v) = with_override(self.name, |o| o.and_then(|s| T::parse_knob(s))) {
            return v;
        }
        if let Some(v) = scoped_value(self.name) {
            return v;
        }
        if let Ok(s) = std::env::var(self.name)
            && let Some(v) = T::parse_knob(&s)
        {
            return v;
        }
        if let Some(v) = config_global_value(self.name) {
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
    /// Read while a model is transformed or prepared, so a [`KnobScope`] may
    /// set it for one model.
    pub scopable: bool,
    /// Whether a source string is a valid value.
    pub parses: fn(&str) -> bool,
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

/// Render all declared knobs as a human-readable list.
pub fn list() -> String {
    use std::fmt::Write;
    let shown = |s: String| if s.is_empty() { "(unset)".to_string() } else { s };
    let mut out = String::new();
    for k in all() {
        let current = shown((k.current)());
        let default = shown((k.default)());
        write!(out, "{} = {}  [{}", k.name, current, k.type_name).unwrap();
        if current != default {
            write!(out, ", default {default}").unwrap();
        }
        writeln!(out, "]\n    {}\n", k.doc).unwrap();
    }
    out
}

/// Declare a knob: a typed runtime setting resolved from override / environment
/// / default, registered for listing.
///
/// The static is named exactly like the environment variable (the `TRACT_`
/// prefix included), so one grep finds both the declaration and any use, and
/// the env-var name is derived from the identifier — there is no separate
/// string to drift.
///
/// Prefix the name with `scopable` for a knob read only while a model is
/// transformed or prepared, never at eval nor cached for the process: such a
/// knob honours a [`KnobScope`].
///
/// ```ignore
/// declare_knob!(TRACT_MY_FLAG, bool, false, "Enable the thing.");
/// declare_knob!(scopable TRACT_MY_RULE_OFF, bool, false, "Skip my rewrite rule.");
/// // ... later ...
/// if TRACT_MY_FLAG.get() { /* ... */ }
/// ```
#[macro_export]
macro_rules! declare_knob {
    (scopable $ident:ident, $ty:ty, $default:expr, $doc:literal) => {
        $crate::declare_knob!(@ $ident, $ty, $default, $doc, true);
    };
    ($ident:ident, $ty:ty, $default:expr, $doc:literal) => {
        $crate::declare_knob!(@ $ident, $ty, $default, $doc, false);
    };
    (@ $ident:ident, $ty:ty, $default:expr, $doc:literal, $scopable:literal) => {
        #[doc = $doc]
        pub static $ident: $crate::knobs::Knob<$ty> =
            $crate::knobs::Knob::new(stringify!($ident), || $default, $doc);

        $crate::knobs::inventory::submit! {
            $crate::knobs::KnobInfo {
                name: stringify!($ident),
                doc: $doc,
                type_name: <$ty as $crate::knobs::KnobValue>::TYPE_NAME,
                scopable: $scopable,
                parses: |s| <$ty as $crate::knobs::KnobValue>::parse_knob(s).is_some(),
                default: || $crate::knobs::KnobValue::render_knob(&$ident.default_value()),
                current: || $crate::knobs::KnobValue::render_knob(&$ident.get()),
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    declare_knob!(TRACT_TEST_KNOB_BOOL, bool, false, "Test bool knob.");
    declare_knob!(TRACT_TEST_KNOB_INT, usize, 7, "Test int knob.");
    declare_knob!(TRACT_TEST_KNOB_SET_INT, usize, 7, "Test int knob for set-by-name.");
    declare_knob!(scopable TRACT_TEST_KNOB_SCOPED, usize, 7, "Test scopable knob.");
    declare_knob!(TRACT_TEST_KNOB_GLOBAL, usize, 7, "Test global knob.");

    #[test]
    fn scope_applies_inside_enter_only() {
        let scope = KnobScope::new([("TRACT_TEST_KNOB_SCOPED", "3")]).unwrap();
        assert_eq!(scope.enter(|| TRACT_TEST_KNOB_SCOPED.get()), 3);
        assert_eq!(TRACT_TEST_KNOB_SCOPED.get(), 7);
        let other = std::thread::spawn(move || scope.enter(|| TRACT_TEST_KNOB_SCOPED.get()));
        assert_eq!(other.join().unwrap(), 3);
    }

    #[test]
    fn scope_rejects_bad_entries() {
        assert!(KnobScope::new([("TRACT_TEST_KNOB_GLOBAL", "3")]).is_err());
        assert!(KnobScope::new([("TRACT_TEST_KNOB_SCOPED", "three")]).is_err());
        assert!(KnobScope::new([("TRACT_TEST_KNOB_DOES_NOT_EXIST", "3")]).is_err());
    }

    #[test]
    fn config_global_below_override_and_scope() {
        set_config_global("TRACT_TEST_KNOB_GLOBAL", "5").unwrap();
        assert_eq!(TRACT_TEST_KNOB_GLOBAL.get(), 5);
        set_config_global("TRACT_TEST_KNOB_GLOBAL", "5").unwrap();
        assert!(set_config_global("TRACT_TEST_KNOB_GLOBAL", "6").is_err());
        TRACT_TEST_KNOB_GLOBAL.set(9);
        assert_eq!(TRACT_TEST_KNOB_GLOBAL.get(), 9);
        TRACT_TEST_KNOB_GLOBAL.clear();
    }

    #[test]
    fn default_then_override_then_clear() {
        assert!(!TRACT_TEST_KNOB_BOOL.get());
        assert_eq!(TRACT_TEST_KNOB_INT.get(), 7);

        TRACT_TEST_KNOB_BOOL.set(true);
        assert!(TRACT_TEST_KNOB_BOOL.get());
        TRACT_TEST_KNOB_BOOL.clear();
        assert!(!TRACT_TEST_KNOB_BOOL.get());
    }

    #[test]
    fn set_by_name() {
        assert!(set_str("TRACT_TEST_KNOB_SET_INT", "42"));
        assert_eq!(TRACT_TEST_KNOB_SET_INT.get(), 42);
        assert!(clear_str("TRACT_TEST_KNOB_SET_INT"));
        assert_eq!(TRACT_TEST_KNOB_SET_INT.get(), 7);
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
