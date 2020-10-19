#[macro_export]
macro_rules! as_op {
    () => {
        fn as_op(&self) -> &dyn Op {
            self
        }

        fn as_op_mut(&mut self) -> &mut dyn Op {
            self
        }
    };
}

#[macro_export]
macro_rules! op_as_typed_op {
    () => {
        fn as_typed(&self) -> Option<&dyn TypedOp> {
            Some(self)
        }
    };
}

#[macro_export]
macro_rules! not_a_typed_op {
    () => {
        fn as_typed(&self) -> Option<&dyn TypedOp> {
            None
        }
    };
}

#[macro_export]
macro_rules! op_core {
    () => {
        fn op_families(&self) -> &'static [&'static str] {
            &["core"]
        }
    };
}

#[macro_export]
macro_rules! op_core_mir {
    () => {
        fn op_families(&self) -> &'static [&'static str] {
            &["core", "mir"]
        }
    };
}

#[macro_export]
macro_rules! op_core_lir {
    () => {
        fn op_families(&self) -> &'static [&'static str] {
            &["core", "lir"]
        }
    };
}

#[macro_export]
macro_rules! op_core_lir_mir {
    () => {
        fn op_families(&self) -> &'static [&'static str] {
            &["core", "lir", "mir"]
        }
    };
}

#[macro_export]
macro_rules! args_1 {
    ($inputs:expr) => {{
        if $inputs.len() != 1 {
            $crate::internal::bail!("Expected 1 arg, got {:?}", $inputs)
        }
        let result = $inputs.pop().unwrap();
        ::std::mem::drop($inputs);
        result
    }};
}

#[macro_export]
macro_rules! args_2 {
    ($inputs:expr) => {{
        if $inputs.len() != 2 {
            $crate::internal::bail!("Expected 2 arg, got {:?}", $inputs)
        }
        $inputs.reverse();
        let result = ($inputs.pop().unwrap(), $inputs.pop().unwrap());
        ::std::mem::drop($inputs);
        result
    }};
}

#[allow(unused_macros)]
#[macro_export]
macro_rules! args_3 {
    ($inputs:expr) => {{
        if $inputs.len() != 3 {
            $crate::internal::bail!("Expected 3 arg, got {:?}", $inputs)
        }
        $inputs.reverse();
        let result = ($inputs.pop().unwrap(), $inputs.pop().unwrap(), $inputs.pop().unwrap());
        ::std::mem::drop($inputs);
        result
    }};
}

#[allow(unused_macros)]
#[macro_export]
macro_rules! args_4 {
    ($inputs:expr) => {{
        if $inputs.len() != 4 {
            $crate::internal::bail!("Expected 4 arg, got {:?}", $inputs)
        }
        $inputs.reverse();
        let result = (
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
        );
        ::std::mem::drop($inputs);
        result
    }};
}

#[allow(unused_macros)]
#[macro_export]
macro_rules! args_5 {
    ($inputs:expr) => {{
        if $inputs.len() != 5 {
            $crate::internal::bail!("Expected 5 arg, got {:?}", $inputs)
        }
        $inputs.reverse();
        let result = (
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
        );
        ::std::mem::drop($inputs);
        result
    }};
}

#[allow(unused_macros)]
#[macro_export]
macro_rules! args_6 {
    ($inputs:expr) => {{
        if $inputs.len() != 6 {
            $crate::internal::bail!("Expected 6 arg, got {:?}", $inputs)
        }
        $inputs.reverse();
        let result = (
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
        );
        ::std::mem::drop($inputs);
        result
    }};
}

#[allow(unused_macros)]
#[macro_export]
macro_rules! args_7 {
    ($inputs:expr) => {{
        if $inputs.len() != 7 {
            $crate::internal::bail!("Expected 7 arg, got {:?}", $inputs)
        }
        $inputs.reverse();
        let result = (
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
        );
        ::std::mem::drop($inputs);
        result
    }};
}

#[allow(unused_macros)]
#[macro_export]
macro_rules! args_8 {
    ($inputs:expr) => {{
        if $inputs.len() != 8 {
            $crate::internal::bail!("Expected 8 arg, got {:?}", $inputs)
        }
        $inputs.reverse();
        let result = (
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
        );
        ::std::mem::drop($inputs);
        result
    }};
}

#[macro_export]
macro_rules! impl_op_same_as {
    () => {
        fn same_as(&self, other: &dyn Op) -> bool {
            if let Some(other) = other.downcast_ref::<Self>() {
                self == other
            } else {
                false
            }
        }
    };
}

#[macro_export]
macro_rules! assert_close {
    ($left:expr, $right:expr) => ({
        match (&$left, &$right) {
            (left_val, right_val) => {
                if let Err(e) = left_val.close_enough(right_val, true) {
                    panic!(r#"assertion failed: `(left ~ right)`
  left: `{:?}`,
 right: `{:?}`
 {:?}"#, left_val, right_val, e)
                }
            }
        }
    });
    ($left:expr, $right:expr,) => ({
        assert_eq!($left, $right)
    });
    ($left:expr, $right:expr, $($arg:tt)+) => ({
        match (&($left), &($right)) {
            (left_val, right_val) => {
                if let Err(e) = left_val.close_enough(right_val, true) {
                    panic!(r#"assertion failed: `(left ~ right)`
  left: `{:?}`,
 right: `{:?}`: {}
 {:?}"#, left_val, right_val,
                           format_args!($($arg)+), e)
                }
            }
        }
    });
}
