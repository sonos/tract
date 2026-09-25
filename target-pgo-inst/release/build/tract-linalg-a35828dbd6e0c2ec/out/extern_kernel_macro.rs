
    macro_rules! extern_kernel {
        (fn $name: ident($($par_name:ident : $par_type: ty ),*) -> $rv: ty) => {
            paste! {
                unsafe extern "C" { pub fn [<$name _ 0_23_4_pre>]($(par_name: $par_type),*) -> $rv; }
                pub use [<$name _ 0_23_4_pre>] as $name;
            }
        }
    }