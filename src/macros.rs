#[macro_export]
macro_rules! tvec {
    // count helper: transform any expression into 1
    (@one $x:expr) => (1usize);
    ($elem:expr; $n:expr) => ({
        $crate::ops::TVec::from_elem($elem, $n)
    });
    ($($x:expr),*$(,)*) => ({
        let count = 0usize $(+ tvec!(@one $x))*;
        #[allow(unused_mut)]
        let mut vec = $crate::ops::TVec::new();
        if count <= vec.inline_size() {
            $(vec.push($x);)*
            vec
        } else {
            $crate::ops::TVec::from_vec(vec![$($x,)*])
        }
    });
}
