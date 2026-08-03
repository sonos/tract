// Bodies for the FusedKerSpec arms that apply one operation uniformly across a
// kernel's accumulators. Each macro expands to exactly the statement sequence it
// replaces — one binding for the shared operand, then one assignment per
// accumulator in the order given — so the emitted wasm is unchanged.
//
// The arms that carry a multiply-add or walk the C tile by stride stay written
// out in each kernel: AddMatMul and AddRowColProducts must agree on whether the
// madd is fused (see `dispatch_tests::add_row_col_products_and_add_mat_mul_agree_on_fusion`),
// and AddUnicast, Store and LoadTile each read the tile differently per kernel.
//
// Callers need `std::arch::wasm32::*` in scope, and pass accumulators by name so
// that grepping for an accumulator still finds every arm that touches it.

/// `acc = shared` for each accumulator.
macro_rules! wasm_set {
    ($v:expr; $($acc:ident),+ $(,)?) => {{
        let s = $v;
        $($acc = s;)+
    }};
}

/// `acc = op(shared, acc)` — the operand order every arm uses except the
/// flipped `*SubF` variants.
macro_rules! wasm_bin_sv {
    ($op:path, $v:expr; $($acc:ident),+ $(,)?) => {{
        let s = $v;
        $($acc = $op(s, $acc);)+
    }};
}

/// `acc = op(acc, shared)` — the `*SubF` order.
macro_rules! wasm_bin_vs {
    ($op:path, $v:expr; $($acc:ident),+ $(,)?) => {{
        let s = $v;
        $($acc = $op($acc, s);)+
    }};
}

/// `acc_i = op(v128_load(p + i), acc_i)`, where consecutive accumulators read
/// consecutive vectors. Used where a per-row or per-column operand is laid out
/// along the same axis the accumulators are.
macro_rules! wasm_bin_load_indexed {
    (@step $op:path, $p:ident, $i:expr;) => {};
    (@step $op:path, $p:ident, $i:expr; $acc:ident $(, $rest:ident)*) => {
        $acc = $op(v128_load($p.add($i)), $acc);
        wasm_bin_load_indexed!(@step $op, $p, $i + 1; $($rest),*);
    };
    ($op:path, $ptr:expr; $($acc:ident),+ $(,)?) => {{
        let p = $ptr as *const v128;
        wasm_bin_load_indexed!(@step $op, p, 0usize; $($acc),+);
    }};
}

/// `acc_i = op(acc_i, v128_load(p + i))` — the flipped counterpart.
macro_rules! wasm_bin_load_indexed_vs {
    (@step $op:path, $p:ident, $i:expr;) => {};
    (@step $op:path, $p:ident, $i:expr; $acc:ident $(, $rest:ident)*) => {
        $acc = $op($acc, v128_load($p.add($i)));
        wasm_bin_load_indexed_vs!(@step $op, $p, $i + 1; $($rest),*);
    };
    ($op:path, $ptr:expr; $($acc:ident),+ $(,)?) => {{
        let p = $ptr as *const v128;
        wasm_bin_load_indexed_vs!(@step $op, p, 0usize; $($acc),+);
    }};
}

/// `acc_i = *(p + i)` — a contiguous reload of the whole accumulator set.
macro_rules! wasm_load_indexed {
    (@step $p:ident, $i:expr;) => {};
    (@step $p:ident, $i:expr; $acc:ident $(, $rest:ident)*) => {
        $acc = *$p.add($i);
        wasm_load_indexed!(@step $p, $i + 1; $($rest),*);
    };
    ($ptr:expr; $($acc:ident),+ $(,)?) => {{
        let p = $ptr as *const v128;
        wasm_load_indexed!(@step p, 0usize; $($acc),+);
    }};
}

/// `acc = if acc > 0 { acc } else { alpha * acc }`, lane-wise.
macro_rules! wasm_leaky_relu {
    ($a:expr; $($acc:ident),+ $(,)?) => {{
        let s = f32x4_splat($a);
        let zero = f32x4_splat(0.0);
        $(
            let m = f32x4_gt($acc, zero);
            $acc = v128_bitselect($acc, f32x4_mul(s, $acc), m);
        )+
    }};
}

/// `acc_i = op(splat(p[i]), acc_i)`, used where a per-row operand runs across
/// the accumulators rather than along the lanes inside one.
macro_rules! wasm_bin_splat_indexed {
    (@step $op:path, $p:ident, $i:expr;) => {};
    (@step $op:path, $p:ident, $i:expr; $acc:ident $(, $rest:ident)*) => {
        $acc = $op(f32x4_splat(*$p.add($i)), $acc);
        wasm_bin_splat_indexed!(@step $op, $p, $i + 1; $($rest),*);
    };
    ($op:path, $ptr:expr; $($acc:ident),+ $(,)?) => {{
        let p = $ptr;
        wasm_bin_splat_indexed!(@step $op, p, 0usize; $($acc),+);
    }};
}

/// `acc_i = op(acc_i, splat(p[i]))` — the flipped counterpart.
macro_rules! wasm_bin_splat_indexed_vs {
    (@step $op:path, $p:ident, $i:expr;) => {};
    (@step $op:path, $p:ident, $i:expr; $acc:ident $(, $rest:ident)*) => {
        $acc = $op($acc, f32x4_splat(*$p.add($i)));
        wasm_bin_splat_indexed_vs!(@step $op, $p, $i + 1; $($rest),*);
    };
    ($op:path, $ptr:expr; $($acc:ident),+ $(,)?) => {{
        let p = $ptr;
        wasm_bin_splat_indexed_vs!(@step $op, p, 0usize; $($acc),+);
    }};
}

/// Per-row operand against accumulators held as one `(low, high)` pair per row:
/// row `i` is splatted once and applied to both halves.
macro_rules! wasm_bin_row_pairs {
    (@step $op:path, $p:ident, $i:expr;) => {};
    (@step $op:path, $p:ident, $i:expr; ($lo:ident, $hi:ident) $(, $rest:tt)*) => {
        let s = f32x4_splat(*$p.add($i));
        $lo = $op(s, $lo);
        $hi = $op(s, $hi);
        wasm_bin_row_pairs!(@step $op, $p, $i + 1; $($rest),*);
    };
    ($op:path, $ptr:expr; $(($lo:ident, $hi:ident)),+ $(,)?) => {{
        let p = $ptr;
        wasm_bin_row_pairs!(@step $op, p, 0usize; $(($lo, $hi)),+);
    }};
}

/// `wasm_bin_row_pairs!` with the operands flipped.
macro_rules! wasm_bin_row_pairs_vs {
    (@step $op:path, $p:ident, $i:expr;) => {};
    (@step $op:path, $p:ident, $i:expr; ($lo:ident, $hi:ident) $(, $rest:tt)*) => {
        let s = f32x4_splat(*$p.add($i));
        $lo = $op($lo, s);
        $hi = $op($hi, s);
        wasm_bin_row_pairs_vs!(@step $op, $p, $i + 1; $($rest),*);
    };
    ($op:path, $ptr:expr; $(($lo:ident, $hi:ident)),+ $(,)?) => {{
        let p = $ptr;
        wasm_bin_row_pairs_vs!(@step $op, p, 0usize; $(($lo, $hi)),+);
    }};
}

/// Per-column operand against `(low, high)` accumulator pairs: the first vector
/// of columns hits every low half, the second every high half.
macro_rules! wasm_bin_col_pairs {
    ($op:path, $ptr:expr; $(($lo:ident, $hi:ident)),+ $(,)?) => {{
        let p = $ptr as *const v128;
        let clo = v128_load(p);
        let chi = v128_load(p.add(1));
        $(
            $lo = $op(clo, $lo);
            $hi = $op(chi, $hi);
        )+
    }};
}

/// `wasm_bin_col_pairs!` with the operands flipped.
macro_rules! wasm_bin_col_pairs_vs {
    ($op:path, $ptr:expr; $(($lo:ident, $hi:ident)),+ $(,)?) => {{
        let p = $ptr as *const v128;
        let clo = v128_load(p);
        let chi = v128_load(p.add(1));
        $(
            $lo = $op($lo, clo);
            $hi = $op($hi, chi);
        )+
    }};
}
