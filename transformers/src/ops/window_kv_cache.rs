//! Bounded (sliding-window) KV cache — a fixed-capacity ring buffer for models
//! trained with sliding-window attention (Mistral, Gemma-style local/global, …).
//!
//! Instead of growing the cache with the sequence (O(T) memory + O(T) per-step
//! attention), keep a fixed `window` slots and overwrite the oldest on append. So a
//! decode of arbitrary length runs at **constant** memory and per-step cost — and it's
//! **lossless**, because the model was trained to attend only within the window.
//!
//! Key trick that makes the ring buffer cheap: **decode attention is order-invariant
//! over keys** — `O = Σ_j softmax_j · V_j` is unchanged if you permute (K,V) together
//! (the set of scores, hence the softmax weights, is identical). So we never have to
//! un-rotate the buffer: the consumer attends over the W physical slots in whatever
//! order they sit, and the result equals attending over the ordered last-W (up to
//! floating-point summation order). Validated in the tests.
//!
//! Companion to the in-place cache (#2321): this is "the in-place cache with a cap and
//! wraparound." For prefill (multi-query) the window also needs a banded causal mask;
//! that lives on the attention op, not here.

use tract_nnef::internal::*;

/// Fixed-capacity sliding-window KV cache (ring buffer) along `axis`.
#[derive(Clone, Debug)]
pub struct WindowKvCache {
    pub axis: usize,
    pub window: usize,
    buf: Option<Tensor>, // capacity `window` along `axis`, allocated on first push
    len: usize,          // valid slots, ≤ window
    cursor: usize,       // next write position, in 0..window
}

impl WindowKvCache {
    pub fn new(axis: usize, window: usize) -> Self {
        assert!(window > 0, "window must be > 0");
        WindowKvCache { axis, window, buf: None, len: 0, cursor: 0 }
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Always the window capacity once allocated — memory is bounded regardless of T.
    pub fn capacity(&self) -> usize {
        self.buf.as_ref().map(|b| b.shape()[self.axis]).unwrap_or(0)
    }

    fn ensure_buf(&mut self, like: &Tensor) -> TractResult<()> {
        if self.buf.is_none() {
            let mut shape: TVec<usize> = like.shape().into();
            shape[self.axis] = self.window;
            self.buf = Some(unsafe { Tensor::uninitialized_dt(like.datum_type(), &shape)? });
        }
        Ok(())
    }

    /// Append `input` along `axis`, overwriting the oldest slots once full. O(min(new, window)).
    pub fn push(&mut self, input: &Tensor) -> TractResult<()> {
        let new = input.shape()[self.axis];
        if new == 0 {
            return Ok(());
        }
        self.ensure_buf(input)?;
        let w = self.window;

        if new >= w {
            // Only the last `w` of the input survive; lay them out [0..w], cursor resets.
            let buf = self.buf.as_mut().unwrap();
            buf.assign_slice(0..w, input, (new - w)..new, self.axis)?;
            self.cursor = 0;
            self.len = w;
            return Ok(());
        }

        // new < w: write at the cursor, wrapping around the end.
        let end = self.cursor + new;
        let buf = self.buf.as_mut().unwrap();
        if end <= w {
            buf.assign_slice(self.cursor..end, input, 0..new, self.axis)?;
            self.cursor = if end == w { 0 } else { end };
        } else {
            let first = w - self.cursor;
            buf.assign_slice(self.cursor..w, input, 0..first, self.axis)?;
            buf.assign_slice(0..(new - first), input, first..new, self.axis)?;
            self.cursor = new - first;
        }
        self.len = (self.len + new).min(w);
        Ok(())
    }

    /// Zero-copy view of the `len` valid slots. Once full this is the whole buffer in
    /// *physical* (rotated) order — correct for decode attention by order-invariance.
    pub fn valid_view<T: Datum>(&self) -> TractResult<tract_ndarray::ArrayViewD<'_, T>> {
        let buf = self.buf.as_ref().context("empty window cache")?;
        let mut v = buf.to_plain_array_view::<T>()?;
        v.slice_axis_inplace(tract_ndarray::Axis(self.axis), (0..self.len).into());
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tract_nnef::tract_ndarray::{Array4, ArrayView4, s};

    fn tok(shape: &[usize], v: f32) -> Tensor {
        let n: usize = shape.iter().product();
        Tensor::from_shape(shape, &vec![v; n]).unwrap()
    }

    // ---- ring-buffer mechanics: holds exactly the last `window` items (as a set) ----
    #[test]
    fn window_holds_last_w_as_a_set() -> TractResult<()> {
        let w = 4;
        let mut c = WindowKvCache::new(2, w); // [B,H,S,D], seq axis
        let mut full: Vec<f32> = vec![];
        for t in 0..10 {
            c.push(&tok(&[1, 1, 1, 1], t as f32))?;
            full.push(t as f32);
            assert!(c.len() <= w, "len bounded by window");
            // valid_view set == last min(t+1,w) tokens
            let view = c.valid_view::<f32>()?;
            let mut got: Vec<f32> = view.iter().copied().collect();
            got.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut want: Vec<f32> = full.iter().rev().take(w).copied().collect();
            want.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert_eq!(got, want, "step {t}: window must hold the last {w} as a set");
        }
        assert_eq!(c.capacity(), w, "memory stays bounded at the window");
        Ok(())
    }

    #[test]
    fn prefill_chunk_larger_than_window_keeps_last_w() -> TractResult<()> {
        let w = 3;
        let mut c = WindowKvCache::new(2, w);
        // one chunk of 7 tokens with distinct values, axis=2
        let chunk = Tensor::from_shape(&[1, 1, 7, 1], &[0f32, 1., 2., 3., 4., 5., 6.])?;
        c.push(&chunk)?;
        let mut got: Vec<f32> = c.valid_view::<f32>()?.iter().copied().collect();
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(got, vec![4.0, 5.0, 6.0], "keeps the last 3");
        Ok(())
    }

    // ---- the correctness property: decode attention over the (rotated) window ==
    //      attention over the ordered last-W slice, by order-invariance ----
    fn attention(
        q: ArrayView4<f32>,
        k: ArrayView4<f32>,
        v: ArrayView4<f32>,
        scale: f32,
    ) -> Array4<f32> {
        let (b, h, sq, d) = q.dim();
        let mut out = Array4::<f32>::zeros((b, h, sq, d));
        for bi in 0..b {
            for hi in 0..h {
                let qm = q.slice(s![bi, hi, .., ..]);
                let km = k.slice(s![bi, hi, .., ..]);
                let vm = v.slice(s![bi, hi, .., ..]);
                let mut sc = qm.dot(&km.t());
                sc *= scale;
                for mut row in sc.rows_mut() {
                    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut s = 0.0;
                    row.iter_mut().for_each(|x| {
                        *x = (*x - m).exp();
                        s += *x;
                    });
                    row.iter_mut().for_each(|x| *x /= s);
                }
                out.slice_mut(s![bi, hi, .., ..]).assign(&sc.dot(&vm));
            }
        }
        out
    }

    #[test]
    fn windowed_attention_matches_last_w_full() -> TractResult<()> {
        let (h, d, w) = (2usize, 8usize, 6usize);
        let scale = 1.0 / (d as f32).sqrt();
        // deterministic varied K/V per token so order actually differs after wrap
        let seq = |s: usize, base: f32| -> Tensor {
            let data: Vec<f32> = (0..h * s * d).map(|i| base + (i as f32 * 0.013).sin()).collect();
            Tensor::from_shape(&[1, h, s, d], &data).unwrap()
        };
        let mut kc = WindowKvCache::new(2, w);
        let mut vc = WindowKvCache::new(2, w);
        let mut kfull: Option<Tensor> = None;
        let mut vfull: Option<Tensor> = None;
        use tract_nnef::tract_core::ops::array::TypedConcat;
        for t in 0..20 {
            let knew = seq(1, 1.0 + t as f32 * 0.1);
            let vnew = seq(1, 5.0 - t as f32 * 0.07);
            kc.push(&knew)?;
            vc.push(&vnew)?;
            let grow = |acc: Option<Tensor>, x: Tensor| -> TractResult<Tensor> {
                Ok(match acc {
                    None => x,
                    Some(a) => TypedConcat { axis: 2 }
                        .eval(tvec![a.into(), x.into()])?
                        .remove(0)
                        .into_tensor(),
                })
            };
            kfull = Some(grow(kfull.take(), knew)?);
            vfull = Some(grow(vfull.take(), vnew)?);

            let q = seq(1, 9.0 + t as f32 * 0.05);
            let qv = q.to_plain_array_view::<f32>()?.into_dimensionality()?;

            // windowed (rotated physical order)
            let o_win = attention(
                qv,
                kc.valid_view::<f32>()?.into_dimensionality()?,
                vc.valid_view::<f32>()?.into_dimensionality()?,
                scale,
            );
            // reference: ordered last-W of the full cache
            let len = kc.len();
            let kf = kfull.as_ref().unwrap();
            let s = kf.shape()[2];
            let kslice = kf.slice(2, s - len, s)?;
            let vslice = vfull.as_ref().unwrap().slice(2, s - len, s)?;
            let o_ref = attention(
                qv,
                kslice.to_plain_array_view::<f32>()?.into_dimensionality()?,
                vslice.to_plain_array_view::<f32>()?.into_dimensionality()?,
                scale,
            );
            let a = Tensor::from(o_win);
            let b = Tensor::from(o_ref);
            a.close_enough(&b, Approximation::Approximate)
                .with_context(|| format!("windowed != last-W at step {t}"))?;
        }
        Ok(())
    }
}
