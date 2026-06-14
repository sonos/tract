//! Speculative decoding for the causal LLM example.
//!
//! A cheap *drafter* proposes up to `k` continuation tokens; the target model
//! verifies all of them in a single forward pass (which costs about the same as
//! one decode step, since decode is memory-bandwidth bound). The longest correct
//! prefix is accepted plus one *correction* token the target picks itself, so a
//! step advances by `accepted + 1` tokens while reading the weights once.
//!
//! Greedy verification (`greedy_verify`) is exactly lossless: the emitted
//! sequence is identical to plain greedy decoding. Stochastic verification
//! (`sample_verify`) preserves the target's sampling distribution via the
//! modified-rejection scheme of Leviathan et al. 2023 / Chen et al. 2023.
//!
//! The verification math here is pure and model-free so it can be unit-tested in
//! isolation; the model wiring lives in [`crate::CausalLlmState`].

use anyhow::Result;

/// Proposes candidate continuation tokens for the target model to verify.
///
/// A drafter sees the full confirmed token sequence and returns up to `k`
/// guesses for what comes next. Returning fewer (or zero) tokens is allowed and
/// makes the step fall back toward ordinary decoding.
pub trait Drafter {
    /// Propose up to `k` tokens continuing `context` (the full confirmed
    /// sequence so far). An empty result is valid.
    fn draft(&mut self, context: &[u32], k: usize) -> Result<Vec<u32>>;

    /// Notify the drafter that `confirmed` is the new full sequence after the
    /// target accepted/corrected the last proposal, letting a stateful drafter
    /// resynchronise. Default: no-op.
    fn commit(&mut self, _confirmed: &[u32]) -> Result<()> {
        Ok(())
    }

    /// Reset any per-sequence state. Default: no-op.
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Prompt-lookup / n-gram drafter: find the most recent earlier occurrence of
/// the current suffix and propose the tokens that followed it.
///
/// Needs no second model. It pays off when the output echoes the context
/// (summarisation, RAG, code editing, repetition) and proposes nothing when the
/// suffix has never been seen, in which case the step degrades to plain decode.
#[derive(Clone, Debug)]
pub struct NgramDrafter {
    /// Longest suffix length to try to match; shorter suffixes are tried on
    /// failure down to `min_ngram`.
    pub max_ngram: usize,
    /// Shortest suffix length to accept as a match.
    pub min_ngram: usize,
}

impl Default for NgramDrafter {
    fn default() -> Self {
        Self { max_ngram: 3, min_ngram: 1 }
    }
}

impl NgramDrafter {
    pub fn new(max_ngram: usize, min_ngram: usize) -> Self {
        Self { max_ngram: max_ngram.max(1), min_ngram: min_ngram.max(1) }
    }
}

impl Drafter for NgramDrafter {
    fn draft(&mut self, context: &[u32], k: usize) -> Result<Vec<u32>> {
        if k == 0 || context.is_empty() {
            return Ok(vec![]);
        }
        let n = context.len();
        for ng in (self.min_ngram..=self.max_ngram.min(n)).rev() {
            let suffix = &context[n - ng..];
            // Search earlier positions, most recent first, for a matching window
            // whose continuation we can copy.
            if n < ng + 1 {
                continue;
            }
            for start in (0..n - ng).rev() {
                if &context[start..start + ng] == suffix {
                    let cont_start = start + ng;
                    let take = k.min(n - cont_start);
                    if take > 0 {
                        return Ok(context[cont_start..cont_start + take].to_vec());
                    }
                }
            }
        }
        Ok(vec![])
    }
}

/// Outcome of greedy verification of a draft against the target's own picks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GreedyVerdict {
    /// Number of leading draft tokens that matched the target's greedy choice.
    pub accepted: usize,
    /// The target's greedy token at position `accepted` — the correction (when
    /// some draft was rejected) or the bonus (when every draft was accepted).
    pub correction: u32,
}

/// Greedy verification.
///
/// `target_preds[j]` is the target's greedy argmax of the next token given the
/// confirmed sequence followed by `drafts[0..j]`. It must hold exactly
/// `drafts.len() + 1` entries (the last one is the bonus prediction). The
/// returned `accepted + 1` tokens (`drafts[0..accepted]` then `correction`) are
/// guaranteed identical to what plain greedy decoding would emit.
pub fn greedy_verify(drafts: &[u32], target_preds: &[u32]) -> GreedyVerdict {
    debug_assert_eq!(target_preds.len(), drafts.len() + 1);
    let mut accepted = 0;
    while accepted < drafts.len() && target_preds[accepted] == drafts[accepted] {
        accepted += 1;
    }
    GreedyVerdict { accepted, correction: target_preds[accepted] }
}

/// Argmax over a logits row. Ties resolve to the highest index, matching
/// `Iterator::max_by_key` over `FloatOrd` used by the baseline decode path, so
/// greedy speculative output stays bit-identical to plain greedy.
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v >= best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

/// Outcome of stochastic (rejection-sampling) verification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampleVerdict {
    pub accepted: usize,
    pub correction: u32,
}

/// Modified-rejection verification preserving the target distribution.
///
/// For each draft token `d_i`, accept it with probability
/// `min(1, p_target(d_i) / p_draft(d_i))`; on the first rejection, sample the
/// correction from the normalised residual `max(0, p_target - p_draft)`. If all
/// drafts are accepted, sample the bonus from `p_target` at the final position.
///
/// `draft_probs[i]` and `target_probs[i]` are full next-token distributions
/// (length = vocab) aligned with `drafts[i]`; `target_probs` has one extra row
/// for the bonus. `uniforms` supplies the acceptance test draws and must have at
/// least `drafts.len()` entries; `residual_pick` draws the residual/bonus token
/// given a distribution. Splitting out the randomness keeps this deterministic
/// and testable.
pub fn sample_verify(
    drafts: &[u32],
    draft_probs: &[Vec<f32>],
    target_probs: &[Vec<f32>],
    uniforms: &[f32],
    mut residual_pick: impl FnMut(&[f32]) -> u32,
) -> SampleVerdict {
    debug_assert_eq!(target_probs.len(), drafts.len() + 1);
    debug_assert_eq!(draft_probs.len(), drafts.len());
    for i in 0..drafts.len() {
        let d = drafts[i] as usize;
        let q = draft_probs[i][d].max(0.0);
        let p = target_probs[i][d].max(0.0);
        let accept_ratio = if q > 0.0 { (p / q).min(1.0) } else { 1.0 };
        if uniforms[i] <= accept_ratio {
            continue;
        }
        let residual = residual_distribution(&target_probs[i], &draft_probs[i]);
        return SampleVerdict { accepted: i, correction: residual_pick(&residual) };
    }
    SampleVerdict { accepted: drafts.len(), correction: residual_pick(&target_probs[drafts.len()]) }
}

/// Normalised `max(0, p - q)`; falls back to `p` if the residual is all zeros.
fn residual_distribution(p: &[f32], q: &[f32]) -> Vec<f32> {
    let mut r: Vec<f32> = p.iter().zip(q).map(|(p, q)| (p - q).max(0.0)).collect();
    let sum: f32 = r.iter().sum();
    if sum > 0.0 {
        for v in &mut r {
            *v /= sum;
        }
    } else {
        r.copy_from_slice(p);
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ngram_proposes_recent_continuation() {
        let mut d = NgramDrafter::new(3, 1);
        // suffix "10 20" last occurred at index 0; copy what followed it
        let ctx = vec![10, 20, 30, 10, 20];
        assert_eq!(d.draft(&ctx, 1).unwrap(), vec![30]);
        assert_eq!(d.draft(&ctx, 3).unwrap(), vec![30, 10, 20]);
    }

    #[test]
    fn ngram_empty_when_no_match() {
        let mut d = NgramDrafter::new(3, 2);
        let ctx = vec![1, 2, 3, 4, 5];
        assert_eq!(d.draft(&ctx, 3).unwrap(), Vec::<u32>::new());
    }

    #[test]
    fn ngram_respects_k() {
        let mut d = NgramDrafter::new(2, 1);
        // suffix "9" first seen at 0, continuation "1 2 3 ..." truncated to k
        let ctx = vec![9, 1, 2, 3, 4, 9];
        assert_eq!(d.draft(&ctx, 2).unwrap(), vec![1, 2]);
    }

    #[test]
    fn greedy_accept_all_gives_bonus() {
        // target predicts exactly the drafts, plus a bonus
        let drafts = [5, 6, 7];
        let preds = [5, 6, 7, 8];
        let v = greedy_verify(&drafts, &preds);
        assert_eq!(v, GreedyVerdict { accepted: 3, correction: 8 });
    }

    #[test]
    fn greedy_reject_at_first_mismatch() {
        let drafts = [5, 6, 7];
        let preds = [5, 99, 7, 8];
        let v = greedy_verify(&drafts, &preds);
        assert_eq!(v, GreedyVerdict { accepted: 1, correction: 99 });
    }

    #[test]
    fn greedy_reject_immediately() {
        let drafts = [5, 6];
        let preds = [42, 6, 7];
        let v = greedy_verify(&drafts, &preds);
        assert_eq!(v, GreedyVerdict { accepted: 0, correction: 42 });
    }

    #[test]
    fn sample_accepts_when_target_dominates() {
        // q small, p large at the draft token -> ratio >= 1 -> always accept
        let drafts = [1u32];
        let draft_probs = vec![vec![0.5, 0.1, 0.4]];
        let target_probs = vec![vec![0.0, 1.0, 0.0], vec![0.2, 0.3, 0.5]];
        let v = sample_verify(&drafts, &draft_probs, &target_probs, &[0.99], |p| argmax(p));
        assert_eq!(v.accepted, 1);
        assert_eq!(v.correction, 2); // bonus = argmax of last target row
    }

    #[test]
    fn sample_rejects_and_corrects_from_residual() {
        // draft token 0 has q=1 but p=0 -> ratio 0 -> reject; residual = (p-q)+ ~ token 2
        let drafts = [0u32];
        let draft_probs = vec![vec![1.0, 0.0, 0.0]];
        let target_probs = vec![vec![0.0, 0.0, 1.0], vec![0.0, 0.0, 1.0]];
        let v = sample_verify(&drafts, &draft_probs, &target_probs, &[0.5], |p| argmax(p));
        assert_eq!(v, SampleVerdict { accepted: 0, correction: 2 });
    }
}
