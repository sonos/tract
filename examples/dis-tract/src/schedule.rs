//! Which sequence steps next, and with what — the coordinator's policy, separated
//! from the transport that carries it.
//!
//! A sequence's steps are strictly ordered: token `t+1` is the argmax of step `t`, so
//! one sequence never occupies more than one stage at a time. Throughput comes from
//! interleaving *different* sequences down the same pipeline, which is what the
//! per-sequence KV in each worker exists for. This module owns that interleaving and
//! nothing else: it takes results in and hands steps back, so it can be driven by a
//! test as easily as by zenoh.

use std::collections::VecDeque;

use crate::protocol::{GenerateRequest, Phase, StepMeta};

/// One step to publish to stage 0.
#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    pub seq_id: u64,
    pub meta: StepMeta,
    pub ids: Vec<i64>,
}

/// What a result did for its sequence.
#[derive(Debug)]
pub enum Progress<R> {
    /// A prefill chunk landed. Its logits predict a token the prompt already
    /// contains, so there is nothing to emit and nothing to time as decode.
    Prefilling,
    /// A generated token. The sequence keeps its slot.
    Token(i64),
    /// The sequence hit a stop token or its turn budget; it holds nothing now.
    Done(Box<Sequence<R>>),
}

/// One generation in flight. `R` is whatever the caller needs to answer the client
/// with — a zenoh `Query` in the coordinator, anything at all in a test.
#[derive(Debug)]
pub struct Sequence<R> {
    pub id: u64,
    pub reply: R,
    pub stream_key: String,
    pub prompt_tokens: usize,
    pub out_toks: Vec<i64>,
    /// Time to first token: the whole prefill, however many chunks it took.
    pub ttft_ms: f64,
    dec_sum_ms: f64,
    dec_cnt: u64,
    stop: Vec<i64>,
    max_tokens: usize,
    /// Prompt ids not yet fed. Non-empty means the next step is a prefill chunk.
    pending_prompt: VecDeque<i64>,
    next_ids: Vec<i64>,
    turn: u64,
    in_flight: bool,
    /// Steps handed out so far, so the busiest sequence yields to the least-served
    /// one when there are more sequences than pipeline slots.
    dispatched: u64,
}

impl<R> Sequence<R> {
    /// This sequence's own decode rate. Interleaving does not improve it — a sequence
    /// still waits a full pipeline traversal per token — so it is reported per reply
    /// rather than as the cluster's headline number.
    pub fn decode_tok_s(&self) -> f64 {
        if self.dec_cnt > 0 { 1000.0 * self.dec_cnt as f64 / self.dec_sum_ms } else { 0.0 }
    }
}

/// Interleaves sequences over a pipeline of a fixed depth.
pub struct Scheduler<R> {
    active: Vec<Sequence<R>>,
    next_id: u64,
    /// Steps allowed in flight at once — the number of stages, since a step occupies
    /// exactly one stage at a time.
    window: usize,
    /// Sequences allowed to hold a KV slot. Above `window` they take turns, which
    /// costs KV on every worker but lets more clients make progress at once.
    max_active: usize,
    /// Prompt tokens per prefill step, or 0 to feed the whole prompt in one step.
    /// Chunking stops a long prompt from holding a stage for its entire prefill and
    /// blocking every other sequence behind it.
    prefill_chunk: usize,
}

impl<R> Scheduler<R> {
    pub fn new(window: usize, max_active: usize, prefill_chunk: usize) -> Self {
        Scheduler {
            active: vec![],
            next_id: 1,
            window: window.max(1),
            max_active: max_active.max(1),
            prefill_chunk,
        }
    }

    pub fn active(&self) -> usize {
        self.active.len()
    }

    pub fn has_room(&self) -> bool {
        self.active.len() < self.max_active
    }

    pub fn waiting(&self) -> bool {
        self.active.iter().any(|s| s.in_flight)
    }

    pub fn sequence(&self, seq_id: u64) -> Option<&Sequence<R>> {
        self.active.iter().find(|s| s.id == seq_id)
    }

    /// Take on a request. The returned id names its KV slot on every stage; the caller
    /// must reset that slot before the sequence's first step reaches a worker.
    pub fn admit(&mut self, req: GenerateRequest, stream_key: String, reply: R) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let prompt_tokens = req.prompt.len();
        let mut pending: VecDeque<i64> = req.prompt.into();
        let next_ids = take_chunk(&mut pending, self.prefill_chunk);
        self.active.push(Sequence {
            id,
            reply,
            stream_key,
            prompt_tokens,
            out_toks: Vec::with_capacity(req.max_tokens),
            ttft_ms: 0.0,
            dec_sum_ms: 0.0,
            dec_cnt: 0,
            stop: req.stop,
            max_tokens: req.max_tokens,
            pending_prompt: pending,
            next_ids,
            turn: 0,
            in_flight: false,
            dispatched: 0,
        });
        id
    }

    /// The steps to publish now: every sequence that is not already waiting on a
    /// result, least-served first, up to the free room in the window.
    pub fn dispatch(&mut self) -> Vec<Step> {
        let free = self.window.saturating_sub(self.active.iter().filter(|s| s.in_flight).count());
        let mut ready: Vec<usize> =
            (0..self.active.len()).filter(|&i| !self.active[i].in_flight).collect();
        ready.sort_by_key(|&i| (self.active[i].dispatched, self.active[i].id));
        ready.truncate(free);
        ready.sort_unstable();

        ready
            .into_iter()
            .map(|i| {
                let s = &mut self.active[i];
                let phase = if s.turn == 0 || !s.pending_prompt.is_empty() || s.out_toks.is_empty()
                {
                    Phase::Prefill
                } else {
                    Phase::Decode
                };
                s.in_flight = true;
                s.dispatched += 1;
                Step {
                    seq_id: s.id,
                    meta: StepMeta { turn: s.turn, phase, seq_ids: vec![s.id] },
                    ids: s.next_ids.clone(),
                }
            })
            .collect()
    }

    /// Feed back the token a step produced. `elapsed_ms` is that step's own latency.
    /// Returns `None` for a sequence that is no longer active — a late result from an
    /// abandoned generation, which must not be applied to whoever reused the slot.
    pub fn on_result(&mut self, seq_id: u64, token: i64, elapsed_ms: f64) -> Option<Progress<R>> {
        let i = self.active.iter().position(|s| s.id == seq_id)?;
        let s = &mut self.active[i];
        s.in_flight = false;
        s.turn += 1;

        if !s.pending_prompt.is_empty() {
            // Still feeding the prompt: this chunk's logits predict a token the prompt
            // already supplies. Time it as prefill, emit nothing.
            s.ttft_ms += elapsed_ms;
            s.next_ids = take_chunk(&mut s.pending_prompt, self.prefill_chunk);
            return Some(Progress::Prefilling);
        }

        if s.out_toks.is_empty() {
            s.ttft_ms += elapsed_ms;
        } else {
            s.dec_sum_ms += elapsed_ms;
            s.dec_cnt += 1;
        }
        s.next_ids = vec![token];

        let stop_hit = s.stop.contains(&token);
        if !stop_hit {
            s.out_toks.push(token);
        }
        // The budget counts generated tokens, not steps: prefill chunks are steps too,
        // and a chunked prompt must not eat into what the caller asked to generate.
        if stop_hit || s.out_toks.len() >= s.max_tokens {
            return Some(Progress::Done(Box::new(self.active.remove(i))));
        }
        Some(Progress::Token(token))
    }

    /// Give up on every sequence: the stages are shared, so a wedged or departed one
    /// takes down everything riding the pipeline, not just the outstanding step.
    pub fn drain(&mut self) -> Vec<Sequence<R>> {
        self.active.drain(..).collect()
    }
}

fn take_chunk(pending: &mut VecDeque<i64>, chunk: usize) -> Vec<i64> {
    let n = if chunk == 0 { pending.len() } else { chunk.min(pending.len()) };
    pending.drain(..n).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(prompt: &[i64], max_tokens: usize, stop: &[i64]) -> GenerateRequest {
        GenerateRequest { prompt: prompt.to_vec(), max_tokens, stream_id: 0, stop: stop.to_vec() }
    }

    fn sched(window: usize, max_active: usize, chunk: usize) -> Scheduler<&'static str> {
        Scheduler::new(window, max_active, chunk)
    }

    fn admit(s: &mut Scheduler<&'static str>, prompt: &[i64], max_tokens: usize) -> u64 {
        s.admit(req(prompt, max_tokens, &[]), "stream".into(), "client")
    }

    #[test]
    fn a_sequence_never_has_two_steps_in_flight() {
        let mut s = sched(2, 2, 0);
        let a = admit(&mut s, &[1, 2], 8);
        assert_eq!(s.dispatch().len(), 1);
        // Nothing more for `a` until its result comes back.
        assert!(s.dispatch().is_empty());
        s.on_result(a, 7, 1.0);
        assert_eq!(s.dispatch().len(), 1);
    }

    #[test]
    fn sequences_interleave_up_to_the_window() {
        let mut s = sched(2, 4, 0);
        let a = admit(&mut s, &[1], 8);
        let b = admit(&mut s, &[2], 8);
        let ids: Vec<u64> = s.dispatch().iter().map(|st| st.seq_id).collect();
        assert_eq!(ids, vec![a, b], "both sequences should be in flight at once");

        // One comes back; only that one re-enters the pipeline.
        s.on_result(a, 7, 1.0);
        let ids: Vec<u64> = s.dispatch().iter().map(|st| st.seq_id).collect();
        assert_eq!(ids, vec![a]);
    }

    #[test]
    fn the_window_bounds_steps_in_flight() {
        let mut s = sched(1, 3, 0);
        admit(&mut s, &[1], 8);
        admit(&mut s, &[2], 8);
        admit(&mut s, &[3], 8);
        assert_eq!(s.dispatch().len(), 1, "a one-stage pipeline holds one step");
    }

    /// With more sequences than pipeline slots, the least-served sequence goes next.
    /// Round-robin by construction: no sequence can be passed over twice running.
    #[test]
    fn a_narrow_window_still_serves_every_sequence() {
        let mut s = sched(1, 3, 0);
        let a = admit(&mut s, &[1], 100);
        let b = admit(&mut s, &[2], 100);
        let c = admit(&mut s, &[3], 100);

        let mut order = vec![];
        for _ in 0..6 {
            let steps = s.dispatch();
            assert_eq!(steps.len(), 1);
            order.push(steps[0].seq_id);
            s.on_result(steps[0].seq_id, 9, 1.0);
        }
        assert_eq!(order, vec![a, b, c, a, b, c], "starved sequence");
    }

    #[test]
    fn the_whole_prompt_goes_in_one_step_when_chunking_is_off() {
        let mut s = sched(2, 2, 0);
        admit(&mut s, &[1, 2, 3, 4, 5], 8);
        let steps = s.dispatch();
        assert_eq!(steps[0].ids, vec![1, 2, 3, 4, 5]);
        assert_eq!(steps[0].meta.phase, Phase::Prefill);
    }

    #[test]
    fn chunked_prefill_feeds_the_prompt_in_pieces() {
        let mut s = sched(2, 2, 2);
        let a = admit(&mut s, &[1, 2, 3, 4, 5], 8);

        let mut chunks = vec![];
        loop {
            let steps = s.dispatch();
            chunks.push(steps[0].ids.clone());
            assert_eq!(steps[0].meta.phase, Phase::Prefill);
            // Every prefill chunk but the last predicts a token the prompt already has.
            match s.on_result(a, 99, 10.0) {
                Some(Progress::Prefilling) => continue,
                Some(Progress::Token(t)) => {
                    assert_eq!(t, 99);
                    break;
                }
                other => panic!("unexpected progress: {other:?}"),
            }
        }
        assert_eq!(chunks, vec![vec![1, 2], vec![3, 4], vec![5]]);
        let seq = s.sequence(a).unwrap();
        assert_eq!(seq.out_toks, vec![99], "only the last chunk yields a token");
        assert_eq!(seq.ttft_ms, 30.0, "ttft covers every prefill chunk");
        assert_eq!(seq.decode_tok_s(), 0.0, "prefill is not decode");
    }

    /// What chunking actually buys: no single step carries an unbounded slice of
    /// prompt, so the time a long prompt can hold a stage — and block everyone behind
    /// it — is bounded by the chunk rather than by the prompt. The interleaving order
    /// alone would prove nothing; it is the same either way, since every active
    /// sequence already gets one step per round.
    #[test]
    fn chunked_prefill_bounds_the_prompt_one_step_can_hold_a_stage_with() {
        let long = &[1, 2, 3, 4, 5, 6, 7, 8];
        let widest = |chunk: usize| {
            let mut s = sched(1, 2, chunk);
            admit(&mut s, long, 8);
            admit(&mut s, &[9], 8);
            let mut widest = 0;
            for _ in 0..4 {
                let steps = s.dispatch();
                assert_eq!(steps.len(), 1);
                widest = widest.max(steps[0].ids.len());
                s.on_result(steps[0].seq_id, 42, 1.0);
            }
            widest
        };
        assert_eq!(widest(0), long.len(), "unchunked, one step swallows the whole prompt");
        assert_eq!(widest(2), 2, "chunked, no step exceeds the chunk");
    }

    #[test]
    fn a_stop_token_ends_the_sequence_without_emitting_it() {
        let mut s = sched(2, 2, 0);
        let a = s.admit(req(&[1], 100, &[7]), "stream".into(), "client");
        s.dispatch();
        s.on_result(a, 5, 1.0);
        s.dispatch();
        let Some(Progress::Done(seq)) = s.on_result(a, 7, 1.0) else {
            panic!("stop token should finish the sequence")
        };
        assert_eq!(seq.out_toks, vec![5], "the stop token is not part of the reply");
        assert_eq!(s.active(), 0);
    }

    /// Prefill chunks are steps too. If the budget counted steps, a chunked prompt
    /// would silently eat the tokens the caller asked to generate.
    #[test]
    fn chunked_prefill_does_not_spend_the_token_budget() {
        let mut s = sched(2, 2, 2);
        let a = admit(&mut s, &[1, 2, 3, 4, 5, 6, 7, 8], 3);
        let mut generated = 0;
        for _ in 0..20 {
            if s.dispatch().is_empty() {
                break;
            }
            match s.on_result(a, 42, 1.0) {
                Some(Progress::Prefilling) => {}
                Some(Progress::Token(_)) => generated += 1,
                Some(Progress::Done(seq)) => {
                    generated += 1;
                    assert_eq!(seq.out_toks.len(), 3);
                    break;
                }
                None => panic!("sequence vanished"),
            }
        }
        assert_eq!(generated, 3, "a 4-chunk prompt stole tokens from the budget");
    }

    #[test]
    fn the_turn_budget_ends_the_sequence() {
        let mut s = sched(2, 2, 0);
        let a = admit(&mut s, &[1], 3);
        for _ in 0..2 {
            s.dispatch();
            assert!(matches!(s.on_result(a, 5, 1.0), Some(Progress::Token(_))));
        }
        s.dispatch();
        let Some(Progress::Done(seq)) = s.on_result(a, 5, 1.0) else {
            panic!("max_tokens should finish the sequence")
        };
        assert_eq!(seq.out_toks.len(), 3);
    }

    #[test]
    fn a_result_for_an_abandoned_sequence_is_refused() {
        let mut s = sched(2, 2, 0);
        let a = admit(&mut s, &[1], 8);
        s.dispatch();
        assert_eq!(s.drain().len(), 1);
        // The slot may already belong to someone else; applying this would corrupt it.
        assert!(s.on_result(a, 5, 1.0).is_none());
    }

    #[test]
    fn admission_is_bounded() {
        let mut s = sched(2, 2, 0);
        admit(&mut s, &[1], 8);
        assert!(s.has_room());
        admit(&mut s, &[2], 8);
        assert!(!s.has_room());
    }
}
