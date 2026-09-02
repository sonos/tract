use std::fmt::Debug;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread;
use std::time::Duration;

use crate::internal::*;

/// The lanes of one laned state: which are taken, and which of them a turn
/// seats.
///
/// Plain data. Taking a lane does not touch the state's buffers, and clearing
/// what a stream left in a lane it gave up is the table's caller's, since it
/// writes the state -- device memory for a state on a GPU -- and must run where
/// the state lives. So a lane handed to a new stream carries the previous one's
/// history until that caller resets it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaneTable {
    taken: Vec<bool>,
}

impl LaneTable {
    pub fn new(max_lanes: usize) -> TractResult<LaneTable> {
        ensure!(max_lanes > 0, "A laned state needs at least one lane");
        Ok(LaneTable { taken: vec![false; max_lanes] })
    }

    /// The extent of the lane axis of the state's per-lane buffers, fixed for
    /// the life of the state.
    pub fn max_lanes(&self) -> usize {
        self.taken.len()
    }

    pub fn taken(&self) -> usize {
        self.taken.iter().filter(|t| **t).count()
    }

    /// The lowest free lane, `None` when every lane is taken -- whether that
    /// blocks the new stream or fails it is the caller's policy. Lowest first,
    /// so that a turn seating every lane seats a run of consecutive lanes.
    pub fn take(&mut self) -> Option<LaneId> {
        let lane = self.taken.iter().position(|t| !t)?;
        self.taken[lane] = true;
        Some(LaneId(lane))
    }

    /// Hand `lane` back, for [`LaneTable::take`] to give to another stream.
    pub fn give_back(&mut self, lane: LaneId) -> TractResult<()> {
        ensure!(self.is_taken(lane), "Lane {} is not taken, so it can not be given back", lane.0);
        self.taken[lane.0] = false;
        Ok(())
    }

    pub fn is_taken(&self, lane: LaneId) -> bool {
        self.taken.get(lane.0).copied().unwrap_or(false)
    }

    /// Seat `lanes`, in that order: seat `ix` of the coming turn carries the
    /// `ix`th of them. Every one must be taken, so that a stream which ended
    /// can not be seated by a stale handle of it.
    pub fn seat(&self, lanes: impl IntoIterator<Item = LaneId>) -> TractResult<Seating> {
        let lanes: Vec<LaneId> = lanes.into_iter().collect();
        for lane in &lanes {
            ensure!(self.is_taken(*lane), "Seating lane {}, which no stream took", lane.0);
        }
        Seating::new(self.max_lanes(), lanes)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn takes_the_lowest_free_lane() -> TractResult<()> {
        let mut table = LaneTable::new(3)?;
        assert_eq!(table.take(), Some(LaneId(0)));
        assert_eq!(table.take(), Some(LaneId(1)));
        table.give_back(LaneId(0))?;
        assert_eq!(table.take(), Some(LaneId(0)));
        assert_eq!(table.taken(), 2);
        Ok(())
    }

    #[test]
    fn runs_out_of_lanes() -> TractResult<()> {
        let mut table = LaneTable::new(1)?;
        assert_eq!(table.take(), Some(LaneId(0)));
        assert_eq!(table.take(), None);
        Ok(())
    }

    #[test]
    fn gives_back_a_taken_lane_only() -> TractResult<()> {
        let mut table = LaneTable::new(2)?;
        assert!(table.give_back(LaneId(0)).is_err());
        table.take();
        table.give_back(LaneId(0))?;
        assert!(table.give_back(LaneId(0)).is_err());
        assert!(table.give_back(LaneId(7)).is_err());
        Ok(())
    }

    #[test]
    fn seats_taken_lanes_in_order() -> TractResult<()> {
        let mut table = LaneTable::new(4)?;
        table.take();
        table.take();
        table.take();
        table.give_back(LaneId(1))?;
        let seating = table.seat([LaneId(2), LaneId(0)])?;
        assert_eq!(seating.max_lanes(), 4);
        assert_eq!(seating.occupancy(), 2);
        assert_eq!(seating.address(0), (Some(0), Some(2)));
        assert_eq!(seating.address(1), (Some(1), Some(0)));
        assert!(table.seat([LaneId(0), LaneId(1)]).is_err());
        assert!(table.seat([LaneId(0), LaneId(0)]).is_err());
        Ok(())
    }
}

crate::declare_knob!(
    TRACT_MAX_SEATS,
    usize,
    256,
    "Most streams a laned runtime serves in one turn, clamped to the state's lanes."
);

crate::declare_knob!(
    TRACT_TURN_LINGER_US,
    usize,
    0,
    "How long a laned runtime waits for more streams once one is ready to run."
);

/// A model prepared to serve many streams at once: one state, one lane per
/// stream, and turns seating whoever is ready.
///
/// `spawn` hands out a [`SessionHandle`] per stream, each holding a lane, and
/// every `run` on a handle is a request to the worker thread which owns the
/// state and the [`LaneTable`] both. The worker takes the turns queued at that
/// moment, at most one per lane and at most [`TRACT_MAX_SEATS`] of them,
/// concatenates their inputs along axis 0, publishes the seating and runs the
/// state once, then hands each stream back its own row.
///
/// A stream feeds one row per turn: axis 0 carries streams, not data. Inputs and
/// outputs whose axis 0 is a symbol are the batched ones; the rest are shared,
/// fed from the first seat and handed back to every stream.
#[derive(Clone)]
pub struct LanedRunnable {
    shared: Arc<Shared>,
}

struct Shared {
    /// [`std::sync::mpsc::Sender`] is not `Sync`, and a `Runnable` is: handles
    /// take their own clone of it, under the lock, once.
    requests: Mutex<Sender<Request>>,
    model: Option<Arc<TypedModel>>,
    plan: Option<Arc<TypedSimplePlan>>,
    max_lanes: usize,
    counts: Arc<Counts>,
}

/// What the worker has served, for whoever tunes the turn policy: mean
/// occupancy is `seats / turns`.
#[derive(Debug, Default)]
struct Counts {
    turns: AtomicU64,
    seats: AtomicU64,
}

impl LanedRunnable {
    /// Serve `max_lanes` streams through `inner`, which must be prepared from a
    /// model carrying a batch axis: at least one input and one output with a
    /// symbol on axis 0, and one symbol for all of them.
    pub fn wrap(inner: Box<dyn Runnable>, max_lanes: usize) -> TractResult<LanedRunnable> {
        let model = inner.typed_model().cloned();
        let plan = inner.typed_plan().cloned();
        let mut symbols: Vec<Symbol> = vec![];
        let mut batch_in: Vec<bool> = vec![];
        for ix in 0..inner.input_count() {
            let symbol = batch_symbol(inner.input_fact(ix)?);
            batch_in.push(symbol.is_some());
            symbols.extend(symbol);
        }
        let mut batch_out: Vec<bool> = vec![];
        for ix in 0..inner.output_count() {
            let symbol = batch_symbol(inner.output_fact(ix)?);
            batch_out.push(symbol.is_some());
            symbols.extend(symbol);
        }
        symbols.sort();
        symbols.dedup();
        ensure!(
            symbols.len() == 1,
            "A laned model carries one batch symbol on axis 0, this one carries {symbols:?}"
        );
        ensure!(batch_out.iter().any(|b| *b), "A laned model must batch one output at least");
        let counts = Arc::new(Counts::default());
        let max_seats = TRACT_MAX_SEATS.get().min(max_lanes);
        let linger = Duration::from_micros(TRACT_TURN_LINGER_US.get() as u64);
        let (requests, queue) = channel::<Request>();
        let (spawned, ready) = channel::<TractResult<()>>();
        let worker_counts = counts.clone();
        thread::Builder::new().name("tract-lanes".into()).spawn(move || {
            let mut state = match inner.spawn().and_then(|mut state| {
                let lanes: Vec<LaneId> = (0..max_lanes).map(LaneId).collect();
                state.reset_lanes(&lanes).context("Preparing a laned model")?;
                Ok(state)
            }) {
                Ok(state) => {
                    let _ = spawned.send(Ok(()));
                    state
                }
                Err(e) => {
                    let _ = spawned.send(Err(e));
                    return;
                }
            };
            worker(
                &mut *state,
                queue,
                Table { batch_in, batch_out, max_seats, linger, max_lanes, counts: worker_counts },
            );
        })?;
        ready.recv().map_err(|_| format_err!("The laned worker died spawning the state"))??;
        Ok(LanedRunnable {
            shared: Arc::new(Shared {
                requests: Mutex::new(requests),
                model,
                plan,
                max_lanes,
                counts,
            }),
        })
    }

    pub fn max_lanes(&self) -> usize {
        self.shared.max_lanes
    }

    /// Turns run and seats filled since the model was prepared: how wide the
    /// turns the queue actually offers are.
    pub fn turns_and_seats(&self) -> (u64, u64) {
        (
            self.shared.counts.turns.load(Ordering::Relaxed),
            self.shared.counts.seats.load(Ordering::Relaxed),
        )
    }

    fn request(&self) -> TractResult<Sender<Request>> {
        Ok(self.shared.requests.lock().map_err(|_| format_err!("Poisoned laned sender"))?.clone())
    }
}

/// The symbol axis 0 of `fact` carries, or `None` for a tensor every seat
/// shares. A stored fact can claim a symbol on an axis of extent one, so this
/// says how the caller talks, not what the graph does with it.
fn batch_symbol(fact: &TypedFact) -> Option<Symbol> {
    match fact.shape.dims().first() {
        Some(TDim::Sym(sym)) => Some(sym.clone()),
        _ => None,
    }
}

impl Debug for LanedRunnable {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "LanedRunnable({} lanes)", self.shared.max_lanes)
    }
}

impl Runnable for LanedRunnable {
    fn spawn(&self) -> TractResult<Box<dyn State>> {
        let requests = self.request()?;
        let (taken, lane) = channel();
        requests.send(Request::Take(taken)).map_err(|_| format_err!("The laned worker is gone"))?;
        let lane = lane.recv().map_err(|_| format_err!("The laned worker dropped a lane"))??;
        Ok(Box::new(SessionHandle {
            lease: Arc::new(Lease { lane, requests }),
            runnable: self.clone(),
        }))
    }

    fn typed_plan(&self) -> Option<&Arc<TypedSimplePlan>> {
        self.shared.plan.as_ref()
    }

    fn typed_model(&self) -> Option<&Arc<TypedModel>> {
        self.shared.model.as_ref()
    }
}

/// One stream's view of a [`LanedRunnable`]: the lane it holds, and the queue to
/// the worker. Cloning it shares the lane -- clones are the same stream, and the
/// lane goes back to the table once the last of them is dropped.
#[derive(Clone, Debug)]
pub struct SessionHandle {
    lease: Arc<Lease>,
    runnable: LanedRunnable,
}

#[derive(Debug)]
struct Lease {
    lane: LaneId,
    requests: Sender<Request>,
}

impl Drop for Lease {
    fn drop(&mut self) {
        let _ = self.requests.send(Request::GiveBack(self.lane));
    }
}

impl State for SessionHandle {
    fn run(&mut self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (done, outputs) = channel();
        self.lease
            .requests
            .send(Request::Turn(Turn { lane: self.lease.lane, inputs, done }))
            .map_err(|_| format_err!("The laned worker is gone"))?;
        outputs.recv().map_err(|_| format_err!("The laned worker dropped a turn"))?
    }

    fn runnable(&self) -> &dyn Runnable {
        &self.runnable
    }
}

enum Request {
    Take(Sender<TractResult<LaneId>>),
    GiveBack(LaneId),
    Turn(Turn),
}

struct Turn {
    lane: LaneId,
    inputs: TVec<TValue>,
    done: Sender<TractResult<TVec<TValue>>>,
}

/// What the worker needs beyond the state and its lanes: which tensors carry the
/// batch axis, and the turn policy.
struct Table {
    batch_in: Vec<bool>,
    batch_out: Vec<bool>,
    max_seats: usize,
    linger: Duration,
    max_lanes: usize,
    counts: Arc<Counts>,
}

fn worker(state: &mut dyn State, queue: Receiver<Request>, table: Table) {
    let mut lanes = match LaneTable::new(table.max_lanes) {
        Ok(lanes) => lanes,
        Err(_) => return,
    };
    let mut queued: Vec<Turn> = vec![];
    loop {
        if queued.is_empty() {
            match queue.recv() {
                Ok(request) => serve(state, &mut lanes, &mut queued, request),
                Err(_) => return,
            }
            if !table.linger.is_zero() {
                thread::sleep(table.linger);
            }
        }
        while let Ok(request) = queue.try_recv() {
            serve(state, &mut lanes, &mut queued, request);
        }
        let mut seated: Vec<Turn> = vec![];
        let mut waiting: Vec<Turn> = vec![];
        for turn in queued.drain(..) {
            if seated.len() < table.max_seats && !seated.iter().any(|s| s.lane == turn.lane) {
                seated.push(turn);
            } else {
                waiting.push(turn);
            }
        }
        queued = waiting;
        if seated.is_empty() {
            continue;
        }
        table.counts.turns.fetch_add(1, Ordering::Relaxed);
        table.counts.seats.fetch_add(seated.len() as u64, Ordering::Relaxed);
        match run_turn(state, &lanes, &seated, &table) {
            Ok(per_seat) => {
                for (turn, outputs) in seated.into_iter().zip(per_seat) {
                    let _ = turn.done.send(Ok(outputs));
                }
            }
            Err(e) => {
                let e = format!("{e:#}");
                for turn in seated {
                    let _ = turn.done.send(Err(format_err!("Laned turn failed: {e}")));
                }
            }
        }
    }
}

/// Take or give back a lane there and then; queue a turn for the coming one.
/// Taking a lane resets it, which is why it happens here rather than in the
/// handle: it writes the state.
fn serve(state: &mut dyn State, lanes: &mut LaneTable, queued: &mut Vec<Turn>, request: Request) {
    match request {
        Request::Take(taken) => {
            let lane = lanes.take().ok_or_else(|| {
                format_err!("Every one of the {} lanes is taken", lanes.max_lanes())
            });
            let lane = lane.and_then(|lane| {
                state.reset_lanes(&[lane]).map(|_| lane).inspect_err(|_| {
                    let _ = lanes.give_back(lane);
                })
            });
            let _ = taken.send(lane);
        }
        Request::GiveBack(lane) => {
            let _ = lanes.give_back(lane);
        }
        Request::Turn(turn) => queued.push(turn),
    }
}

fn run_turn(
    state: &mut dyn State,
    lanes: &LaneTable,
    seated: &[Turn],
    table: &Table,
) -> TractResult<Vec<TVec<TValue>>> {
    let seating = lanes.seat(seated.iter().map(|turn| turn.lane))?;
    let mut batched: TVec<TValue> = tvec!();
    for turn in seated {
        ensure!(
            turn.inputs.len() == table.batch_in.len(),
            "A turn feeds {} inputs, the model takes {}",
            turn.inputs.len(),
            table.batch_in.len()
        );
    }
    for (ix, is_batched) in table.batch_in.iter().enumerate() {
        if *is_batched {
            let rows: TVec<&Tensor> = seated.iter().map(|turn| &*turn.inputs[ix]).collect();
            for row in &rows {
                ensure!(
                    row.rank() > 0 && row.shape()[0] == 1,
                    "A stream feeds one row per turn, input {ix} carries {:?}",
                    row.shape()
                );
            }
            batched.push(Tensor::stack_tensors(0, &rows)?.into_tvalue());
        } else {
            batched.push(seated[0].inputs[ix].clone());
        }
    }
    state.seat(seating)?;
    let outputs = state.run(batched)?;
    let mut per_seat: Vec<TVec<TValue>> = seated.iter().map(|_| tvec!()).collect();
    for (ix, output) in outputs.into_iter().enumerate() {
        if table.batch_out.get(ix).copied().unwrap_or(false) {
            ensure!(
                output.shape()[0] == seated.len(),
                "The turn seats {} streams, output {ix} carries {:?}",
                seated.len(),
                output.shape()
            );
            for (seat, outputs) in per_seat.iter_mut().enumerate() {
                outputs.push(output.slice(0, seat, seat + 1)?.into_tvalue());
            }
        } else {
            for outputs in per_seat.iter_mut() {
                outputs.push(output.clone());
            }
        }
    }
    Ok(per_seat)
}

#[cfg(test)]
mod laned_test {
    use super::*;
    use crate::ops::math::mul;

    /// `[BATCH, 3] * 2`, prepared on the cpu runtime: stateless, so its lanes
    /// address nothing and only the seating of the batch axis is exercised.
    fn doubler(max_lanes: usize) -> TractResult<LanedRunnable> {
        let mut model = TypedModel::default();
        let batch = model.symbols.sym("B");
        let input = model.add_source("input", f32::fact(dims!(batch, 3)))?;
        let two = model.add_const("two", tensor2(&[[2f32]]))?;
        let doubled = model.wire_node("doubled", mul(), &[input, two])?;
        model.select_output_outlets(&doubled)?;
        let inner = DefaultRuntime.prepare(model)?;
        LanedRunnable::wrap(inner, max_lanes)
    }

    fn turn(handle: &mut Box<dyn State>, stream: usize, turn: usize) -> TractResult<()> {
        let input = tensor2(&[[stream as f32, turn as f32, 1.]]);
        let output = handle.run(tvec!(input.into_tvalue()))?;
        assert_eq!(&*output[0], &tensor2(&[[2. * stream as f32, 2. * turn as f32, 2.]]));
        Ok(())
    }

    /// A dropped handle hands its lane back through the queue, so the lane is
    /// free at some point after the drop rather than at it.
    fn spawn_once_free(runnable: &LanedRunnable) -> TractResult<Box<dyn State>> {
        for _ in 0..100 {
            if let Ok(handle) = runnable.spawn() {
                return Ok(handle);
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        runnable.spawn()
    }

    #[test]
    fn one_stream_at_a_time() -> TractResult<()> {
        let runnable = doubler(2)?;
        let mut handle = runnable.spawn()?;
        for t in 0..4 {
            turn(&mut handle, 0, t)?;
        }
        Ok(())
    }

    #[test]
    fn every_stream_gets_its_own_row() -> TractResult<()> {
        let runnable = doubler(8)?;
        let streams: Vec<_> = (0..8)
            .map(|stream| {
                let runnable = runnable.clone();
                std::thread::spawn(move || -> TractResult<()> {
                    let mut handle = runnable.spawn()?;
                    for t in 0..32 {
                        turn(&mut handle, stream, t)?;
                    }
                    Ok(())
                })
            })
            .collect();
        for stream in streams {
            stream.join().unwrap()?;
        }
        Ok(())
    }

    #[test]
    fn a_turn_seats_the_streams_that_are_ready() -> TractResult<()> {
        TRACT_TURN_LINGER_US.set(20_000);
        let runnable = doubler(8);
        TRACT_TURN_LINGER_US.clear();
        let runnable = runnable?;
        let streams: Vec<_> = (0..8)
            .map(|stream| {
                let runnable = runnable.clone();
                std::thread::spawn(move || -> TractResult<()> {
                    let mut handle = runnable.spawn()?;
                    for t in 0..4 {
                        turn(&mut handle, stream, t)?;
                    }
                    Ok(())
                })
            })
            .collect();
        for stream in streams {
            stream.join().unwrap()?;
        }
        let (turns, seats) = runnable.turns_and_seats();
        assert!(seats > turns, "{seats} seats over {turns} turns, none of them shared");
        Ok(())
    }

    #[test]
    fn a_dropped_stream_gives_its_lane_back() -> TractResult<()> {
        let runnable = doubler(1)?;
        let mut handle = runnable.spawn()?;
        turn(&mut handle, 0, 0)?;
        assert!(runnable.spawn().is_err());
        let clone = dyn_clone::clone_box(&*handle);
        drop(handle);
        assert!(runnable.spawn().is_err());
        drop(clone);
        let mut handle = spawn_once_free(&runnable)?;
        turn(&mut handle, 1, 0)?;
        Ok(())
    }
}
