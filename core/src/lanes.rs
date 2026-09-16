use std::collections::{HashMap, VecDeque};
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
/// `spawn` hands out a [`LanedStateHandle`] per stream, each holding a lane, and
/// every `run` on a handle is a request to the worker thread which owns the
/// state and the [`LaneTable`] both. The worker takes the turns queued at that
/// moment, at most one per lane and at most [`TRACT_MAX_SEATS`] of them,
/// concatenates their inputs along axis 0, publishes the seating and runs the
/// state once, then hands each stream back its own seat.
///
/// A stream feeds one seat per turn: axis 0 carries streams, not data. Inputs and
/// outputs whose axis 0 is a symbol are the batched ones; the rest are shared,
/// so one value of such an input serves the whole turn and every seat must feed
/// the same one, and such an output is handed back to every stream.
#[derive(Clone)]
pub struct LanedRunnable {
    shared: Arc<Shared>,
}

struct Shared {
    /// [`std::sync::mpsc::Sender`] is not `Sync`, and a `Runnable` is: handles
    /// take their own clone of it, under the lock, once. `None` once the
    /// runnable is being dropped, which is what closes the queue.
    requests: Mutex<Option<Sender<Request>>>,
    /// Joined when the runnable is dropped, so the worker is gone before
    /// whatever the caller does next.
    worker: Mutex<Option<thread::JoinHandle<()>>>,
    inner: Arc<dyn Runnable>,
    model: Option<Arc<TypedModel>>,
    plan: Option<Arc<TypedSimplePlan>>,
    batch: Symbol,
    max_lanes: usize,
    counts: Arc<Counts>,
}

/// The worker owns per-thread device state -- a CUDA stream, its cuBLAS and
/// cuDNN handles -- whose destructors run as the thread exits. Nothing joined
/// it before, so a process that returned from `main` while the worker was
/// still winding down ran those destructors against libraries already tearing
/// themselves down in their own `atexit` handlers, and segfaulted in
/// `cudnnDestroy` about one run in fifteen. Closing the queue is what stops
/// the worker, so the sender goes first and the join waits for at most the
/// turn in flight.
impl Drop for Shared {
    fn drop(&mut self) {
        if let Ok(mut requests) = self.requests.lock() {
            requests.take();
        }
        let worker = self.worker.lock().ok().and_then(|mut worker| worker.take());
        if let Some(worker) = worker {
            let _ = worker.join();
        }
    }
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
    pub fn wrap(inner: Arc<dyn Runnable>, max_lanes: usize) -> TractResult<LanedRunnable> {
        let model = inner.typed_model().cloned();
        let plan = inner.typed_plan().cloned();
        let mut symbols: Vec<Symbol> = vec![];
        let mut facts: Vec<(String, TypedFact)> = vec![];
        let mut batch_in: Vec<bool> = vec![];
        for ix in 0..inner.input_count() {
            let fact = inner.input_fact(ix)?;
            let symbol = batch_symbol(fact);
            batch_in.push(symbol.is_some());
            symbols.extend(symbol);
            facts.push((format!("input {ix}"), fact.clone()));
        }
        let mut batch_out: Vec<bool> = vec![];
        for ix in 0..inner.output_count() {
            let fact = inner.output_fact(ix)?;
            let symbol = batch_symbol(fact);
            batch_out.push(symbol.is_some());
            symbols.extend(symbol);
            facts.push((format!("output {ix}"), fact.clone()));
        }
        symbols.sort();
        symbols.dedup();
        if symbols.len() != 1 {
            bail!(off_axis_zero(&facts, &symbols));
        }
        ensure!(batch_out.iter().any(|b| *b), "A laned model must batch one output at least");
        let batch = symbols.remove(0);
        let counts = Arc::new(Counts::default());
        let max_seats = TRACT_MAX_SEATS.get().min(max_lanes);
        let linger = Duration::from_micros(TRACT_TURN_LINGER_US.get() as u64);
        let (requests, queue) = channel::<Request>();
        let (spawned, ready) = channel::<TractResult<()>>();
        let worker_counts = counts.clone();
        let worker_inner = inner.clone();
        let worker_thread = thread::Builder::new().name("tract-lanes".into()).spawn(move || {
            let mut state = match worker_inner.spawn().and_then(|mut state| {
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
                requests: Mutex::new(Some(requests)),
                worker: Mutex::new(Some(worker_thread)),
                inner,
                model,
                plan,
                batch,
                max_lanes,
                counts,
            }),
        })
    }

    pub fn max_lanes(&self) -> usize {
        self.shared.max_lanes
    }

    /// The model as it was prepared, serving one stream at a time: what a turn
    /// of one seat has to agree with.
    pub fn inner(&self) -> &Arc<dyn Runnable> {
        &self.shared.inner
    }

    /// The symbol axis 0 of the batched tensors carries. A stream feeds one seat
    /// per turn, so it stands for the turn's occupancy, never for a stream's
    /// own shapes.
    pub fn batch_symbol(&self) -> &Symbol {
        &self.shared.batch
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
        self.shared
            .requests
            .lock()
            .map_err(|_| format_err!("Poisoned laned sender"))?
            .clone()
            .context("The laned runnable is gone")
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

/// Why `facts` do not carry one batch symbol on axis 0: where the symbols they
/// do carry sit, and that putting one on axis 0 is a graph edit.
fn off_axis_zero(facts: &[(String, TypedFact)], symbols: &[Symbol]) -> TractError {
    let elsewhere: Vec<String> = facts
        .iter()
        .filter_map(|(what, fact)| {
            let dims = fact.shape.dims();
            let axis = dims.iter().skip(1).position(|dim| matches!(dim, TDim::Sym(_)))? + 1;
            Some(format!("{what} carries {} on axis {axis}", dims[axis]))
        })
        .collect();
    let elsewhere = elsewhere.join(", ");
    if symbols.is_empty() {
        let found = if elsewhere.is_empty() {
            "none of them carries a symbol at all".to_string()
        } else {
            elsewhere
        };
        format_err!(
            "A laned model carries its batch symbol on axis 0 of the inputs and outputs it \
             batches, and {found}. Axis 0 is where a seat's values are a contiguous run, so move \
             the batch there as a graph edit -- Batchify, or an AddAxis/MoveAxis the optimiser \
             can absorb -- rather than have every turn transpose"
        )
    } else {
        let symbols: Vec<String> = symbols.iter().map(|s| s.to_string()).collect();
        format_err!(
            "A laned model carries one batch symbol on axis 0, this one carries {}. One symbol \
             has to stand for the whole turn's occupancy, so share it across the batched inputs \
             and outputs in the export",
            symbols.join(" and ")
        )
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
        requests
            .send(Request::Spawn(taken))
            .map_err(|_| format_err!("The laned worker is gone"))?;
        let lane = lane.recv().map_err(|_| format_err!("The laned worker dropped a lane"))??;
        Ok(Box::new(LanedStateHandle {
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
pub struct LanedStateHandle {
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
        let _ = self.requests.send(Request::Drop(self.lane));
    }
}

impl State for LanedStateHandle {
    fn run(&mut self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (done, outputs) = channel();
        self.lease
            .requests
            .send(Request::Call(Call { lane: self.lease.lane, inputs, done }))
            .map_err(|_| format_err!("The laned worker is gone"))?;
        outputs.recv().map_err(|_| format_err!("The laned worker dropped a turn"))?
    }

    fn runnable(&self) -> &dyn Runnable {
        &self.runnable
    }
}

/// What a handle asks the worker for, one variant per event of the handle's
/// life: a lane when it is spawned, a turn per call, and its lane back when it
/// is dropped.
enum Request {
    Spawn(Sender<TractResult<LaneId>>),
    Call(Call),
    Drop(LaneId),
}

/// One `run()` on a handle. Its inputs carry one seat or several, and it is
/// answered once every one of them has been served -- over as many turns as
/// the free lanes took to seat them all.
struct Call {
    lane: LaneId,
    inputs: TVec<TValue>,
    done: Sender<TractResult<TVec<TValue>>>,
}

/// One seat of one call, waiting for a lane: the lane it sits in -- its
/// caller's own until a turn borrows another for it -- its place in the call,
/// and the slice of the call's inputs it feeds.
struct Seat {
    call: u64,
    lane: LaneId,
    ix: usize,
    inputs: TVec<TValue>,
}

/// What a call still owes its caller: its seats' outputs, in the order it asked
/// for them, and where to answer.
struct Answer {
    served: Vec<Option<TVec<TValue>>>,
    done: Sender<TractResult<TVec<TValue>>>,
}

/// The seats the worker has to seat, and the calls they answer. A call explodes
/// into seats as it arrives, so what a turn picks from is a queue of seats: it
/// fills to the free lanes from the head, and a call wider than they are is
/// split at the boundary rather than held until it fits whole.
#[derive(Default)]
struct Queue {
    seats: VecDeque<Seat>,
    answers: HashMap<u64, Answer>,
    calls: u64,
}

impl Queue {
    fn push(&mut self, call: Call, batch_in: &[bool]) {
        let id = self.calls;
        self.calls += 1;
        match explode(&call, batch_in) {
            Ok(seats) => {
                self.answers
                    .insert(id, Answer { served: vec![None; seats.len()], done: call.done });
                for (ix, inputs) in seats.into_iter().enumerate() {
                    self.seats.push_back(Seat { call: id, lane: call.lane, ix, inputs });
                }
            }
            Err(e) => {
                let _ = call.done.send(Err(e));
            }
        }
    }

    /// Hand `seat` its turn's outputs, and answer the call when that was the
    /// last seat it was waiting for. A failed seat fails the whole call at
    /// once, and drops the seats of it still queued.
    fn serve(&mut self, seat: Seat, outputs: TractResult<TVec<TValue>>, batch_out: &[bool]) {
        if !self.answers.contains_key(&seat.call) {
            return;
        }
        let outputs = match outputs {
            Ok(outputs) => outputs,
            Err(e) => {
                let answer = self.answers.remove(&seat.call).unwrap();
                self.seats.retain(|queued| queued.call != seat.call);
                let _ = answer.done.send(Err(e));
                return;
            }
        };
        let answer = self.answers.get_mut(&seat.call).unwrap();
        answer.served[seat.ix] = Some(outputs);
        if answer.served.iter().all(Option::is_some) {
            let answer = self.answers.remove(&seat.call).unwrap();
            let served: Vec<TVec<TValue>> =
                answer.served.into_iter().map(|outputs| outputs.unwrap()).collect();
            let _ = answer.done.send(assemble(served, batch_out));
        }
    }
}

/// The inputs of each seat of `call`: the batched ones sliced along axis 0, the
/// shared ones as they came. Every batched input says how many seats the call
/// asks for, so they must agree, and a call with none asks for one.
fn explode(call: &Call, batch_in: &[bool]) -> TractResult<Vec<TVec<TValue>>> {
    ensure!(
        call.inputs.len() == batch_in.len(),
        "A call feeds {} inputs, the model takes {}",
        call.inputs.len(),
        batch_in.len()
    );
    let mut seats: Option<usize> = None;
    for (ix, input) in call.inputs.iter().enumerate().filter(|(ix, _)| batch_in[*ix]) {
        ensure!(input.rank() > 0, "Input {ix} carries the batch axis, so it can not be a scalar");
        let asked = input.shape()[0];
        ensure!(
            seats.is_none_or(|seats| seats == asked),
            "A call asks for as many seats as its batched inputs carry, and input {ix} carries \
             {asked} against {} before it",
            seats.unwrap_or(0)
        );
        seats = Some(asked);
    }
    let seats = seats.unwrap_or(1);
    ensure!(seats > 0, "A call asks for one seat at least");
    if seats == 1 {
        return Ok(vec![call.inputs.clone()]);
    }
    (0..seats)
        .map(|seat| {
            call.inputs
                .iter()
                .zip(batch_in)
                .map(|(input, is_batched)| {
                    if *is_batched {
                        Ok(input.slice(0, seat, seat + 1)?.into_tvalue())
                    } else {
                        Ok(input.clone())
                    }
                })
                .collect()
        })
        .collect()
}

/// One call's outputs from its seats': the batched ones stacked back along axis
/// 0 in the order the call asked for its seats, the shared ones as the first
/// seat got them.
fn assemble(served: Vec<TVec<TValue>>, batch_out: &[bool]) -> TractResult<TVec<TValue>> {
    if served.len() == 1 {
        return Ok(served.into_iter().next().unwrap());
    }
    let first = &served[0];
    (0..first.len())
        .map(|ix| {
            if batch_out.get(ix).copied().unwrap_or(false) {
                let seats: TVec<&Tensor> = served.iter().map(|outputs| &*outputs[ix]).collect();
                Ok(Tensor::stack_tensors(0, &seats)?.into_tvalue())
            } else {
                Ok(first[ix].clone())
            }
        })
        .collect()
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
    let mut queued = Queue::default();
    loop {
        // The linger belongs to the turn a request opens, not to the request:
        // taking or giving back a lane must not delay the turns behind it, and
        // turns left waiting by a full one have lingered already.
        while queued.seats.is_empty() {
            match queue.recv() {
                Ok(request) => serve(state, &mut lanes, &mut queued, &table, request),
                Err(_) => return,
            }
            if !queued.seats.is_empty() && !table.linger.is_zero() {
                thread::sleep(table.linger);
            }
        }
        while let Ok(request) = queue.try_recv() {
            serve(state, &mut lanes, &mut queued, &table, request);
        }
        let (seated, borrowed) = fill(state, &mut lanes, &mut queued, &table);
        if seated.is_empty() {
            continue;
        }
        table.counts.turns.fetch_add(1, Ordering::Relaxed);
        table.counts.seats.fetch_add(seated.len() as u64, Ordering::Relaxed);
        let served = run_turn(state, &lanes, &seated, &table);
        for lane in borrowed {
            let _ = lanes.give_back(lane);
        }
        match served {
            Ok(per_seat) => {
                for (seat, outputs) in seated.into_iter().zip(per_seat) {
                    queued.serve(seat, Ok(outputs), &table.batch_out);
                }
            }
            Err(e) => {
                let e = format!("{e:#}");
                for seat in seated {
                    queued.serve(
                        seat,
                        Err(format_err!("Laned turn failed: {e}")),
                        &table.batch_out,
                    );
                }
            }
        }
    }
}

/// Hand out a lane or take it back there and then; queue the seats of a call
/// for the coming turns. Taking a lane resets it, which is why it happens here
/// rather than in the handle: it writes the state.
fn serve(
    state: &mut dyn State,
    lanes: &mut LaneTable,
    queued: &mut Queue,
    table: &Table,
    request: Request,
) {
    match request {
        Request::Spawn(taken) => {
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
        Request::Call(call) => queued.push(call, &table.batch_in),
        Request::Drop(lane) => {
            let _ = lanes.give_back(lane);
        }
    }
}

/// Seat the head of the queue, and the lanes borrowed to do it. A seat sits in
/// its caller's own lane, and a call's second and later seats of one turn in
/// lanes borrowed from whatever is free -- which resets them, so a seat is
/// always served by a lane holding nothing of another caller's.
fn fill(
    state: &mut dyn State,
    lanes: &mut LaneTable,
    queued: &mut Queue,
    table: &Table,
) -> (Vec<Seat>, Vec<LaneId>) {
    let mut seated: Vec<Seat> = vec![];
    let mut taken: Vec<LaneId> = vec![];
    let mut borrowed: Vec<LaneId> = vec![];
    let mut waiting: VecDeque<Seat> = VecDeque::new();
    while let Some(mut seat) = queued.seats.pop_front() {
        if seated.len() >= table.max_seats {
            waiting.push_back(seat);
            continue;
        }
        let lane = if taken.contains(&seat.lane) {
            match lanes.take() {
                Some(lane) => match state.reset_lanes(&[lane]) {
                    Ok(()) => {
                        borrowed.push(lane);
                        Some(lane)
                    }
                    Err(_) => {
                        let _ = lanes.give_back(lane);
                        None
                    }
                },
                None => None,
            }
        } else {
            Some(seat.lane)
        };
        match lane {
            Some(lane) => {
                seat.lane = lane;
                taken.push(lane);
                seated.push(seat);
            }
            None => waiting.push_back(seat),
        }
    }
    queued.seats = waiting;
    (seated, borrowed)
}

fn run_turn(
    state: &mut dyn State,
    lanes: &LaneTable,
    seated: &[Seat],
    table: &Table,
) -> TractResult<Vec<TVec<TValue>>> {
    let seating = lanes.seat(seated.iter().map(|seat| seat.lane))?;
    let mut batched: TVec<TValue> = tvec!();
    for (ix, is_batched) in table.batch_in.iter().enumerate() {
        if *is_batched {
            let seats: TVec<&Tensor> = seated.iter().map(|seat| &*seat.inputs[ix]).collect();
            for (seat, input) in seats.iter().enumerate() {
                ensure!(
                    input.rank() > 0 && input.shape()[0] == 1,
                    "Seat {seat} feeds {:?} of input {ix}, which a turn seats one at a time",
                    input.shape()
                );
            }
            batched.push(Tensor::stack_tensors(0, &seats)?.into_tvalue());
        } else {
            let shared = &seated[0].inputs[ix];
            for (seat, turn) in seated.iter().enumerate().skip(1) {
                ensure!(
                    turn.inputs[ix] == *shared,
                    "Input {ix} carries no batch axis, so one value of it serves the whole \
                     turn, and seats 0 and {seat} feed it different ones"
                );
            }
            batched.push(shared.clone());
        }
    }
    state.seat(seating)?;
    let outputs = state.run(batched)?;
    let mut per_seat: Vec<TVec<TValue>> = seated.iter().map(|_| tvec!()).collect();
    for (ix, output) in outputs.into_iter().enumerate() {
        if table.batch_out.get(ix).copied().unwrap_or(false) {
            ensure!(
                output.shape()[0] == seated.len(),
                "The turn fills {} seats, output {ix} carries {:?}",
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

// The suite is disabled on Wasm because a laned runnable spawns a thread.
#[cfg(all(test, not(target_family = "wasm")))]
mod laned_test {
    use super::*;
    use crate::ops::math::{add, mul};

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
        LanedRunnable::wrap(inner.into(), max_lanes)
    }

    fn turn(handle: &mut Box<dyn State>, stream: usize, turn: usize) -> TractResult<()> {
        let input = tensor2(&[[stream as f32, turn as f32, 1.]]);
        let output = handle.run(tvec!(input.into_tvalue()))?;
        assert_eq!(&*output[0], &tensor2(&[[2. * stream as f32, 2. * turn as f32, 2.]]));
        Ok(())
    }

    /// `TRACT_TURN_LINGER_US` is process-wide, so the tests which widen the
    /// turns hold this while they build their runnable and run their streams.
    static LINGER: Mutex<()> = Mutex::new(());

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
    fn every_stream_gets_its_own_seat() -> TractResult<()> {
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
        let _linger = LINGER.lock().unwrap_or_else(|e| e.into_inner());
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

    /// `[B, 3] * 2 + bias`, `bias` carrying no batch axis: the shape of a
    /// shared input, which one value of serves the whole turn.
    fn biased(max_lanes: usize) -> TractResult<LanedRunnable> {
        let mut model = TypedModel::default();
        let batch = model.symbols.sym("B");
        let input = model.add_source("input", f32::fact(dims!(batch, 3)))?;
        let bias = model.add_source("bias", f32::fact(dims!(1, 1)))?;
        let two = model.add_const("two", tensor2(&[[2f32]]))?;
        let doubled = model.wire_node("doubled", mul(), &[input, two])?;
        let biased = model.wire_node("biased", add(), &[doubled[0], bias])?;
        model.select_output_outlets(&biased)?;
        let inner = DefaultRuntime.prepare(model)?;
        LanedRunnable::wrap(inner.into(), max_lanes)
    }

    /// One turn per stream, all of them at once, the `stream`th feeding
    /// `biases[stream]`. Every lane is taken before any turn is queued, so the
    /// linger has the turns to seat together rather than a `spawn` to serve.
    fn biased_turns(runnable: &LanedRunnable, biases: &[f32]) -> TractResult<Vec<TractResult<()>>> {
        let handles: Vec<Box<dyn State>> =
            biases.iter().map(|_| runnable.spawn()).collect::<TractResult<_>>()?;
        let streams: Vec<_> = handles
            .into_iter()
            .zip(biases.iter().copied())
            .map(|(mut handle, bias)| {
                std::thread::spawn(move || -> TractResult<()> {
                    handle.run(tvec!(
                        tensor2(&[[1f32, 2., 3.]]).into_tvalue(),
                        tensor2(&[[bias]]).into_tvalue()
                    ))?;
                    Ok(())
                })
            })
            .collect();
        Ok(streams.into_iter().map(|stream| stream.join().unwrap()).collect())
    }

    #[test]
    fn seats_agreeing_on_a_shared_input_share_a_turn() -> TractResult<()> {
        let _linger = LINGER.lock().unwrap_or_else(|e| e.into_inner());
        TRACT_TURN_LINGER_US.set(100_000);
        let runnable = biased(2);
        TRACT_TURN_LINGER_US.clear();
        let runnable = runnable?;
        let served = biased_turns(&runnable, &[7., 7.])?;
        assert!(served.iter().all(|s| s.is_ok()), "{served:?}");
        assert_eq!(runnable.turns_and_seats(), (1, 2));
        Ok(())
    }

    #[test]
    fn seats_disagreeing_on_a_shared_input_fail_the_turn() -> TractResult<()> {
        let _linger = LINGER.lock().unwrap_or_else(|e| e.into_inner());
        TRACT_TURN_LINGER_US.set(100_000);
        let runnable = biased(2);
        TRACT_TURN_LINGER_US.clear();
        let runnable = runnable?;
        let served = biased_turns(&runnable, &[7., 8.])?;
        assert_eq!(runnable.turns_and_seats(), (1, 2));
        for stream in &served {
            let error = format!("{:#}", stream.as_ref().unwrap_err());
            assert!(error.contains("seats 0 and 1 feed it different ones"), "{error}");
        }
        Ok(())
    }

    /// A call feeding `seats` seats of the doubler at once: seat `s` feeds
    /// `[s, s + 1, s + 2]`, and the call is answered in that order.
    fn wide_call(handle: &mut Box<dyn State>, seats: usize) -> TractResult<()> {
        let fed: Vec<f32> =
            (0..seats).flat_map(|s| [s as f32, s as f32 + 1., s as f32 + 2.]).collect();
        let doubled: Vec<f32> = fed.iter().map(|x| 2. * x).collect();
        let output = handle.run(tvec!(Tensor::from_shape(&[seats, 3], &fed)?.into_tvalue()))?;
        assert_eq!(&*output[0], &Tensor::from_shape(&[seats, 3], &doubled)?);
        Ok(())
    }

    #[test]
    fn a_call_asks_for_as_many_seats_as_it_feeds() -> TractResult<()> {
        let runnable = doubler(4)?;
        let mut handle = runnable.spawn()?;
        wide_call(&mut handle, 3)?;
        assert_eq!(runnable.turns_and_seats(), (1, 3));
        Ok(())
    }

    #[test]
    fn a_call_wider_than_the_lanes_is_split_across_turns() -> TractResult<()> {
        let runnable = doubler(2)?;
        let mut handle = runnable.spawn()?;
        wide_call(&mut handle, 5)?;
        let (turns, seats) = runnable.turns_and_seats();
        assert_eq!(seats, 5);
        assert_eq!(turns, 3, "5 seats over 2 lanes are 3 turns, got {turns}");
        Ok(())
    }

    #[test]
    fn a_seat_borrows_a_free_lane_only() -> TractResult<()> {
        let runnable = doubler(2)?;
        let mut handle = runnable.spawn()?;
        let mut other = runnable.spawn()?;
        turn(&mut other, 1, 0)?;
        wide_call(&mut handle, 4)?;
        let (turns, seats) = runnable.turns_and_seats();
        assert_eq!(seats, 5);
        assert_eq!(turns, 5, "another stream holds the second lane, so each seat is a turn");
        Ok(())
    }

    /// `[B, 3] + [B, 3]`: two batched inputs, which a call has to feed the same
    /// number of seats of.
    fn adder(max_lanes: usize) -> TractResult<LanedRunnable> {
        let mut model = TypedModel::default();
        let batch = model.symbols.sym("B");
        let left = model.add_source("left", f32::fact(dims!(batch, 3)))?;
        let right = model.add_source("right", f32::fact(dims!(batch, 3)))?;
        let sum = model.wire_node("sum", add(), &[left, right])?;
        model.select_output_outlets(&sum)?;
        let inner = DefaultRuntime.prepare(model)?;
        LanedRunnable::wrap(inner.into(), max_lanes)
    }

    #[test]
    fn a_call_whose_batched_inputs_disagree_on_seats_fails() -> TractResult<()> {
        let runnable = adder(4)?;
        let mut handle = runnable.spawn()?;
        let error = handle
            .run(tvec!(
                tensor2(&[[1f32, 2., 3.], [4., 5., 6.]]).into_tvalue(),
                tensor2(&[[7f32, 8., 9.]]).into_tvalue()
            ))
            .unwrap_err();
        let error = format!("{error:#}");
        assert!(error.contains("input 1 carries 1 against 2"), "{error}");
        Ok(())
    }

    #[test]
    fn a_batch_off_axis_zero_says_which_axis_it_sits_on() -> TractResult<()> {
        let mut model = TypedModel::default();
        let batch = model.symbols.sym("B");
        let input = model.add_source("input", f32::fact(dims!(3, batch)))?;
        let two = model.add_const("two", tensor2(&[[2f32]]))?;
        let doubled = model.wire_node("doubled", mul(), &[input, two])?;
        model.select_output_outlets(&doubled)?;
        let inner = DefaultRuntime.prepare(model)?;
        let error = format!("{:#}", LanedRunnable::wrap(inner.into(), 2).unwrap_err());
        assert!(error.contains("input 0 carries B on axis 1"), "{error}");
        assert!(error.contains("Batchify"), "{error}");
        Ok(())
    }

    #[test]
    fn two_batch_symbols_are_named() -> TractResult<()> {
        let mut model = TypedModel::default();
        let left_batch = model.symbols.sym("L");
        let right_batch = model.symbols.sym("R");
        let left = model.add_source("left", f32::fact(dims!(left_batch, 3)))?;
        let right = model.add_source("right", f32::fact(dims!(right_batch, 3)))?;
        let sum = model.wire_node("sum", add(), &[left, right])?;
        model.select_output_outlets(&sum)?;
        let inner = DefaultRuntime.prepare(model)?;
        let error = format!("{:#}", LanedRunnable::wrap(inner.into(), 2).unwrap_err());
        assert!(error.contains("L and R"), "{error}");
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
