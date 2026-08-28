# Lexicon

Words with one meaning each. Pick these when naming things in `core`, `pulse`,
`pulse-opl`, `gpu` and the backends; do not spend them on anything else.

## Execution scopes

- **turn** — one `run()` call on a state. `TurnState` carries what lives for a
  turn: resolved symbols, intermediate values, scratch. `TurnStateHandler` fires
  once before and once after each turn.
- **session** — the scope a state covers across many turns.
  `SimpleState::op_states` holds it: op state built once at spawn and carried from
  turn to turn.

The split is in the type: `SimpleState { op_states, turn_state }` is exactly
session-scoped versus turn-scoped. State belongs in the container matching its
scope — an op that needs only per-turn workspace does not belong in `op_states`.

## Reserved

- **lane** — the address at which one session's state lives inside a state shared
  by several sessions. Reserved for batched streaming runtimes; nothing
  implements it yet.
- **seat** — a session's position within one turn's batch, where a turn carries
  several sessions. Reserved with `lane`.

## Taken, do not reuse

- **stream** — in `pulse`, the axis a model is pulsified along (`StreamInfo`,
  `stream_sym`). Never a client connection or an audio session.
- **slot** — a node's input or output port (`OutletId::slot`).
- **row** — a row of a matrix or a kernel tile, in `linalg`.
- **rank** — the number of axes of a tensor.

Applications built on tract have their own Session and turn, at their own
granularity. tract does not pick terminology to avoid colliding with them.
