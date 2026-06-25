# Harness Server Specification

## Overview

The Harness Server is a multi-agent runtime that loads participants, routes messages between them by role, and manages the dialogue loop. Multiple participants may subscribe to the same role and all receive messages sent to that role (fan-out). The harness runs as a persistent server — participants connect as clients (via WebSocket) or are embedded in-process. The harness itself is not a participant; it is the message broker.

## Dependencies

- `specs/agent.md` — Kalvin rationalisation API, events, Cogitator
- `specs/kvalue.md` — KValue (KLine + significance): the exchange unit for the trainer's submissions, countersigns, and RationaliseEvent payloads.
- `specs/kscript.md` — compilation pipeline (KScript source → CompiledEntry)
- `specs/harness.md` — tracking state, satisfaction, ratification (absorbed into Trainer)

## Definitions

### Harness Configuration

A YAML or JSON file loaded at startup. Lists participants with their role and connection type (embedded or client). Multiple participants may share the same role — all receive messages sent to that role.

```yaml
participants:
  - role: trainee
    type: embedded
    class: KAgent
  - role: trainer
    type: embedded
    class: Trainer
  - role: supervisor
    type: client
    class: TUIParticipant
  - role: supervisor
    type: client
    class: SlackParticipant
```

### Message

A unit of inter-participant communication. Routed by role.

```
Message:
  role:     str       # recipient role (trainee, trainer, supervisor)
  action:   str       # interpreted by recipient
  message:  any       # payload (KScript source, RationaliseEvent, freeform text, etc.)
```

The harness does not interpret `action` or `message` — it routes by `role` only.

### Message Bus

A thread-safe role-based message router with fan-out dispatch. Participants subscribe by role. Messages are dispatched to all subscribers of the target role on a single harness event loop via an internal queue.

```
MessageBus:
  subscribe(role, handler) → None
  send(Message) → None
  run() → None              # event loop: dequeue and dispatch
```

- Thread-safe: any participant may call `send()` from any thread.
- Fan-out: all handlers subscribed to a role receive messages sent to that role.
- Single dispatch: all handler invocations execute on the harness event loop thread.
- Diagnostic listeners may subscribe to a special wildcard role to observe all messages.

### Participant Protocol

Every participant (embedded or client) implements:

```
Participant:
  role: str
  on_message(Message) → None     # receive a message from the bus
```

Embedded participants are called directly. Client participants receive messages via their WebSocket connection.

### WebSocket Wire Protocol

Connected clients communicate over WebSocket. JSON frames.

**Registration** (client → server, once on connect):

```json
{ "register": "supervisor" }
```

Multiple clients may register for the same role. All receive messages sent to that role.

**Message** (bidirectional):

```json
{ "role": "trainee", "action": "submit", "message": "MHALL = SVO" }
```

**Error** (server → client):

```json
{ "error": "unknown role: foo" }
```

After registration, all frames on that socket are implicitly from the registered role. The harness maps sockets to roles internally, allowing multiple sockets per role.

### Kalvin Adapter

Kalvin's interface to the harness bus. A thin layer that:

1. Receives harness messages sent to role `trainee`.
2. Interprets the `action`:
   - `submit` — compile KScript source via the KScript pipeline, submit each compiled entry to `kagent.rationalise()` one at a time.
   - `countersign` — materialise the bus payload to a KValue and call `kagent.countersign(kvalue)`. The payload may arrive as a live KValue, a wire dict, or a legacy KLine (wrapped at S1); see @kvalue spec §KP-2.
   - `rationalise` — materialise the bus payload to a KValue and call `kagent.rationalise(kvalue)` directly. Unlike `submit` (which re-derives significance from structure) and `countersign` (which builds the reciprocal at S1), this delivers the KValue as-is: the significance on the KValue is the sender's declared assessment, carried straight into the Phase 1b significance-comparison gate (@agent spec §Rationalisation). Same three payload forms as `countersign`.
3. Receives Kalvin's callbacks directly (no internal EventBus) and wraps them into harness messages dispatched to the original sender's role.
4. Maintains a sender map: when entries arrive from role X, the adapter records X as the sender. Kalvin's callbacks are sent to role X.

### Adapter Callback Interface

Kalvin calls the adapter directly instead of publishing to an internal EventBus:

```
KAgentAdapterCallback:
  on_event(event: RationaliseEvent) → None
```

Kalvin is constructed with an adapter instance. The adapter's `on_event` wraps the event into a message and sends it to the sender's role via the bus.

### Compilation Errors

If a `submit` action contains KScript that fails to compile, the adapter sends an error message back to the sender:

```json
{
  "role": "trainer",
  "action": "error",
  "message": "ParseError at line 2: unexpected token"
}
```

### Trainer Participant

An embedded participant that drives the training loop. See `@specs/trainer.md` (TBD) for full specification.

Key behaviours:

- **Curriculum-driven mode**: submits the next lesson from the curriculum to Kalvin.
- **Reactive mode**: cogitates (via LLM agent) on S2/S3 events and generates reactive scaffolding. Gated by `trainer.llm.enabled` (default `true`); when `false`, reactive decisions are delegated to the supervisor instead. See `@specs/reactive-delegation.md`.
- **Ratification**: auto-countersigns proposals that structurally match expectations. When auto-countersign succeeds, the proposal is already ratified and no `ratify_request` is sent to the supervisor.
- **Escalation**: sends messages to role `supervisor` when stuck (budget exhaustion or low LLM agent confidence). All supervisor participants receive escalation notifications. Suppressed in delegated mode (`@specs/reactive-delegation.md`).
- **Event relay**: relays Kalvin ground/frame/error events to role `supervisor` so supervisors observe the full training session. Ground events forwarded as `event` action. Frame events (S2/S3 proposals) forwarded as both `event` and `ratify_request` (carrying the proposal for ratification).
- **Ratify request**: when a frame event carries a ratifiable proposal **and auto-countersign did not succeed**, sends a `ratify_request` action to role `supervisor` with the proposal payload. When auto-countersign succeeds the request is suppressed (the proposal needs no supervisor ratification); event relay to the supervisor continues regardless.
- **Recurring-proposal drop**: when the same proposal kline reaches the reactor twice in one lesson (intra-expectation candidate fan-out — one expectation against two candidates yielding the same reshaped proposal), the second sighting re-submits the proposal to Kalvin as a `rationalise` action carrying a declared `SIG_S4`. Kalvin's Phase 1b gate then drops it instead of re-cogitating it. The reactor records each first-sighting proposal in a per-lesson seen-set (cleared on `load_lesson`); a structurally-different proposal is not recurrence and gets a fresh first-sighting. Recurrence counts toward the reactive budget so the escalation safety net survives a pure-recurrence stall.

### S2/S3 Auto-Countersign Suppression

The reactor's `process_s2_s3()` returns a `bool`: `True` when
auto-countersign succeeded **or** a recurring proposal was dropped, `False`
when reactive handling was invoked (scaffolding or escalation). It checks
auto-countersign first; on success it returns `True` immediately without
invoking reactive handling. On a non-match it next checks recurrence: a
second sighting of the same proposal this lesson re-submits it as a
`rationalise` action at declared `SIG_S4` and returns `True` (no supervisor
needed). Otherwise the Trainer captures the return and only sends
`ratify_request` when auto-countersign did **not** succeed **and** the
proposal was not a recurrence. Event relay to the supervisor is
unconditional — the supervisor observes every event even when no
ratification is needed.

- **SAC-1.** `Reactor.process_s2_s3()` MUST return `True` when
  auto-countersign succeeds or a recurring proposal is dropped, and `False`
  when reactive handling is invoked.
- **SAC-2.** When auto-countersign succeeds, reactive handling MUST NOT be
  invoked.
- **SAC-2a.** On a second sighting of the same proposal kline within a
  lesson, the reactor MUST re-submit it as a `rationalise` action carrying
  declared `SIG_S4` and MUST NOT invoke scaffolding; the recurrence MUST
  count toward the reactive budget.
- **SAC-2b.** `load_lesson` MUST clear the per-lesson seen-set so a
  proposal that recurred in one lesson gets a fresh first sighting in the
  next.
- **SAC-3.** The Trainer MUST NOT send `ratify_request` for S2/S3 events
  where auto-countersign succeeded or the proposal was dropped as
  recurrence.
- **SAC-4.** The Trainer MUST still send `ratify_request` for S2/S3 events
  where auto-countersign failed **and** the proposal was a first sighting
  (not recurrence).
- **SAC-5.** Event relay to the supervisor MUST continue regardless of
  auto-countersign outcome.
- **Session management**: one training session at a time. Supports pause and stop via human input.
- **State persistence**: curriculum position and event log persisted for restart recovery.

Trainer maintains per-session submitted, satisfied, and pending sets — absorbing the tracking state from the current harness spec (`@specs/harness.md`).

### Supervisor Participant

A participant subscribed to role `supervisor` that monitors the training session and may intercede when needed. The supervisor role is independent of the medium — TUI, Slack, or a future AI agent participant all have the same capabilities. The supervisor receives all session events (progress, Kalvin events, escalation, ratify requests) and can send commands (session control, guidance, ratification) through a shared command protocol.

An ideal training session sees the supervisor in a mostly monitoring role, interceding only on escalated events.

### Shared Command Protocol

Both TUI and Slack participants use a shared command parser (`src/training/participants/commands.py`) to interpret human input. The parser maps free-text to structured commands, which are then dispatched as bus messages.

**Parsed commands:**

| Input | Command | Bus message |
|-------|---------|-------------|
| `start` | `StartCommand` | `{role: "trainer", action: "input", message: "start"}` |
| `stop` | `StopCommand` | `{role: "trainer", action: "input", message: "stop"}` |
| `pause` | `PauseCommand` | `{role: "trainer", action: "input", message: "pause"}` |
| `resume` | `ResumeCommand` | `{role: "trainer", action: "input", message: "resume"}` |
| `goal: <text>` | `GoalCommand` | `{role: "trainer", action: "input", message: "goal: <text>"}` |
| `ratify` | `RatifyCommand` | `{role: "trainee", action: "countersign", message: <latest proposal>}` |
| `scaffold:<kscript>` | `ScaffoldCommand` | `{role: "trainee", action: "submit", message: <kscript>}` (delegated reactive decision — see `@specs/reactive-delegation.md`) |
| `<file path>` | `FileGoalCommand` | `{role: "trainer", action: "input", message: "<path>"}` |
| `<anything else>` | `GuidanceCommand` | `{role: "trainer", action: "input", message: "<text>"}` |

The `ratify` command uses the latest pending proposal tracked by the participant. Each supervisor participant tracks the most recent `ratify_request` received from the Trainer and uses it when a ratify command is issued.

**Messages received by all supervisor participants (via role `supervisor`):**

| Bus action | Payload | When |
|------------|---------|------|
| `progress` | `{status, lesson_label, lessons_total, lessons_completed}` | Session lifecycle events |
| `event` | `RationaliseEvent` (ground/frame/error) | Kalvin event relay from Trainer |
| `escalation` | `{reason, detail, lesson_position}` | Trainer stuck, needs help |
| `ratify_request` | `{proposal, query, significance}` (optionally enriched with `misfit`, `curriculum_context` in delegated mode) | Proposal ready for ratification / reactive decision (`@specs/reactive-delegation.md`) |

**Messages sent by all supervisor participants:**

| Bus action | Target role | Payload | Purpose |
|------------|------------|---------|--------|
| `input` | `trainer` | free-text | Session control, goals, guidance |
| `countersign` | `trainee` | KValue proposal | Ratification |

### Slack Participant

A client participant that translates between Slack API and harness messages. Registers as role `supervisor`. Has full supervisor capability — renders all session events in Slack and forwards human commands through the shared command protocol.

**Rendering:**

- `progress` — posted as Slack message with status summary.
- `event` — posted as Slack message with Kalvin event summary.
- `escalation` — posted as prominent Slack message requiring attention.
- `ratify_request` — posted as Slack message with proposal details, hinting that `ratify` command is available.

**Input handling:**

- Human Slack messages are fed through the shared command parser.
- `ratify` command sends `countersign` to the trainee with the latest pending proposal.
- All other commands route to the Trainer as `input` actions.

Uses a dedicated Slack channel. No addressing syntax needed.

### TUI Participant

A client participant that renders events for the supervisor, provides structured ratification controls, and allows the human to compose and send commands through the shared command protocol. Connects to the harness server via WebSocket. Registers as role `supervisor`. Does not own Kalvin or make curriculum decisions.

**Rendering:**

- All incoming events (progress, event, escalation, ratify_request) displayed in the EventLog.
- RatifyBar activates when a `ratify_request` is received.

**Input handling:**

- InputBar text is fed through the shared command parser.
- `ratify` command sends `countersign` to the trainee with the latest pending proposal (same path as RatifyBar button).
- All other commands route to the Trainer as `input` actions.

**UI regions:**

- **EventLog** — scrollable log displaying all received harness events (timestamp, action, message summary).
- **InputBar** — text input field. Text is parsed via the shared command protocol on submit. Field clears after submission.
- **RatifyBar** — button (disabled by default) that becomes active when a `ratify_request` arrives. Clicking it or typing `ratify` in the InputBar sends a `countersign` action to the trainee.
- **Header / Footer** — app title and keyboard shortcut hints (`ctrl+q` quit, `ctrl+r` ratify, `ctrl+s` send input).

## Behavioural Rules

### Message Routing

1. Messages are routed by `role` only. The harness does not inspect `action` or `message`.
2. A message sent to an unknown role (no subscribers) results in an error sent back to the sender.
3. Kalvin's adapter always addresses responses to the original sender's role.
4. All subscribers registered for a role receive messages sent to that role (fan-out dispatch).

### Participant Lifecycle

5. Embedded participants are instantiated on harness startup from the configuration file.
6. Client participants connect via WebSocket and register for a role.
7. Multiple WebSocket clients may register for the same role. All receive messages for that role.
8. A client that disconnects does not remove the role's subscription. Messages to that role are delivered to remaining subscribers. If all subscribers for a role disconnect, messages are silently dropped until a subscriber reconnects.

### Session Model

9. One training session at a time. The Trainer queues new goals.
10. A session ends when the curriculum is complete, the human sends "stop", or the harness shuts down.
11. The Trainer persists curriculum state for restart recovery.

### Threading

12. All message dispatch occurs on a single harness event loop.
13. The internal bus queue is thread-safe — any thread may send.
14. Kalvin's Cogitator thread calls the adapter directly; the adapter sends to the bus queue.

## Test Matrix

| ID      | Criterion                                                                  | Origin ref |
| ------- | -------------------------------------------------------------------------- | ---------- |
| HRNS-1  | Message bus routes by role to correct subscriber(s)                        | —          |
| HRNS-2  | Thread-safe: message sent from Cogitator thread arrives correctly          | —          |
| HRNS-3  | Unknown role (no subscribers) produces error response to sender            | —          |
| HRNS-4  | WebSocket client registers for a role; subsequent frames have implicit sender | —       |
| HRNS-5  | Harness loads embedded participants from config file on startup            | —          |
| HRNS-6  | Harness accepts WebSocket client connections                               | —          |
| HRNS-7  | Kalvin's adapter compiles KScript and submits entries one at a time        | —          |
| HRNS-8  | Kalvin's adapter sends compilation errors back to sender                   | —          |
| HRNS-9  | Kalvin's adapter maintains sender map; responses addressed to sender role  | —          |
| HRNS-10 | Kalvin's adapter handles countersign action                                | —          |
| HRNS-11 | Diagnostic listener receives all messages when subscribed to wildcard      | —          |
| HRNS-12 | Trainer auto-countersigns structurally matching proposals                  | —          |
| HRNS-13 | Trainer enters reactive mode on S2/S3 events                               | —          |
| HRNS-14 | Trainer escalates to role `supervisor` on budget exhaustion                | —          |
| HRNS-15 | Trainer persists curriculum state across harness restart                   | —          |
| HRNS-16 | Trainer accepts one session at a time; queues additional goals             | —          |
| HRNS-17 | Slack participant forwards human input via shared command parser           | —          |
| HRNS-18 | Slack participant renders all supervisor actions (progress, event, escalation, ratify_request) | — |
| HRNS-19 | Session pause: Trainer stops submitting but stays active                   | —          |
| HRNS-20 | Session stop: Trainer ends session, persists state, goes dormant           | —          |
| HRNS-21 | Disconnected client: messages delivered to remaining subscribers; silently dropped if no subscribers remain | — |
| HRNS-22 | Kalvin calls adapter directly (no internal EventBus)                       | —          |
| HRNS-23 | Single dispatch thread: all handlers execute on harness event loop         | —          |
| HRNS-24 | Trainer determines lesson completion by satisfaction count (len(satisfied) >= len(submitted)), not event count | —          |
| HRNS-25 | TUI participant renders all received harness events in EventLog            | —          |
| HRNS-25a | TUI participant sends input through shared command parser                    | —          |
| HRNS-26 | TUI participant sends free-form human input to Trainer as `input` action   | —          |
| HRNS-27 | TUI participant sends `countersign` to Kalvin on ratify action (button or `ratify` command) | — |
| HRNS-28 | TUI InputBar clears after successful send                                  | —          |
| HRNS-29 | All subscribers to a role receive messages sent to that role (fan-out)     | —          |
| HRNS-30 | Multiple WebSocket clients may register for the same role simultaneously   | —          |
| HRNS-31 | Trainer sends progress and escalation to role `supervisor`; all supervisor subscribers receive | — |
| HRNS-32 | Shared command parser maps free-text to structured commands                  | —          |
| HRNS-33 | Trainer relays Kalvin ground/frame/error events to role `supervisor`         | —          |
| HRNS-34 | Slack participant sends `countersign` via `ratify` command                   | —          |
| HRNS-35 | `process_s2_s3` returns `True` when auto-countersign succeeds (SAC-1)        | —          |
| HRNS-36 | `process_s2_s3` returns `False` when auto-countersign fails (SAC-1)          | —          |
| HRNS-37 | Reactive handling not called when auto-countersign succeeds (SAC-2)          | —          |
| HRNS-38 | `ratify_request` suppressed when auto-countersign succeeds (SAC-3)           | —          |
| HRNS-39 | `ratify_request` sent when auto-countersign fails AND proposal is a first sighting (SAC-4) | —          |
| HRNS-40 | Event relay sent regardless of auto-countersign outcome (SAC-5)             | —          |
| HRNS-41 | Kalvin's adapter handles `rationalise` action (live KValue + wire dict)      | §Kalvin Adapter |
| HRNS-42 | `rationalise` wire dict missing `significance` raises TypeError             | §Kalvin Adapter |
| HRNS-43 | `rationalise` does not call `countersign`/`submit`                           | §Kalvin Adapter |
| HRNS-44 | Second-sighting recurrence sends `rationalise` at declared SIG_S4 (SAC-2a)  | §Trainer |
| HRNS-45 | Recurrence does not invoke scaffolding (SAC-2a)                              | §Trainer |
| HRNS-46 | Recurrence counts toward the reactive budget (SAC-2a)                        | §Trainer |
| HRNS-47 | Structurally-different proposal is not recurrence (fresh first sighting)     | §Trainer |
| HRNS-48 | `load_lesson` clears the per-lesson seen-set (SAC-2b)                        | §Trainer |
| HRNS-49 | Pure-recurrence at budget cliff still escalates budget_exhaustion            | §Trainer |

## Out of Scope

- Dynamic participant registration at runtime (participants defined at startup only)
- Multiple concurrent training sessions
- TUI participant rendering internals (widget styling, layout)
- LLM agent prompt engineering and response parsing (see `@specs/trainer.md` TBD)
- Slack API integration details (webhook handling, rate limits, threading)
- Authentication or access control between participants
- Kalvin internal changes (rationalisation pipeline, Cogitator, significance)
