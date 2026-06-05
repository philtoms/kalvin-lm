# Harness Server Specification

## Overview

The Harness Server is a multi-agent runtime that loads participants, routes messages between them by topic, and manages the dialogue loop. Multiple participants may subscribe to the same topic and all receive messages sent to that topic (fan-out). The harness runs as a persistent server — participants connect as clients (via WebSocket) or are embedded in-process. The harness itself is not a participant; it is the message broker.

## Dependencies

- `specs/agent.md` — Kalvin rationalisation API, events, Cogitator
- `specs/kscript.md` — compilation pipeline (KScript source → CompiledEntry)
- `specs/harness.md` — tracking state, satisfaction, ratification (absorbed into Trainer)

## Definitions

### Harness Configuration

A YAML or JSON file loaded at startup. Lists participants with their topic and connection type (embedded or client). Multiple participants may share the same topic — all receive messages sent to that topic.

```yaml
participants:
  - topic: kalvin
    type: embedded
    class: KAgent
  - topic: trainer
    type: embedded
    class: Trainer
  - topic: ui
    type: client
    class: TUIParticipant
  - topic: ui
    type: client
    class: SlackParticipant
```

### Message

A unit of inter-participant communication. Routed by topic.

```
Message:
  topic:    str       # recipient topic (participant type)
  action:   str       # interpreted by recipient
  message:  any       # payload (KScript source, RationaliseEvent, freeform text, etc.)
```

The harness does not interpret `action` or `message` — it routes by `topic` only.

### Message Bus

A thread-safe topic-based message router with fan-out dispatch. Participants subscribe by topic. Messages are dispatched to all subscribers of the target topic on a single harness event loop via an internal queue.

```
MessageBus:
  subscribe(topic, handler) → None
  send(Message) → None
  run() → None              # event loop: dequeue and dispatch
```

- Thread-safe: any participant may call `send()` from any thread.
- Fan-out: all handlers subscribed to a topic receive messages sent to that topic.
- Single dispatch: all handler invocations execute on the harness event loop thread.
- Diagnostic listeners may subscribe to a special wildcard topic to observe all messages.

### Participant Protocol

Every participant (embedded or client) implements:

```
Participant:
  topic: str
  on_message(Message) → None     # receive a message from the bus
```

Embedded participants are called directly. Client participants receive messages via their WebSocket connection.

### WebSocket Wire Protocol

Connected clients communicate over WebSocket. JSON frames.

**Registration** (client → server, once on connect):

```json
{ "register": "ui" }
```

Multiple clients may register for the same topic. All receive messages sent to that topic.

**Message** (bidirectional):

```json
{ "topic": "kalvin", "action": "submit", "message": "MHALL = SVO" }
```

**Error** (server → client):

```json
{ "error": "unknown topic: foo" }
```

After registration, all frames on that socket are implicitly from the registered topic. The harness maps sockets to topics internally, allowing multiple sockets per topic.

### Kalvin Adapter

Kalvin's interface to the harness bus. A thin layer that:

1. Receives harness messages sent to topic `kalvin`.
2. Interprets the `action`:
   - `submit` — compile KScript source via the KScript pipeline, submit each compiled entry to `kagent.rationalise()` one at a time.
   - `countersign` — call `kagent.countersign(kline)` with the provided kline.
3. Receives Kalvin's callbacks directly (no internal EventBus) and wraps them into harness messages dispatched to the original sender's topic.
4. Maintains a sender map: when entries arrive from topic X, the adapter records X as the sender. Kalvin's callbacks are sent to topic X.

### Adapter Callback Interface

Kalvin calls the adapter directly instead of publishing to an internal EventBus:

```
KAgentAdapterCallback:
  on_event(event: RationaliseEvent) → None
```

Kalvin is constructed with an adapter instance. The adapter's `on_event` wraps the event into a message and sends it to the sender's topic via the bus.

### Compilation Errors

If a `submit` action contains KScript that fails to compile, the adapter sends an error message back to the sender:

```json
{
  "topic": "trainer",
  "action": "error",
  "message": "ParseError at line 2: unexpected token"
}
```

### Trainer Participant

An embedded participant that drives the training loop. See `@specs/trainer.md` (TBD) for full specification.

Key behaviours:

- **Curriculum-driven mode**: submits the next lesson from the curriculum to Kalvin.
- **Reactive mode**: cogitates (via LLM agent) on S2/S3 events and generates reactive scaffolding.
- **Ratification**: auto-countersigns proposals that structurally match expectations.
- **Escalation**: sends messages to topic `ui` when stuck (budget exhaustion or low LLM agent confidence). All participants subscribed to topic `ui` receive escalation notifications.
- **Session management**: one training session at a time. Supports pause and stop via human input.
- **State persistence**: curriculum position and event log persisted for restart recovery.

Trainer maintains per-session submitted, satisfied, and pending sets — absorbing the tracking state from the current harness spec (`@specs/harness.md`).

### Slack Participant

A client participant that translates between Slack API and harness messages. Registers as topic `ui`, receiving the same messages as the TUI participant.

**Actions received from the bus (via topic `ui`):**

- `notify` — render message content for the human in Slack.
- Other actions (e.g. `progress`) are silently ignored by the Slack participant.

**Actions sent to the bus:**

- `input` — forward human Slack message to the Trainer (topic `trainer`).

Uses a dedicated Slack channel. No addressing syntax needed — all human messages route to the Trainer.

### TUI Participant

A client participant that renders events for the human, provides ratification controls, and allows the human to compose and send free-form text messages to the Trainer. Connects to the harness server via WebSocket. Registers as topic `ui`. Does not own Kalvin or make curriculum decisions.

**Actions received from the bus (via topic `ui`):**

- `notify` — render message content in the event log for the human.
- Any other action — render in the event log (topic, action, and message summary).

**Actions sent to the bus:**

- `input` — forward human-composed free-form text to the Trainer (`{topic: "trainer", action: "input", message: <text>}`).
- `countersign` — ratify a Kalvin proposal (`{topic: "kalvin", action: "countersign", message: <event_data>}`).

**UI regions:**

- **EventLog** — scrollable log displaying all received harness events (timestamp, action, message summary).
- **InputBar** — text input field where the human types free-form messages. Pressing Enter or a Send button dispatches the text as `{topic: "trainer", action: "input", message: <text>}` via the WebSocket connection.
- **RatifyBar** — button (disabled by default) that becomes active when a Kalvin proposal event arrives. Clicking it sends a `countersign` action to Kalvin.
- **Header / Footer** — app title and keyboard shortcut hints (`ctrl+q` quit, `ctrl+r` ratify, `ctrl+s` send input).

**Input behaviour:**

- The InputBar accepts single-line free-form text.
- On submit (Enter key or Send button), the text is sent as an `input` action addressed to the Trainer, then the input field is cleared.
- This mirrors the Slack participant's `_send_to_trainer` behaviour — no addressing syntax is needed; all human input routes to the Trainer.

## Behavioural Rules

### Message Routing

1. Messages are routed by `topic` only. The harness does not inspect `action` or `message`.
2. A message sent to an unknown topic (no subscribers) results in an error sent back to the sender.
3. Kalvin's adapter always addresses responses to the original sender's topic.
4. All subscribers registered for a topic receive messages sent to that topic (fan-out dispatch).

### Participant Lifecycle

5. Embedded participants are instantiated on harness startup from the configuration file.
6. Client participants connect via WebSocket and register for a topic.
7. Multiple WebSocket clients may register for the same topic. All receive messages for that topic.
8. A client that disconnects does not remove the topic's subscription. Messages to that topic are delivered to remaining subscribers. If all subscribers for a topic disconnect, messages are silently dropped until a subscriber reconnects.

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
| HRNS-1  | Message bus routes by topic to correct subscriber(s)                       | —          |
| HRNS-2  | Thread-safe: message sent from Cogitator thread arrives correctly          | —          |
| HRNS-3  | Unknown topic (no subscribers) produces error response to sender           | —          |
| HRNS-4  | WebSocket client registers for a topic; subsequent frames have implicit sender | —      |
| HRNS-5  | Harness loads embedded participants from config file on startup            | —          |
| HRNS-6  | Harness accepts WebSocket client connections                               | —          |
| HRNS-7  | Kalvin's adapter compiles KScript and submits entries one at a time        | —          |
| HRNS-8  | Kalvin's adapter sends compilation errors back to sender                   | —          |
| HRNS-9  | Kalvin's adapter maintains sender map; responses addressed to sender topic | —          |
| HRNS-10 | Kalvin's adapter handles countersign action                                | —          |
| HRNS-11 | Diagnostic listener receives all messages when subscribed to wildcard      | —          |
| HRNS-12 | Trainer auto-countersigns structurally matching proposals                  | —          |
| HRNS-13 | Trainer enters reactive mode on S2/S3 events                               | —          |
| HRNS-14 | Trainer escalates to topic `ui` on budget exhaustion                       | —          |
| HRNS-15 | Trainer persists curriculum state across harness restart                   | —          |
| HRNS-16 | Trainer accepts one session at a time; queues additional goals             | —          |
| HRNS-17 | Slack participant forwards human input to Trainer (topic `trainer`)        | —          |
| HRNS-18 | Slack participant renders notify messages for human                        | —          |
| HRNS-19 | Session pause: Trainer stops submitting but stays active                   | —          |
| HRNS-20 | Session stop: Trainer ends session, persists state, goes dormant           | —          |
| HRNS-21 | Disconnected client: messages delivered to remaining subscribers; silently dropped if no subscribers remain | — |
| HRNS-22 | Kalvin calls adapter directly (no internal EventBus)                       | —          |
| HRNS-23 | Single dispatch thread: all handlers execute on harness event loop         | —          |
| HRNS-24 | Trainer counts submitted entries; knows when lesson is complete            | —          |
| HRNS-25 | TUI participant renders all received harness events in EventLog            | —          |
| HRNS-26 | TUI participant sends free-form human input to Trainer as `input` action   | —          |
| HRNS-27 | TUI participant sends `countersign` to Kalvin on ratify action             | —          |
| HRNS-28 | TUI InputBar clears after successful send                                  | —          |
| HRNS-29 | All subscribers to a topic receive messages sent to that topic (fan-out)   | —          |
| HRNS-30 | Multiple WebSocket clients may register for the same topic simultaneously  | —          |
| HRNS-31 | Trainer sends progress and escalation to topic `ui`; all `ui` subscribers receive both | — |

## Out of Scope

- Dynamic participant registration at runtime (participants defined at startup only)
- Multiple concurrent training sessions
- TUI participant rendering internals (widget styling, layout)
- LLM agent prompt engineering and response parsing (see `@specs/trainer.md` TBD)
- Slack API integration details (webhook handling, rate limits, threading)
- Authentication or access control between participants
- Kalvin internal changes (rationalisation pipeline, Cogitator, significance)
