# Multi-Agent Harness Implementation Plan

**Parent:** [`docs/roadmap.md`](../../docs/roadmap.md) Phase D
**Status:** complete
**Spec refs:** `specs/harness-server.md`
**Depends on:** Phase A+ (structural grounding, extended cogitation), Phase B (current TUI harness)

## Spec References

- `@specs/harness-server.md` — HRNS-1 through HRNS-24
- `@specs/agent.md` — KAgent rationalisation API, events (AGT-1..AGT-40)
- `@specs/kscript.md` — compilation pipeline (KS-1..KS-33)
- `CONTEXT.md` — Agents and the Harness section

## Implementation Tasks

### Task 1: Rename Agent to KAgent (`src/kalvin/agent.py`)

- **Spec ref:** @CONTEXT.md §KAgent
- **Test mapping:** Existing AGT tests unchanged — rename only
- **Details:**
  - Rename class `Agent` → `KAgent`
  - Remove internal `EventBus`; accept an adapter callback in constructor
  - Replace all `self._event_bus.publish(event)` with `self._adapter.on_event(event)`
  - Update all imports across `src/` and `tests/`
  - `CogitationHandler` protocol stays — the adapter implements it

### Task 2: Message Bus (`src/training/harness/bus.py`)

- **Spec ref:** @specs/harness-server §Message Bus — HRNS-1, HRNS-2, HRNS-3, HRNS-11, HRNS-23
- **Test mapping:** HRNS-1, HRNS-2, HRNS-3, HRNS-11, HRNS-23
- **Details:**
  - Thread-safe queue (e.g. `queue.Queue`)
  - `subscribe(address, handler)` — register handler for address
  - `subscribe("*", handler)` — wildcard for diagnostic listeners
  - `send(Message)` — enqueue message
  - `run()` — event loop: dequeue, dispatch to handler(s) for address
  - Unknown address → send error back to sender (requires sender context on Message)

### Task 3: Message dataclass (`src/training/harness/message.py`)

- **Spec ref:** @specs/harness-server §Message
- **Details:**
  - `Message(address: str, action: str, message: Any, sender: Optional[str] = None)`
  - `sender` is set by the bus on inbound messages (derived from registration)
  - Immutable dataclass

### Task 4: KAgent Adapter (`src/training/harness/adapter.py`)

- **Spec ref:** @specs/harness-server §KAgent Adapter — HRNS-7, HRNS-8, HRNS-9, HRNS-10, HRNS-22
- **Test mapping:** HRNS-7, HRNS-8, HRNS-9, HRNS-10, HRNS-22
- **Details:**
  - Implements `Participant` protocol (`on_message`)
  - Implements `CogitationHandler` protocol (`on_s1`, `on_expansion`)
  - Holds reference to KAgent and MessageBus
  - Sender map: `dict[EntryKey, str]` — maps submitted entries to sender address
  - `on_message(msg)`:
    - `action == "submit"`: compile KScript via `compile_source()`, submit each entry to `kagent.rationalise()`, record sender per entry
    - `action == "countersign"`: call `kagent.countersign(kline)`
  - `on_event(event)`: wrap `RationaliseEvent` into `Message`, look up sender, send via bus
  - Compilation errors: send error message back to sender
  - KAgent calls `adapter.on_event()` directly — no internal EventBus

### Task 5: Harness Server (`src/training/harness/server.py`)

- **Spec ref:** @specs/harness-server §Harness Configuration, §Participant Lifecycle — HRNS-5, HRNS-6
- **Test mapping:** HRNS-5, HRNS-6
- **Details:**
  - Load YAML/JSON config file listing participants
  - Instantiate embedded participants, wire to bus
  - Start WebSocket server for client participants
  - Start bus event loop on main thread
  - Graceful shutdown: persist Trainer state, disconnect clients

### Task 6: WebSocket Protocol (`src/training/harness/protocol.py`)

- **Spec ref:** @specs/harness-server §WebSocket Wire Protocol — HRNS-4, HRNS-6, HRNS-21
- **Test mapping:** HRNS-4, HRNS-6, HRNS-21
- **Details:**
  - WebSocket server using `websockets` or `aiohttp`
  - On connect: expect `{"register": "<address>"}` frame
  - After registration: inbound frames are messages with implicit sender
  - Outbound: forward bus messages to client as JSON frames
  - Disconnect: mark address as disconnected; silently drop messages until reconnect
  - JSON frame format: `{"address": "...", "action": "...", "message": ...}`

### Task 7: Trainer — Curriculum Execution (`src/training/trainer/trainer.py`)

- **Spec ref:** @specs/harness-server §Trainer Participant — HRNS-12, HRNS-13, HRNS-15, HRNS-16, HRNS-19, HRNS-20, HRNS-24
- **Test mapping:** HRNS-12, HRNS-13, HRNS-15, HRNS-16, HRNS-19, HRNS-20, HRNS-24
- **Details:**
  - Implements `Participant` protocol (`on_message`)
  - Holds: current session, curriculum, submitted/satisfied/pending sets, event log
  - `on_message(msg)`:
    - From `kalvin`: process KAgent event
      - Fast path (S1): auto-satisfy, advance curriculum if lesson complete
      - Slow path (S2/S3): enter reactive mode
      - Compilation error: log and escalate
    - From `slack` (`action: input`): interpret via LLM agent
      - Goal + KScript: start new session
      - Guidance: generate reactive scaffolding
      - "pause": stop submitting, stay active
      - "stop": end session, persist, go dormant
  - Curriculum-driven mode: send next lesson to `{address: kalvin, action: submit, message: <kscript>}`
  - Reactive mode: cogitate via LLM agent, send scaffolding to kalvin
  - Escalation: send `{address: slack, action: notify, message: ...}`
  - Ratification: auto-countersign on structural match, send `{address: kalvin, action: countersign, message: <kline>}`
  - Entry counting: tracks how many entries submitted per lesson, counts events until all received

### Task 8: Trainer — Curriculum State (`src/training/trainer/curriculum.py`)

- **Spec ref:** @specs/harness-server §Trainer Participant — HRNS-15, HRNS-16
- **Test mapping:** HRNS-15, HRNS-16
- **Details:**
  - `Curriculum`: ordered list of lessons (KScript source strings), current position
  - `CurriculumState`: current position, submitted/satisfied/pending sets, event log
  - `save(path)` / `load(path)`: JSON persistence for restart recovery
  - Session management: one active session, queue for pending goals

### Task 9: Trainer — LLM agent Integration (`src/training/trainer/cogitation.py`)

- **Spec ref:** @specs/harness-server §Trainer Participant — HRNS-13, HRNS-14
- **Test mapping:** HRNS-13, HRNS-14
- **Details:**
  - Generates prompts for LLM agent with:
    - Current event context (misfit diagnosis, expectation vs proposal)
    - Curriculum context (what the Trainer is trying to teach)
    - Conversation history (previous human guidance)
  - Sets up tool calling for scaffolding extraction
  - Calls LLM API
  - Extracts and compiles recommended scaffolding
  - Returns confidence level alongside scaffolding
  - Low confidence → trigger escalation

### Task 10: Slack Participant (`src/training/supervisors/slack_agent.py`)

- **Spec ref:** @specs/harness-server §Slack Participant — HRNS-17, HRNS-18
- **Test mapping:** HRNS-17, HRNS-18
- **Details:**
  - WebSocket client connecting to harness server
  - Registers as `"slack"` on connect
  - `on_message(msg)` with `action: notify`: render to Slack channel via Slack API
  - Slack webhook/event listener: forward human input as `{address: trainer, action: input, message: <text>}`
  - Dedicated training channel; no addressing syntax

### Task 11: TUI Participant (`src/training/supervisors/tui_client.py`, `src/training/supervisors/tui_regions.py`)

- **Spec ref:** @specs/harness-server §TUI Participant — HRNS-25, HRNS-26, HRNS-27, HRNS-28
- **Test mapping:** HRNS-25, HRNS-26, HRNS-27, HRNS-28
- **Details:**
  - WebSocket client connecting to harness server
  - Registers as `"ui"` on connect
  - **EventLog** — renders all received harness events (timestamp, action, message summary)
  - **InputBar** — text input + Send button for free-form messages to Trainer. On submit sends `{address: "trainer", action: "input", message: <text>}`, clears the field
  - **RatifyBar** — ratification button (disabled by default), enables on KAgent proposal events. Sends `{address: "kalvin", action: "countersign", message: <event_data>}`
  - Keyboard shortcuts: `ctrl+q` quit, `ctrl+r` ratify, `ctrl+s` send input
  - Parity with Slack participant: human input routes to Trainer as `input` action

### Task 12: Harness Configuration and Entry Point (`src/training/harness/__main__.py`)

- **Spec ref:** @specs/harness-server §Harness Configuration
- **Details:**
  - CLI entry point: `python -m training.harness --config training.harness.yaml`
  - Load config, instantiate server, start
  - Signal handling: SIGTERM → persist Trainer state, graceful shutdown

## Test Mapping

| Spec ID | Test file                         | Test function                                                          | Status |
| ------- | --------------------------------- | ---------------------------------------------------------------------- | ------ |
| HRNS-1  | tests/test_bus.py                 | test_route_by_address                                                  | ⬜     |
| HRNS-2  | tests/test_bus.py                 | test_threadsafe_send                                                   | ⬜     |
| HRNS-3  | tests/test_bus.py                 | test_unknown_address_error                                             | ⬜     |
| HRNS-4  | tests/test_protocol.py            | test_client_registration                                               | ⬜     |
| HRNS-5  | tests/test_server.py              | test_load_embedded_participants                                        | ⬜     |
| HRNS-6  | tests/test_server.py              | test_websocket_client_connect                                          | ⬜     |
| HRNS-7  | tests/test_adapter.py             | test_submit_compiles_and_submits                                       | ⬜     |
| HRNS-8  | tests/test_adapter.py             | test_compilation_error_response                                        | ⬜     |
| HRNS-9  | tests/test_adapter.py             | test_sender_map_response_addressing                                    | ⬜     |
| HRNS-10 | tests/test_adapter.py             | test_countersign_action                                                | ⬜     |
| HRNS-11 | tests/test_bus.py                 | test_wildcard_diagnostic_listener                                      | ⬜     |
| HRNS-12 | tests/test_trainer.py             | test_auto_countersign_structural_match                                 | ⬜     |
| HRNS-13 | tests/test_trainer.py             | test_reactive_mode_on_s2_s3                                            | ⬜     |
| HRNS-14 | tests/test_trainer.py             | test_escalation_on_budget_exhaustion                                   | ⬜     |
| HRNS-15 | tests/test_curriculum.py          | test_state_persistence_across_restart                                  | ⬜     |
| HRNS-16 | tests/test_trainer.py             | test_one_session_at_a_time                                             | ⬜     |
| HRNS-17 | tests/test_slack_agent.py         | test_slack_forwards_human_input                                        | ✅     |
| HRNS-18 | tests/test_slack_agent.py         | test_slack_renders_notify                                              | ✅     |
| HRNS-19 | tests/test_trainer.py             | test_session_pause                                                     | ⬜     |
| HRNS-20 | tests/test_trainer.py             | test_session_stop                                                      | ⬜     |
| HRNS-21 | tests/test_protocol.py            | test_disconnect_silent_drop                                            | ⬜     |
| HRNS-22 | tests/test_adapter.py             | test_kagent_calls_adapter_directly                                     | ⬜     |
| HRNS-23 | tests/test_bus.py                 | test_single_dispatch_thread                                            | ⬜     |
| HRNS-24 | tests/test_trainer.py             | test_entry_counting_lesson_complete                                    | ⬜     |
| HRNS-25 | tests/test_tui_client.py          | test_renders_received_events                                           | ✅     |
| HRNS-26 | tests/test_tui_client.py          | test_sends_freeform_input_to_trainer                                   | ✅     |
| HRNS-27 | tests/test_tui_client.py          | test_sends_countersign_on_ratify                                       | ✅     |
| HRNS-28 | tests/test_tui_client.py          | test_input_bar_clears_after_send                                       | ✅     |
| HRNS-35 | tests/test_s3_auto_countersign.py | `test_returns_true_on_auto_countersign`                                | ✅     |
| HRNS-36 | tests/test_s3_auto_countersign.py | `test_returns_false_on_no_match`                                       | ✅     |
| HRNS-37 | tests/test_s3_auto_countersign.py | `test_no_escalation_on_auto_countersign`                               | ✅     |
| HRNS-38 | tests/test_s3_auto_countersign.py | `test_ratify_suppressed_on_auto_countersign`                           | ✅     |
| HRNS-39 | tests/test_s3_auto_countersign.py | `test_ratify_sent_when_auto_countersign_fails`                         | ✅     |
| HRNS-40 | tests/test_s3_auto_countersign.py | `test_relay_on_auto_countersign` + `test_relay_on_no_auto_countersign` | ✅     |

## Design Decisions

1. **Renamed Agent → KAgent** — disambiguates from the broader multi-agent sense. Internal rename only; all existing tests continue to pass.

2. **Removed internal EventBus** — the KAgent calls the adapter directly. The adapter is responsible for routing events onto the harness bus. Eliminates a layer of indirection and makes the KAgent's output path explicit.

3. **Addressed message bus with single dispatch thread** — thread-safe queue + single event loop avoids concurrency bugs while allowing the Cogitator thread to emit events freely.

4. **Adapter as the KAgent's output interface** — the adapter implements the callback protocol the KAgent needs. The KAgent remains unaware of the harness.

5. **Trainer absorbs harness tracking state** — submitted, satisfied, and pending sets move from the TUI to the Trainer. The TUI becomes a thin rendering client.

6. **KScript source as the message format for KAgent submissions** — keeps messages human-readable for diagnostics. Compilation is the adapter's responsibility.

7. **WebSocket for client participants** — full duplex, persistent connections, natural for streaming events.

8. **Static participant configuration** — loaded at startup from YAML/JSON. Dynamic registration is out of scope for MVP.

9. **LLM agent details deferred** — the Trainer's cogitation module defines the interface (prompt → scaffolding + confidence) but the prompt engineering and tool calling specifics will be designed in a separate grill session.

10. **One session at a time** — prevents interleaving confusion for the KAgent and simplifies the Trainer's counting mechanism.

11. **`process_s2_s3` returns `bool`, guard in Trainer.** The reactor only
    needs to communicate two states — auto-countersign succeeded or not —
    so a `bool` is the simplest type. The decision to suppress
    `ratify_request` belongs to the Trainer (the component that sends the
    message); the reactor reports what happened and the Trainer decides
    what to do about it. Event relay stays unconditional to preserve
    observability.

## Build Order

```
Task 1: KAgent rename + EventBus removal
  │
  ├── Task 2: Message Bus
  │     │
  │     └── Task 3: Message dataclass
  │           │
  │           └── Task 4: KAgent Adapter
  │                 │
  │                 └── Task 7: Trainer (core)
  │                       │
  │                       ├── Task 8: Curriculum state
  │                       └── Task 9: LLM agent integration
  │
  ├── Task 5: Harness Server
  │     │
  │     └── Task 6: WebSocket Protocol
  │           │
  │           ├── Task 10: Slack Participant
  │           └── Task 11: TUI Participant
  │
  └── Task 12: Config + CLI entry point
```

**Parallelizable:** Tasks 2–3 (bus/message) and Task 5–6 (server/protocol) can proceed in parallel after Task 1. Task 10 and 11 are independent of each other.

## Estimates

| Task      | Component                  | Estimate       | Risk                              |
| --------- | -------------------------- | -------------- | --------------------------------- |
| 1         | KAgent rename              | 0.5 day        | Low                               |
| 2–3       | Message bus + dataclass    | 1 day          | Low                               |
| 4         | KAgent adapter             | 1.5 days       | Medium (sender map, threading)    |
| 5–6       | Harness server + WebSocket | 2 days         | Medium (async, WebSocket library) |
| 7–8       | Trainer core + curriculum  | 2–3 days       | Medium (state machine, counting)  |
| 9         | LLM agent integration      | 2–3 days       | High (prompt quality unknown)     |
| 10        | Slack participant          | 1–2 days       | Low                               |
| 11        | TUI participant            | 1–2 days       | Low                               |
| 12        | Config + entry point       | 0.5 day        | Low                               |
| **Total** |                            | **12–17 days** |                                   |

## Status

**Complete.** All 12 tasks implemented.
