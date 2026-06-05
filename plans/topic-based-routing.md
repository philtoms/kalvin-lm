# Topic-Based Bus Routing Plan

**Status:** planning
**Spec refs:** `@specs/harness-server.md` — HRNS-1 through HRNS-31 (updated), especially HRNS-29, HRNS-30, HRNS-31
**Supersedes:** routing portions of `plans/implement-harness-server.md` Tasks 2, 3, 5, 6

## What Changes

The bus routing key changes from unique participant address to shared topic (participant type). This enables fan-out: multiple participants subscribe to the same topic and all receive messages. The concrete use case is TUI and Slack both subscribing to topic `ui`, both receiving progress and escalation messages.

The bus internals already support multiple handlers per key (`dict[str, list[Callable]]`). The work is: (a) rename `address` → `topic` everywhere, (b) change WebSocketProtocol from single-connection-per-key to multi-connection, (c) move Slack from its own topic (`"slack"`) to `"ui"`, (d) update config validation to allow duplicate topics.

## Spec References

- `@specs/harness-server.md` §Message — field renamed to `topic`
- `@specs/harness-server.md` §Message Bus — `subscribe(topic, ...)`, fan-out dispatch
- `@specs/harness-server.md` §Participant Protocol — `topic: str`
- `@specs/harness-server.md` §WebSocket Wire Protocol — multiple clients per topic
- `@specs/harness-server.md` §Trainer — escalation to topic `ui`
- `@specs/harness-server.md` §Slack Participant — registers as topic `ui`
- HRNS-29: fan-out dispatch
- HRNS-30: multiple WebSocket clients per topic
- HRNS-31: Trainer progress/escalation to `ui` topic

## Implementation Tasks

### Task T1: Message — rename `address` to `topic`

- **Spec ref:** @specs/harness-server.md §Message
- **Files:** `src/harness/message.py`
- **Details:**
  - Rename field `address` → `topic` on the `Message` dataclass
  - Update `__repr__` to show `topic` instead of `address`
  - Update docstring
- **Test mapping:** all tests constructing `Message(address=...)` must update to `Message(topic=...)`

### Task T2: MessageBus — rename `address` to `topic`

- **Spec ref:** @specs/harness-server.md §Message Bus — HRNS-1, HRNS-2, HRNS-3, HRNS-11, HRNS-23, HRNS-29
- **Files:** `src/harness/bus.py`
- **Details:**
  - Rename parameter `address` → `topic` in `subscribe()`
  - Update all internal references (`_handlers`, dispatch logic)
  - Update docstrings
  - No behavioural change — fan-out already works (multiple handlers per key)
  - Add new test: two handlers subscribed to same topic both receive a message (HRNS-29)
- **Test mapping:** HRNS-1, HRNS-2, HRNS-3, HRNS-11, HRNS-23, HRNS-29 → `tests/test_bus.py`

### Task T3: Participant protocol — rename `address` to `topic`

- **Spec ref:** @specs/harness-server.md §Participant Protocol
- **Files:** `src/harness/protocols.py`
- **Details:**
  - Rename `address: str` → `topic: str` on the `Participant` protocol
  - Update docstring
- **Downstream:** all classes implementing `Participant` must rename their `address` property to `topic`

### Task T4: WebSocketProtocol — multi-client per topic

- **Spec ref:** @specs/harness-server.md §WebSocket Wire Protocol — HRNS-4, HRNS-6, HRNS-21, HRNS-30
- **Files:** `src/harness/protocol.py`
- **Details:**
  - Change `self._connections: dict[str, ClientConnection]` → `dict[str, list[ClientConnection]]`
  - Change `self._reverse: dict[int, str]` stays the same (ws id → topic)
  - Change `self._participants: dict[str, _ClientParticipant]` stays the same (one participant wrapper per topic, not per connection)
  - Registration: allow multiple clients for the same topic. On registration, if topic already has a participant wrapper, reuse it. Add the new connection to the topic's connection list.
  - `_send_to_client_sync`: iterate all connections for the topic, send to each
  - Disconnect: remove the specific connection from the topic's list, keep the participant wrapper
  - Update `_serialise_message`: use `topic` instead of `address`
  - Update `_parse_message_frame`: use `topic` instead of `address`
  - Error frame: `"unknown topic: ..."`
- **Test mapping:** HRNS-4, HRNS-6, HRNS-21, HRNS-30 → `tests/test_protocol.py`

### Task T5: Harness Server — config uses `topic`, allows duplicates

- **Spec ref:** @specs/harness-server.md §Harness Configuration
- **Files:** `src/harness/server.py`
- **Details:**
  - `ParticipantConfig`: rename `address` → `topic`
  - `_validate_config`: allow duplicate topics (remove `seen_addresses` uniqueness check). Validate that duplicate topics only occur for `type: client` participants (embedded participants should have unique topics).
  - `_setup`: use `pcfg.topic` instead of `pcfg.address`
  - `HarnessServer.__init__`: update internal references
- **Test mapping:** HRNS-5, HRNS-30 → `tests/test_server.py`

### Task T6: KAgent Adapter — rename `address` to `topic`

- **Spec ref:** @specs/harness-server.md §Kalvin Adapter — HRNS-7, HRNS-8, HRNS-9, HRNS-10
- **Files:** `src/harness/adapter.py`
- **Details:**
  - Rename constructor parameter `address` → `topic`, default `"kalvin"`
  - Rename `self._address` → `self._topic`
  - Rename `address` property → `topic`
  - Update all `Message(address=...)` → `Message(topic=...)`
  - Update docstrings
  - Sender map stores topic strings instead of address strings
- **Test mapping:** HRNS-7, HRNS-8, HRNS-9, HRNS-10 → `tests/test_adapter.py`

### Task T7: Trainer — rename `address` to `topic`, escalation to `ui`

- **Spec ref:** @specs/harness-server.md §Trainer Participant — HRNS-14, HRNS-31
- **Files:** `src/trainer/trainer.py`
- **Details:**
  - Rename constructor parameter `address` → `topic`, default `"trainer"`
  - Rename `self._address` → `self._topic`
  - Rename `address` property → `topic`
  - Update `_emit_progress`: `Message(topic="ui", ...)`
  - Update `_emit_polling_status`: `Message(topic="ui", ...)`
  - Update `_submit_next_lesson`: `Message(topic="kalvin", ...)`, `sender=self._topic`
  - Update `bus.subscribe(self._topic, ...)`
- **Test mapping:** HRNS-12, HRNS-14, HRNS-16, HRNS-31 → `tests/test_trainer.py`, `tests/test_harness*.py`

### Task T8: Reactor — rename `address` to `topic`, escalation to `ui`

- **Spec ref:** @specs/harness-server.md §Trainer Participant — HRNS-14
- **Files:** `src/trainer/reactor.py`
- **Details:**
  - Rename constructor parameter `address` → `topic`, default `"trainer"`
  - Rename `self._address` → `self._topic`
  - `_auto_countersign`: `Message(topic="kalvin", ...)`, `sender=self._topic`
  - `_handle_reactive`: `Message(topic="kalvin", ...)`, `sender=self._topic`
  - `_escalate`: change `address="slack"` → `topic="ui"` (key behavioural change)
- **Test mapping:** HRNS-13, HRNS-14, HRNS-31 → `tests/test_reactor.py`

### Task T9: TUI Client — rename `address` to `topic`

- **Spec ref:** @specs/harness-server.md §TUI Participant — HRNS-25, HRNS-26, HRNS-27
- **Files:** `src/participants/tui_client.py`
- **Details:**
  - `HarnessClient`: rename `address` parameter → `topic`, default `"ui"`
  - Update registration frame: `{"register": self._topic}`
  - `send()` method: use `topic` parameter name
  - `TUIApp`: rename `address` → `topic`
  - `on_ratify_bar_ratify_clicked`: `self._client.send("kalvin", ...)`
  - `on_input_bar_submitted`: `self._client.send("trainer", ...)`
  - All `Message(address=...)` in outgoing frames → `Message(topic=...)` (frame key in JSON is `"topic"`)
- **Test mapping:** HRNS-25, HRNS-26, HRNS-27, HRNS-28 → `tests/test_tui_client.py`

### Task T10: Slack Agent — register as `ui`, rename `address` to `topic`

- **Spec ref:** @specs/harness-server.md §Slack Participant — HRNS-17, HRNS-18, HRNS-31
- **Files:** `src/participants/slack_agent.py`
- **Details:**
  - Registration: change `{"register": "slack"}` → `{"register": "ui"}`
  - `_send_to_trainer`: frame uses `"topic": "trainer"` instead of `"address": "trainer"`
  - Log message: update "registered as 'slack'" → "registered for topic 'ui'"
  - `_receive_loop`: update frame parsing for `"topic"` key
- **Test mapping:** HRNS-17, HRNS-18 → `tests/test_slack_agent.py`

### Task T11: Harness config (`harness.yaml`)

- **Spec ref:** @specs/harness-server.md §Harness Configuration
- **Files:** `harness.yaml`
- **Details:**
  - Rename `address` → `topic` on all participant entries
  - Change Slack entry from `topic: slack` to `topic: ui`
  - Result: both TUI and Slack have `topic: ui`

### Task T12: CLI entry point — rename `address` to `topic`

- **Spec ref:** @specs/harness-server.md §Harness Configuration
- **Files:** `src/harness/__main__.py`
- **Details:**
  - `_AlreadySubscribed`: rename `address` property → `topic`
  - Factories: rename parameter `address` → `topic`, pass to constructors
  - Update all Message constructions

### Task T13: Update all tests

- **Details:**
  - `tests/test_bus.py`: `Message(address=...)` → `Message(topic=...)`, handler parameter name
  - `tests/test_protocol.py`: registration frames, message frames, `address` → `topic`
  - `tests/test_adapter.py`: `KAgentAdapter(bus, address=...)` → `KAgentAdapter(bus, topic=...)`
  - `tests/test_trainer.py`: `Trainer(bus, ..., address=...)` → `Trainer(bus, ..., topic=...)`
  - `tests/test_reactor.py`: `Reactor(bus, state, address=...)` → `Reactor(bus, state, topic=...)`
  - `tests/test_tui_client.py`: `HarnessClient(url, address=...)` → `HarnessClient(url, topic=...)`
  - `tests/test_slack_agent.py`: verify registration frame uses `"ui"`
  - `tests/test_server.py`: `ParticipantConfig(address=...)` → `ParticipantConfig(topic=...)`
  - `tests/test_harness*.py`: update all `Message(address=...)` constructions
  - Add new tests:
    - HRNS-29: `test_fan_out_dispatch` — two handlers same topic both receive
    - HRNS-30: `test_multiple_clients_same_topic` — two WebSocket connections same topic
    - HRNS-31: `test_trainer_progress_to_all_ui_subscribers` — trainer sends progress, all `ui` subscribers receive

## Test Mapping

| Spec ID | Test file                 | Test function                            | Status |
| ------- | ------------------------- | ---------------------------------------- | ------ |
| HRNS-1  | tests/test_bus.py         | test_route_by_address → rename           | ⬜     |
| HRNS-2  | tests/test_bus.py         | test_threadsafe_send                     | ⬜     |
| HRNS-3  | tests/test_bus.py         | test_unknown_address_error → rename      | ⬜     |
| HRNS-4  | tests/test_protocol.py    | test_client_registration → update        | ⬜     |
| HRNS-5  | tests/test_server.py      | test_load_embedded_participants → update | ⬜     |
| HRNS-6  | tests/test_server.py      | test_websocket_client_connect            | ⬜     |
| HRNS-7  | tests/test_adapter.py     | test_submit_compiles_and_submits → update| ⬜     |
| HRNS-8  | tests/test_adapter.py     | test_compilation_error_response → update | ⬜     |
| HRNS-9  | tests/test_adapter.py     | test_sender_map_response_addressing → up | ⬜     |
| HRNS-10 | tests/test_adapter.py     | test_countersign_action → update         | ⬜     |
| HRNS-11 | tests/test_bus.py         | test_wildcard_diagnostic_listener        | ⬜     |
| HRNS-12 | tests/test_trainer.py     | test_auto_countersign → update           | ⬜     |
| HRNS-13 | tests/test_reactor.py     | test_reactive_mode → update              | ⬜     |
| HRNS-14 | tests/test_reactor.py     | test_escalation_to_ui_topic              | ⬜     |
| HRNS-15 | tests/test_harness_persistence.py | test_state_persistence → update | ⬜     |
| HRNS-16 | tests/test_trainer.py     | test_one_session_at_a_time → update      | ⬜     |
| HRNS-17 | tests/test_slack_agent.py | test_slack_forwards_human_input → update | ⬜     |
| HRNS-18 | tests/test_slack_agent.py | test_slack_renders_notify → update       | ⬜     |
| HRNS-19 | tests/test_trainer.py     | test_session_pause → update              | ⬜     |
| HRNS-20 | tests/test_trainer.py     | test_session_stop → update               | ⬜     |
| HRNS-21 | tests/test_protocol.py    | test_disconnect_silent_drop → update     | ⬜     |
| HRNS-22 | tests/test_adapter.py     | test_kagent_calls_adapter_directly       | ⬜     |
| HRNS-23 | tests/test_bus.py         | test_single_dispatch_thread              | ⬜     |
| HRNS-24 | tests/test_trainer.py     | test_entry_counting → update             | ⬜     |
| HRNS-25 | tests/test_tui_client.py  | test_renders_received_events → update    | ⬜     |
| HRNS-26 | tests/test_tui_client.py  | test_sends_freeform_input → update       | ⬜     |
| HRNS-27 | tests/test_tui_client.py  | test_sends_countersign → update          | ⬜     |
| HRNS-28 | tests/test_tui_client.py  | test_input_bar_clears → update           | ⬜     |
| HRNS-29 | tests/test_bus.py         | test_fan_out_dispatch                    | 🆕     |
| HRNS-30 | tests/test_protocol.py    | test_multiple_clients_same_topic         | 🆕     |
| HRNS-31 | tests/test_trainer.py     | test_progress_to_all_ui_subscribers      | 🆕     |

## Design Decisions

1. **`topic` not `type`** — `topic` is standard pub/sub terminology. `type` is a Python builtin and already used in the config for `embedded`/`client`. Avoids ambiguity and keyword conflict.

2. **Clean break, no migration shim** — rename `address` → `topic` everywhere in one pass. No backward compatibility layer. The project has no external API consumers.

3. **Escalation target changes from `"slack"` to `"ui"`** — escalation messages go to all `ui` subscribers. Slack renders `notify` actions; TUI shows them in the event log. The human sees escalation in whichever interface they're using.

4. **WebSocketProtocol: one `_ClientParticipant` wrapper per topic** — the bus-facing wrapper is shared across all connections for a topic. The wrapper iterates all active connections on `on_message`. This keeps bus subscription simple (one handler per topic) while supporting multiple connections.

5. **Duplicate topics allowed for client participants only** — embedded participants should have unique topics (one Kalvin, one Trainer). Client participants may share topics (multiple UIs). Validation in config loading enforces this.

6. **Sender is the sender's topic** — when Trainer sends to Kalvin, `sender="trainer"`. Kalvin responds to topic `"trainer"`. This works because each embedded participant has a unique topic. For client participants, the sender is the topic they registered for (e.g. `"ui"`).

## Build Order

```
T1: Message (rename field)
  │
  ├── T2: MessageBus (rename param)
  │
  └── T3: Participant protocol (rename field)
        │
        ├── T4: WebSocketProtocol (multi-client)
        │     │
        │     ├── T9: TUI Client
        │     └── T10: Slack Agent
        │
        ├── T5: Harness Server (config)
        │
        ├── T6: KAgent Adapter
        │
        ├── T7: Trainer
        │     │
        │     └── T8: Reactor
        │
        └── T12: CLI entry point
              │
              └── T11: harness.yaml

T13: Update all tests (interleaved with each task)
```

**Recommended approach:** T1→T2→T3 first (core renames), then T4→T5→T6→T7→T8 (server-side), then T9→T10 (clients), then T11→T12 (config/CLI). T13 runs throughout.

## Status

Planning. Pending cascade Step 3 (consistency check) and Step 4 (commit).
