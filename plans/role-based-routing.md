# Role-Based Bus Routing Plan

**Status:** planning
**Spec refs:** `@specs/harness-server.md` — HRNS-1 through HRNS-34
**Supersedes:** `plans/topic-based-routing.md` (renamed), routing portions of `plans/implement-harness-server.md` Tasks 2, 3, 5, 6

## What Changes

The bus routing key changes from unique participant address to shared role. This enables fan-out: multiple participants subscribe to the same role and all receive messages. The concrete use case is TUI and Slack both subscribing to role `supervisor`, both receiving progress, event relay, escalation, and ratify request messages.

The bus internals already support multiple handlers per key (`dict[str, list[Callable]]`). The work is: (a) rename `address` → `role` everywhere, (b) change WebSocketProtocol from single-connection-per-key to multi-connection, (c) move Slack from its own role (`"slack"`) to `"supervisor"`, (d) update config validation to allow duplicate roles, (e) add shared command protocol, (f) add Kalvin event relay to supervisors, (g) add ratification alignment for both participants.

The role names reflect the training session domain: `trainee` (Kalvin), `trainer` (Trainer), `supervisor` (TUI, Slack, future AI agents).

## Spec References

- `@specs/harness-server.md` §Message — field named `role`
- `@specs/harness-server.md` §Message Bus — `subscribe(role, ...)`, fan-out dispatch
- `@specs/harness-server.md` §Participant Protocol — `role: str`
- `@specs/harness-server.md` §WebSocket Wire Protocol — multiple clients per role
- `@specs/harness-server.md` §Trainer — escalation to role `supervisor`, event relay
- `@specs/harness-server.md` §Supervisor Participant — role definition, shared command protocol
- `@specs/harness-server.md` §Slack Participant — registers as role `supervisor`
- `@specs/harness-server.md` §TUI Participant — registers as role `supervisor`
- HRNS-29: fan-out dispatch
- HRNS-30: multiple WebSocket clients per role
- HRNS-31: Trainer progress/escalation to `supervisor` role
- HRNS-32: shared command parser
- HRNS-33: Trainer relays Kalvin events to `supervisor` role
- HRNS-34: Slack sends `countersign` via `ratify` command

## Implementation Tasks

### Task T1: Message — rename `address` to `role`

- **Spec ref:** @specs/harness-server.md §Message
- **Files:** `src/harness/message.py`
- **Details:**
  - Rename field `address` → `role` on the `Message` dataclass
  - Update `__repr__` to show `role` instead of `address`
  - Update docstring
- **Test mapping:** all tests constructing `Message(address=...)` must update to `Message(role=...)`

### Task T1a: Introduce role constants

- **Spec ref:** @specs/harness-server.md §Supervisor Participant, §Harness Configuration
- **Files:** `src/harness/constants.py` (new)
- **Details:**
  - Define role constants as single source of truth:
    - `TRAINEE_ROLE = "trainee"`
    - `TRAINER_ROLE = "trainer"`
    - `SUPERVISOR_ROLE = "supervisor"`
  - All modules reference these constants instead of string literals
- **Test mapping:** no direct test — consumed by T4, T6, T7, T8, T9, T10

### Task T2: MessageBus — rename `address` to `role`

- **Spec ref:** @specs/harness-server.md §Message Bus — HRNS-1, HRNS-2, HRNS-3, HRNS-11, HRNS-23, HRNS-29
- **Files:** `src/harness/bus.py`
- **Details:**
  - Rename parameter `address` → `role` in `subscribe()`
  - Update all internal references (`_handlers`, dispatch logic)
  - Update docstrings
  - No behavioural change — fan-out already works (multiple handlers per key)
  - Add new test: two handlers subscribed to same role both receive a message (HRNS-29)
- **Test mapping:** HRNS-1, HRNS-2, HRNS-3, HRNS-11, HRNS-23, HRNS-29 → `tests/test_bus.py`

### Task T3: Participant protocol — rename `address` to `role`

- **Spec ref:** @specs/harness-server.md §Participant Protocol
- **Files:** `src/harness/protocols.py`
- **Details:**
  - Rename `address: str` → `role: str` on the `Participant` protocol
  - Update docstring
- **Downstream:** all classes implementing `Participant` must rename their `address` property to `role`

### Task T4: WebSocketProtocol — multi-client per role

- **Spec ref:** @specs/harness-server.md §WebSocket Wire Protocol — HRNS-4, HRNS-6, HRNS-21, HRNS-30
- **Files:** `src/harness/protocol.py`
- **Details:**
  - Change `self._connections: dict[str, ClientConnection]` → `dict[str, list[ClientConnection]]`
  - Change `self._reverse: dict[int, str]` stays the same (ws id → role)
  - Change `self._participants: dict[str, _ClientParticipant]` stays the same (one participant wrapper per role, not per connection)
  - Registration: allow multiple clients for the same role. On registration, if role already has a participant wrapper, reuse it. Add the new connection to the role's connection list.
  - `_send_to_client_sync`: iterate all connections for the role, send to each
  - Disconnect: remove the specific connection from the role's list, keep the participant wrapper
  - Update `_serialise_message`: use `role` instead of `address`
  - Update `_parse_message_frame`: use `role` instead of `address`
  - Error frame: `"unknown role: ..."`
- **Test mapping:** HRNS-4, HRNS-6, HRNS-21, HRNS-30 → `tests/test_protocol.py`

### Task T5: Harness Server — config uses `role`, allows duplicates

- **Spec ref:** @specs/harness-server.md §Harness Configuration
- **Files:** `src/harness/server.py`
- **Details:**
  - `ParticipantConfig`: rename `address` → `role`
  - `_validate_config`: allow duplicate roles (remove `seen_addresses` uniqueness check). Validate that duplicate roles only occur for `type: client` participants (embedded participants should have unique roles).
  - `_setup`: use `pcfg.role` instead of `pcfg.address`
  - `HarnessServer.__init__`: update internal references
- **Test mapping:** HRNS-5, HRNS-30 → `tests/test_server.py`

### Task T6: KAgent Adapter — rename `address` to `role`, default `"trainee"`

- **Spec ref:** @specs/harness-server.md §Kalvin Adapter — HRNS-7, HRNS-8, HRNS-9, HRNS-10
- **Files:** `src/harness/adapter.py`
- **Details:**
  - Rename constructor parameter `address` → `role`, default `"trainee"` (was `"kalvin"`)
  - Rename `self._address` → `self._role`
  - Rename `address` property → `role`
  - Update all `Message(address=...)` → `Message(role=...)`
  - Update docstrings
  - Sender map stores role strings instead of address strings
- **Test mapping:** HRNS-7, HRNS-8, HRNS-9, HRNS-10 → `tests/test_adapter.py`

### Task T7: Trainer — rename `address` to `role`, target `supervisor` role

- **Spec ref:** @specs/harness-server.md §Trainer Participant — HRNS-14, HRNS-31, HRNS-33
- **Files:** `src/trainer/trainer.py`
- **Details:**
  - Rename constructor parameter `address` → `role`, default `"trainer"`
  - Rename `self._address` → `self._role`
  - Rename `address` property → `role`
  - Update `_emit_progress`: `Message(role=SUPERVISOR_ROLE, ...)`
  - Update `_emit_polling_status`: `Message(role=SUPERVISOR_ROLE, ...)`
  - Update `_submit_next_lesson`: `Message(role=TRAINEE_ROLE, ...)`, `sender=self._role`
  - Update `bus.subscribe(self._role, ...)`
  - Add event relay: after `_handle_kagent_event`, relay the event to role `supervisor` as `event` action (HRNS-33)
  - Add `ratify_request`: when a frame event (S2/S3) is handled, send `ratify_request` to role `supervisor` (HRNS-33)
- **Test mapping:** HRNS-12, HRNS-14, HRNS-16, HRNS-31, HRNS-33 → `tests/test_trainer.py`, `tests/test_harness*.py`

### Task T8: Reactor — rename `address` to `role`, escalation to `supervisor`

- **Spec ref:** @specs/harness-server.md §Trainer Participant — HRNS-14
- **Files:** `src/trainer/reactor.py`
- **Details:**
  - Rename constructor parameter `address` → `role`, default `"trainer"`
  - Rename `self._address` → `self._role`
  - `_auto_countersign`: `Message(role=TRAINEE_ROLE, ...)`, `sender=self._role`
  - `_handle_reactive`: `Message(role=TRAINEE_ROLE, ...)`, `sender=self._role`
  - `_escalate`: change `address="slack"` → `role=SUPERVISOR_ROLE` (key behavioural change)
- **Test mapping:** HRNS-13, HRNS-14, HRNS-31 → `tests/test_reactor.py`

### Task T9: TUI Client — rename `address` to `role`, register as `supervisor`

- **Spec ref:** @specs/harness-server.md §TUI Participant — HRNS-25, HRNS-25a, HRNS-26, HRNS-27, HRNS-28
- **Files:** `src/participants/tui_client.py`
- **Details:**
  - `HarnessClient`: rename `address` parameter → `role`, default `SUPERVISOR_ROLE`
  - Update registration frame: `{"register": self._role}`
  - `send()` method: use `role` parameter name
  - `TUIApp`: rename `address` → `role`
  - `on_ratify_bar_ratify_clicked`: `self._client.send(TRAINEE_ROLE, "countersign", ...)`
  - `on_input_bar_submitted`: feed through shared command parser (T14)
  - All outgoing frames use `"role"` JSON key
  - Track latest `ratify_request` for `ratify` command and RatifyBar
- **Test mapping:** HRNS-25, HRNS-25a, HRNS-26, HRNS-27, HRNS-28 → `tests/test_tui_client.py`

### Task T10: Slack Agent — register as `supervisor`, rename `address` to `role`

- **Spec ref:** @specs/harness-server.md §Slack Participant — HRNS-17, HRNS-18, HRNS-31, HRNS-34
- **Files:** `src/participants/slack_agent.py`
- **Details:**
  - Registration: change `{"register": "slack"}` → `{"register": "supervisor"}`
  - `_send_to_trainer`: frame uses `"role": "trainer"` instead of `"address": "trainer"`
  - Log message: update "registered as 'slack'" → "registered for role 'supervisor'"
  - `_receive_loop`: update frame parsing for `"role"` key
  - Track latest `ratify_request` for `ratify` command
  - Feed human input through shared command parser (T14)
  - Send `countersign` to trainee when `ratify` command parsed
  - Render all supervisor actions: progress, event, escalation, ratify_request
- **Test mapping:** HRNS-17, HRNS-18, HRNS-34 → `tests/test_slack_agent.py`

### Task T11: Harness config (`harness.yaml`)

- **Spec ref:** @specs/harness-server.md §Harness Configuration
- **Files:** `harness.yaml`
- **Details:**
  - Rename `address` → `role` on all participant entries
  - Change Kalvin entry from `role: kalvin` to `role: trainee`
  - Change Slack entry from `role: slack` to `role: supervisor`
  - Result: TUI and Slack both have `role: supervisor`

### Task T12: CLI entry point — rename `address` to `role`

- **Spec ref:** @specs/harness-server.md §Harness Configuration
- **Files:** `src/harness/__main__.py`
- **Details:**
  - `_AlreadySubscribed`: rename `address` property → `role`
  - Factories: rename parameter `address` → `role`, pass to constructors
  - Update all Message constructions

### Task T13: Update all tests

- **Details:**
  - `tests/test_bus.py`: `Message(address=...)` → `Message(role=...)`, handler parameter name
  - `tests/test_protocol.py`: registration frames, message frames, `address` → `role`
  - `tests/test_adapter.py`: `KAgentAdapter(bus, address=...)` → `KAgentAdapter(bus, role=...)`
  - `tests/test_trainer.py`: `Trainer(bus, ..., address=...)` → `Trainer(bus, ..., role=...)`
  - `tests/test_reactor.py`: `Reactor(bus, state, address=...)` → `Reactor(bus, state, role=...)`
  - `tests/test_tui_client.py`: `HarnessClient(url, address=...)` → `HarnessClient(url, role=...)`
  - `tests/test_slack_agent.py`: verify registration frame uses `"supervisor"`
  - `tests/test_server.py`: `ParticipantConfig(address=...)` → `ParticipantConfig(role=...)`
  - `tests/test_harness*.py`: update all `Message(address=...)` constructions
  - Add new tests:
    - HRNS-29: `test_fan_out_dispatch` — two handlers same role both receive
    - HRNS-30: `test_multiple_clients_same_role` — two WebSocket connections same role
    - HRNS-31: `test_trainer_progress_to_all_supervisor_subscribers` — trainer sends progress, all `supervisor` subscribers receive
    - HRNS-33: `test_trainer_relay_kagent_events_to_supervisor` — ground/frame events relayed
    - HRNS-34: `test_slack_ratify_command` — Slack sends `countersign` via `ratify` command

### Task T14: Shared command parser

- **Spec ref:** @specs/harness-server.md §Shared Command Protocol — HRNS-32
- **Files:** `src/participants/commands.py` (new), `tests/test_commands.py` (new)
- **Details:**
  - Define command dataclasses: `StartCommand`, `StopCommand`, `PauseCommand`, `ResumeCommand`, `GoalCommand`, `RatifyCommand`, `FileGoalCommand`, `GuidanceCommand`
  - `parse_command(text: str) -> Command` — maps free-text to command
  - `Command.to_messages(latest_proposal)` — returns list of `(role, action, message)` tuples for bus dispatch
  - Command recognition rules:
    - `"start"` → `StartCommand`
    - `"stop"` → `StopCommand`
    - `"pause"` → `PauseCommand`
    - `"resume"` → `ResumeCommand`
    - `"goal: <text>"` → `GoalCommand(text)`
    - `"ratify"` → `RatifyCommand`
    - path-like string (ends in `.md`, starts with `/` or `./`, or exists on disk) → `FileGoalCommand`
    - everything else → `GuidanceCommand(text)`
  - `RatifyCommand.to_messages(latest_proposal)` → `[(TRAINEE_ROLE, "countersign", latest_proposal)]`
  - All other commands → `[(TRAINER_ROLE, "input", original_text)]`
- **Test mapping:** HRNS-32 → `tests/test_commands.py`
- **Consumed by:** T9 (TUI InputBar), T10 (Slack message handler)

## Test Mapping

| Spec ID | Test file                 | Test function                            | Status |
| ------- | ------------------------- | ---------------------------------------- | ------ |
| HRNS-1  | tests/test_bus.py         | test_route_by_role                       | ⬜     |
| HRNS-2  | tests/test_bus.py         | test_threadsafe_send                     | ⬜     |
| HRNS-3  | tests/test_bus.py         | test_unknown_role_error                  | ⬜     |
| HRNS-4  | tests/test_protocol.py    | test_client_registration                 | ⬜     |
| HRNS-5  | tests/test_server.py      | test_load_embedded_participants          | ⬜     |
| HRNS-6  | tests/test_server.py      | test_websocket_client_connect            | ⬜     |
| HRNS-7  | tests/test_adapter.py     | test_submit_compiles_and_submits         | ⬜     |
| HRNS-8  | tests/test_adapter.py     | test_compilation_error_response          | ⬜     |
| HRNS-9  | tests/test_adapter.py     | test_sender_map_response_addressing      | ⬜     |
| HRNS-10 | tests/test_adapter.py     | test_countersign_action                  | ⬜     |
| HRNS-11 | tests/test_bus.py         | test_wildcard_diagnostic_listener        | ⬜     |
| HRNS-12 | tests/test_trainer.py     | test_auto_countersign                    | ⬜     |
| HRNS-13 | tests/test_reactor.py     | test_reactive_mode                       | ⬜     |
| HRNS-14 | tests/test_reactor.py     | test_escalation_to_supervisor_role       | ⬜     |
| HRNS-15 | tests/test_harness_persistence.py | test_state_persistence           | ⬜     |
| HRNS-16 | tests/test_trainer.py     | test_one_session_at_a_time               | ⬜     |
| HRNS-17 | tests/test_slack_agent.py | test_slack_forwards_via_command_parser   | ⬜     |
| HRNS-18 | tests/test_slack_agent.py | test_slack_renders_all_supervisor_actions| ⬜     |
| HRNS-19 | tests/test_trainer.py     | test_session_pause                       | ⬜     |
| HRNS-20 | tests/test_trainer.py     | test_session_stop                        | ⬜     |
| HRNS-21 | tests/test_protocol.py    | test_disconnect_silent_drop              | ⬜     |
| HRNS-22 | tests/test_adapter.py     | test_kagent_calls_adapter_directly       | ⬜     |
| HRNS-23 | tests/test_bus.py         | test_single_dispatch_thread              | ⬜     |
| HRNS-24 | tests/test_trainer.py     | test_entry_counting                      | ⬜     |
| HRNS-25 | tests/test_tui_client.py  | test_renders_received_events             | ⬜     |
| HRNS-25a| tests/test_tui_client.py  | test_input_uses_command_parser           | ⬜     |
| HRNS-26 | tests/test_tui_client.py  | test_sends_freeform_input                | ⬜     |
| HRNS-27 | tests/test_tui_client.py  | test_sends_countersign                   | ⬜     |
| HRNS-28 | tests/test_tui_client.py  | test_input_bar_clears                    | ⬜     |
| HRNS-29 | tests/test_bus.py         | test_fan_out_dispatch                    | 🆕     |
| HRNS-30 | tests/test_protocol.py    | test_multiple_clients_same_role          | 🆕     |
| HRNS-31 | tests/test_trainer.py     | test_progress_to_all_supervisor_subscribers | 🆕  |
| HRNS-32 | tests/test_commands.py    | test_command_parser                      | 🆕     |
| HRNS-33 | tests/test_trainer.py     | test_relay_kagent_events_to_supervisor   | 🆕     |
| HRNS-34 | tests/test_slack_agent.py | test_slack_ratify_command                | 🆕     |

## Design Decisions

1. **`role` not `topic`** — the routing key represents the participant's role in the training session (trainee, trainer, supervisor). This is domain language, not infrastructure jargon. `topic` and `address` are pub/sub terms that leak implementation detail into the domain model.

2. **`trainee` not `kalvin`** — the role name describes what the participant *does*, not what it *is*. Kalvin is the implementation class; `trainee` is the role. Future trainee implementations would use the same role.

3. **Clean break, no migration shim** — rename `address` → `role` everywhere in one pass. No backward compatibility layer. The project has no external API consumers.

4. **Escalation target changes from `"slack"` to `"supervisor"`** — escalation messages go to all `supervisor` subscribers. Slack renders them; TUI shows them. The supervisor sees escalation in whichever interface they're using.

5. **WebSocketProtocol: one `_ClientParticipant` wrapper per role** — the bus-facing wrapper is shared across all connections for a role. The wrapper iterates all active connections on `on_message`. This keeps bus subscription simple (one handler per role) while supporting multiple connections.

6. **Duplicate roles allowed for client participants only** — embedded participants should have unique roles (one trainee, one trainer). Client participants may share roles (multiple supervisors). Validation in config loading enforces this.

7. **Sender is the sender's role** — when Trainer sends to trainee, `sender="trainer"`. Trainee responds to role `"trainer"`. This works because each embedded participant has a unique role. For client participants, the sender is the role they registered for (e.g. `"supervisor"`).

8. **Shared command parser lives in `src/participants/`** — it's a participant-level concern, not a harness concern. Both TUI and Slack import and use it. The Trainer does not need to know about it — commands are still delivered as `input` actions on the `trainer` role, except `ratify` which goes directly to `trainee`.

9. **Event relay is Trainer's responsibility** — the Trainer already receives all Kalvin events. Rather than making Kalvin aware of supervisors, the Trainer relays events to the `supervisor` role. This maintains Kalvin's ignorance of the training loop.

10. **Role constants in `src/harness/constants.py`** — single source of truth for role strings. All modules reference constants instead of string literals.

## Build Order

```
T1: Message (rename field)
  │
  ├── T1a: Role constants (new file)
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
        ├── T6: KAgent Adapter (role="trainee")
        │
        ├── T7: Trainer (target supervisor, event relay)
        │     │
        │     └── T8: Reactor (escalation to supervisor)
        │
        └── T12: CLI entry point
              │
              └── T11: harness.yaml

T14: Shared command parser (parallel with T4–T12)
T13: Update all tests (interleaved with each task)
```

**Recommended approach:** T1→T1a→T2→T3 first (core renames + constants), then T14 (command parser, no dependencies), then T4→T5→T6→T7→T8 (server-side), then T9→T10 (clients), then T11→T12 (config/CLI). T13 runs throughout.

## Status

Planning. Pending cascade Step 3 (consistency check) and Step 4 (commit).
