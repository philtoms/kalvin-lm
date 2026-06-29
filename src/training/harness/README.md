# Harness — Multi-Agent Runtime

The harness is the persistent server that hosts Kalvin's multi-agent training system. It loads participants, routes role-based messages between them via a thread-safe message bus, and exposes a WebSocket endpoint for remote clients.

All participants are **equal peers**: each has a unique role, sends messages through the bus, and receives messages addressed to its role. No participant knows it's in a training loop — the dialogue between them _is_ the training.

```
Harness Server (python -m training.harness)
  │
  ├── MessageBus (thread-safe role-based router, runs on its own thread)
  │
  ├── Embedded participants (loaded in-process):
  │     ├── Kalvin   (role: "trainee")   — rationalisation engine
  │     └── Trainer  (role: "trainer")   — curriculum driver + reactive scaffolding
  │
  └── WebSocket server (ws://host:port):
        ├── Slack agent (role: "supervisor") — human-in-the-loop via Slack
        └── TUI agent   (role: "supervisor") — human-in-the-loop via terminal
```

> **Note:** Both Slack and TUI clients register as `"supervisor"`. The bus
> delivers messages to all handlers subscribed to a role, so multiple
> clients on the same role each receive a copy.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Curriculum](#curriculum)
3. [Configuration](#configuration)
4. [API Keys & Environment Variables](#api-keys--environment-variables)
5. [Running the Harness](#running-the-harness)
6. [Architecture](#architecture)
7. [Participants](#participants)
8. [Wire Protocol](#wire-protocol)
9. [How to Participate (Write a Custom Client)](#how-to-participate-write-a-custom-client)
10. [Message Flow](#message-flow)
11. [Graceful Shutdown & State Persistence](#graceful-shutdown--state-persistence)
12. [Testing](#testing)

---

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. (Optional) Install trainer dependencies for reactive scaffolding
uv sync --extra trainer

# 3. Set required environment variables
export KALVIN_LLM_API_KEY=...          # for Trainer reactive scaffolding (optional)
export SLACK_BOT_TOKEN=xoxb-...        # only if using Slack participant
export SLACK_APP_TOKEN=xapp-...        # only if using Slack participant

# 4. Run the harness
uv run python -m training.harness --config training.harness.yaml
```

The harness starts, loads the embedded Kalvin and Trainer, and listens for WebSocket connections from the Slack and TUI clients.

To start training, send a goal via Slack or the TUI:

```
goal: build a curriculum around the nursery rhyme "Mary had a little lamb"
```

Or supply a path to an existing curriculum file:

```
goal: curricula/mary-had-a-little-lamb.md
```

See [Curriculum](#curriculum) for the full lifecycle.

---

## Curriculum

A curriculum is a **living structured document** — a markdown file that drives the entire training session. It contains three sections:

| Section       | Purpose                                                                  |
| ------------- | ------------------------------------------------------------------------ |
| **Objective** | What this curriculum teaches. One or two paragraphs.                     |
| **Approach**  | The pedagogical strategy — the order and rationale for the lessons.      |
| **Lessons**   | Ordered KScript entries, each with a human-readable heading and context. |

### Example

````markdown
# Curriculum: Syntactic Foundations of "Mary Had a Little Lamb"

## Objective

Teach Kalvin to decompose a simple English sentence into its grammatical
components, so that it can recognise subject-verb-object structure in
unfamiliar sentences later in the curriculum.

## Approach

Start by grounding the individual tokens as identities (S4). Then build
the composite signature for the full sentence and decompose it into its
constituent roles (Subject, Verb, Object). Each lesson introduces one
layer of structure, with scaffolding that anticipates the countersign
relationships.

## Lessons

### Lesson 1: Ground the tokens

Establish each word as an identity kline so Kalvin has raw material
to work with.

```kscript
Mary
had
a
little
lamb
```
````

### Lesson 2: Compose the full sentence

Build the composite signature M-H-A-L-L and ground it as a
self-referential unit.

```kscript
MHALL = M H A L L
```

### Lesson 3: Introduce grammatical roles

Decompose the sentence into subject, verb, and object.

```kscript
MHALL => S V O
  S(ubject) = M
  V(erb) = H
  O(bject) = ALL
    A > D(et)
    L > M(od)
    L > O
```

### Lesson 4: Reinforce with countersigns

Bidirectional links cement the relationships.

```kscript
M == S
H == V
ALL == O
```

```

### Starting a New Curriculum

There are three ways to start a training session:

1. **Natural language goal.** Send a high-level request via Slack or the TUI:
```

goal: build a curriculum around "Mary had a little lamb" that includes
steps to isolate the syntax of the text

```
The Trainer uses the LLM client to generate a full curriculum document, writes
it to the `curricula/` directory, and starts training immediately.

2. **Existing file.** Supply the path to a pre-written curriculum:
```

goal: curricula/mary-had-a-little-lamb.md

```
The Trainer loads the file and starts training.

3. **Resume.** If no goal is provided and the Trainer has persisted state
from a previous session, it resumes automatically from the saved position.

### Reviewing Progress

The Trainer publishes **progress events** after each lesson completes. These
appear in the Slack channel and the TUI event log, showing:

- Which lesson just completed (label and title)
- Status: satisfied, pending, or escalated
- A brief summary

You can also open the curriculum file directly — since it's a regular
markdown document, it's readable in any editor. The `curricula/` directory
contains all generated and curated curriculum files.

### Amending a Running Curriculum

The curriculum is a **living document** — it can be changed at any time
during a training session. Two approaches:

1. **Edit the file directly.** Open the curriculum markdown file in any
editor and make your changes. The Trainer re-reads the file before each
lesson, so your changes are picked up at the next natural checkpoint.

2. **Ask the Trainer.** Send a message via Slack or the TUI:
```

Add a bridging lesson between lesson 2 and lesson 3 that grounds
the individual tokens as SVO components before composition

````
The Trainer uses the LLM client to generate the amendment and writes it into
the curriculum file.

In both cases, the Trainer resumes from the first unsubmitted lesson in
the updated document. Kalvin's monotonic submitted set ensures that only
new klines are compiled and submitted — nothing re-runs.

### Lesson Labelling

Lessons are identified by stable labels derived from their headings:

- **Whole numbers** (1, 2, 3) indicate distinct conceptual steps.
- **Sub-labels** (2a, 2b) indicate lessons semantically related to their
parent — refinements, bridges, or remediations.
- If a new lesson is logically subsequent but **not** semantically related,
the document is renumbered instead.

```markdown
### Lesson 2: Compose the full sentence
### Lesson 2a: Bridge token identity to composition   ← refines lesson 2
### Lesson 3: Introduce grammatical roles             ← renumbered, new concept
````

The curriculum should always read as a logical and temporal narrative.

### Reactive Scaffolding

When Kalvin hits an S2/S3 event (partial understanding), the Trainer
enters reactive mode and generates scaffolding via the LLM client. This
scaffolding is written into the curriculum as a new lesson — making the
Trainer's reactive work visible, persistent, and auditable.

### Rollback and Replay

The curriculum **never** reverts — it only evolves forward. If Kalvin's
performance is outside expectations, the operator resets Kalvin's model
state and replays the current curriculum from lesson 1. Because the
submitted set is monotonic, replay is safe and efficient.

---

## Configuration

The harness reads a YAML (or JSON) configuration file. The default path is `training.harness.yaml` in the project root.

### Full Example (`training.harness.yaml`)

```yaml
# Server settings (overridable via CLI flags)
server:
  host: "localhost"
  port: 8765

# Trainer settings
trainer:
  curriculum_file: "curricula/first-steps.md" # Pre-made curriculum, or "" to wait for a goal
  curricula_dir: "curricula" # Directory for generated curriculum files
  # State file is auto-derived from curriculum_file: e.g. curricula/first-steps.md → curricula/first-steps.json
  max_reactive_rounds: 5 # Max reactive scaffolding rounds before escalation
  # llm:                                        # Uncomment to override defaults
  #   base_url: "https://api.z.ai/api/coding/paas/v4"
  #   model: "glm-5.1"
  # API key is read from KALVIN_LLM_API_KEY env var — never put it here

# Participant definitions
participants:
  # Embedded participants — loaded in-process by the harness
  - role: trainee
    type: embedded
    class: KAgent

  - role: trainer
    type: embedded
    class: Trainer

  # Client participants — connect via WebSocket
  - role: supervisor
    type: client
    class: SlackParticipant

  - role: supervisor
    type: client
    class: TUIParticipant
```

### Configuration Sections

| Section          | Key                   | Default       | Description                                                             |
| ---------------- | --------------------- | ------------- | ----------------------------------------------------------------------- |
| `server`         | `host`                | `"localhost"` | WebSocket server bind address                                           |
| `server`         | `port`                | `8765`        | WebSocket server bind port                                              |
| `trainer`        | `curriculum_file`     | `""`          | Path to a pre-made curriculum file; empty string waits for a goal       |
| `trainer`        | `curricula_dir`       | `"curricula"` | Directory for generated curriculum files                                |
| `trainer`        | `max_reactive_rounds` | `5`           | Reactive scaffolding budget before escalation                           |
| `trainer`        | `llm.base_url`        | _(see code)_  | OpenAI-compatible API endpoint                                          |
| `trainer`        | `llm.model`           | `"glm-5.1"`   | Model name for LLM calls                                                |
| `participants[]` | `role`                | —             | Bus role for this participant (used for routing)                        |
| `participants[]` | `type`                | —             | `"embedded"` (loaded in-process) or `"client"` (connects via WebSocket) |
| `participants[]` | `class`               | —             | Registered class name (e.g. `KAgent`, `Trainer`, `SlackParticipant`)    |

### Validation Rules

- Every embedded participant must have a unique `role`. Duplicates for embedded participants are rejected at startup.
- Client participants may share a `role` (multiple WebSocket connections per role are supported).
- Every participant must have `type` = `"embedded"` or `"client"`.
- Every participant must have a `class` (the registered factory name).
- Embedded participants must have a corresponding factory registered in the CLI (`__main__.py`). Unknown classes are rejected at startup.

---

## API Keys & Environment Variables

### LLM Client (Trainer / Cogitation)

The Trainer's reactive mode uses the `Cogitator` with an `OpenAICompatibleClient`. To enable this:

| Variable             | Required | Description                                                                        |
| -------------------- | -------- | ---------------------------------------------------------------------------------- |
| `KALVIN_LLM_API_KEY` | No       | API key passed to the LLM client. If unset, reactive mode escalates to supervisor. |

The `base_url` and `model` are read from the `trainer.llm` config section and default to the ZhipuAI GLM-5.1 endpoint. Any OpenAI-compatible endpoint works.

### Slack Participant (optional)

| Variable          | Required             | Description                                                                              |
| ----------------- | -------------------- | ---------------------------------------------------------------------------------------- |
| `SLACK_BOT_TOKEN` | Yes (if using Slack) | Slack Bot OAuth token (`xoxb-...`). Used to post messages to channels.                   |
| `SLACK_APP_TOKEN` | Yes (if using Slack) | Slack App-Level token (`xapp-...`). Used for Socket Mode to receive supervisor messages. |

These can also be passed as constructor arguments to `SlackParticipant` instead of environment variables.

### Required Python Packages

| Package      | When Needed                      | Install                   |
| ------------ | -------------------------------- | ------------------------- |
| `websockets` | Always (harness core)            | `uv sync`                 |
| `pyyaml`     | Always (config loading)          | `uv sync`                 |
| `openai`     | Reactive scaffolding (LLM agent) | `uv sync --extra trainer` |
| `slack-sdk`  | Slack participant                | `uv sync`                 |
| `textual`    | TUI participant                  | `uv sync`                 |

---

## Running the Harness

### Basic

```bash
uv run python -m training.harness --config training.harness.yaml
```

### Override Host / Port

```bash
# CLI flags override the config file's server.host and server.port
uv run python -m training.harness --config training.harness.yaml --host 0.0.0.0 --port 9000
```

### All CLI Flags

```
usage: harness [-h] [--config CONFIG] [--host HOST] [--port PORT]

Multi-agent harness runtime

options:
  -h, --help       show this help message and exit
  --config CONFIG  Path to YAML/JSON config file (default: training.harness.yaml)
  --host HOST      Override WebSocket server host
  --port PORT      Override WebSocket server port
```

### What Happens at Startup

1. **Config loaded** — `training.harness.yaml` is read and validated.
2. **Bus created** — A `MessageBus` is instantiated.
3. **Embedded participants wired** — Kalvin and Trainer factories are called:
   - `KAgentAdapter` subscribes to role `"trainee"` on the bus.
   - `Trainer` subscribes to role `"trainer"` on the bus.
4. **Bus thread started** — The bus event loop runs on a daemon thread.
5. **WebSocket server started** — Listens on the configured host:port for client participants.
6. **Blocked** — The main thread runs the async event loop, waiting for SIGINT / SIGTERM.

After startup, the Trainer checks for saved state. If found, it resumes the previous session. Otherwise, it waits for a goal from a supervising participant (see [Curriculum](#curriculum)).

---

## Architecture

### Module Map

| File           | Purpose                                                                                                                                                                               |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `__main__.py`  | CLI entry point. Loads config, wires participant factories, builds the LLM client, runs the server.                                                                                   |
| `server.py`    | `HarnessServer` — loads config, instantiates embedded participants, starts WebSocket server, runs the bus. Also contains `load_config()` and `ConfigError`.                           |
| `bus.py`       | `MessageBus` — thread-safe role-based message router with single-dispatch event loop. Supports wildcard (`"*"`) subscribers for diagnostics.                                          |
| `message.py`   | `Message` — immutable dataclass: `role`, `action`, `message`, `sender`. The bus routes by `role` only; `action` and `message` are interpreted by the recipient.                       |
| `adapter.py`   | `KAgentAdapter` — bridge between Kalvin's rationalisation pipeline and the bus. Compiles KScript source, submits entries to KAgent, and routes events back to the original sender.    |
| `protocol.py`  | `WebSocketProtocol` — handles WebSocket client connections: registration, bidirectional JSON frame routing, silent-drop disconnect semantics. Supports multiple connections per role. |
| `protocols.py` | `Participant` protocol — the interface every participant must implement: `role` + `on_message(msg)`.                                                                                  |
| `constants.py` | Canonical role constants: `TRAINEE_ROLE`, `TRAINER_ROLE`, `SUPERVISOR_ROLE`.                                                                                                          |

### Thread Model

```
Main Thread (asyncio)          Bus Thread (sync)
─────────────────────          ─────────────────
WebSocket server               MessageBus.run()
  handle_connection()            _dispatch()
  _send_to_client_sync()         → handler.on_message()
  asyncio event loop               → KAgentAdapter.on_message()
                                   → Trainer.on_message()
```

- `MessageBus.send()` is thread-safe (uses `queue.Queue`).
- All handler dispatch runs on the bus thread.
- WebSocket sends from the bus thread are bridged via `asyncio.run_coroutine_threadsafe()`.

---

## Participants

### Embedded Participants

These are loaded in-process and wired directly to the bus at startup.

#### Kalvin (`role: "trainee"`)

The rationalisation engine. The `KAgentAdapter` receives bus messages and delegates to the core `KAgent`:

| Incoming Action | Behaviour                                                                                                                                         |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `submit`        | Compile KScript source from `msg.message`, record sender per entry, call `kagent.rationalise(entry)` for each. Events flow back via `on_event()`. |
| `countersign`   | Call `kagent.countersign(kline)` with the KLine in `msg.message`.                                                                                 |

Events from Kalvin are routed back to the original sender (stored in a sender map keyed by entry identity).

#### Trainer (`role: "trainer"`)

Drives the training loop. Holds LLM access for curriculum generation and reactive scaffolding:

| Incoming Action    | Behaviour                                                                                                                      |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| `ground` / `frame` | Kalvin events. S1 events auto-satisfy. S2/S3 events trigger reactive mode (auto-countersign → cogitation → escalation).        |
| `input`            | Supervisor input from Slack/TUI. Supports: `goal: <text or path>`, `pause`, `stop`, `resume`, amendment requests, or guidance. |
| `error`            | Kalvin compilation error. Logged and counted toward lesson completion.                                                         |
| `progress`         | Published by the Trainer after each lesson completes. Consumed by TUI and Slack to display status.                             |

**Trainer lifecycle:**

1. **Startup** — checks for saved state (resume) or waits for a goal.
2. **Goal received** — natural language → generate curriculum via LLM, or file path → load directly.
3. Submits lessons from the curriculum to `"trainee"` via `submit` messages. Re-reads the curriculum file before each lesson to pick up amendments.
4. Listens for ground/frame events:
   - **S1** (fully grounded) → auto-satisfy, advance curriculum.
   - **S2/S3** → try auto-countersign, then reactive mode:
     - Up to `max_reactive_rounds` of cogitation.
     - If cogitation generates scaffolding → write as a new lesson in the curriculum, then submit.
     - If stuck → **escalate** to `"supervisor"` with a `notify` message.
5. Publishes **progress events** after each lesson completes.
6. When curriculum is complete → end session, persist state, process queued goals.

Any participant can request a curriculum amendment by messaging the Trainer. The Trainer uses the LLM client to generate the amendment and writes it to the curriculum file. Amendments take effect at the next lesson boundary.

### Client Participants

These connect to the harness via WebSocket and register with a role.

#### Slack Participant (`role: "supervisor"`)

Bridges Slack and the harness:

| Direction       | Action   | Behaviour                                                                                             |
| --------------- | -------- | ----------------------------------------------------------------------------------------------------- |
| Harness → Slack | `notify` | Posts the message content to the configured Slack channel.                                            |
| Slack → Harness | `input`  | Forwards supervisor messages to `"trainer"` as `{role: "trainer", action: "input", message: <text>}`. |

**Running the Slack participant:**

```python
from training.supervisors import SlackParticipant

agent = SlackParticipant(
    harness_url="ws://localhost:8765",
    channel_id="C01234567",        # Slack channel ID
    # slack_token and app_token default to SLACK_BOT_TOKEN / SLACK_APP_TOKEN env vars
)
await agent.start()
```

#### TUI Participant (`role: "supervisor"`)

A Textual TUI that displays Kalvin's events and provides ratification (countersign) controls:

- **EventLog** — scrollable log of all events received from the harness.
- **RatifyBar** — button/keyboard shortcut (`Ctrl+R`) to send a `countersign` message to `"trainee"`.

**Running the TUI participant:**

```bash
uv run python -m training.supervisors.tui_client
```

Or programmatically:

```python
from training.supervisors import TUIApp

app = TUIApp(harness_url="ws://localhost:8765")
app.run()
```

---

## Wire Protocol

Client participants communicate with the harness over WebSocket using JSON frames.

### Registration (first frame)

```json
{ "register": "supervisor" }
```

The first frame from any client **must** be a registration frame. Multiple clients may register with the same role — the bus will deliver messages to all of them.

### Sending a Message

```json
{
  "role": "trainee",
  "action": "submit",
  "message": "(Mary had a little lamb)\nMHALL = SVO => ..."
}
```

After registration, all outbound frames are messages. The `sender` field is set implicitly by the harness to the registered role.

### Receiving a Message

```json
{
  "role": "supervisor",
  "action": "notify",
  "message": {
    "reason": "budget_exhaustion",
    "detail": "",
    "lesson_position": 3
  },
  "sender": "trainer"
}
```

Inbound frames are routed by `role` — the harness delivers them to all handlers subscribed to that role.

### Error Frames

```json
{ "error": "malformed frame: not valid JSON" }
```

Sent by the harness when a frame cannot be parsed or is structurally invalid. If the first frame is not a valid registration, the connection is closed with code `4001`.

### Disconnect Semantics

- On disconnect, the bus subscription is **not** removed.
- Messages to a disconnected client are **silently dropped** until the client reconnects with the same role.
- Reconnection picks up the existing subscription — no re-registration of handlers needed.

---

## How to Participate (Write a Custom Client)

Any program that can open a WebSocket and send JSON can be a harness participant. Here's a minimal Python example:

```python
import asyncio
import json
import websockets

async def my_participant():
    async with websockets.connect("ws://localhost:8765") as ws:
        # 1. Register
        await ws.send(json.dumps({"register": "my-agent"}))
        print("Registered as 'my-agent'")

        # 2. Send a message to Kalvin
        await ws.send(json.dumps({
            "role": "trainee",
            "action": "submit",
            "message": "(hello world)\nHW = SVO => S(ubject)=H V(erb)=W O(bject)=W",
        }))

        # 3. Listen for responses
        async for raw in ws:
            frame = json.loads(raw)
            if "error" in frame:
                print(f"Error: {frame['error']}")
            else:
                print(f"Received: action={frame['action']} from={frame.get('sender')}")
                print(f"  message: {frame.get('message')}")

asyncio.run(my_participant())
```

### Participant Protocol

Every participant (embedded or client) implements the `Participant` protocol:

```python
class Participant(Protocol):
    role: str
    def on_message(self, message: Message) -> None: ...
```

### Writing an Embedded Participant

To add a new embedded participant that runs in-process:

1. **Create a class** with `role` and `on_message(msg)`:

   ```python
   class MyParticipant:
       def __init__(self, bus: MessageBus, role: str = "my-agent"):
           self.role = role
           bus.subscribe(role, self.on_message)

       def on_message(self, msg: Message) -> None:
           if msg.action == "ping":
               bus.send(Message(role=msg.sender, action="pong", message="alive"))
   ```

2. **Register a factory** in `__main__.py`:

   ```python
   def my_factory(role: str, bus: MessageBus) -> _AlreadySubscribed:
       participant = MyParticipant(bus, role=role)
       return _AlreadySubscribed(participant)

   server.register_participant_class("MyParticipant", my_factory)
   ```

3. **Add to `training.harness.yaml`**:

   ```yaml
   participants:
     - role: my-agent
       type: embedded
       class: MyParticipant
   ```

### Writing a Client Participant

Client participants are simpler — just a WebSocket program:

1. Connect to `ws://<host>:<port>`.
2. Send `{"register": "<your-role>"}`.
3. Send/receive JSON message frames.

See the [Wire Protocol](#wire-protocol) section for frame formats.

---

## Message Flow

A typical training cycle looks like this:

```
 Supervisor (Slack)     Trainer              Kalvin
     │                    │                    │
     │  "goal: teach      │                    │
     │   colours"         │                    │
     │───────────────────>│                    │
     │                    │                    │
     │                    │  submit (lesson 1) │
     │                    │───────────────────>│
     │                    │                    │ rationalise()
     │                    │  frame (S1)        │
     │                    │<───────────────────│
     │                    │                    │
     │                    │ auto-satisfy,      │
     │                    │ advance curriculum │
     │                    │                    │
     │                    │  submit (lesson 2) │
     │                    │───────────────────>│
     │                    │                    │ rationalise()
     │                    │  frame (S2)        │
     │                    │<───────────────────│
     │                    │                    │
     │                    │ reactive mode:     │
     │                    │ cogitate → submit  │
     │                    │───────────────────>│
     │                    │                    │ rationalise()
     │                    │  frame (S1)        │
     │                    │<───────────────────│
     │                    │                    │
     │                    │ [stuck after N     │
     │                    │  reactive rounds]  │
     │                    │                    │
     │  notify            │                    │
     │  (escalation)      │                    │
     │<───────────────────│                    │
     │                    │                    │
     │  "try breaking     │                    │
     │   it down"         │                    │
     │───────────────────>│                    │
     │                    │ (guidance stored   │
     │                    │  for cogitation)   │
```

---

## Graceful Shutdown & State Persistence

The harness handles `SIGINT` (Ctrl+C) and `SIGTERM` gracefully:

1. **WebSocket server** is closed — existing connections are dropped.
2. **Bus event loop** is stopped.
3. **Trainer state** is persisted to `<curriculum_file>.json` (e.g. `curricula/first-steps.md` → `curricula/first-steps.json`). The state file path is auto-derived from the curriculum filename — no separate config key is needed. This includes:
   - Curriculum file path
   - Current lesson label (stable identity)
   - Submitted / satisfied / pending entry sets
   - Append-only event log

On restart, the Trainer loads its persisted state and resumes the curriculum from the saved position. The curriculum file itself is never rolled back — only Kalvin's model state can be reset for a full replay.

---

## Testing

```bash
# Run all harness tests
uv run pytest tests/test_harness*.py tests/test_bus.py tests/test_server.py tests/test_adapter.py tests/test_protocol.py tests/test_protocols.py tests/test_harness_cli.py tests/test_harness_main.py

# Run with verbose output
uv run pytest tests/test_harness*.py -v

# Run a specific test file
uv run pytest tests/test_harness_run.py -v
```

### Test Files

| File                          | Coverage                                                     |
| ----------------------------- | ------------------------------------------------------------ |
| `test_bus.py`                 | MessageBus subscribe, send, dispatch, wildcard, stop         |
| `test_harness.py`             | Config loading, validation, ParticipantConfig                |
| `test_server.py`              | HarnessServer setup, embedded participant wiring             |
| `test_adapter.py`             | KAgentAdapter submit, countersign, event routing             |
| `test_protocol.py`            | WebSocketProtocol registration, frame parsing                |
| `test_protocols.py`           | Participant protocol compliance                              |
| `test_harness_run.py`         | End-to-end: start harness, route messages through full stack |
| `test_harness_cli.py`         | CLI argument parsing                                         |
| `test_harness_main.py`        | **main**.py factory wiring, LLM client building              |
| `test_harness_events.py`      | Event classification (S1/S2/S3)                              |
| `test_harness_tracking.py`    | CurriculumState tracking sets                                |
| `test_harness_persistence.py` | State save/load round-trip                                   |
