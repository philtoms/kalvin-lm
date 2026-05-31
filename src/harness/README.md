# Harness — Multi-Agent Runtime

The harness is the persistent server that hosts Kalvin's multi-agent training system. It loads participants, routes addressed messages between them via a thread-safe message bus, and exposes a WebSocket endpoint for remote clients.

All participants are **equal peers**: each has a unique address, sends addressed messages through the bus, and receives messages addressed to it. No participant knows it's in a training loop — the dialogue between them _is_ the training.

```
Harness Server (python -m harness)
  │
  ├── MessageBus (thread-safe addressed router, runs on its own thread)
  │
  ├── Embedded participants (loaded in-process):
  │     ├── Kalvin   (address: "kalvin")   — rationalisation engine
  │     └── Trainer  (address: "trainer")  — curriculum driver + reactive scaffolding
  │
  └── WebSocket server (ws://host:port):
        ├── Slack agent (address: "slack") — human-in-the-loop via Slack
        └── TUI agent   (address: "ui")    — human-in-the-loop via terminal
```

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
export SLACK_BOT_TOKEN=xoxb-...      # only if using Slack participant
export SLACK_APP_TOKEN=xapp-...      # only if using Slack participant

# 4. Run the harness
uv run python -m harness --config harness.yaml
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

| Section | Purpose |
|---------|----------|
| **Objective** | What this curriculum teaches. One or two paragraphs. |
| **Approach** | The pedagogical strategy — the order and rationale for the lessons. |
| **Lessons** | Ordered KScript entries, each with a human-readable heading and context. |

### Example

```markdown
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
   The Trainer uses GLM-5.1 to generate a full curriculum document, writes
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
   ```
   The Trainer uses GLM-5.1 to generate the amendment and writes it into
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
```

The curriculum should always read as a logical and temporal narrative.

### Reactive Scaffolding

When Kalvin hits an S2/S3 event (partial understanding), the Trainer
enters reactive mode and generates scaffolding via GLM-5.1. This
scaffolding is written into the curriculum as a new lesson — making the
Trainer's reactive work visible, persistent, and auditable.

### Rollback and Replay

The curriculum **never** reverts — it only evolves forward. If Kalvin's
performance is outside expectations, the operator resets Kalvin's model
state and replays the current curriculum from lesson 1. Because the
submitted set is monotonic, replay is safe and efficient.

---

## Configuration

The harness reads a YAML (or JSON) configuration file. The default path is `harness.yaml` in the project root.

### Full Example (`harness.yaml`)

```yaml
# Server settings (overridable via CLI flags)
server:
  host: "localhost"
  port: 8765

# Trainer settings
trainer:
  state_path: "trainer_state.json" # Path for state persistence across restarts
  max_reactive_rounds: 5 # Max reactive scaffolding rounds before escalation

# Participant definitions
participants:
  # Embedded participants — loaded in-process by the harness
  - address: kalvin
    type: embedded
    class: KAgent

  - address: trainer
    type: embedded
    class: Trainer

  # Client participants — connect via WebSocket
  - address: slack
    type: client
    class: SlackParticipant

  - address: ui
    type: client
    class: TUIParticipant
```

### Configuration Sections

| Section          | Key                   | Default                | Description                                                             |
| ---------------- | --------------------- | ---------------------- | ----------------------------------------------------------------------- |
| `server`         | `host`                | `"localhost"`          | WebSocket server bind address                                           |
| `server`         | `port`                | `8765`                 | WebSocket server bind port                                              |
| `trainer`        | `state_path`          | `"trainer_state.json"` | File for Trainer state persistence                                      |
| `trainer`        | `max_reactive_rounds` | `5`                    | Reactive scaffolding budget before escalation                           |
| `participants[]` | `address`             | —                      | Unique bus address for this participant                                 |
| `participants[]` | `type`                | —                      | `"embedded"` (loaded in-process) or `"client"` (connects via WebSocket) |
| `participants[]` | `class`               | —                      | Registered class name (e.g. `KAgent`, `Trainer`, `SlackParticipant`)    |

### Validation Rules

- Every participant must have a unique `address`. Duplicates are rejected at startup.
- Every participant must have `type` = `"embedded"` or `"client"`.
- Every participant must have a `class` (the registered factory name).
- Embedded participants must have a corresponding factory registered in the CLI (`__main__.py`). Unknown classes are rejected at startup.

---

## API Keys & Environment Variables

The harness itself requires no API keys. API keys are only needed by specific participants:

### Slack Participant (optional)

| Variable          | Required             | Description                                                                         |
| ----------------- | -------------------- | ----------------------------------------------------------------------------------- |
| `SLACK_BOT_TOKEN` | Yes (if using Slack) | Slack Bot OAuth token (`xoxb-...`). Used to post messages to channels.              |
| `SLACK_APP_TOKEN` | Yes (if using Slack) | Slack App-Level token (`xapp-...`). Used for Socket Mode to receive human messages. |

These can also be passed as constructor arguments to `SlackParticipant` instead of environment variables.

### Trainer / Cogitation (optional)

The Trainer's reactive mode uses the `Cogitator` with an `OpenAICompatibleClient` that calls a openai-compatible LLM API. To enable this:

| Variable    | Required            | Description                                                                                                                                                                 |
| ----------- | ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| _(API key)_ | If using cogitation | Passed to `OpenAICompatibleClient(api_key=...)` — defaults to the ZhipuAI GLM-5.1 endpoint at `https://open.bigmodel.cn/api/paas/v4`. Any OpenAI-compatible endpoint works. |

The API key is **not** read from an environment variable automatically — it must be injected via the cogitation function or client constructor. In the default wiring (`__main__.py`), the `Trainer` is created without a `cogitate_fn`, meaning reactive mode will immediately escalate to the human. To enable LLM cogitation, wire a `Cogitator` with your API key into the Trainer factory.

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
uv run python -m harness --config harness.yaml
```

### Override Host / Port

```bash
# CLI flags override the config file's server.host and server.port
uv run python -m harness --config harness.yaml --host 0.0.0.0 --port 9000
```

### All CLI Flags

```
usage: harness [-h] [--config CONFIG] [--host HOST] [--port PORT]

Multi-agent harness runtime

options:
  -h, --help       show this help message and exit
  --config CONFIG  Path to YAML/JSON config file (default: harness.yaml)
  --host HOST      Override WebSocket server host
  --port PORT      Override WebSocket server port
```

### What Happens at Startup

1. **Config loaded** — `harness.yaml` is read and validated.
2. **Bus created** — A `MessageBus` is instantiated.
3. **Embedded participants wired** — Kalvin and Trainer factories are called:
   - `KAgentAdapter` subscribes to address `"kalvin"` on the bus.
   - `Trainer` subscribes to address `"trainer"` on the bus.
4. **Bus thread started** — The bus event loop runs on a daemon thread.
5. **WebSocket server started** — Listens on the configured host:port for client participants.
6. **Blocked** — The main thread runs the async event loop, waiting for SIGINT / SIGTERM.

After startup, the Trainer checks for saved state. If found, it resumes the previous session. Otherwise, it waits for a goal from a human participant (see [Curriculum](#curriculum)).

---

## Architecture

### Module Map

| File           | Purpose                                                                                                                                                                            |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `__main__.py`  | CLI entry point. Loads config, wires participant factories, runs the server.                                                                                                       |
| `server.py`    | `HarnessServer` — loads config, instantiates embedded participants, starts WebSocket server, runs the bus. Also contains `load_config()` and `ConfigError`.                        |
| `bus.py`       | `MessageBus` — thread-safe addressed message router with single-dispatch event loop. Supports wildcard (`"*"`) subscribers for diagnostics.                                        |
| `message.py`   | `Message` — immutable dataclass: `address`, `action`, `message`, `sender`. The bus routes by `address` only; `action` and `message` are interpreted by the recipient.              |
| `adapter.py`   | `KAgentAdapter` — bridge between Kalvin's rationalisation pipeline and the bus. Compiles KScript source, submits entries to KAgent, and routes events back to the original sender. |
| `protocol.py`  | `WebSocketProtocol` — handles WebSocket client connections: registration, bidirectional JSON frame routing, silent-drop disconnect semantics.                                      |
| `protocols.py` | `Participant` protocol — the interface every participant must implement: `address` + `on_message(msg)`.                                                                            |

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

#### Kalvin (`address: "kalvin"`)

The rationalisation engine. The `KAgentAdapter` receives bus messages and delegates to the core `KAgent`:

| Incoming Action | Behaviour                                                                                                                                         |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `submit`        | Compile KScript source from `msg.message`, record sender per entry, call `kagent.rationalise(entry)` for each. Events flow back via `on_event()`. |
| `countersign`   | Call `kagent.countersign(kline)` with the KLine in `msg.message`.                                                                                 |

Events from Kalvin are routed back to the original sender (stored in a sender map keyed by entry identity).

#### Trainer (`address: "trainer"`)

Drives the training loop. Holds LLM access for curriculum generation and reactive scaffolding:

| Incoming Action       | Behaviour                                                                                                               |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `ground` / `frame`    | Kalvin events. S1 events auto-satisfy. S2/S3 events trigger reactive mode (auto-countersign → cogitation → escalation). |
| `input`               | Human input from Slack/TUI. Supports: `goal: <text or path>`, `pause`, `stop`, `resume`, amendment requests, or guidance. |
| `error`               | Kalvin compilation error. Logged and counted toward lesson completion.                                                  |
| `progress`            | Published by the Trainer after each lesson completes. Consumed by TUI and Slack to display status.                      |

**Trainer lifecycle:**

1. **Startup** — checks for saved state (resume) or waits for a goal.
2. **Goal received** — natural language → generate curriculum via GLM-5.1, or file path → load directly.
3. Submits lessons from the curriculum to `"kalvin"` via `submit` messages. Re-reads the curriculum file before each lesson to pick up amendments.
4. Listens for ground/frame events:
   - **S1** (fully grounded) → auto-satisfy, advance curriculum.
   - **S2/S3** → try auto-countersign, then reactive mode:
     - Up to `max_reactive_rounds` of cogitation.
     - If cogitation generates scaffolding → write as a new lesson in the curriculum, then submit.
     - If stuck → **escalate** to human via `"slack"` with a `notify` message.
5. Publishes **progress events** after each lesson completes.
6. When curriculum is complete → end session, persist state, process queued goals.

Any participant can request a curriculum amendment by messaging the Trainer. The Trainer uses GLM-5.1 to generate the amendment and writes it to the curriculum file. Amendments take effect at the next lesson boundary.

### Client Participants

These connect to the harness via WebSocket and register with an address.

#### Slack Participant (`address: "slack"`)

Bridges Slack and the harness:

| Direction       | Action   | Behaviour                                                                                           |
| --------------- | -------- | --------------------------------------------------------------------------------------------------- |
| Harness → Slack | `notify` | Posts the message content to the configured Slack channel.                                          |
| Slack → Harness | `input`  | Forwards human messages to `"trainer"` as `{address: "trainer", action: "input", message: <text>}`. |

**Running the Slack participant:**

```python
from participants import SlackParticipant

agent = SlackParticipant(
    harness_url="ws://localhost:8765",
    channel_id="C01234567",        # Slack channel ID
    # slack_token and app_token default to SLACK_BOT_TOKEN / SLACK_APP_TOKEN env vars
)
await agent.start()
```

#### TUI Participant (`address: "ui"`)

A Textual TUI that displays Kalvin's events and provides ratification (countersign) controls:

- **EventLog** — scrollable log of all events received from the harness.
- **RatifyBar** — button/keyboard shortcut (`Ctrl+R`) to send a `countersign` message to `"kalvin"`.

**Running the TUI participant:**

```bash
uv run python -m participants.tui_client
```

Or programmatically:

```python
from participants import TUIApp

app = TUIApp(harness_url="ws://localhost:8765")
app.run()
```

---

## Wire Protocol

Client participants communicate with the harness over WebSocket using JSON frames.

### Registration (first frame)

```json
{ "register": "slack" }
```

The first frame from any client **must** be a registration frame. The address must be unique — duplicate registrations are rejected with:

```json
{ "error": "address 'slack' already registered" }
```

followed by a WebSocket close (code `4002`).

### Sending a Message

```json
{
  "address": "kalvin",
  "action": "submit",
  "message": "(Mary had a little lamb)\nMHALL = SVO => ..."
}
```

After registration, all outbound frames are messages. The `sender` field is set implicitly by the harness to the registered address.

### Receiving a Message

```json
{
  "address": "slack",
  "action": "notify",
  "message": {
    "reason": "budget_exhaustion",
    "detail": "",
    "lesson_position": 3
  },
  "sender": "trainer"
}
```

Inbound frames are routed by `address` — the harness delivers them to the matching subscriber.

### Error Frames

```json
{ "error": "malformed frame: not valid JSON" }
```

Sent by the harness when a frame cannot be parsed or is structurally invalid.

### Disconnect Semantics

- On disconnect, the bus subscription is **not** removed.
- Messages to a disconnected client are **silently dropped** until the client reconnects with the same address.
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
            "address": "kalvin",
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
    address: str
    def on_message(self, message: Message) -> None: ...
```

### Writing an Embedded Participant

To add a new embedded participant that runs in-process:

1. **Create a class** with `address` and `on_message(msg)`:

   ```python
   class MyParticipant:
       def __init__(self, bus: MessageBus, address: str = "my-agent"):
           self.address = address
           bus.subscribe(address, self.on_message)

       def on_message(self, msg: Message) -> None:
           if msg.action == "ping":
               bus.send(Message(address=msg.sender, action="pong", message="alive"))
   ```

2. **Register a factory** in `__main__.py`:

   ```python
   def my_factory(address: str, bus: MessageBus) -> _AlreadySubscribed:
       participant = MyParticipant(bus, address=address)
       return _AlreadySubscribed(participant)

   server.register_participant_class("MyParticipant", my_factory)
   ```

3. **Add to `harness.yaml`**:

   ```yaml
   participants:
     - address: my-agent
       type: embedded
       class: MyParticipant
   ```

### Writing a Client Participant

Client participants are simpler — just a WebSocket program:

1. Connect to `ws://<host>:<port>`.
2. Send `{"register": "<your-address>"}`.
3. Send/receive JSON message frames.

See the [Wire Protocol](#wire-protocol) section for frame formats.

---

## Message Flow

A typical training cycle looks like this:

```
 Human (Slack)          Trainer              Kalvin
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
3. **Trainer state** is persisted to `trainer_state.json` (if `state_path` is configured). This includes:
   - Curriculum file path
   - Current lesson label (stable identity)
   - Submitted / satisfied / pending entry sets
   - Append-only event log

On restart, the Trainer loads its persisted state and resumes the curriculum from the saved position. The curriculum file itself is never rolled back — only Kalvin's model state can be reset for a full replay.

---

## Testing

```bash
# Run all harness tests
uv run pytest tests/test_harness*.py tests/test_bus.py tests/test_server.py tests/test_adapter.py tests/test_protocol.py tests/test_harness_cli.py

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
| `test_harness_run.py`         | End-to-end: start harness, route messages through full stack |
| `test_harness_cli.py`         | CLI argument parsing                                         |
| `test_harness_events.py`      | Event classification (S1/S2/S3)                              |
| `test_harness_tracking.py`    | CurriculumState tracking sets                                |
| `test_harness_persistence.py` | State save/load round-trip                                   |
