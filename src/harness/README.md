# Harness — Multi-Agent Runtime

The harness is the persistent server that hosts Kalvin's multi-agent training system. It loads participants, routes addressed messages between them via a thread-safe message bus, and exposes a WebSocket endpoint for remote clients.

All participants are **equal peers**: each has a unique address, sends addressed messages through the bus, and receives messages addressed to it. No participant knows it's in a training loop — the dialogue between them *is* the training.

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
2. [Configuration](#configuration)
3. [API Keys & Environment Variables](#api-keys--environment-variables)
4. [Running the Harness](#running-the-harness)
5. [Architecture](#architecture)
6. [Participants](#participants)
7. [Wire Protocol](#wire-protocol)
8. [How to Participate (Write a Custom Client)](#how-to-participate-write-a-custom-client)
9. [Message Flow](#message-flow)
10. [Graceful Shutdown & State Persistence](#graceful-shutdown--state-persistence)
11. [Testing](#testing)

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
  curriculum_path: ""                # Path to curriculum JSON (optional)
  state_path: "trainer_state.json"   # Path for state persistence across restarts
  max_reactive_rounds: 5             # Max reactive scaffolding rounds before escalation

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

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `server` | `host` | `"localhost"` | WebSocket server bind address |
| `server` | `port` | `8765` | WebSocket server bind port |
| `trainer` | `curriculum_path` | `""` | Path to a curriculum JSON file (optional) |
| `trainer` | `state_path` | `"trainer_state.json"` | File for Trainer state persistence |
| `trainer` | `max_reactive_rounds` | `5` | Reactive scaffolding budget before escalation |
| `participants[]` | `address` | — | Unique bus address for this participant |
| `participants[]` | `type` | — | `"embedded"` (loaded in-process) or `"client"` (connects via WebSocket) |
| `participants[]` | `class` | — | Registered class name (e.g. `KAgent`, `Trainer`, `SlackParticipant`) |

### Validation Rules

- Every participant must have a unique `address`. Duplicates are rejected at startup.
- Every participant must have `type` = `"embedded"` or `"client"`.
- Every participant must have a `class` (the registered factory name).
- Embedded participants must have a corresponding factory registered in the CLI (`__main__.py`). Unknown classes are rejected at startup.

---

## API Keys & Environment Variables

The harness itself requires no API keys. API keys are only needed by specific participants:

### Slack Participant (optional)

| Variable | Required | Description |
|----------|----------|-------------|
| `SLACK_BOT_TOKEN` | Yes (if using Slack) | Slack Bot OAuth token (`xoxb-...`). Used to post messages to channels. |
| `SLACK_APP_TOKEN` | Yes (if using Slack) | Slack App-Level token (`xapp-...`). Used for Socket Mode to receive human messages. |

These can also be passed as constructor arguments to `SlackParticipant` instead of environment variables.

### Trainer / Cogitation (optional)

The Trainer's reactive mode uses the `Cogitator` with an `OpenAICompatibleClient` that calls a GLM-5.1-compatible LLM API. To enable this:

| Variable | Required | Description |
|----------|----------|-------------|
| *(API key)* | If using cogitation | Passed to `OpenAICompatibleClient(api_key=...)` — defaults to the ZhipuAI GLM-5.1 endpoint at `https://open.bigmodel.cn/api/paas/v4`. Any OpenAI-compatible endpoint works. |

The API key is **not** read from an environment variable automatically — it must be injected via the cogitation function or client constructor. In the default wiring (`__main__.py`), the `Trainer` is created without a `cogitate_fn`, meaning reactive mode will immediately escalate to the human. To enable GLM-5.1 cogitation, wire a `Cogitator` with your API key into the Trainer factory.

### Required Python Packages

| Package | When Needed | Install |
|---------|-------------|---------|
| `websockets` | Always (harness core) | `uv sync` |
| `pyyaml` | Always (config loading) | `uv sync` |
| `openai` | Reactive scaffolding (GLM-5.1) | `uv sync --extra trainer` |
| `slack-sdk` | Slack participant | `uv sync` |
| `textual` | TUI participant | `uv sync` |

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

---

## Architecture

### Module Map

| File | Purpose |
|------|---------|
| `__main__.py` | CLI entry point. Loads config, wires participant factories, runs the server. |
| `server.py` | `HarnessServer` — loads config, instantiates embedded participants, starts WebSocket server, runs the bus. Also contains `load_config()` and `ConfigError`. |
| `bus.py` | `MessageBus` — thread-safe addressed message router with single-dispatch event loop. Supports wildcard (`"*"`) subscribers for diagnostics. |
| `message.py` | `Message` — immutable dataclass: `address`, `action`, `message`, `sender`. The bus routes by `address` only; `action` and `message` are interpreted by the recipient. |
| `adapter.py` | `KAgentAdapter` — bridge between Kalvin's rationalisation pipeline and the bus. Compiles KScript source, submits entries to KAgent, and routes events back to the original sender. |
| `protocol.py` | `WebSocketProtocol` — handles WebSocket client connections: registration, bidirectional JSON frame routing, silent-drop disconnect semantics. |
| `protocols.py` | `Participant` protocol — the interface every participant must implement: `address` + `on_message(msg)`. |

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

| Incoming Action | Behaviour |
|-----------------|-----------|
| `submit` | Compile KScript source from `msg.message`, record sender per entry, call `kagent.rationalise(entry)` for each. Events flow back via `on_event()`. |
| `countersign` | Call `kagent.countersign(kline)` with the KLine in `msg.message`. |

Events from Kalvin are routed back to the original sender (stored in a sender map keyed by entry identity).

#### Trainer (`address: "trainer"`)

Drives the training loop:

| Incoming Action | Behaviour |
|-----------------|-----------|
| `ground` / `frame` | Kalvin events. S1 events auto-satisfy. S2/S3 events trigger reactive mode (auto-countersign → cogitation → escalation). |
| `input` | Human input from Slack. Supports commands: `goal: <text>`, `pause`, `stop`, `resume`, or freeform guidance text. |
| `error` | Kalvin compilation error. Logged and counted toward lesson completion. |

**Trainer lifecycle:**

1. Receives `goal:` input → starts a training session.
2. Submits lessons from the curriculum to `"kalvin"` via `submit` messages.
3. Listens for ground/frame events:
   - **S1** (fully grounded) → auto-satisfy, advance curriculum.
   - **S2/S3** → try auto-countersign, then reactive mode:
     - Up to `max_reactive_rounds` of cogitation.
     - If cogitation generates scaffolding → submit to Kalvin.
     - If stuck → **escalate** to human via `"slack"` with a `notify` message.
4. When curriculum is complete → end session, persist state, process queued goals.

### Client Participants

These connect to the harness via WebSocket and register with an address.

#### Slack Participant (`address: "slack"`)

Bridges Slack and the harness:

| Direction | Action | Behaviour |
|-----------|--------|-----------|
| Harness → Slack | `notify` | Posts the message content to the configured Slack channel. |
| Slack → Harness | `input` | Forwards human messages to `"trainer"` as `{address: "trainer", action: "input", message: <text>}`. |

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
{"register": "slack"}
```

The first frame from any client **must** be a registration frame. The address must be unique — duplicate registrations are rejected with:

```json
{"error": "address 'slack' already registered"}
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
  "message": {"reason": "budget_exhaustion", "detail": "", "lesson_position": 3},
  "sender": "trainer"
}
```

Inbound frames are routed by `address` — the harness delivers them to the matching subscriber.

### Error Frames

```json
{"error": "malformed frame: not valid JSON"}
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
   - Curriculum position
   - All lessons
   - Submitted / satisfied / pending entry sets
   - Append-only event log

On restart, the Trainer can call `CurriculumState.load(path)` to resume from the persisted state.

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

| File | Coverage |
|------|----------|
| `test_bus.py` | MessageBus subscribe, send, dispatch, wildcard, stop |
| `test_harness.py` | Config loading, validation, ParticipantConfig |
| `test_server.py` | HarnessServer setup, embedded participant wiring |
| `test_adapter.py` | KAgentAdapter submit, countersign, event routing |
| `test_protocol.py` | WebSocketProtocol registration, frame parsing |
| `test_harness_run.py` | End-to-end: start harness, route messages through full stack |
| `test_harness_cli.py` | CLI argument parsing |
| `test_harness_events.py` | Event classification (S1/S2/S3) |
| `test_harness_tracking.py` | CurriculumState tracking sets |
| `test_harness_persistence.py` | State save/load round-trip |
