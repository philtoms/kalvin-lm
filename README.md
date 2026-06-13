# Kalvin

A rationalising system whose entire world is built from **klines** — node-like structures with a signature (identity) and nodes (relationships). Kalvin receives new information, rationalises it against existing knowledge, and produces a response with a **significance** measurement indicating how well it understood.

## What Kalvin Does

Kalvin is not an oracle. Every response carries significance — a structural measurement of how grounded the response is in what Kalvin already knows:

| Level  | Meaning              | Kalvin says                     |
| ------ | -------------------- | ------------------------------- |
| **S1** | Fully grounded       | "I know this."                  |
| **S2** | Partially understood | "I understand some of it."      |
| **S3** | Recognised aspects   | "This reminds me of something." |
| **S4** | Completely novel     | "I've never seen this before."  |

This makes every response actionable. The other agent in the dialogue knows exactly where understanding is strong and where it breaks down.

## The Multi-Agent System

Kalvin operates inside a **harness** — a persistent server that loads agents and routes messages between them. All participants are equal: each has a unique address, sends addressed messages through the harness, and receives messages addressed to it.

```
Harness Server
  ├── Kalvin (the rationalisation engine)
  ├── Trainer (drives the training loop, generates scaffolding)
  ├── Slack agent (human-in-the-loop via Slack)
  └── TUI agent (human-in-the-loop via terminal)
```

No participant knows it's in a training loop. Each simply receives and responds. The dialogue between them is the training.

### How Training Works

1. The **human** provides a goal and initial KScript via Slack.
2. The **Trainer** breaks the goal into a curriculum and submits lessons to Kalvin.
3. **Kalvin** rationalises each lesson, emitting events with significance.
4. If lessons land at S1, the Trainer advances. If S2/S3, the Trainer enters reactive mode:
   - Cogitates (via LLM agent) on what scaffolding to write.
   - Submits reactive scaffolding to Kalvin.
   - Auto-countersigns proposals that structurally match expectations.
5. If stuck after N rounds, the Trainer **escalates** to the human via Slack.

## KScript — The Language of Teaching

KScript is a DSL for constructing knowledge graphs. It compiles declarative scripts into klines that Kalvin rationalises.

```kscript
(Mary had a little lamb)
MHALL = SVO =>
   S(ubject) = M
   V(erb) = H
   O(bject) = ALL =>
     A > D(et)
     L > M(od)
     L > O
```

| Operator    | Syntax     | Significance | Meaning                |
| ----------- | ---------- | ------------ | ---------------------- |
| Countersign | `A == B`   | S1           | Mutual / bidirectional |
| Undersign   | `A = B`    | S1           | Unconditional          |
| Canonize    | `A => B C` | S2           | Canonical              |
| Connotate   | `A > B`    | S3           | Connotative            |
| Unsigned    | `A`        | S4           | Identity only          |

Indented blocks are **scaffolding** — context that steers Kalvin toward understanding the parent line.

## Getting Started

### Install

```bash
uv sync
```

### Run Tests

```bash
uv run pytest
```

> **NLP tokenizer tests** require BPE + grammar data assets that live under
> the gitignored `data/tokenizer/` directory (binary assets, not checked in).
> These tests **gracefully skip** when the assets are absent, so a fresh clone
> reports a clean pass with skips rather than errors. To run them, generate the
> assets once:
>
> ```bash
> bash scripts/rebuild-tokenizer-data.sh
> ```

### KScript CLI

```bash
# Compile a script
uv run python -m kscript script.ks

# Output formats
uv run python -m kscript script.ks -out output.json
uv run python -m kscript script.ks -out output.bin
```

### KScript TUI (standalone)

```bash
uv run python -m ui.kscript
```

### Harness Server (multi-agent)

```bash
uv run python -m harness --config harness.yaml
```

## Project Structure

```
src/
├── kalvin/               # Core rationalisation engine
│   ├── kline.py          #   KLine data structure
│   ├── signature.py      #   Signature computation
│   ├── tokenizer.py      #   Tokenizer interface
│   ├── mod_tokenizer.py  #   Mod32/Mod64 tokenizers
│   ├── model.py          #   3-tier knowledge graph (STM → Frame → Base)
│   ├── agent.py          #   KAgent — rationalisation pipeline + Cogitator
│   ├── events.py         #   Event definitions
│   └── ...
├── kscript/              # KScript DSL
│   ├── lexer.py          #   Lexer (source → tokens)
│   ├── parser.py         #   Parser (tokens → AST)
│   ├── compiler.py       #   Compiler (AST → compiled entries)
│   └── decompiler.py     #   Decompiler (entries → KScript)
├── harness/              # Multi-agent harness server
│   ├── server.py         #   Harness server + config loading
│   ├── bus.py            #   Addressed message bus
│   ├── adapter.py        #   KAgent ↔ bus adapter
│   └── protocol.py       #   WebSocket wire protocol
├── trainer/              # Trainer participant
│   ├── trainer.py        #   Curriculum execution + reactive mode
│   ├── curriculum.py     #   Curriculum state + persistence
│   └── cogitation.py     #   LLM agent integration
└── participants/         # Client participants
    ├── slack_agent.py    #   Slack ↔ harness
    └── tui_client.py     #   TUI ↔ harness
```

## Documentation

| Document                                                     | Purpose                                            |
| ------------------------------------------------------------ | -------------------------------------------------- |
| [`CONTEXT.md`](CONTEXT.md)                                   | Domain glossary — precise definitions of all terms |
| [`docs/kalvin-introduction.md`](docs/kalvin-introduction.md) | What Kalvin is and why it works this way           |
| [`docs/kalvin-origin.md`](docs/kalvin-origin.md)             | Conceptual model and philosophy                    |
| [`docs/roadmap.md`](docs/roadmap.md)                         | Build phases, dependency graph, current status     |
| [`docs/adr/`](docs/adr/)                                     | Architectural decision records                     |
| [`specs/`](specs/)                                           | Testable behavioural contracts                     |
| [`plans/`](plans/)                                           | Implementation strategy and test mapping           |

## Development

```bash
uv run pytest                    # Run tests
uv run ruff format .             # Format code
uv run ruff check .              # Lint
```
