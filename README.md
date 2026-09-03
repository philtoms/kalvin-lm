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
2. The **Trainer** breaks the goal into scripts and submits them to Kalvin, lesson by lesson.
3. **Kalvin** rationalises each lesson, emitting events with significance.
4. If lessons land at S1, the Trainer advances. If S2/S3, the Trainer enters reactive mode:
   - Cogitates (via LLM agent) on what scaffolding to write.
   - Submits reactive scaffolding to Kalvin.
   - Auto-countersigns proposals that structurally match expectations.
5. If stuck after N rounds, the Trainer **escalates** to the human via Slack.

## KScript — The Language of Teaching

KScript is a DSL for writing structured klines. It compiles declarative scripts into klines that Kalvin rationalises.

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

| Operator    | Syntax     | Significance | Meaning                             |
| ----------- | ---------- | ------------ | ----------------------------------- |
| Countersign | `A == B`   | S1           | Mutual / bidirectional              |
| Denote      | `A = B`    | S3           | Objective — A denotes B (B is an A) |
| Canonize    | `A => B C` | S2           | Canonical                           |
| Connote     | `A > B`    | S3           | Connotative — A connotes B          |
| Undefined   | `A`        | S4           | Identity only                       |

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

> **Tokenizer tests** require BPE + grammar data assets that live under
> the gitignored `data/tokenizer/` directory (binary assets, not checked in).
> These tests **gracefully skip** when the assets are absent, so a fresh clone
> reports a clean pass with skips rather than errors. To run them, generate the
> assets once:
>
> ```bash
> bash scripts/rebuild-tokenizer-data.sh
> ```

### Dialogue Harness

The dialogue harness compiles a KScript script, feeds each entry to the
rationalising engine one at a time, and presents the resulting trace. It is
synchronous and non-judging — the trainer (a pi agent, outside the loop)
reads the trace and decides what to edit.

```bash
PYTHONPATH=src python -m dialogue.harness path/to/script.ks
PYTHONPATH=src python -m dialogue.harness path/to/script.ks -v   # show hex signatures
PYTHONPATH=src python -m dialogue.harness path/to/script.ks -s expand  # cogitation strategy
```

KScript itself is a library (`ks.compiler.compile_source`); there is no
standalone compiler CLI. Compiled klines are produced in-process by the
harness, the multi-agent runtime, or directly:

```python
from ks.compiler import compile_source
entries = compile_source("(hello world)\nHW = HELLO WORLD")
```

### Multi-Agent Harness Server

```bash
uv run python -m training.harness --config training.harness.yaml
uv run python -m training.harness --host 0.0.0.0 --port 9000   # overrides
```

### Auto-Tune

```bash
uv run python -m training.auto_tune --help
```

## Project Structure

```
src/
├── kalvin/               # Core rationalisation engine
│   ├── kline.py          #   KLine structure + predicates (is_canon/is_misfit/…)
│   ├── kvalue.py         #   KValue — the value a node/signature carries
│   ├── abstract.py       #   Abstract interfaces (KSignifier, …)
│   ├── agent_codec.py    #   Agent (de)serialisation codec
│   ├── cogitator.py      #   Cogitation — model-traversal slow path
│   ├── expand.py         #   Expand cogitation strategy
│   ├── proposals.py      #   Proposal construction for ratification
│   ├── significance.py   #   Significance levels (S1–S4) + normalisation
│   ├── signifier.py      #   NLPSignifier — NLP-backed signifier
│   ├── stm.py            #   Short-term memory
│   ├── model.py          #   Tiered memory
│   ├── rationaliser.py   #   Rationaliser pipeline + Cogitator wiring
│   ├── events.py         #   Event definitions + EventBus
│   ├── tokenizer.py      #   Tokenizer interface
│   ├── nlp_tokenizer.py  #   NLP-backed tokenizer
│   └── paths.py          #   Resolved filesystem paths
├── ks/                   # KScript DSL
│   ├── lexer.py          #   Lexer (source → tokens)
│   ├── parser.py         #   Parser (tokens → AST)
│   ├── ast.py            #   AST node definitions
│   ├── ast_emitter.py    #   ASTEmitter (AST → symbolic entries)
│   ├── token.py          #   KScript token definitions
│   ├── binding_scope.py  #   Word binding resolution
│   ├── token_encoder.py  #   TokenEncoder (symbolic → encoded KLines)
│   └── compiler.py       #   Compiler (orchestrator; source → KLines)
├── dialogue/             # Dialogue harness + rationalising engine
│   ├── engine.py         #   Stateless engine: (state, incoming) → (batch, observations)
│   ├── engine_state.py   #   EngineState — engine-held memory
│   ├── harness.py        #   Harness: compile → feed → present
│   ├── runner.py         #   Coverage-tracking wildcard subscriber over the bus
│   ├── actors.py         #   Actor base + role impls
│   ├── rationalise.py    #   Rationaliser + RationaliserState
│   ├── synthesize.py     #   Synthesizer (real-actor side of the triad)
│   ├── decoder.py        #   Turn decoding (DecodedTurn)
│   └── pivot_fill.py     #   Expand-fit misfit strategy
└── training/             # Multi-agent training runtime
    ├── harness/          #   Harness server
    │   ├── __main__.py   #     CLI entry point
    │   ├── server.py     #     Harness server + config loading
    │   ├── bus.py        #     Addressed message bus
    │   ├── message.py    #     Message type
    │   ├── adapter.py    #     Rationaliser ↔ bus adapter
    │   ├── llm.py        #     OpenAICompatibleClient
    │   ├── protocol.py   #     WebSocket wire protocol
    │   ├── protocols.py  #     Typed participant protocols
    │   └── constants.py  #     Shared constants
    ├── trainer/          #   Trainer participant
    │   ├── trainer.py             #   Script execution + decision gating
    │   ├── reactor.py             #   Mechanical S2/S3 handling (auto-countersign, dedup)
    │   ├── curriculum.py          #   Curriculum state + persistence
    │   ├── curriculum_document.py #   Markdown curriculum parser + amendments
    │   └── curriculum_generator.py#   LLM-based curriculum generation
    ├── supervisors/      #   Client supervisors (Slack, TUI, CLI, LLM)
    │   ├── slack_agent.py     #   Slack ↔ harness
    │   ├── tui_client.py      #   TUI ↔ harness
    │   ├── tui_regions.py     #   Textual widgets for the TUI
    │   ├── cli_supervisor.py  #   Headless file-based supervisor (auto-tune)
    │   ├── cli_events.py      #   Auto-tune event enrichment
    │   ├── commands.py        #   Shared free-text → command parser
    │   └── llm_supervisor.py  #   LLM decider + reactive scaffolding pipeline
    └── auto_tune/        #   Auto-tune orchestrator
        ├── __main__.py   #     `python -m training.auto_tune` entry point
        ├── cli.py        #     CLI
        ├── lifecycle.py  #     Run lifecycle
        ├── orchestrate.py #     Session orchestration
        ├── session.py    #     Session state
        └── snapshots.py  #     Code/config snapshots across runs
```

## Documentation

| Document                   | Purpose                  |
| -------------------------- | ------------------------ |
| [`CONTEXT.md`](CONTEXT.md) | domain glossary          |
| [`.llm-wiki/`](.llm-wiki)  | LLM Wiki knowledge vault |

Source is the truth document for behaviour. `CONTEXT.md`'s **Project
Navigation** section maps the source tree to its concerns. The LLM Wiki
(`.llm-wiki/wiki/`) holds distilled entity/concept pages derived from the
glossary and from captured sources.

## Development

```bash
uv run pytest                    # Run tests
uv run ruff format .             # Format code
uv run ruff check .              # Lint
```
