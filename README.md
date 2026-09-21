# Kalvin

A rationalising system whose entire world is built from **klines** — node-like structures with a signature (identity) and nodes (relationships). Kalvin receives new information, rationalises it against existing knowledge, and produces a response with a **significance** measurement indicating how well it understood.

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

## Project Structure

```
src/
├── kalvin/               # Core rationalisation engine
│   ├── engine.py         #   Rationalising engine: (state, incoming) → batch
│   ├── engine_state.py   #   EngineState — engine-held memory
│   ├── kline.py          #   KLine structure + predicates (is_canon/is_misfit/…)
│   ├── kvalue.py         #   KValue — the value a node/signature carries
│   ├── derivation.py     #   Derivation — the algebra core (§6–10)
│   ├── hop.py            #   Hop + run_hops — the strategy unit over derivations
│   ├── abstract.py       #   Abstract interfaces (KSignifier, …)
│   ├── agent_codec.py    #   Agent (de)serialisation codec
│   ├── cogitator.py      #   Cogitation — model-traversal slow path
│   ├── expand.py         #   Expand cogitation strategy
│   ├── proposals.py      #   Proposal construction for ratification
│   ├── significance.py   #   Significance levels (S1–S4) + normalisation
│   ├── signifier.py      #   NLPSignifier — NLP-backed signifier
│   ├── tokenizer.py      #   Tokenizer interface
│   ├── bpe_tokenizer.py  #   BPE tokenizer (production)
│   ├── nlp_tokenizer.py  #   NLP-backed tokenizer
│   ├── mod_tokenizer.py  #   Modular bit-packed tokenizer
│   ├── stm.py            #   Short-term memory
│   ├── model.py          #   Tiered memory
│   ├── rationaliser.py   #   Rationaliser pipeline + Cogitator wiring
│   ├── events.py         #   Event definitions + EventBus
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

dev/
└── dialogue/             # Development harness (engine tuning)
    ├── harness.py        #   The non-judging compile→feed→present loop + CLI
    ├── structural.py     #   Structural supervisor (harness escalation seam)
    ├── decoder.py        #   Dialogue-table decoder (table → DecodedTurns)
    └── probe_*.py        #   Ad-hoc investigative scripts (kept)
```

## Documentation

| Document                                     | Purpose                  |
| -------------------------------------------- | ------------------------ |
| [`docs/kalvin-algebra.md`](kalvin-algbra.md) | normative definition     |
| [`CONTEXT.md`](CONTEXT.md)                   | domain glossary          |
| [`.llm-wiki/`](.llm-wiki)                    | LLM Wiki knowledge vault |

Source is the truth document for implemented behaviour. The LLM Wiki
(`.llm-wiki/wiki/`) holds distilled entity/concept pages derived from the
glossary and from captured sources.

## Development

```bash
uv run pytest                    # Run tests
uv run ruff format .             # Format code
uv run ruff check .              # Lint
```
