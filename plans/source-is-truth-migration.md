# Plan — Source-is-Truth Migration

**Status:** disposable working plan. Update freely. Delete on completion.
**Created:** 2026-08-06

## Goal

Retire the cascade documentation model (`specs/`, `plans/`, `docs/cascade-development.md`,
`docs/ARCHIVE.md`) in favour of **source-as-truth**. What survives:

- `CONTEXT.md` — kept, and gains a new **Project Navigation** section mapping the code.
- `docs/kalvin-vision.md` — kept (the WHY).
- `AGENTS.md` — kept, but its instruction to "read docs/cascade-development.md" is removed and
  the documentation-maintenance model is rewritten to source-is-truth.

Everything else is deleted once its source has been appraised as a viable truth document.

## Termination principle

The version of the truth we are seeking is **in the code**. A spec/plan is **retired** once the
agent has read both the spec and its source and confirmed that the source genuinely implements the
behaviour the spec describes. Once code is appraised as a viable truth document, the spec is simply
*deleted* — its content is not transplanted elsewhere.

This is not a docstring-inflation exercise. Specs are not homed into comments. The appraisal's job
is to **verify** the code, not to reproduce the spec inside it. Do **not** add comments or extend
docstrings to preserve spec content; that defeats the point.

The only exceptions, applied sparingly:

- A **genuine correctness gap** — the code is wrong or stale relative to what the spec nailed down.
  Fix the *code* (behaviour), not by annotating it.
- A point the code expresses only implicitly, where a reader could not recover it without the spec,
  **and** a single short standard-practice docstring genuinely aids navigation. This is the rare
case, not the default. When in doubt, leave the code alone.
- A **domain-term** that belongs in `CONTEXT.md`'s glossary (it already may). Migrate the term, not
  the spec.

Reading and aligning specs against source **is** the work of the appraisal phase; rewriting source
to mirror specs is not.

## End-state layout

```
CONTEXT.md                         (glossary + operating notes + Project Navigation)
docs/kalvin-vision.md              (WHY)
docs/cascade-development.md        → DELETED
docs/ARCHIVE.md                    → folded into a final Archive row, then DELETED
specs/                             → DELETED (all 19 files)
plans/                             → DELETED (all 13 files, including this one last)
```

## Plan-tracking conventions

- Use a single checklist per spec. A spec moves: `PENDING → APPRAISING → RETIRED`.
- "RETIRED" = deleted in a commit; note any carry-over in the commit message and, where non-trivial,
  add a row to a transient migration log at the bottom of this file.
- `git` commits require explicit user confirmation (per AGENTS.md) unless in a worktree.
- Update `docs/ARCHIVE.md` with one summarising row per retired spec as we go (it is itself deleted
  at the very end, but serves as the in-flight ledger until then).

---

## Phase 0 — Scaffold

- [x] Survey current docs, source tree, and cross-references. (done)
- [x] Draft this plan.
- [ ] **Commit current uncommitted work first** (`specs/cogitator.md`, `specs/model.md`,
      `src/kalvin/proposals.py` are dirty) — needs user confirmation.
- [ ] Add a transient **Project Navigation** skeleton section to `CONTEXT.md` (sections only,
      placeholders). Filled incrementally as each source area is appraised.

## Phase 1 — Source appraisal & spec retirement

One workstream per source area. Each workstream: read spec(s) + read source + tests → verify
the code implements the spec (fix code only on a genuine correctness gap; do **not** annotate to
preserve spec content) → retire spec(s) (delete + ARCHIVE row + fix dangling source/test
references) → fill the matching Project Navigation entry.

Order is by dependency, leaf-first (foundational structures before the things built on them),
so each appraisal has stable referents.

### 1a — Signature & Token primitives  *(specs/signature.md, specs/tokenizer.md, specs/nlp_tokenizer.md)* ✅
- Source: `src/kalvin/signifier.py`, `src/kalvin/tokenizer.py`, `src/kalvin/nlp_tokenizer.py`, `src/kalvin/abstract.py`.
- Note: `agent_codec.py` was listed here but is Agent-persistence serialization — refiled to 1c.
- [x] Appraise (read spec + source + tests; verify; fix code only on genuine gaps).
- [x] Fill navigation entries.
- [x] Retire specs → ARCHIVE rows → fix references.

### 1b — KLine & KValue  *(specs/kline.md, specs/kvalue.md)* ✅
- Source: `src/kalvin/kline.py`, `src/kalvin/kvalue.py`, `src/kalvin/abstract.py`, `src/kalvin/events.py`.
- [x] Appraise (read spec + source + tests; verify; fix code only on genuine gaps).
- [x] Fill navigation entries.
- [x] Retire specs → ARCHIVE rows → fix references.

### 1c — Model, STM, Significance, Expansion  *(specs/model.md, specs/stm.md)* ✅
- Source: `src/kalvin/model.py`, `src/kalvin/stm.py`, `src/kalvin/significance.py`, `src/kalvin/expand.py`,
  `src/kalvin/proposals.py`, `src/kalvin/agent_codec.py`, `src/kalvin/paths.py`.
- Note: `specs/rationaliser.md §significance` was listed here but rationaliser is its own spec — handled in 1d.
  `agent_codec.py` refiled here from 1a (Agent-persistence serialization).
- Appraisal note: spec's standalone `is_s1` free function does not exist in code — S1 recognition is `model.grounded()` + structural predicates at call sites. Naming divergence only; no behaviour gap, no code change.
- [x] Appraise (read spec + source + tests; verify; fix code only on genuine gaps).
- [x] Fill navigation entries.
- [x] Retire specs → ARCHIVE rows → fix references.

### 1d — Cogitator & Rationaliser  *(specs/cogitator.md, specs/rationaliser.md)* ✅
- Source: `src/kalvin/cogitator.py`, `src/kalvin/rationaliser.py`, `src/kalvin/proposals.py`.
- Appraisal notes:
  - `cogitator.py` docstring referenced `specs/cogitator-drain.md` — a phantom spec that never existed. Drain semantics live in `cogitator.md` §Lifecycle and `cogitator.py:drain()`. Dangling pointer removed.
  - `rationaliser.md` test-row AGT-7 (assign-sig-if-missing) was stale vs the spec body and the code's `assert` — code is the truth, no change.
- [x] Appraise (read spec + source + tests; verify; fix code only on genuine gaps).
- [x] Fill navigation entries.
- [x] Retire specs → ARCHIVE rows → fix references.

### 1e — KScript  *(specs/kscript.md)* ✅
- Source: `src/ks/` (lexer, parser, ast, ast_emitter, binding_scope, compiler, token, token_encoder).
- Largest single spec (900 lines). Appraised across all 8 modules + 8 test files (327 tests).
- [x] Appraise (read spec + source + tests; verify; fix code only on genuine gaps).
- [x] Fill navigation entries.
- [x] Retire spec → ARCHIVE row → fix references.

### 1f — Training: trainer, reactor, curriculum  *(specs/curriculum.md, specs/trainer-satisfaction.md, specs/training-log.md)* ✅
- Source: `src/training/trainer/` (curriculum_document, curriculum, curriculum_generator, reactor, trainer).
- Appraisal note: `trainer-satisfaction.md` described a transitional "paced-loop" Prompted/Withheld partition + Held-Index design its own header marked superseded by the dialogue path. The production `Trainer` uses Reactor auto-countersign + supervisor escalation instead. Code is the truth; spec was stale on the paced-loop mechanics.
- [x] Appraise (read spec + source + tests; verify; fix code only on genuine gaps).
- [x] Fill navigation entries.
- [x] Retire specs → ARCHIVE rows → fix references.

### 1g — Harness & supervisors  *(specs/harness-server.md, specs/supervisor-decision.md)* ✅
- Source: `src/training/harness/`, `src/training/supervisors/`.
- Note: `supervisor-decision.md`'s header marked it superseded by the dialogue path but still the running production path — code is the truth.
- [x] Appraise (read spec + source + tests; verify; fix code only on genuine gaps).
- [x] Fill navigation entries.
- [x] Retire specs → ARCHIVE rows → fix references.

### 1h — Dialogue subsystem  *(specs/dialogue-driven-training.md, specs/dialogue-cogitation.md)* ✅
- Source: `src/dialogue/` (runner, actors, decoder, rationalise, synthesize).
- Note: both specs self-declared as "working sketches, not frozen contracts" stating the code is the source of truth.
- [x] Appraise (read spec + source + tests; verify; fix code only on genuine gaps).
- [x] Fill navigation entries.
- [x] Retire specs → ARCHIVE rows → fix references.

### 1i — Auto-tune  *(specs/auto-tune.md)* ✅
- Source: `src/training/auto_tune/` (session, lifecycle, orchestrate, snapshots, cli) + `src/training/supervisors/cli_supervisor.py`/`cli_events.py`.
- Also folded in the deferred `specs/signifier.md` (missed in P1a; source appraised there).
- Also swept `@kvalue spec` stragglers missed in P1b across `adapter.py` (6), `protocol.py` (2), `significance.py` (2), `events.py` (1), `cli_events.py` (1), `test_agent.py` (1).
- [x] Appraise (read spec + source + tests; verify; fix code only on genuine gaps).
- [x] Fill navigation entries.
- [x] Retire specs → ARCHIVE rows → fix references.

---

**Phase 1 complete.** All 19 specs retired. `specs/` is empty. The only remaining spec/plan references in the repo are inside `plans/` (retired wholesale in Phase 2) and `docs/ARCHIVE.md` (the ledger) / `docs/cascade-development.md` (deleted in Phase 3).

## Phase 2 — Plans retirement ✅

Plans are HOW; once specs are gone and source is truth, plans are spent. Retired wholesale.

- [x] All 13 plans deleted (impl/* ×3, implement-*.md ×8, role-based-routing.md, remove-compound-token.md) → one summarising ARCHIVE row → delete.
- [x] `plans/source-is-truth-migration.md` retained (this plan; deleted in Phase 3).
- [x] Full suite: 1220 passed.

## Phase 3 — Cascade teardown & CONTEXT finalisation

- [ ] Delete `docs/cascade-development.md`.
- [ ] Rewrite `AGENTS.md` "Coding activity" section for source-is-truth:
      remove the "read cascade-development.md" step; replace the documentation-maintenance model
      with "source is the truth document — when changing behaviour, update source and CONTEXT in
      the same change." Keep the docstring minimalism rule.
- [ ] Also update the in-repo skills that reference cascade-development
      (`.pi/skills/auto-tune/SKILL.md`, `.pi/skills/dialogue-dev/SKILL.md`, `.pi/skills/docs-reset/SKILL.md`)
      if they instruct reading the cascade doc — verify and patch.
- [ ] Finalise `CONTEXT.md` **Project Navigation** section (complete, accurate, no placeholders).
- [ ] Delete `docs/ARCHIVE.md` (its ledger is now permanent history in git).
- [ ] Delete this plan file.
- [ ] Final repo-wide sweep: confirm no remaining references to `specs/`, `plans/`,
      `docs/cascade-development.md` in source/tests/docs.
- [ ] Update `README.md` if it references the cascade structure.

## Phase 4 — Verify & close

- [ ] Full test run passes.
- [ ] `rg -n "specs/|plans/|cascade-development" .` returns nothing outside `.git`/history.
- [ ] `CONTEXT.md` reads cleanly as the single navigation + glossary source.
- [ ] Ask user to confirm before the final teardown commit.

---

## Migration log (transient)

Append one line per retired artifact: `artifact → carry-over destination (if any)`.

- `specs/signature.md` → deleted; truth in `kline.py`/`signifier.py`/`abstract.py`; term in CONTEXT (Signature). Tag `source-is-truth-2026-08-06`.
- `specs/tokenizer.md` → deleted; truth in `abstract.py` (`KTokenizer`), `tokenizer.py`. Tag `source-is-truth-2026-08-06`.
- `specs/nlp_tokenizer.md` → deleted; truth in `nlp_tokenizer.py`. Tag `source-is-truth-2026-08-06`.
- `specs/kline.md` → deleted; truth in `kline.py` (KLine + structural predicates); terms in CONTEXT (Structure). Tag `source-is-truth-2026-08-06`.
- `specs/kvalue.md` → deleted; truth in `kvalue.py` (KValue), `events.py` (RationaliseEvent); term in CONTEXT (KValue). Tag `source-is-truth-2026-08-06`.
  - Stripped dangling `@specs/…` refs from 2 source docstrings (`kline.py`, `kvalue.py`) and 4 test docstrings (`test_kvalue.py`, `test_kline.py` ×2, `test_misfit.py`, `test_agent_codec.py`).
  - Cross-refs in other pending specs (`dialogue-driven-training`, `training-log`, `trainer-satisfaction`, `harness-server`) left to dissolve in their own workstreams.
- `specs/stm.md` → deleted; truth in `stm.py`. Tag `source-is-truth-2026-08-06`.
- `specs/model.md` → deleted; truth in `model.py`, `significance.py`, `expand.py`, `proposals.py`. Spec's `is_s1` was a naming abstraction the code expresses via `model.grounded()` — no behaviour gap. Tag `source-is-truth-2026-08-06`.
  - Stripped 7 dangling refs: `model.py` (3: module docstring ×2, `unpack` docstring), `significance.py` (1 comment), `test_model.py`/`test_expand.py`/`test_countersign_resolution.py` (3).
- `specs/cogitator.md` → deleted; truth in `cogitator.py` (+`expand.py`/`proposals.py`). Tag `source-is-truth-2026-08-06`.
- `specs/rationaliser.md` → deleted; truth in `rationaliser.py`. Spec AGT-7 stale vs code's `assert`. Tag `source-is-truth-2026-08-06`.
  - Stripped 14 dangling refs across `cogitator.py` (4, incl. phantom `cogitator-drain.md` pointer), `rationaliser.py` (5: module docstring + 3 inline comments + 1), `kline.py` (1), `harness/adapter.py` (1), `test_ks.py`/`test_ks_token_encoder.py`/`test_cogitator_drain.py` (3).
- `specs/kscript.md` → deleted; truth in `src/ks/` (8 modules). Tag `source-is-truth-2026-08-06`.
  - Stripped 7 dangling refs across `token.py`, `lexer.py`, `ast.py`, `compiler.py`, `__init__.py` (module docstrings), `token_encoder.py` (1 comment), `test_ks.py` (header).
- `specs/curriculum.md` → deleted; truth in `curriculum_document.py`, `curriculum.py`, `curriculum_generator.py`. Tag `source-is-truth-2026-08-06`.
- `specs/trainer-satisfaction.md` → deleted; truth in `trainer.py`, `reactor.py`. Spec's paced-loop partition was a superseded transitional design; production uses Reactor + supervisor escalation. Tag `source-is-truth-2026-08-06`.
- `specs/training-log.md` → deleted; truth in the `logging` calls across `trainer.py`, `reactor.py`, `harness/adapter.py`. Tag `source-is-truth-2026-08-06`.
  - Stripped 1 dangling ref (`test_training_log.py` header). Reactor/trainer `@specs/supervisor-decision.md` refs deferred to 1g.
- `specs/harness-server.md` → deleted; truth in `src/training/harness/` + `src/training/supervisors/`. Tag `source-is-truth-2026-08-06`.
- `specs/supervisor-decision.md` → deleted; truth in the Trainer decision-gate logic + `llm_supervisor.py`. Spec header marked it superseded-but-running; code is the truth. Tag `source-is-truth-2026-08-06`.
  - Stripped 28 dangling refs across `reactor.py` (3), `trainer.py` (3), `commands.py` (3), `tui_client.py` (3), `llm_supervisor.py` (2), `slack_agent.py` (2), `cli_supervisor.py` (1), `supervisors/__init__.py` (1), `harness/__main__.py` (1), `harness/README.md` (4), and 7 test files (`test_reactor`, `test_tui_client` ×2, `test_training_log`, `test_slack_agent` ×2, `test_commands`, `test_s3_auto_countersign`, `test_trainer`).
- `specs/dialogue-driven-training.md` → deleted; truth in `src/dialogue/`. Self-declared working sketch. Tag `source-is-truth-2026-08-06`.
- `specs/dialogue-cogitation.md` → deleted; truth in `src/dialogue/rationalise.py`. Self-declared most-speculative sketch. Tag `source-is-truth-2026-08-06`.
  - Stripped 10 dangling refs across `actors.py` (3), `decoder.py`/`runner.py`/`rationalise.py`/`synthesize.py` (module headers), `runner.py` (1 inline), CONTEXT.md (Trainer glossary → redirected to `src/dialogue/`), `test_runner.py` (header).

- `specs/signifier.md` → deleted; truth in `signifier.py` + `abstract.py`. Missed in P1a, folded into P1i. Tag `source-is-truth-2026-08-06`.
  - Stripped 4 dangling refs in `signifier.py` (3) + `test_signifier.py` (1).
- `specs/auto-tune.md` → deleted; truth in `src/training/auto_tune/` + `cli_supervisor.py`/`cli_events.py`. Tag `source-is-truth-2026-08-06`.
  - Stripped 11 dangling refs across `lifecycle.py`/`session.py`×2/`snapshots.py`/`orchestrate.py` (module docstrings), `cli_events.py`/`cli_supervisor.py` (headers), `protocol.py` (2), `test_auto_tune_summary.py`/`test_auto_tune_lifecycle.py` (headers).
  - Bonus: swept 11 `@kvalue spec` stragglers missed in P1b across `adapter.py` (6), `protocol.py` (2), `significance.py` (2), `events.py` (1), `cli_events.py` (1), `test_agent.py` (1).

### ✅ Correction resolved

`specs/signifier.md` (missed in P1a) folded into P1i above. Phase 1 is complete: `specs/` is empty.
