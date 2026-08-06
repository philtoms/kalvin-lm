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

### 1d — Cogitator & Rationaliser  *(specs/cogitator.md, specs/rationaliser.md)*
- Source: `src/kalvin/cogitator.py`, `src/kalvin/rationaliser.py`, `src/kalvin/proposals.py`.
- [ ] Appraise (read spec + source + tests; verify; fix code only on genuine gaps).
- [ ] Fill navigation entries.
- [ ] Retire specs → ARCHIVE rows → fix references.

### 1e — KScript  *(specs/kscript.md)*
- Source: `src/ks/` (lexer, parser, ast, ast_emitter, binding_scope, compiler, token, token_encoder).
- Largest single spec (900 lines). Many source/test references point here.
- [ ] Appraise (read spec + source + tests; verify; fix code only on genuine gaps).
- [ ] Fill navigation entries.
- [ ] Retire spec → ARCHIVE row → fix references.

### 1f — Training: trainer, reactor, curriculum  *(specs/curriculum.md, specs/trainer-satisfaction.md, specs/training-log.md)*
- Source: `src/training/trainer/`.
- [ ] Appraise (read spec + source + tests; verify; fix code only on genuine gaps).
- [ ] Fill navigation entries.
- [ ] Retire specs → ARCHIVE rows → fix references.

### 1g — Harness & supervisors  *(specs/harness-server.md, specs/supervisor-decision.md)*
- Source: `src/training/harness/`, `src/training/supervisors/`.
- [ ] Appraise (read spec + source + tests; verify; fix code only on genuine gaps).
- [ ] Fill navigation entries.
- [ ] Retire specs → ARCHIVE rows → fix references.

### 1h — Dialogue subsystem  *(specs/dialogue-driven-training.md, specs/dialogue-cogitation.md)*
- Source: `src/dialogue/` (runner, actors, decoder, rationalise, synthesize).
- [ ] Appraise (read spec + source + tests; verify; fix code only on genuine gaps).
- [ ] Fill navigation entries.
- [ ] Retire specs → ARCHIVE rows → fix references.

### 1i — Auto-tune  *(specs/auto-tune.md)*
- Source: `src/training/auto_tune/`.
- [ ] Appraise (read spec + source + tests; verify; fix code only on genuine gaps).
- [ ] Fill navigation entries.
- [ ] Retire spec → ARCHIVE row → fix references.

## Phase 2 — Plans retirement

Plans are HOW; once specs are gone and source is truth, plans are spent. Retire them wholesale,
one ARCHIVE row per plan (or one summarising row per directory). No per-plan appraisal needed —
their value was transitional.

- [ ] `plans/impl/*` → ARCHIVE rows → delete.
- [ ] `plans/implement-*.md`, `plans/role-based-routing.md`, `plans/remove-compound-token.md` → ARCHIVE rows → delete.

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
