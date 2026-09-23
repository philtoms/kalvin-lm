---
type: source
title: "Observation: src/dialogue dissolved: engine into src/kalvin, harness into dev/dialogue"
tags:
  - refactor
  - architecture
  - layout
status: observation
created: 2026-09-21
updated: 2026-09-21
slug: obs-2026-09-21-src-dialogue-dissolved-engine-into-src-kalvin-harness-into-d
relevance: critical
observed_at: 2026-09-21T15:16:42.155Z
source_context: Refactoring src/dialogue into main code lines
---

# 🔴 Observation: src/dialogue dissolved: engine into src/kalvin, harness into dev/dialogue

Refactor complete, uncommitted: src/dialogue/ dissolved. Engine side → src/kalvin/engine.py + src/kalvin/engine_state.py (imports rewritten to kalvin.*; src/kalvin/hop.py TYPE_CHECKING import de-inverted). Harness side → dev/dialogue/harness.py + structural.py + decoder.py (structural = harness escalation-seam support; decoder = dialogue-table config layer). dev/ and dev/dialogue/ gained __init__.py (regular packages); pyproject pytest pythonpath now ["src", "."]; tests import dev.dialogue.harness via repo-root path insertion from tests/__init__.py package mechanics. Harness CLI now: PYTHONPATH=src:. python -m dev.dialogue.harness <script.ks> (-e structural supervisor verified live). ~31 probes + 5 tests + dev/ks/build_state.py mechanically re-imported; stale dialogue.cogitator probe imports retargeted to kalvin.cogitator. Verification: 95/95 tests, ruff at pre-existing baseline (252), probes fail only on pre-existing causes (missing gitignored data/scripts/wdmh-underfit.ks, Derivation.slot_walk monkeypatch drift, trace_full.py's long-dead dialogue.actors import). Docs updated: CONTEXT.md (3 paths), README.md (tree), .pi/skills/dialogue-dev/SKILL.md (commands + file map).

*Relevance: critical*
*Context: Refactoring src/dialogue into main code lines*
*Tags: refactor architecture layout*

---
*Observed: 2026-09-21T15:16:42.155Z*
