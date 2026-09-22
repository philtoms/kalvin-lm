---
type: source
title: "Observation: Orphaned pipeline modules removed (model, agent_codec, expand, proposals, stm)"
tags:
  - cleanup
  - dead-code
  - removal
status: observation
created: 2026-09-22
updated: 2026-09-22
slug: obs-2026-09-22-orphaned-pipeline-modules-removed-model-agent-codec-expand-p
relevance: high
observed_at: 2026-09-22T08:22:03.573Z
source_context: Removing orphaned Model/agent_codec/expand/proposals/stm modules
---

# ⭐ Observation: Orphaned pipeline modules removed (model, agent_codec, expand, proposals, stm)

Removed the orphaned old-pipeline modules: git rm of src/kalvin/model.py, agent_codec.py, expand.py, proposals.py, and stm.py (stm was Model's locked store — exclusively imported by model.py, so it went with it). tests/test_gamma.py was surgically split: the five test_expand_* tests deleted with expand; the canonical γ-algebra tests (gamma_aggregate/decay/word_atom_count/gamma_to_byte) and the Memory acq_depth snapshot tests remain — suite 95→90. significance.py docstring de-referenced (expand pipeline + Model.is_countersigned mentions); adapter.py's thread-model paragraph rewritten honestly: the shared Memory is mutated from the bus thread (rationalise) and the WorkRunner thread (cogitation) with NO locking — GIL-atomic individual ops, composite passes not transactional. Collateral: scripts/kalvin_test.py still imports kalvin.expand/model — it was already broken (D_MAX drift) predating this session; needs a port to the new Engine or deletion (user decision pending). Post-removal: 90 tests pass, mhall byte-identical, training imports ok, mypy dropped 79→53 errors (26 lived in the deleted files).

*Relevance: high*
*Context: Removing orphaned Model/agent_codec/expand/proposals/stm modules*
*Tags: cleanup dead-code removal*

---
*Observed: 2026-09-22T08:22:03.573Z*
