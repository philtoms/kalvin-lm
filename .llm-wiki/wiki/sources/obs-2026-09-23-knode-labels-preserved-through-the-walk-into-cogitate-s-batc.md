---
type: source
title: "Observation: KNode labels preserved through the walk into cogitate's batch"
tags:
  - knode
  - labels
  - cogitate
  - batch
  - walk-bridges
  - fix
status: observation
created: 2026-09-23
updated: 2026-09-23
slug: obs-2026-09-23-knode-labels-preserved-through-the-walk-into-cogitate-s-batc
relevance: medium
observed_at: 2026-09-23T15:10:59.381Z
source_context: Fixing KNode label loss in cogitate's batch
---

# 🔍 Observation: KNode labels preserved through the walk into cogitate's batch

Fixed: cogitate's batch was losing KNode labels. Root cause: Derivation preserved queued.nodes' KNode objects at init (self.nodes = list(queued.nodes)) and replace_fwd/replace_rev/canonicalisations all build outputs from those objects — but the WALK path coerced to plain int at every hop: _walk_neighbours yielded int(n)/int(k.signature), descend/_meet frontiers re-coerced, and _walk's b_values appended ints — so bridge heads and walked-in witness atoms entered A's node list as bare ints, stripping labels (KNode is an int subclass; int() strips the subclass+label). Also _propose built the proposal via canon_key(kline.signature) — plain int signature. Fixes: value-carrying spots now pass objects through (_walk_neighbours yields k.nodes items and k.signature directly; frontiers/b_values uncoerced — comparisons/hashes unaffected since KNode hashes as int), and _propose re-wraps the canon signature as KNode(sig, label). Also corrected hop._reentry's stale comment (dbg rides for presentation only — selection no longer reads it). Verified with a cogitate spy: every node in every batched proposal is a KNode with a label; 90/90 tests; mhall MHALL:[Subject, Verb, Object] S1 255 and wdmh WDMH:[ALL, had, Mary] S1 255 unchanged.

*Relevance: medium*
*Context: Fixing KNode label loss in cogitate's batch*
*Tags: knode labels cogitate batch walk-bridges fix*

---
*Observed: 2026-09-23T15:10:59.381Z*
