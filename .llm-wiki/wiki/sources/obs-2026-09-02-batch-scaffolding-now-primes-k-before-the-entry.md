---
type: source
title: "Observation: Batch scaffolding now primes K before the entry"
tags:
  - harness
  - scaffolding
  - dialogue
status: observation
created: 2026-09-02
updated: 2026-09-02
slug: obs-2026-09-02-batch-scaffolding-now-primes-k-before-the-entry
relevance: high
observed_at: 2026-09-02T12:23:09.590Z
source_context: Reordering batch scaffolding before entry in dialogue harness
---

# ⭐ Observation: Batch scaffolding now primes K before the entry

Batch scaffolding mode now feeds scaffolding BEFORE the entry: in run(), the feed→ask→answer loop was extracted into Harness._drive(step, queue, heads, exact, words, answered); batch mode calls it first with the group-minus-opener batch, then with [opener] (skipped if the opener was primed/grounded). Priming-filter and terminal-word identity ride-along moved into a local build_batch(sources) closure. mhall trace shows T01 = scaffolding (SVO canon etc.), T02 = entry MHALL:[SVO]. On-demand trace byte-identical to before the change. ruff/mypy clean (pre-existing I001/bus.py aside).

*Relevance: high*
*Context: Reordering batch scaffolding before entry in dialogue harness*
*Tags: harness scaffolding dialogue*

---
*Observed: 2026-09-02T12:23:09.590Z*
