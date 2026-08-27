---
type: source
title: "Observation: KNode became a labelled int subclass; labels wired at TokenEncoder"
tags:
  - knode
  - kline
  - labels
  - token-encoder
  - refactoring
status: observation
created: 2026-08-26
updated: 2026-08-26
slug: obs-2026-08-26-knode-became-a-labelled-int-subclass-labels-wired-at-tokenen
relevance: high
observed_at: 2026-08-26T13:19:04.160Z
source_context: KNode label struct refactor and wiring labels through the ks compiler
---

# ⭐ Observation: KNode became a labelled int subclass; labels wired at TokenEncoder

KNode changed from TypeAlias=int to class KNode(int) with a label attribute, .value property, and .with_label(). Chose int-subclass over a {value,label} dataclass so hashing, dict/set keys, SIG_MASK arithmetic, and JSON serialisation stay int-identical — zero call-site changes across the ~34 files using nodes. Labels are attached at the two TokenEncoder mint sites (src/ks/token_encoder.py: _encode_node single-token case wraps KNode(tokens[0], word); _emit_mts_for_tokens wraps the compound signature when dbg_label present), so labelled nodes flow into KLine.nodes and through STM/LTM/frame. Verified: mhall run shows node labels ('Mary','had','what',...) present in engine memory; 993 tests pass unchanged (pre-existing WIP failures unaffected). Caveat: plain ints (e.g. from JSON state snapshots) have no .label — read with getattr(node,'label',''). CONTEXT.md Node glossary updated. Also this session: renamed src/dialogue/training.py → teaching.py (shadowed the top-level training package when harness.py was run as a file).

*Relevance: high*
*Context: KNode label struct refactor and wiring labels through the ks compiler*
*Tags: knode kline labels token-encoder refactoring*

---
*Observed: 2026-08-26T13:19:04.160Z*
