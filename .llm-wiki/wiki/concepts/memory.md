---
type: concept
title: Memory
created: 2026-08-17
updated: 2026-08-17
---

---
type: concept
title: Memory
description: The tiered structure inside the Model — STM, Frame, and LTM as modes of relation to held klines (recent attention, current focus, held knowledge), not storage locations.
created: 2026-08-14
updated: 2026-08-14
---

# Memory

The tiered structure **inside** the [[concepts/model|Model]] — not a substrate
beneath it.

## Definition

The tiers are **modes of relation** to held [[concepts/kline|klines]], not
buckets:

- **[[entities/stm-short-term-memory|STM]]** — recent attention: what Kalvin
  was just thinking about. Written by attention itself, which is how traversal
  is temporally situated and how Kalvin can notice a revisit.
- **[[concepts/frame|Frame]]** — current focus and its shift: where Kalvin's
  attention lies now, and what focused attention has produced there.
- **[[entities/ltm-long-term-memory|LTM]]** — held knowledge: what Kalvin
  counts on as [[concepts/grounding|grounded]].

The same kline content stands in different epistemic relations depending on its
tier: a kline in STM is *recently attended to*; in LTM it is *held as known*.
These relations only exist for something that rationalises — which is why
memory cannot be outside the model.

A tier change (promotion, framing, eviction) is a change in how Kalvin relates
to a kline — rationalisation work, not storage bookkeeping. Untiered klines in
a serialised file are a serialisation; they become memory again only when
loaded into a model that can attend to them.

_Avoid_: substrate / storage tier (implies memory sits beneath the model);
cache / buffer (implementation terms for a relation, not the relation).

## Links

- [CONTEXT.md](../../../../CONTEXT.md) — canonical definition (glossary, Memory entry)
- [[concepts/model]] — what memory is inside of
- [[entities/stm-short-term-memory]], [[concepts/frame]], [[entities/ltm-long-term-memory]] — the three relations
- [[concepts/monotonic-growth]] — Frame and LTM only grow
