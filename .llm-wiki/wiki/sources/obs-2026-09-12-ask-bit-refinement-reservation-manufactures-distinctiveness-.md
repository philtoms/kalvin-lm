---
type: source
title: "Observation: Ask-bit refinement: reservation manufactures distinctiveness and a permanent ungroundable gap"
tags:
  - ask
  - formalisation
  - ks2
  - packing
status: observation
created: 2026-09-12
updated: 2026-09-12
slug: obs-2026-09-12-ask-bit-refinement-reservation-manufactures-distinctiveness-
relevance: high
observed_at: 2026-09-12T10:15:19.444Z
source_context: Refining the ask-bit deviation from the ks2 conformance review
---

# ⭐ Observation: Ask-bit refinement: reservation manufactures distinctiveness and a permanent ungroundable gap

Refinement of the pending ASK_BPE_TOKEN removal (obs-2026-09-10): the deviation is not the bit's existence — an externally set (KScript) value that composes like any other atom would be fine — but the reservation that guarantees its distinctiveness. Word bits allocated 0-30 so no node ever carries bit 63; consequences: (1) is_ask is a clean predicate only because of the reservation — un-reserving doesn't make the bit harmless, it silently aliases with the 32nd distinct word, so the design space is reservation-or-removal with no stable middle; (2) the ask bit sits inside _TYPE_MASK and behaves as a word atom in signifies/residual/signature_of, but since no compiled node carries it, σ(ν) can never contain the ask atom — an ask-marked kline is permanently s ≠ σ(ν): never Canon, never S1, a permanent 1-atom ungroundable gap against its own canonical form; the grounding layer routes around this via dict-key equality (_fast_route), i.e. protocol compensates for corrupted fit; (3) one fewer word bit. Noted fossil: ks2 Def 1's A = {a₀…a₃₀} is exactly the 31 usable word bits — the reservation's shadow in the formalisation; only finiteness is load-bearing so the count is not normative. Removal consumers beyond the 2026-09-10 list: is_ask gates at engine.py:143 and cogitator.py:41 (must key on emptiness/S4), compiler _emit_ask must produce s:[] Unknown shape not s|ASK:[nodes], harness _graph_name ? prefix, word allocation reclaims bits 0-31, persisted bit-63 signatures need migration/staleness acceptance.

*Relevance: high*
*Context: Refining the ask-bit deviation from the ks2 conformance review*
*Tags: ask formalisation ks2 packing*

---
*Observed: 2026-09-12T10:15:19.444Z*
