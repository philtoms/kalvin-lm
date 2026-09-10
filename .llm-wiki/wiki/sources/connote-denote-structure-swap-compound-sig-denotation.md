---
type: source
title: CONNOTE/DENOTE structures swapped — denotation is compound-signature S2
status: insight
category: kscript
created: 2026-09-09
updated: 2026-09-09
slug: connote-denote-structure-swap-compound-sig-denotation
---

# CONNOTE/DENOTE structures swapped — denotation is compound-signature S2

KScript operator structures swapped per user spec: CONNOTES `A > B` now emits plain `A:[B]` (was compound-in-slot `A:[AB]`); RCONNOTES `A < B` emits `B:[A]` (preserving `A < B ≡ B > A`); DENOTES `A = B` emits `AB:[B]` — the signature is the compound of both operands and the node is the denoted value. The `concat` field in SymbolicEntry moved from the node slot to the SIGNATURE slot: TokenEncoder composes synthesized compound signatures via `_compose_concat(entry.concat, entry.sig)`, registered in `_compound_sigs`, never taking a word bit. DENOTES reclassified as S2: `sig_level` (kalvin/kline.py) single-node branch now returns S2 when `signifier.node_in(nodes[0], signature)` (underfit containment), S3 when disjoint — making the structural derivation agree with the pre-existing `band_significance("DENOTES") = SIG_S2` stamp. Containment mirrors flipped in dialogue: `EngineState.is_connotation` and harness `_structure_class` now treat node-inside-sig (AB:[B]) as denotation, disjoint (A:[B]) as connotation. Changed: src/ks/ast_emitter.py (DENOTES/CONNOTES/RCONNOTES branches; RCONNOTES passes concat only for multi-node), src/ks/token_encoder.py, src/kalvin/kline.py, src/dialogue/engine_state.py, src/dialogue/harness.py, src/training/trainer/curriculum_generator.py prompt, CONTEXT.md (Misfit/Relationship/Token ID/Relational Tokens glossary). Verified via inline scripts (tests absent since commit c31c1ce): compiled shapes, word-bit accounting (AB composes from A|B, no bit), decoder roundtrip (relation_by_label resolves compound), connotateY A→B→C traversal, engine rationalise smoke. Probe failures (probe_coverage_*, probe_what_*, probe_wdmh_expand, probe_reverse_edges, trace_full) are pre-existing at HEAD — old tuple-unpacking connotateY callers and missing dialogue.actors. [[sources/obs-2026-09-08-connotes-semantics-inverted-to-compound-in-slot-a-ab]] is now superseded by this swap.

*Category: kscript*

---
*Captured: 2026-09-09*

## Related

_Add links to related pages._
