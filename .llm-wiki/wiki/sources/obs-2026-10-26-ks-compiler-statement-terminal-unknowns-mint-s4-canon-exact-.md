---
type: source
title: "Observation: ks compiler: statement-terminal unknowns mint S4; canon-exact asks feed S1"
tags:
  - ks
  - compiler
  - s4
  - unknown
  - terminals
  - asks
  - significance
status: observation
created: 2026-10-26
updated: 2026-10-26
slug: obs-2026-10-26-ks-compiler-statement-terminal-unknowns-mint-s4-canon-exact-
relevance: high
observed_at: 2026-10-26T15:05:39.998Z
source_context: Implementing the terminal-unknown S4 rule in the ks compiler
---

# ⭐ Observation: ks compiler: statement-terminal unknowns mint S4; canon-exact asks feed S1

New compile rules implemented per user spec: (1) a statement ending in a leaf Signature terminal whose resolved word has no identity/canon of its own (op in IDENTITY/ASK/CANONICALISES/UNKNOWN with sig==word among final entries) mints an ADDITIVE empty-form entry {word:[]} op=UNKNOWN fed at SIG_S4 — placed immediately after its statement, one per distinct word per file, deduped across imports (first occurrence wins). Implemented as a post-pass in ASTEmitter.emit() (_emit_terminal_unknowns) collecting (insert-after index, resolved word) from _process_scope_body (only when scope.op is not None and items[-1] is a plain Signature — nested-scope terminals like Object < Query and Query < ALL are skipped). mhall.ks now compiles Det:[], Mod:[], Object:[] after their DENOTES statements. (2) A canon-exact ASK (is_exact over atom space, ASK marker masked) is fed at SIG_S1 instead of SIG_S4 in token_encoder._encode_entries_for_entry — the fed band is the ask's content. mhall's MHALL ask AND wdmh's (what did Mary have) ask now feed 255. _OP_TO_SIG gained "UNKNOWN": SIG_S4 (src/kalvin/significance.py). This breaks the "empty-form S4 only in src/trs" invariant recorded in obs-2026-10-26: the ks compiler now emits empty-node klines.

*Relevance: high*
*Context: Implementing the terminal-unknown S4 rule in the ks compiler*
*Tags: ks compiler s4 unknown terminals asks significance*

---
*Observed: 2026-10-26T15:05:39.998Z*
