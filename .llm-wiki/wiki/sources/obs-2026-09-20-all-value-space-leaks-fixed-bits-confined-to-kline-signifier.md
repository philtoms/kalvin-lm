---
type: source
title: "Observation: All value-space leaks fixed: bits confined to kline/signifier/tokenizer/compiler homes"
tags:
  - engine
  - refactoring
  - values
  - seam
status: observation
created: 2026-09-20
updated: 2026-09-20
slug: obs-2026-09-20-all-value-space-leaks-fixed-bits-confined-to-kline-signifier
relevance: critical
observed_at: 2026-09-20T22:24:10.054Z
source_context: Fixing remaining value-space leaks in hop.py and dialogue modules
---

# 🔴 Observation: All value-space leaks fixed: bits confined to kline/signifier/tokenizer/compiler homes

Remaining value-space leaks fixed — bit knowledge now lives only in its legitimate homes. Added to kline.py (the marker's home, where ASK_SIG is defined): canon_key(value) (& ~ASK_SIG) and mark_ask(value) (| ASK_SIG). Routed: dialogue/harness.py (5 sites), engine_state.py is_answered, engine.py proposal minting, decoder.py ask minting → canon_key/mark_ask. hop.py fully seam-routed: candidate_goals canon exclusion via canon_key, the Def 22 coverage test via signifier.signifies (FIXED latent leak: the old `int(n) & kc` let BPE token-id bits satisfy coverage — cargo now inert per the doc), γ scores via signifier.measure; trawl() gained a signifier param — the reached set composes full values and membership reads signifier.signifies (masked at read, cargo inert). dialogue/derivation.py fork mirror-applied (_atom_bits deleted, acq ledger via units, _jaccard via measure). pivot_fill.py: _BPE_MASK deleted (dead code). significance.py: WORD_BITS constant deleted — zero importers remained. word_atom_count survives as the seam-backed μ alias for expand/harness/cogitator. Tests: test_hop trawl calls threaded SIG, test_trawl_touches_at_word_bits renamed test_trawl_touches_at_content (its token-id-carry-no-correspondence assertion now passes via the seam rather than masked reach). All 70 tests pass. Bit-touching sites remaining, all legitimate homes: kline.py (ASK_SIG + marker helpers), signifier.py (_TYPE_MASK + measure/units), tokenizers (node packing — value producers at the world boundary), ks/token_encoder.py (word-bit basis assignment + the ask mint: "only compilation mints it").

*Relevance: critical*
*Context: Fixing remaining value-space leaks in hop.py and dialogue modules*
*Tags: engine refactoring values seam*

---
*Observed: 2026-09-20T22:24:10.054Z*
