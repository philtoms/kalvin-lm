# Behaviour Notes

Memory aids surfaced in session — **Rules Uncovered** (settled behaviour, candidates for
CONTEXT.md), the **active state of K** (the current frontier — not yet rules),
and **process discipline**. One line per rule. Append freely; One active state item; move to
Rules when it resolves.

## Rules Uncovered

### Engine — routing

- `route()` dispatches on **structural** significance (`sig_level`), not the producer's compiled stamp.
- The fast route admits identities and seen-signature canons when their signature is seen; unseen-signature canons take the slow route (nodes unpacked as asks); unseen-signature identities are dropped.
- An incoming S4 (`{X:[]}`) is a reply to K's own framed ask — feeding one never discovers a signature.
- A signature is only discovered when the slow route unpacks it from an S2/S3 incoming; unreferenced signatures stay invisible.

### Engine — grounding & cogitation

- `_is_groundable` branches, in order: identity → signature grounded; canon → all nodes grounded; single-node relationship → reciprocal grounded; any misfit → signature grounded AND all nodes grounded.
- The `_promote` cascade grounds groundable entries to fixed point at S1; it does not emit S2 proposals.
- `cogitate` runs one full LIFO pass over the work-list (no short-circuit): per entry it asks (S4), countersigns (S3), proposes (S2), or grounds (S1).
- The engine speaks in semantic predicates (`is_identity`, `is_unknown`, `is_canon`, `is_relationship`), never raw `kline.nodes`.

### Engine — state

- `observations` and `_incoming` reset to fresh lists at the top of every `rationalise()` call (per-turn scoping).

### Harness

- Non-judging: compiles, feeds, retrieves, presents. No verdict, no band-matching.
- S4 identity asks whose signature the curriculum defines as `X:[X]` are answered inline, mechanically, per-step dedup.
- Feeding an S1 identity alone doesn't ground it; the engine grounds it only once K has framed the signature first.

### Compilation — annotation & scope

- `KDbg.annotation` carries the owning scope's annotation (parens stripped); `KDbg.scope` is 0 for source, 1 for MTS.
- Each kline owns its own annotation; MTS spawned by a signature inherits it.
- Compiler output order ≠ authored order: all source entries first, then all MTS. Symbolic-entry indices do not align with compiled-KValue indices.
- A single-token node word that never heads an entry is labelled via `TokenEncoder.node_labels`; the harness decoder is unsafe for compound-signature-as-node values.

## Active state of K

⚠️ **The two cogitation strategies diverge on S2 emissions, not on the grounded model.** `similar_fit` and `expand` (the harness default) reach the same grounded model on `mhall` and both leave `WDMH:[Mary, had, a, little, lamb]` ungrounded. `similar_fit` emits 12 S2 proposals (the canonical synthesis at step 16, plus spurious recombinations under wrong signatures); `expand` emits 1 (`WDMH:[Mary]` at step 10) and never surfaces the synthesis.

Open: is `expand`'s silent synthesis correct (the WDMH↔MHALL pair grades S1/S3, so nothing is proposed) or a regression (the S2 path should propose the recombination regardless of band)? Read what `expand` grades for `(WDMH:[Mary,DH], MHALL:[Mary,had,a,little,lamb])` and why it is not S2.

## Process — discipline

- **Engine first.** The engine is the target; the curriculum is the lever. Suspect the engine before retreating to `.ks` authoring.
- **Do not assume existing code is correct** — the source is what we are improving; a trace that vanishes or stalls treats the engine as suspect first.
- Check the source before asserting behaviour as fact.
- ⚠️ marks an open suspicion — a rule under active questioning, not a settled fact.
