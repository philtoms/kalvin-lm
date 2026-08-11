# Behaviour Notes

Memory aids surfaced in session — **Rules Uncovered** (settled behaviour, candidates for
CONTEXT.md), the **active state of K** (the current frontier — not yet rules),
and **process discipline**. One line per rule. Append freely; One active state item; move to
Rules when it resolves.

## Rules Uncovered

### Engine — routing

- `route()` dispatches when **structural** significance (`sig_level`) **agrees with** the query's stamped significance (classified via `BandLayout`): both must be S1/S4 for the fast route, else slow route.
- The fast route grounds a seen-signature **identity** unconditionally (self-referential `{S:[S]}`); a seen-signature **canon** grounds only once `_is_groundable` holds (all its nodes are in LTM), else it slow-routes to await them.
- An incoming S4 (`{X:[]}`) is a reply to K's own framed ask — feeding one never discovers a signature.
- A signature is only discovered when the slow route unpacks it from an S2/S3 incoming; unreferenced signatures stay invisible.

### Engine — grounding & cogitation

- **Universal grounding rule:** a signature grounds only once every one of its nodes is in LTM. An identity is the exception — self-referential (`{S:[S]}`), it grounds unconditionally when promoted.
- `_is_groundable` is that rule: identity → True; anything else → all nodes in LTM.
- The `_promote` cascade grounds groundable entries to fixed point at S1; it does not emit S2 proposals.
- `cogitate` runs one full LIFO pass over the work-list (no short-circuit): per entry it asks (S4), countersigns (S3), proposes (S2), or grounds (S1). The loop re-checks each index because the `_promote` cascade can remove arbitrary work-list entries mid-pass.
- The engine speaks in semantic predicates (`is_identity`, `is_unknown`, `is_canon`, `is_relationship`), never raw `kline.nodes`.

### Engine — state

- `observations` resets to a fresh list at the top of every `rationalise()` call (per-turn scoping).
- EngineState holds four stores mirroring the kalvin memory tiers: `work_list` (the cogitator queue — incoming entries and the ungrounded sigs/nodes unpacked from them), `ltm` (ratified klines), `frame` (outgoing proposals and identity requests), and `stm` (Short-Term Memory, reserved for the expansion strategies' exclusive use).
- `work_list` and `stm` are maintained independently — work-list writes do not cascade to STM, and no logic reads or writes STM yet.
- The scoped reads (`is_in_ltm`, `is_seen`, `signature_seen`) each check one store; none is a union across stores.

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
