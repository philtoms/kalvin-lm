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
- `_is_groundable` is that rule: identity → True; anything else → all nodes in LTM — except a multi-node non-canon (an S2 misfit), which is never cascade-groundable and must take the S2 path.
- `pop_identity` drops **any** STM entry under a signature, not just unknown asks — a surviving misfit is silently deleted when another kline under the same signature grounds.
- The `_promote` cascade grounds groundable entries to fixed point at S1; it does not emit S2 proposals.
- `cogitate` runs one full **oldest-first** pass over STM (no short-circuit; FIFO, not LIFO — the entry waiting longest cogitates first): per entry it asks (S4), countersigns (S3), proposes (S2), or grounds (S1). The loop re-checks each index because the `_promote` cascade can remove arbitrary STM entries mid-pass.
- The misfit (S2) arm has two probes: `propose_gap` (multi-node gate — a single-node misfit is the countersign arm's shape) expands the entry's own fit directly; `propose` falls back to node-overlap candidates. A no-proposal misfit stays in STM for a later turn.
- An unknown (`{S: []}`) is never groundable — not even by the `_ground` cascade (an empty node list once slipped through `all([]) == True`).
- The engine speaks in semantic predicates (`is_identity`, `is_unknown`, `is_canon`, `is_relationship`), never raw `kline.nodes`.

### Engine — state

- `observations` resets to a fresh list at the top of every `rationalise()` call (per-turn scoping).
- EngineState holds three stores realising the kalvin memory relations: `stm` (Short-Term Memory — what cogitation is attending to: incoming entries and the ungrounded sigs/nodes unpacked from them; formerly `work_list`, renamed once recognised as already the attention store), `ltm` (ratified klines), and `frame` (outgoing proposals and identity requests). The separate `kalvin.stm.STM` index field was removed — the lean engine had two STM-shaped stores.
- The scoped reads (`is_in_ltm`, `is_seen`, `signature_seen`) each check one store; none is a union across stores.

### Harness

- K-driven dialogue: each sub-script (annotation group) is opened with its first entry; from there the engine drives. Each engine emission is an ask; the harness answers from the script or the run stops.
- Ratifying answers: an identity ask `X:[]` → identity `X:[X]` + the script klines headed `X`; a proposal ask `A:[B]` → the matching script kline + its countersignature `B:[A]`. All other emissions are scaffolding.
- A node of a compound self-ref entry (`DH:[did,have]`) is a script-known word; its identity ask answers with `X:[X]` alone.
- Never judges — feeding and answering only.

### Compilation — annotation & scope

- `KDbg.annotation` carries the owning scope's annotation (parens stripped); `KDbg.scope` is 0 for source, 1 for MTS.
- Each kline owns its own annotation; MTS spawned by a signature inherits it.
- Compiler output order ≠ authored order: all source entries first, then all MTS. Symbolic-entry indices do not align with compiled-KValue indices.
- A single-token node word that never heads an entry is labelled via `TokenEncoder.node_labels`; the harness decoder is unsafe for compound-signature-as-node values.

## Active state of K

⚠️ **Underfit gap-fill bridges via s3 connotations.** Seed: gap-covering grounded kline not connoted by another gap-covering kline (chain head). `_edge_hops` from its nodes builds `s3_connotations` (sig → min hops); every grounded signature whose own `_edge_hops` chain crosses a connotation fills at `connotation_hops + crossing_hops`, graded `decay(total)`. mhall: lamb/pig fill at 2 hops (0xfb), ALL's mid-chain edge `ALL:[Query]` costs an extra crossing (0xf9). Gap-covering fills are suppressed. Open: lamb vs pig tie at equal hops (divergence-based ranking unexplored); multi-proposal ratification semantics; `pop_identity` deleting a surviving misfit on same-signature grounding.

## Process — discipline

- **Engine first.** The engine is the target; the curriculum is the lever. Suspect the engine before retreating to `.ks` authoring.
- **Do not assume existing code is correct** — the source is what we are improving; a trace that vanishes or stalls treats the engine as suspect first.
- Check the source before asserting behaviour as fact.
- ⚠️ marks an open suspicion — a rule under active questioning, not a settled fact.
