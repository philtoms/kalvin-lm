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
- `cogitate` runs one full LIFO pass over STM (no short-circuit): per entry it asks (S4), countersigns (S3), proposes (S2), or grounds (S1). The loop re-checks each index because the `_promote` cascade can remove arbitrary STM entries mid-pass.
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

⚠️ **No relationship ever reaches the S3 countersign arm.** Every relationship entry (`a:[Det]`, `DH:[had]`, `MHALL:[SVO]`) is consumed by the misfit arm in the same turn — popped regardless of whether `propose` returned anything — so `is_countersignable` never sees a live entry. The WDMH↔MHALL synthesis question is unreachable until a no-proposal misfit survives in STM. Candidate: pop only when the strategy produced a batch.

## Process — discipline

- **Engine first.** The engine is the target; the curriculum is the lever. Suspect the engine before retreating to `.ks` authoring.
- **Do not assume existing code is correct** — the source is what we are improving; a trace that vanishes or stalls treats the engine as suspect first.
- Check the source before asserting behaviour as fact.
- ⚠️ marks an open suspicion — a rule under active questioning, not a settled fact.
