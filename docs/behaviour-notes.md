# Behaviour Notes

Memory aids surfaced in session — **Rules Uncovered** (settled behaviour, candidates for
CONTEXT.md), the **active state of K** (the current frontier — not yet rules),
and **process discipline**. One line per rule. Append freely; One active state item; move to
Rules when it resolves.

## Rules Uncovered

### Engine — routing

- `route()` dispatches when **structural** significance (`sig_level`) **agrees with** the query's stamped significance (classified via `BandLayout`): both must be S1/S4 for the fast route, else slow route.
- The fast route grounds a seen-signature **identity** unconditionally (self-referential `{S:[S]}`); a seen-signature **canon** grounds only once `_is_groundable` holds (all its nodes are in LTM), else it slow-routes to await them.
- An incoming S4 is a **rejection**: `route` records it in the state's `refused` set (matched by exact `(signature, nodes)`) and removes the matching STM entry. `propose` filters refused shapes — K does not re-propose a refused kline.
- A signature is only discovered when the slow route unpacks it from an S2/S3 incoming; unreferenced signatures stay invisible.

### Engine — grounding & cogitation

- **Universal grounding rule:** a signature grounds only once every one of its nodes is in LTM. An identity is the exception — self-referential (`{S:[S]}`), it grounds unconditionally when promoted.
- `_is_groundable` is that rule: identity → True; anything else → all nodes in LTM.
- **Denotation gate:** a cascade promotion additionally requires `_is_denoted` — the signature already has some grounded kline under it in LTM, **or** the entry is itself a canon (self-denoting: signature == signature_of(nodes)). Without the canon exemption the canons deadlock (a canon is the only denotation of its signature). Relationships are never self-denoting — their truth comes from ratification, which is why `DH:[had]` waits in STM.
- Grounding happens wherever eligibility is discovered: fast route on feed, cogitate's groundable arm over STM — both via `Engine._ground` (direct grant → observation → fixed-point sweep over groundable+denoted entries).
- `cogitate` runs one full **oldest-first** pass over STM: per entry it asks (S4), grounds (groundable+denoted), proposes (S2), or removes-if-grounded. A pass that removed anything recurses (extended into the same batch). The unknown arm removes-then-asks with `continue` (no index skip).
- The misfit (S2) arm is one `propose(entry)` — `propose_gap` was merged in. A no-proposal misfit stays in STM for a later turn.
- An unknown (`{S: []}`) is never groundable — not even by the `_ground` cascade (an empty node list once slipped through `all([]) == True`).
- The engine speaks in semantic predicates (`is_identity`, `is_unknown`, `is_canon`, `is_relationship`), never raw `kline.nodes`.

### Engine — state

- `observations` resets to a fresh list at the top of every `rationalise()` call (per-turn scoping).
- EngineState stores: `stm` (attention), `ltm` (ratified klines), `frame` (emission memory), `refused` (S4-rejected shapes, keyed `(signature, nodes)` — lifetime scope open).
- `pop_identity` is gone; STM cleanup lives in `cogitate` and the S4 route's `remove_stm`.
- `find()` returns the **last** grounded kline under a signature — bucket order (grounding sequence) leaks into every consumer; consumers needing a canon use `canon_nodes` or scan the bucket.

### Misfit proposals (ExpandFit)

- **Connotations from both sides:** `_crossover_connotations` gathers edge-hop chains from the entry's nodes **and** from the underfit gap's covering bridges. `_edge_hops` traverses canon klines (stops at identity/dead end/cycle).
- **Crossing candidates** are grounded canon klines whose signature or nodes reach a connotation; **fills** are the candidate values that reach a connotation, at `connotation_hops + crossing_hops`.
- Fills are proposed **under the entry's signature** (added to the nodes / swapped for the excess), gated: not gap-covering (the query word itself), not a self-fill (reconstructs the entry), not already in the base nodes, `signifies(signature_of(expanded), entry.signature)`, not terminal.
- **Concrete-first grading:** `_fill_distance` — a fill that is a node of the entry's own canon takes flat distance 1; connotational fills take `hops + 1`. ("Grounded terminal" discriminators fail: grammar types have identities too.)
- **Reentry:** after grading, `propose` recurses on each proposal's kline (depth 2) — each proposal's nodes widen the connotation set, reaching fills one hop further out.
- **Drop rule:** fill-derived proposals drop when the gap could not be filled (`residual(entry.signature, signature_of(nodes)) != 0` — uncovered bits = unassigned work).
- **Pivot alignment:** `_pivot_proposals` — a pivot is a grounded canon sharing a node with the entry's canon. Per canon node: shared → S2 slot (1.0); edge-hop path into the pivot's nodes → S3 (decay(hops), node **replaced** by its pivot counterpart); no path → gap. Canon nodes forming a grounded sub-canon resolve as a **group** through the sub-canon signature's path (did+have → DH → had). Gap slots take the pivot's leftover nodes (S4 fill): one gap takes the whole leftover residual as a grouped fill; N gaps take one each; gaps outnumbering leftovers drop the proposal; **leftovers with no open gap are excluded** (surplus graft was the `ALL:[a,little,lamb,Mary,had]` bug). Slot accounting — not bit residual — decides pivot survival.
- Pivot slot semantics: shared node is **S2 (canonical)**, not S1; path is S3; gap fill is S4.

### Harness

- Curriculum-driven: the CLI takes a curriculum markdown file or a raw `.ks`; each lesson's kscript runs through a shared engine, state persisting across lessons; the label map accumulates the cumulative source.
- K-driven dialogue: each sub-script (annotation group) is opened with its first entry; from there the engine drives. **Run to completion:** an ask the script cannot answer is returned to K at S4 (rejection) and the dialogue continues — the run no longer stops.
- **Terminal-only identities:** `_answer` generates an identity `X:[X]` only for single-token words; a non-terminal's word form must come from the script.
- **Batched replies:** all of a turn's answers feed back as one `rationalise` call; empty replies are dropped from the queue.
- Ask dedup per turn on `(signature, nodes, band)`; the `answered` set dedups across the whole run.
- Trace: `proposes` vs `asks` (nodes present or not); feed items carry their band; band counts classify by significance byte through the spectrum (`BandLayout`), never by shape.
- Never judges — feeding and answering only.

### Compilation — annotation & scope

- `KDbg.annotation` carries the owning scope's annotation (parens stripped); `KDbg.scope` is 0 for source, 1 for MTS.
- Each kline owns its own annotation; MTS spawned by a signature inherits it.
- Compiler output order ≠ authored order: all source entries first, then all MTS. Symbolic-entry indices do not align with compiled-KValue indices.
- A single-token node word that never heads an entry is labelled via `TokenEncoder.node_labels`; the harness decoder is unsafe for compound-signature-as-node values.

## Active state of K

⚠️ **Pivot alignment artifacts + S5T2 churn.** mhall now proposes `WDMH:[had, Mary, a, little, lamb]` (the full alignment: did+have grouped-resolve through DH→had at S3, Mary shared at S2, the `what` gap filled by the grouped residual `[a,little,lamb]` at S4). Open: (a) wdmh still emits the greedy `[Mary,had,a,little]` variant because `find(DH)` returns the last bucket entry (`DH:[did,have]`, a cycle) rather than `DH:[had]` — bucket-order fragility; (b) the pivot arm proposes an entry's own canon when zero gaps exist (`ALL:[a,little,lamb]` self-canon echo — proposes nothing unknown; suppression candidate); (c) S5T2 grounds ALL then asks+proposes the same kline — duplicate STM copies of `ALL:[Query]` (opener + reply feed) each spawn asks, and the misfit arm runs on entries resolved earlier in the same pass; (d) refused-set lifetime (does a later grounding clear refusals whose basis changed?); (e) reentry's exponential widening and refusal circumvention via new shapes (`ALL:[lamb,lamb]` after `ALL:[lamb]` refused).

## Process — discipline

- **Engine first.** The engine is the target; the curriculum is the lever. Suspect the engine before retreating to `.ks` authoring.
- **Do not assume existing code is correct** — the source is what we are improving; a trace that vanishes or stalls treats the engine as suspect first.
- Check the source before asserting behaviour as fact.
- ⚠️ marks an open suspicion — a rule under active questioning, not a settled fact.
