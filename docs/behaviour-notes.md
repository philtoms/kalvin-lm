# Behaviour Notes

Memory aids surfaced in session — rules (settled behaviour, candidates for
CONTEXT.md), the **active state of K** (the current frontier on the
improvement arc — not yet rules), and **process discipline**. Append freely;
prune rules when promoted to CONTEXT.md; move active-state items to Rules
when they resolve into settled behaviour.

## Rules

Settled behavioural facts. Candidates for promotion to CONTEXT.md.

### Engine — routing

- `route()` dispatches on **structural** significance (`sig_level`), not the producer's compiled stamp. A countersign `MHALL:[SVO]` is structurally S3.
- The fast route's admission rule is `_signature_seen` for identities **and** seen-signature canons — both ground when the signature is known. Not "all nodes grounded."
- An **unseen-signature canon** takes the slow route (not dropped): its nodes are unpacked as unknown asks and discovered. This is how a novel signature (e.g. WDMH) enters K's world. Identities of an unseen signature are still dropped (an identity carries nothing to learn).
- An incoming S4 (`{X:[]}`) is treated as a *reply* to K's own framed ask, not a new ask — so feeding an S4 can't introduce/discover a signature either.
- A signature is only **discovered** when the slow route unpacks it as an unknown node from an S2/S3 incoming. A signature never referenced by anything K learns is invisible forever.
- A countersignable pair (`MHALL:[SVO]` + `SVO:[MHALL]`) grounds only when **every operand pairing is grounded**. `_countersignature_proposals` returns one S3 per unresolved pairing (`Mary:[Subject]`, `had:[Verb]`, `ALL:[Object]`); once those ground (via the signature+nodes groundable rule), the countersign completes. The role-mappings and the countersign were one root, not two.

### Engine — grounding & cogitation

- `_is_groundable` is a **slow-route/cogitation** predicate. Its branches, in order: terminal → signature grounded; canon → all nodes grounded; single-node relationship → reciprocal grounded; **any misfit → signature grounded AND all nodes grounded**. The last is the general rule that unblocks role-mappings and denotes/conotes; it applies to any node count.
- The `_promote` cascade **grounds groundable entries until fixed point** — purely records them at S1. A groundable misfit that surfaces inside a cascade is grounded directly; it is not re-rationalised. (The main `cogitate` pass is where misfits get their S2 proposals; the cascade does not emit proposals.)
- **The S2 similar-fit graft is the engine's answering path.** Once the prime grounds, grafting a question-shaped misfit (`WDMH:[Mary, DH]`) against the grounded kline sharing its subject (`MHALL:[Mary, had, a, little, lamb]`) emits the synthesis as an S2 proposal — `WDMH:[Mary, had, a, little, lamb]`. In mhall this surfaces at step 16 (when MHALL grounds), emitted by `cogitate`'s S2 branch on the still-pending WDMH misfit.
- **The S2 graft fires for any misfit sharing a node with a grounded canon — and that is band-correct, not a bug.** The same pass that answers WDMH also emits `DH:[had, Mary, a, little, lamb]`, `SVO:[Mary, had, a, little, lamb]` (step 16) and `L:[Object, Subject, Verb]`, `ALL:[Object, Subject, Verb]` (step 17/20) — recombinations a trainer reads as nonsensical. Confirmed via a question-free curriculum (mhall minus WDMH): the role-maps over-graft identically with no question present, so the behaviour is not question-dependent. These are S2 proposals (the divergent-inference band); none ever ground and none block a correct grounding. A misfit under an ungrounded signature that shares a node with a canon *is* indirectly connected to that canon's other nodes, so the proposal is a defensible association. Whether a proposal is "right" is the trainer's judgement (S2 persists precisely because it is unratified); the engine has no structural signal at graft-time that separates a question from a role-map, and reading intent is not its job. Treat trace noise from the graft as expected, not as an engine defect.
- `cogitate` runs a full LIFO pass over the work-list (no short-circuit): per entry it asks (S4), countersigns (S3 → grounds when complete), proposes (S2 for any misfit), or grounds (S1).
- The engine speaks in semantic predicates (`is_identity`, `is_unknown`, `is_canon`, `is_relationship`), never raw `kline.nodes`.
- "Fast route is for terminals" is **false** — invention. The fast route handles any S1/S4 incoming against the frame.

### Engine — state

- `self.observations` and `self._incoming` reset to fresh lists at the top of every `rationalise()` call (per-turn scoping, no cross-call leak).

### Harness

- The harness is non-judging: it compiles, feeds, retrieves, presents. No verdict, no band-matching. The trainer (a pi agent) judges, outside the loop.
- S4 identity asks (`{X:[]}`) whose signature the curriculum defines as `X:[X]` are answered inline from the compiled entries — unsupervised, mechanical, per-step dedup.
- Feeding an S1 identity alone doesn't ground it; the engine grounds it only once K has framed/asked the signature first.
- The trainer forms expectations from annotation prose ("a fact" → expect grounding; "a question" → expect an answer). Judgement stays conversational, never coded in the harness.

### Compilation — annotation & scope

- `KDbg.annotation` carries the owning scope's annotation text (parens stripped); `KDbg.scope` is 0 for source entries, 1 for MTS/compound-word decomposition.
- Each kline owns its own annotation — no propagation to child scopes. An inline annotation records the **resolved bound word** (`S`+`(ubject)` → "Subject"), not the raw fragment.
- MTS spawned by a signature **inherits that signature's annotation**. MHALL's MTS canon carries "Mary had a little lamb"; Det's MTS carries "".
- The encoder's source-before-MTS partition is **preserved**. Lifting it scrambles the authored step order; the constraint stays in-compiler, not re-derived downstream.
- Compiler output order ≠ authored order: all source entries (every block) first, then all MTS. A single annotation's entries can be non-contiguous in the trace (the prime's source and its MTS split by the test block's source).
- The encoder expands (one symbolic entry → multiple KValues) **and** stable-partitions. Symbolic-entry indices do **not** align with compiled-KValue indices — don't try to reconstruct scope from outside the compiler.
- A single-token node word (e.g. "did", "have") never heads an entry, so its label comes from `TokenEncoder.node_labels` (recorded in `_encode_node`'s single-token branch), not from any KDbg. The harness decoder is **unsafe** for this — packed-signature-as-node values decode to garbage; only the compiler-side map is reliable.

## Active state of K

The current frontier on the engine's improvement arc. **Not rules** — the
state-of-play as we work toward a rule. An item migrates to Rules (or is
deleted) when it resolves.

### ⚠️ Two cogitation strategies for the misfit (S2) arm

`Engine(strategy=...)` selects an S2 ``MisfitStrategy`` (CLI `-s/--strategy`;
harness default `similar_fit`). Each strategy lives in its own module under
`src/dialogue/`:

- **`similar_fit`** (`dialogue/similar_fit.py`) — the original: recombine via
  node-expansion + node-graft.
- **`expand`** (`dialogue/expand_fit.py`, uses `kalvin.expand.expand`) — grade
  each node-sharing grounded `candidate` against the entry via `expand`; the
  best terminal byte's band decides — S1 grounds under the entry's signature,
  S2 proposes, S3/S4 emits nothing.

Shared grounding-store helpers and the read-only ``GroundedModel`` adapter
(``dialogue/misfit.py``) that `expand` expects live alongside the
`MisfitStrategy` protocol. The engine itself only dispatches.

On `mhall` both strategies end with the **same grounded model** (20 klines);
both leave `WDMH:[Mary, had, a, little, lamb]` ungrounded (the synthesis is
a proposal awaiting ratification, never engine-grounded). The divergence is
in **emissions**: `similar_fit` emits 12 S2 proposals across the run
(including the canonical synthesis at step 16, plus two spurious proposals
under wrong signatures `DH:[...]` / `SVO:[...]`); `expand` emits **1** S2
proposal (`WDMH:[Mary]` at step 10) and **never emits the synthesis**. The
synthesis is the answering proposal the trainer expects to see — so `expand`
arrives at the same model silently, without surfacing the answer.

Open question: is the silent synthesis a *correct* consequence of expand's
accounting (the WDMH↔MHALL pair grades S1/S3, not S2, so nothing is proposed)
or a *regression* (the S2 path should propose the recombination regardless of
band)? Resolve by reading what `expand` grades for the
`(WDMH:[Mary,DH], MHALL:[Mary,had,a,little,lamb])` pair and why it is not S2.

## Process — discipline

- **Engine first.** The engine is the target of the work; the curriculum is the lever. Treat engine behaviour as the suspect before retreating to curriculum-authoring. Only edit/create `.ks` when confident the script can test a theory or bring out a different result.
- **Do not assume existing code is correct.** This contradicts "source is the truth document," but we are in a privileged working mode where the source is exactly what we are trying to improve. The fast route's drop-on-unseen-signature is the smoking gun — existing behaviour can be the bug.
- Check the source before asserting behaviour as fact. Inferences stated as established rules cause real bugs (the `_is_groundable` lift; the invented "terminals" rule).
- Don't misread `diff` hunk formatting as duplicate output. Verify counts directly.
- ⚠️ marks an open suspicion — a rule under active questioning, not a settled fact.
