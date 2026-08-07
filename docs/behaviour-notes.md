# Behaviour Notes

One-liner memory aids — rules and observations surfaced in session — kept
here until they earn a home in CONTEXT.md (domain concepts vs behaviour
concepts). Append freely; prune when promoted.

## Engine — routing

- `route()` dispatches on **structural** significance (`sig_level`), not the producer's compiled stamp. A countersign `MHALL:[SVO]` is structurally S3.
- The fast route's admission rule is `_signature_seen` — the same for identities and canons. Not "all nodes grounded."
- `_is_groundable` is a **slow-route/cogitation** predicate (a canon whose nodes are all grounded). Do not lift it into the fast route.
- The engine speaks in semantic predicates (`is_identity`, `is_unknown`, `is_canon`), never raw `kline.nodes`.
- "Fast route is for terminals" is **false** — invention. The fast route handles any S1/S4 incoming against the frame.

## Engine — state

- `self.observations` and `self._incoming` reset to fresh lists at the top of every `rationalise()` call (per-turn scoping, no cross-call leak).

## Harness

- The harness is non-judging: it compiles, feeds, retrieves, presents. No verdict, no band-matching. The trainer (a pi agent) judges, outside the loop.
- S4 identity asks (`{X:[]}`) whose signature the curriculum defines as `X:[X]` are answered inline from the compiled entries — unsupervised, mechanical, per-step dedup.
- Feeding an S1 identity alone doesn't ground it; the engine grounds it only once K has framed/asked the signature first.
- The trainer forms expectations from annotation prose ("a fact" → expect grounding; "a question" → expect an answer). Judgement stays conversational, never coded in the harness.

## Compilation — annotation & scope

- `KDbg.annotation` carries the owning scope's annotation text (parens stripped); `KDbg.scope` is 0 for source entries, 1 for MTS/compound-word decomposition.
- Each kline owns its own annotation — no propagation to child scopes. `S(ubject)=M`'s entry carries "ubject"; a sibling without an annotation stays empty.
- MTS spawned by a signature **inherits that signature's annotation**. MHALL's MTS canon carries "Mary had a little lamb"; Det's MTS carries "".
- The encoder's source-before-MTS partition is **preserved**. Lifting it scrambles the authored step order; the constraint stays in-compiler, not re-derived downstream.
- Compiler output order ≠ authored order: all source entries (every block) first, then all MTS. A single annotation's entries can be non-contiguous in the trace (the prime's source and its MTS split by the test block's source).
- The encoder expands (one symbolic entry → multiple KValues) **and** stable-partitions. Symbolic-entry indices do **not** align with compiled-KValue indices — don't try to reconstruct scope from outside the compiler.

## Process — discipline

- Check the source before asserting behaviour as fact. Inferences stated as established rules cause real bugs (the `_is_groundable` lift; the invented "terminals" rule).
- Don't misread `diff` hunk formatting as duplicate output. Verify counts directly.
