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

## Process — discipline

- Check the source before asserting behaviour as fact. Inferences stated as established rules cause real bugs (the `_is_groundable` lift; the invented "terminals" rule).
- Don't misread `diff` hunk formatting as duplicate output. Verify counts directly.
