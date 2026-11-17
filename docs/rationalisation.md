# Kalvin — the rationalisation model

The rules of the model in `src/rationalisation/` — the Cogitator
(`cogitator.py`) over the shared algebra, domain and report — where
the code is truth and this document is the rules the code implements. It replaces
`docs/attic/trs-significance.md`, which named the mechanism a term
rewriting system. It is not one: nothing here locates a redex and
replaces it, no rule fires by syntactic match, and termination comes
from bounds. What the mechanism is: **a deterministic, bounded,
depth-first search in which unconstructed klines — signatures seeking
construction — meet held klines' claims and receive their offered
decompositions, until the run halts.**

The scripts are the only guarantee that an ask can be framed as a
goal. If a script does not ground, that is the honest state of the
system; the model creates no contrivance to force a resolution.

## 1. Klines, and what they are made of

A **node** is a 64-bit value. Content lives in the word bits; the rest
is encoding cargo, masked off at the harness boundary. Three
operations and a measure, all plain bit arithmetic:

```
x & y        shared content            fit(x, y)
x & ~y       x beyond y                underfit(x, y)
y & ~x       y beyond x                overfit(x, y)
x | y        combined content          merge / the can-fold
popcount(x)  mass of x                 mu(x)
```

`x` is **contained** in `y` when `x & y == x`. Labels are display-only.

A **kline** is `σ:[N]` — a **signature** and an **adjacency**. The two
parts are different kinds of kline material:

- The **signature** is the kline's _claim_: the value it asserts itself
  to be. A signature alone is a kline that has not been formally
  constructed — it exists as a claim, awaiting construction.
- The **adjacency** is the kline's _construction_: a sequence of
  nodes, each node a signature — a reference to a kline that may or may
  not be held anywhere. The sequence is authored order and may cite a
  signature twice; `can(N)` is the adjacency's composition, the OR-fold
  of its nodes: the content the construction asserts. The fold is
  blind to order and repetition — what the sequence preserves is
  identity, not content.

A kline is a **canon** when `σ == can(N)` — the claim is exactly its
construction.

The **domain** `D` is an insertion-ordered set of klines with one
write: `add`, idempotent; D only grows.

The **band** classifies structurally — S4 no adjacency, S1 canon,
S2 overlap, S3 disjoint — and grades; it never steers the search. The
**measures** (`significance`, `entropy`, `effort`) grade on `σ` against
`can(N)` and order a run's proposals by ascending effort.

## 2. The run

The model's runner is the **Cogitator**
(`src/rationalisation/cogitator.py`) — the simple reentry loop:
an entry's signature meets held claims, its fit, underfit and offering
residuals reenter as fresh entries — the decomposition only covers the
fit with the candidate's grounded signature, and a covering is not a
frame — pairings ground when all three sides resolve, stalls wait for
an exchange witness. §2.2–§2.4 describe its machine and rules.

### 2.1 The harness configuration

A script compiles to entries. The harness (`dev/dialogue/harness.py`)
builds **ask==goal scenarios**: every `==` ask with a resolvable goal
is one scenario — the ask presented once, the goal handed to it
directly, never presented and never stored. Every other entry is
**primed** into D verbatim (a harness fact). The harness owns the
domain; nothing learns into it during a run. Scenarios run in ask
order over the one live domain. The harness is the **claim sink**: a
claim is in its hands the moment it grounds — mid-run, not at the
run's return. For now the harness reports claims; it persists
nothing.

### 2.2 The machine's state

One `Cogitator` per rationalisation: a chain of **entries** (each
entry: an ask signature `A`, its remaining candidates, its parent,
the component it was pushed as, and its `at` map — the frame count at
each component's last descent), `stalled` (entries halted at S4, by
signature — the latest per signature), `frame` (covered content, in
mint order), and the run's `events` (reporting only, never read by
the search — the report is events alone). The Cogitator owns no
store: the domain is read-only. The claim is not run state: it leaves
at S2 through the **claim sink** (`on_claim`) — a run may broadcast
many. An unaligned stall follows the same shape: it leaves at the
run's end through the **stall sink** (`on_stall`), bare — `⟨σ, []⟩`.

A **reentry** is a bare region-signature: no witness nodes travel
with it — the child's candidates are whatever the domain holds.

Bounds: `MAX_ENTRIES = 4096` — one counter over the whole loop.

### 2.3 The rules

**WALK** (`rationalise`). An ask run opens with the goal as the one
and only candidate, then walks the entry chain until it unwinds or
the entry cap is reached (a run is never goal-less). Grounding does
not stop the loop: the run continues until the chain unwinds or the
cap; the claim — `KLine(σ_ask, [σ_goal])` — is broadcast to the claim
sink at every S2 alignment with the goal. A run may broadcast many;
the sink receives whatever arrives.

**CANDIDATES** (`_candidates`). An entry's signature `A` primes domain
klines by overlap:

```
c.sig & A != 0         overlap — no-neighbours never reenters
c.nodes is not empty   a bare signature offers no construction
c.nodes != [c.sig]     an identity is not a candidate
c.sig not an ancestor  no candidate may share a signature with an
                       ancestor entry (re-deriving the decomposition)
```

Candidates list in D's insertion order and are consumed from the
end — the most recently held leads.

**PAIR**. The entry's signature meets the candidate's **claim**:

```
F = A & σ          the fit — shared content
U = A & ~σ         the underfit — the entry beyond the claim
O = can(N) & ~A    the offering — the construction composed beyond
```

The claim's own extension (`σ & ~A`) is not the entry's business: the
claim fits, the construction offers, and nothing else is owed.
`F == A` — the claim covers the entry — frames `F` at once.

**REENTER**. A component reenters when unframed and **fresh**: once
per candidate, and again only when frames arrived since its last
descent (the `at` map). Priority is fixed — U, then O, then F. A
candidate that pushes nothing is consumed and its `at` map cleared;
the next candidate may re-descend every component.

**ALIGN**. All three components framed — S2 aligned — frames `A`. If
the candidate's signature is the goal's, the **claim** grounds:
`KLine(σ_ask, [σ_goal])` — the ask's signature over the goal's
signature — **broadcast at S2** through the claim sink, the older
proposal's equal: nothing is stored, the domain is not the
Cogitator's to write. Any entry may ground it — the claim is the
ask's, whatever entry aligned.

**STALL / EXCHANGE**. An entry with no candidates left stalls at S4,
filed under its signature, and backtracks — the parent keeps its
remaining candidates. Two stalls on one signature from different
components sharing a common ancestor witness each other: the
signature frames, the stall resolves, nothing is stored. A stall
that never aligns broadcasts at the run's end through the **stall
sink** — the claim channel's equal — as a bare `⟨σ, []⟩`; the
harness reads it as its connect proposal.

**HALT**. The entry cap is the only other terminal arrival; it stops
the run.

### 2.4 Invariants

- The run never writes D — the domain is the harness's; frames only
  grow; the events list only appends.
- Claims broadcast to the sink at every S2 alignment with the goal;
  unaligned stalls broadcast at the run's end. Neither channel
  assumes uniqueness — the sink holds whatever arrives.
- Every frame is a signature; every reentered region (U, O, F) is a
  signature — meets and residues of signatures. No composite — no
  `σ | can(N)` of any kline — is ever fitted, minted, or reentered.
- The run stores nothing.

## 3. Termination and determinism

**T1.** Every rationalisation halts: each loop iteration either
consumes a candidate, pushes one child entry, or backtracks one link;
a component re-descends only on new frames (the `at` map), and the
entries counter caps the whole loop outright.

**T2.** The outcome is a function of `(D₀, ask, goal)`: every choice
point is ordered (candidates list in D's insertion order and are
consumed from the end, the component priority is fixed U then O then
F, stalls file by signature), and the sets serve membership only.

## 4. Retired instruments

The certificate verifier (`verify.py`) and the existence checker
(`existence.py`) were built to make the reference machine's grounding
checkable without re-running its search. They found and fixed one
soundness defect (exhaustion is not evidence) and measured a
completeness gap (~20% of random scenarios carried a certificate the
deterministic search missed, attributed to the path bar) before being
parked as premature — they encoded the earlier fit, not the current
pairing. Both were retired with the reference machine they
instrumented. The scripts are the only guarantee.
