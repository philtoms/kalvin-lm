# Kalvin — the rationalisation model

The rules of the model in `src/rationalisation/model.py`, where the code is
truth and this document is the rules the code implements. It replaces
`docs/attic/trs-significance.md`, which named the mechanism a term
rewriting system. It is not one: nothing here locates a redex and
replaces it, no rule fires by syntactic match, and termination comes
from bounds. What the mechanism is: **a deterministic, bounded,
depth-first search in which unconstructed klines — signatures seeking
construction — meet held klines' claims and receive their offered
decompositions, until the claim grounds or the run halts.**

The scripts are the only guarantee that an ask can be framed as a
goal. If a script does not ground, that is the honest state of the
system; the model creates no contrivance to force a resolution.

## 1. Klines, and what they are made of

A **node** is a 64-bit value. Content lives in the word bits; the rest
is encoding cargo, masked off at the harness boundary. Three
operations and a measure, all plain bit arithmetic:

```
x & y        shared content            meet(x, y)
x & ~y       x beyond y                residue(x, y)
x | y        combined content          merge / the can-fold
popcount(x)  mass of x                 mu(x)
```

`x` is **contained** in `y` when `x & y == x`. Labels are display-only.

A **kline** is `σ:[N]` — a **signature** and an **adjacency**. The two
parts are different kinds of kline material:

- The **signature** is the kline's *claim*: the value it asserts itself
  to be. A signature alone is a kline that has not been formally
  constructed — it exists as a claim, awaiting construction.
- The **adjacency** is the kline's *construction*: a set of nodes, each
  node a signature — a reference to a kline that may or may not be
  held anywhere. `can(N)` is the adjacency's composition, the OR-fold
  of its nodes: the content the construction asserts.

A kline is a **canon** when `σ == can(N)` — the claim is exactly its
construction. The **fingerprint** `σ | can(N)` is the kline's identity
for loop control and nothing else: it is never content — never fitted
against, never minted, never reentered.

The **domain** `D` is an insertion-ordered set of klines with one
write: `add`, idempotent; D only grows. The **witness** of a signature
is the adjacency of the first-stored canon holding it — the held
meaning of a signature; a signature never stored as a canon has no
witness.

The **band** classifies structurally — S4 no adjacency, S1 canon,
S2 overlap, S3 disjoint — and grades; it never steers the search. The
**measures** (`significance`, `entropy`, `effort`) grade on `σ` against
`can(N)` and order a run's proposals by ascending effort.

## 2. The run

### 2.1 The harness configuration

A script compiles to entries. The harness (`dev/dialogue/harness.py`)
builds **ask==goal scenarios**: every `==` ask with a resolvable goal
is one scenario, and every other entry is **primed** into D verbatim
first (a harness fact — priming is never a store the model earned).
Each scenario presents its goal alone — a canon grounds on receipt
(S1, stored); otherwise it rationalises goal-less — then presents the
ask with the goal attached. Scenarios run in ask order over the one
live domain: an earlier scenario's stores are later scenarios' prior
knowledge.

### 2.2 The machine's state

One `Cogitator` per rationalisation: an `agenda` (a LIFO stack of
**reentries**), `rows` (every pairing, in creation order), `open_rows`,
`frames` (covered content, each with its source row), `stalls`
(reentries halted at S4, by signature), `seen` (pairing fingerprints,
for loop control), a `clock`, and the `claim` — set at most once.

A **reentry** is an unconstructed kline: a region-signature remembering
the row and side it came from. A **row** is one pairing — `F`, `U`,
`O`, its pushed children, and its fate; an event, never a value.

Bounds: `MAX_DEPTH = 10`, `MAX_ROWS = 4096`.

### 2.3 The rules

**POP** (`_run`). Pop the agenda. Stop when the agenda is empty, the
claim has grounded, or the row cap is reached. A reentry already
resolved, or covered by an ancestor's or joint frame, settles without
exploration (**KNOWN**) and propagates to its parent. Otherwise reenter.

**CANDIDATES** (`_reenter`). A reentry's signature `E` primes domain
klines by overlap:

```
c.sig & E != 0            overlap — no-neighbours never pairs
c.adj is not empty        a bare signature offers no construction
c.adj != {c.sig}          an identity is not a partner
c.fingerprint() != E      like for like — the whole offering is the entry
c.sig not framed          a covered signature has nothing to teach
c not in the entry path   no candidate may share a signature with an
                          ancestor entry (re-deriving the decomposition)
fingerprint unseen        each reentry pairs each candidate once
```

plus the **proven skip**: a candidate whose claim covers the whole
entry, whose fit is already framed, and whose beyonds would settle
without children, adds nothing — skip. A reentry whose every candidate
is skipped resolves as **known** only when every skip was proven; any
unproven turn-away halts the reentry — exhaustion is not evidence.
Candidates are taken in D's insertion order; the stack explores the
first candidate's subtree to completion, the u-side leading.

**PAIR** (`_pair`). The entry's signature meets the candidate's
**claim**:

```
F = E & σ          the fit — shared content
U = E & ~σ         the underfit — the entry beyond the claim
```

and the candidate's two **beyonds** — its claim's extension and its
construction's offering — leave the pairing as separate reentries:

```
σ & ~E             the claim's extension — what it asserts beyond the entry
can(N) & ~E        the offering — what its construction composes beyond
```

One reentry per content (the two beyonds may name the same region);
each o-child carries the adjacency nodes contained in it. The opening
pairing is special: at depth 1 the goal is the one and only candidate,
and its row is the `goal_row`. `E == F` — the claim covers the entry —
mints a **frame** of `E`, owned by the row.

**SETTLE** (`_side_ok`). A side region is settled when it is empty;
when *all* its children have resolved; when it is the o-side and the
domain holds its canon (a witnessed signature needs no re-derivation —
u-sides never settle this way: the claim side is earned); or when an
ancestor's or joint frame covers it — never the row's own subtree's.

**FRAME** (`_frame`). A minted frame is covered content the run holds
evidence for. Mints cascade: every open row is rechecked, and each
newly satisfied row resolves. Exchanges mint **joint frames** owned by
no subtree. Every frame's value is a signature; no composite is ever
minted.

**RESOLVE / BACK-PROPAGATE** (`_resolve`, `_after`). Both sides settled
→ the row resolves → its reentry resolves → the parent row's side is
satisfied, upward to the opening pairing.

**CLAIM**. When the opening pairing resolves, the claim grounds:
`Kline(E_ask, {σ_goal})` — the ask's signature over the goal's
signature — stored; the run's only earned store; the loop stops.

**STALL / EXCHANGE** (`_exchange`). A reentry with no candidates at
all stalls at S4, filed under its signature. Two stalls on one
signature from different parents witness each other: both resolve, a
joint frame of that signature records it, nothing is stored. Stalled
overfit meeting stalled underfit on the same content is the S3
exchange — the closing move of the canonical scripts.

**HALT / PROPOSAL**. Terminal arrivals: no adjacency (R4), the depth
bound, exhaustion without proof, and unresolved stalls. Each halt
becomes a proposal — `connect`, `continue`, `acquire` — ordered by
ascending effort.

### 2.4 Invariants

- D only grows; frames only grow; the clock only advances.
- Each reentry resolves at most once; the claim grounds at most once.
- Every frame is a signature; every reentry is a signature. No
  composite value — no `σ | can(N)` of any kline — is ever fitted,
  minted, or reentered.
- Stores during an ask run: exactly the claim, and only on grounding.

### 2.5 The union shape is an optimisation, not structure

An earlier fit ran the two beyonds as one region — the candidate's
fingerprint `σ | can(N)` against `E`, a single o-child of their union.
It closes the same corpus scripts with fewer reentries and faster
runs, and where a composite child's whole content is covered together
it is a genuine economy. But it constructs composite content — values
no kline asserts (the `Mary|had|SVO` shape: a stored claim's signature
and construction fused) — and closing that child then requires
composite evidence. The claim-fit with split beyonds is the structure;
the union remains available as an optimisation of it, at the price of
constructed values reentering the search.

## 3. Termination and determinism

**T1.** Every rationalisation halts: each pop does bounded work,
children are bounded by `MAX_DEPTH`, rows by the cap, frame cascades
and exchange loops strictly consume open work, and the claim stops the
loop outright.

**T2.** The outcome is a function of `(D₀, entry, goal)`: every choice
point is ordered (the agenda is a stack, candidates follow insertion
order, stall groups are lists, ties keep generation order), the sets
serve membership only, and node values hash as plain integers.

## 4. Parked instruments

The certificate verifier (`verify.py`) and the existence checker
(`existence.py`) were built to make grounding checkable without
re-running the search. They found and fixed one soundness defect (the
exhaustion halt above) and measured a completeness gap (~20% of random
scenarios carried a certificate the deterministic search missed,
attributed to the path bar) before being parked as premature — they
encode the earlier fit and do not describe the current pairing. The
scripts are the only guarantee; the instruments wait until the
semantics settles for good.
