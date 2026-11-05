# Kalvin — the rationalisation model

The rules of the model in `src/rationalisation/cogitator.py`, where the
code is truth and this document is the rules the code implements. It replaces
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

The model is the **Cogitator** (`src/rationalisation/cogitator.py`) —
the reentry loop: an entry's signature meets held claims, its underfit
and offering residuals reenter as fresh entries, pairings ground when
both sides resolve, stalls wait for an exchange witness. Three moves
do the work — the descent, the drain, the capture — named in §2.3 and
in the engine's docstring, which is its truth.

The `ReferenceCogitator` (`src/rationalisation/model.py`) is retired
history — the pre-drain baseline, now out of date and ignored; it
remains in the tree only because old probes and tests still compare
against it. It is not part of the model.

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

One `Cogitator` per rationalisation, running one loop over an explicit
**calls stack**. A **call** is one reentry's exploration: the reentry's
signature and depth, its **candidates** (the reentry's own, taken last
first), its **stack** (a copy of the ancestor entries, the current
candidate's fresh entry atop it), its **events** (the candidate's
trace, copied forward), and its **fork** (the pushing call, the side
it was pushed on, the pairing's candidate fingerprint — set at push,
never mutated).

Shared across calls: the **frame** list (covered content, only
growing), the **stalls** (a signature → the stalling call), the
**claim** — set at most once — and the pairing counter. The events are
the grounding call's trace — the candidate path the claim came down; a
run that grounds nothing reports none.

A **reentry** is a signature — an unconstructed region seeking
construction. A **pairing** is one meeting of a reentry's signature
with a candidate's claim — an occurrence, never a value.

Bounds: `MAX_DEPTH = 10`, `MAX_PAIRINGS = 4096`. The pairing cap alone
bounds descent and drain alike; the interpreter's stack is never
spent.

### 2.3 The rules

**THE LOOP.** One loop services the top of the calls stack, and only
the top. A call lands on top of its caller; the caller sits frozen
beneath it — its entry still pushed, its candidate unadvanced — until
the call pops, and the caller's next step re-pairs its own entry
against its own candidate. The loop stops when the claim grounds, the
pairing cap is reached, or the stack empties.

**CANDIDATES** (`_candidates`). A reentry's own candidate list, primed
from the domain:

```
c.sig & E != 0            overlap — no-neighbours never pairs
c.adj is not empty        a bare signature offers no construction
c.adj != {c.sig}          an identity is not a partner
c.fingerprint() != E      like for like — the whole offering is the entry
c.sig != psig             the parent entry — the origin never re-pairs
c.sig != pcsig            the parent candidate
c.sig not framed          a covered signature has nothing to teach
```

At depth 1 the goal is the one and only candidate. These bars select a
reentry's own list; the drain applies a call's candidate to the
ancestors on its stack without them — a zero-fit pairing there forks
the whole entry as its underfit, the self-reentry, by which the other
side of a decomposition re-meets the entry's unresolved content.

**PAIR.** The entry's signature meets the candidate's **claim**:

```
F = E & σ          the fit — shared content
U = E & ~σ         the underfit — the entry beyond the claim
O = can(N) & ~E    the offering — the construction beyond the entry
```

The claim's own extension (`σ & ~E`) is not the entry's business: the
claim fits, the construction offers, and nothing else is owed. Both
sides framed — S2 aligned — the pairing **closes**: a frame of `E`
mints and the close is noted. Otherwise the unsettled sides descend.

**THE DESCENT.** A pairing's unsettled, unmemoed side becomes a new
call — its own candidates, its own stack copy — landing on top of the
caller. The caller's memo marks the side forked: once per candidate,
fresh for the next.

**THE DRAIN.** A call's reentry loop does not stop at its own entry —
it pops through the whole stack, pairing every ancestor against the
call's candidate, close cascades climbing as each mint frames the
entry above. The drain pairs ancestors without their candidate
filters: a zero-fit pairing is legal here, and its underfit is the
whole entry — the self-reentry, the entry meeting itself one level
down under the new candidate's context. Its stall comes from the far
side of the original pairing and witnesses the exchange.

**THE CAPTURE.** A call's entry is its loop variables, which the drain
leaves at the stack's bottom — the opening, depth 1 — so every
candidate after the first re-explores the opening entry, where the
claim grounds and stalls re-file at parent 0.

**CLAIM.** When a pairing of the opening at depth 1 closes against the
goal, the claim grounds: `Kline(E_ask, {σ_goal})` — the ask's
signature over the goal's signature — stored; the run's only earned
store; the run stops outright.

**STALL / EXCHANGE.** A reentry with no candidates at all stalls at
S4, filed under its signature. Two stalls on one signature witness
each other when their reentry chains, walked upstream, part through
the two sides of a shared pairing — the underfit and the offering of
one decomposition meeting again on the same content: the S3
alignment, the closing move of the canonical scripts. The exchange
mints a frame of that signature; nothing is stored.

**HALT / PROPOSAL.** Terminal arrivals: no adjacency (R4), the pairing
cap, exhaustion (a call whose candidates drain — the halt reports the
last drained entry, the opening), and unresolved stalls. Each halt
becomes a proposal — `connect`, `continue`, `acquire` — ordered by
ascending effort.

### 2.4 Invariants

- D only grows; the frame list only grows; the claim grounds at most
  once.
- A fork is set at push and never mutated — entry identities may be
  captured, forks cannot; the witness chains are sound.
- Every frame is a signature; every reentry is a signature. No
  composite value — no `σ | can(N)` of any kline — is ever fitted,
  minted, or reentered.
- Stores during an ask run: exactly the claim, and only on grounding.

## 3. Termination and determinism

**T1.** Every rationalisation halts: each loop step is one pairing,
one push, or one pop; pushes are bounded by the memo and `MAX_DEPTH`,
candidates only shrink, the pairing cap bounds descent and drain
alike, and the claim stops the loop outright.

**T2.** The outcome is a function of `(D₀, entry, goal)`: every choice
point is ordered (the calls stack is LIFO, candidates follow insertion
order taken last first, the drain walks the stack top-down, stalls and
witness chains are deterministic), the sets serve membership only, and
node values hash as plain integers.

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
