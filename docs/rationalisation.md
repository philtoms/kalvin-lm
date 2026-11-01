# Kalvin — the rationalisation model

The rules of the model in `src/rationalisation/model.py`, where the code is truth and
this document is the rules the code implements. It replaces
`docs/attic/trs-significance.md`, which named the mechanism a term
rewriting system. It is not one, and the name is retired: nothing here
locates a redex and replaces it, no rule fires by syntactic match, and
termination comes from bounds, not from a decreasing measure carried by
rewrite steps. What the mechanism is: **a deterministic, bounded,
depth-first search that discharges coverage obligations against a held
domain.** The name for the operation, in the code and here, is
*rationalisation*.

Three guarantees are claimed and argued in §3: the run always halts
(T1), the outcome is a function of the inputs alone (T2), and a grounded
claim always carries a checkable record of why it grounded (T3).
Completeness — that the search finds a grounding whenever one exists
within the bounds — is **not** claimed (§4).

Vocabulary note: every term in this document is the code's own
identifier. Where the code says `side`, `frame`, `witness`, so does this
document. The grading measures (`significance`, `entropy`, `effort`) and
the band (S1–S4) keep their names; they grade and order, they do not
steer the search.

## 1. Values, klines, the domain

A **node** is a 64-bit value. Content lives in the word bits; the rest
is encoding cargo, masked off at the boundary (the harness's
`WORD_MASK`). Three operations and a measure, all plain bit arithmetic:

```
x & y        shared content            meet(x, y)
x & ~y       x beyond y                residue(x, y)
x | y        combined content          merge / can-fold
popcount(x)  mass of x                 mu(x)
```

`x` is **contained** in `y` when `x & y == x`. Labels (`KNode.label`)
are display-only: no label ever affects a decision, a filter, or an
outcome — T2 depends on this.

A **kline** is `sig:[adj]` — a signature value and a set of nodes
(`Kline`). `sig` is non-empty. Three derived forms carry the whole
model:

- `can(adj)` — the OR-fold of the adjacency.
- A kline is a **canon** when `sig == can(adj)`: the signature is
  exactly what its adjacency composes to.
- `side(k) = sig | can(adj)` — the kline's **full content**. Every
  pairing in the search is between sides. (This is the departure from
  the retired spec, which paired `sig` against `can(adj)`; the code
  never does.)

The **domain** `D` is an insertion-ordered set of klines with one write:
`add`, which is idempotent. D only grows. Alongside the set, the domain
keeps a **canonical index**: for each signature value, the adjacency of
the first canon stored with that signature.

```
witness(s) = adj of the first-stored canon with signature s, else ∅
```

The witness is the held meaning of a signature: what the domain knows
`s` decomposes into. A signature never stored as a canon has no witness.

The **band** classifies a kline structurally, in order:

```
S4  adj is empty                     — nothing offered
S1  sig == can(adj)                  — a canon
S2  sig & can(adj) != 0              — partial overlap
S3  otherwise                        — no overlap
```

The band grades (the harness prints it; entries route on it at
presentation — a canon grounds on receipt) but nothing inside the search
branches on it.

The **measures** grade pairings and order proposals. They are computed
on `sig` against `can(adj)` — the claim against the offered content —
not on sides:

```
significance(k) = popcount(sig & can(adj)) / popcount(sig | can(adj))
entropy(k)      = 1 − significance(k)
effort(k)       = (popcount(U) + ½ · popcount(O)) / popcount(sig | can(adj))
```

with `U`, `O` from the decomposition below. Ascending effort orders the
proposals of a run: least entropy first, ties in generation order.

## 2. The run

### 2.1 The harness configuration

A script compiles to entries. The harness (`dev/dialogue/harness.py`)
builds **ask==goal scenarios**: every `==` ask with a resolvable goal is
one scenario, and every other entry is **primed** into D verbatim first
(a harness fact — priming is never a store the model earned). Each
scenario presents its goal alone — if it is a canon it grounds on
receipt (S1, stored); if not, it rationalises goal-less — then presents
the ask with the goal attached. Scenarios run in ask order over the one
live domain: an earlier scenario's stores are later scenarios' prior
knowledge.

### 2.2 The machine's state

One `Cogitator` per rationalisation. Its state:

| state        | what it is                                                        |
| ------------ | ----------------------------------------------------------------- |
| `agenda`     | a stack of reentries (LIFO — last pushed explores first)          |
| `rows`       | every pairing ever made, in creation order                        |
| `open_rows`  | the rows not yet resolved                                         |
| `frames`     | covered content values, in mint order, each with its source row   |
| `stalls`     | reentries stalled at S4, grouped by signature value               |
| `seen`       | (reentry, entry, candidate) fingerprints already paired           |
| `clock`      | one counter — pairing, wait and resolution order                  |
| `claim`      | the grounded claim, set at most once                              |

A **reentry** is `(kline, depth, parent row, side)` — a residual region
re-entered as a fresh entry, remembering which row's which side it came
from. A **row** is one pairing of a reentry's kline against a candidate,
holding `F`, `U`, `O`, its pushed children, and its fate. Rows are
events, not values: two rows may hold identical data and remain distinct.

Bounds: `MAX_DEPTH = 10`, `MAX_ROWS = 4096`.

### 2.3 The rules

Each rule names its anchor in `src/rationalisation/model.py`.

**POP** (`_run`). Pop the agenda. Stop when the agenda is empty, the
claim has grounded, or `len(rows)` reaches the cap. If the reentry is
already resolved, or its region is covered by a frame owned by a proper
ancestor row or by a joint frame, it settles without exploration
(**KNOWN**, `_known`): a known row is recorded — one emission, resolved
— and the resolution propagates to its parent. Otherwise reenter.

**CANDIDATES** (`_reenter`). A reentry's kline `k` pairs only with
domain klines `c` that pass every bar:

```
c.sig & k.sig != 0          overlap — no-neighbours never pairs
c.adj is not empty          a bare signature offers nothing
c.adj != {c.sig}            an identity is not a partner
side(c) != side(k)          like for like never pairs with itself
c.sig not framed            a covered signature has nothing to teach
c not in the reentry's path no candidate may share a signature with an
                          ancestor entry — re-deriving what decomposition
                          established (signature-level by design; see the
                          exhaustion rule below for what a bar may settle)
fingerprint unseen          each reentry pairs each candidate once
```

plus the **proven-fit skip**: a candidate whose fit already frames the
whole entry side and whose both residuals would settle without children
adds nothing — skip. A reentry whose every candidate is skipped
resolves as **known** only when every skip was proven — the region
stands on framed evidence, and a path-barred candidate that is itself a
proven fit counts as proven. A path bar on an unproven region is not
proof: the reentry halts there (**exhaustion is not evidence**).
Candidates are
taken in D's insertion order (a closers-first, least-content sort
exists in the code's history but is not active); the stack then
explores the first candidate's subtree to completion before the next,
and within a subtree the u-side leads.

**PAIR** (`_pair`). For the reentry's kline `k` and candidate `c`, on
sides:

```
F = side(k) &  side(c)          fit — the agreement
U = side(k) & ~side(c)          underfit — the entry beyond the candidate
O = side(c) & ~side(k)          overfit — the candidate beyond the entry
```

The opening pairing is special: at depth 1 the goal is the one and only
candidate, and its row is the `goal_row`. Each unsettled residual
becomes a child reentry carrying the witness nodes it contains
(`contained(n, region)`), the u-side and o-side each at most one child.
The u-side child explores first (LIFO with the push order). A row
satisfied at pairing — both sides already settled — resolves immediately
(`instant`).

**SETTLE** (`_side_ok`). A side region of a row is settled when any of:

1. it is empty;
2. its child reentry is resolved (recursively: the child's own pairing
   resolved, or the child was an exchange party);
3. it is the o-side and `witness(region)` is non-empty — a canonically
   held signature needs no re-derivation. (u-sides never settle this
   way: the asymmetry is deliberate, the claim side must be earned);
4. it is covered by a frame owned by a proper ancestor row of the
   reentry, or by a joint frame. The reentry's own rows and its
   siblings' never cover it — no self-satisfaction.

**FRAME** (`_frame`). When a pairing's fit equals the entry's whole
content — `F == side(k)`, the candidate covers the entry — the fit
becomes a **frame**: a content value the run has settled evidence for,
owned by the row that earned it. Mints cascade: every open row is
rechecked, and each newly satisfied row resolves. A value already framed
is still recorded (chains exclude their own subtree's frames, so
suppressed frames stay load-bearing for other rows). Exchanges mint
**joint frames** owned by no subtree.

**RESOLVE / BACK-PROPAGATE** (`_resolve`, `_after`). A row resolves
when both sides are settled. Its reentry resolves with it; the parent
row — the one that pushed that reentry — has that side satisfied, and
may itself resolve, upward to the opening pairing. Each resolution also
records a resolution frame (`rframes`): a *record* that never settles
anything — it exists so the trace can name what closed.

**CLAIM** (`_resolve`). When the opening pairing — `goal_row` —
resolves, the claim grounds: `Kline(ask.sig, {goal.sig})`, stored (the
run's only earned store). The loop stops at the claim.

**STALL** (`_reenter`). A reentry with no candidates at all stalls at
S4: a row with the whole content underfit, filed under its signature in
`stalls`. (Distinct from KNOWN: a reentry whose candidates all existed
but were barred or proven is *known*, not stalled.)

**EXCHANGE** (`_exchange`). Two reentries stalled on the same signature
value, from different parent rows, witness each other: both resolve, a
joint frame of that value is minted, both chains back-propagate, and
nothing is stored — the frame is the record; the domain learns nothing
from a mutual stall.

**HALT / PROPOSAL** (`_halt`, `proposals`). Terminal arrivals: a kline
with no adjacency (the run's R4), the depth bound, a candidate turned
away without proof (exhausted), and — reported after the loop — every
unresolved stall. Each halt becomes a **proposal**: the
halted kline read as a request, ordered by ascending effort. Kinds:
`connect` (stalled — nobody to pair with), `continue` (bound), `acquire`
(everything else).

### 2.4 Invariants

- D only grows; frames only grow; the clock only advances.
- Each reentry resolves at most once; the claim grounds at most once.
- Resolution frames never settle sides — only earned frames and joint
  frames do.
- Stores during an ask run: exactly the claim, and only on grounding.

## 3. Guarantees

### T1 — Termination

**Every rationalisation halts.** Each agenda pop does bounded work:
candidates are bounded by `|D|`, each pairing pushes at most two
children at `depth + 1`, and pops at depth beyond `MAX_DEPTH` halt
without pushing. Rows are capped (the cap is checked per pop, so one
pop's candidate batch can overshoot by at most `|D|` rows — finite
always). Frame cascades resolve at least one open row per round; the
exchange loop resolves two stalled reentries per round; both therefore
stop. The claim stops the main loop outright. No rule adds to the agenda
without consuming depth, so the agenda cannot grow without bound.

### T2 — Determinism

**The outcome — stores, claim, ends — is a function of `(D₀, entry,
goal)`.** Every decision point is ordered by construction: the agenda is
a stack, candidates follow D's insertion order, stall groups are lists
in stall order, frame sources are lists in mint order, the exchange
picks the first waiter and the first other-parent party, proposal ties
keep generation order (stable sort), and one clock orders all events.
Every set in the machine (`seen`, the domain's membership set) is used
for membership only — never iterated for a choice. `KNode` subclasses
`int`, and integers hash to themselves: no hash randomisation touches
any ordering. Frozenset iteration feeds only value construction (set
contents), never a choice; label text is display-only throughout.
Re-running the same script over the same primed domain reproduces the
run exactly.

Audit caveats, for the property tests that follow: the *presentation*
layer may reorder (`_trace` picks earliest-created sources by policy),
and the row cap's overshoot window means a run can exceed 4096 rows by
a bounded amount — determinism is unaffected, the bound is.

### T3 — Soundness: the covering certificate

Grounding is not a verdict the search asserts; it is a fact the run
records. Define the **justification** of a settled side, by cases:

1. *empty* — the region is `⊥`;
2. *resolved child* — the child reentry resolved: through its own
   pairing (whose sides carry justifications — recurse), through an
   exchange party (case 4), or without exploration at all (KNOWN — its
   region settles by case 4 or case 5);
3. *witness* — the domain holds a canon with this signature: a domain
   fact, checkable by lookup;
4. *frame* — the region is covered by a frame, which is either an
   **earned frame** (some explored pairing where the candidate's side
   contained the entry's whole side — evidence: a held kline, or the
   goal itself — the run's given — whose full content covers the frame
   value) or a **joint frame** (two stalled reentries on the same value
   — evidence: mutual witness, nothing more);
5. *proven* — the exhausted-candidates known: a covering candidate
   exists, the fit value is framed (any owner, evidenced as case 4),
   and the overfit settles by case 1, 3, or 4.

**Theorem.** If the claim grounds, the justifications of the opening
pairing's two sides form a finite tree — the **certificate** — whose
leaves are empty regions, domain canons, earned frames, and exchange
pairs.

*Proof.* By induction over the clock. Each resolution event fires only
after SETTLE checked its disjuncts against records already present:
child resolutions are earlier clock events (induction hypothesis),
witnesses are domain facts, frames were minted earlier by a pairing
whose cover condition held. The claim's resolution is the root; the
tree it roots is finite by T1. ∎

The exchange case is the certificate's honest weakest link: it records
that two subgoals coincided, not that the domain supports them. The
certificate says so, in the open, rather than laundering it through a
store.

**Corollary.** Grounding can be checked without re-running the search:
a verifier given `(D₀, ask, goal)` and the run's frames, sources, and
exchanges either reconstructs the tree or rejects. That verifier is the
next piece of work, with property tests asserting T1–T3 over the script
corpus.

## 4. Not claimed

- **Completeness.** That the search finds a certificate whenever one
  exists within the bounds is not claimed. Every bar in CANDIDATES
  prunes, and each needs its own argument before the claim could be
  made; until then, a script can fail to ground for search reasons as
  well as content reasons, and the two are not yet distinguishable.
- **Trace canonicity.** The presented path (which rows the table shows)
  is a policy over the records — earliest-created sources, one row per
  reentry. The records are deterministic (T2); the policy is a display
  choice.
- **Prediction.** Syntactic conditions on a script that guarantee
  grounding (or refusal) — the goal-directed reading of scripts — wait
  on completeness or stand alone as later work.
