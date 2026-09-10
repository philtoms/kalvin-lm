# Kalvin — A Term Algebra and Rewrite System (first pass)

Goal: a formal definition of Kalvin as a **term algebra** (what exists) and a working Kalvin system as a **term rewrite system** (what may happen). Terminology here is normative for the formalisation; the source follows later.

Status: draft.

---

## 0. Reading guide — four layers

Only the first two layers are the formal system proper; the last two are dynamics over it.

1. **Algebra** — what exists: values, klines, operations, laws (§1).
2. **Rewrite system** — what may happen: licensed steps from a new kline toward held klines (§2).
3. **Strategy & control** — what chooses: candidate selection, bounds, reentry (§3).
4. **Measurement** — what is observed: significance bands and the graded distance (§4).

---

## 1. The Algebra

### 1.1 Sorts

Two sorts. The operations of the algebra split across them, so a one-sorted reading ("the domain is klines") does not type-check.

**Definition 1 (values).** A finite set of atoms `A = {a₀ … a₃₀, ask}` (the atom set is a parameter — word-size dependent; only its _finiteness_ is load-bearing, see §2.5). The **value sort** `V` is the closure of the atoms under composition `∨`. A value is any atom or compound.

**Definition 2 (klines).** The **kline sort** `K` is the set of pairs `(s, ν)` where `s ∈ V` (the head — role name **signature**) and `ν ∈ V*` is a finite sequence of values (the slots — role name **nodes**). Written `s:[n₁,…,nₖ]`. The sequence is free: order and multiplicity are retained at this sort.

The same value may appear as a head in one kline and a slot in another. A kline's slot may hold the signature of another kline — nesting is by reference, so a memory is a DAG of witnesses, not a tree.

### 1.2 The node algebra and its laws

`(V, ∨, ∧, 0)` carries the structure of the powerset Boolean algebra over the atoms: `∨` is composition (union), `∧` is overlap (intersection), `0` the empty value.

**Laws.** `∨` is commutative, associative, idempotent, with unit `0`; `∧` distributes over `∨`; and **atoms are pairwise disjoint**: for distinct atoms `a, b`: `a ∧ b = 0`.

### 1.3 Evaluation — `signature_of` forgets

**Definition 3 (evaluation).** `σ : V* → V` is the homomorphism `σ([n₁,…,nₖ]) = n₁ ∨ … ∨ nₖ` (`σ([]) = 0`).

`σ` is a homomorphism from the free monoid (sequences under concatenation) onto `(V, ∨)`: it **forgets order and multiplicity**. The whole architecture lives in the gap it creates — the kline sort is free, the value sort is maximally quotiented, and `σ` is the evaluation map between them. The term-algebra claim is located precisely here: **klines are the terms; values are the quotient those terms evaluate into.**

### 1.4 Klines are decomposition witnesses

`σ` has no canonical inverse. For `s = a∨b∨c`, every sequence `ν` with `σ(ν) = s` is a decomposition of `s`: `[a,b,c]`, `[a,bc]`, `[ab,c]`, `[abc]`, `[b,a,c]`, … A kline `s:ν` with `σ(ν) = s` is a **witness**: a chosen decomposition, retained. Nothing in `V` reconstructs which witness was chosen — that is what memory is _for_.

> **The central claim.** Kalvin's memory is a set of decomposition witnesses over a forgetting algebra. Significance (§4) measures the failure of a kline to be a **section** (right inverse) of `σ`: canon is exactly the section property, and underfit/overfit are its two failure directions.

### 1.5 Structural predicates — the nine shapes

These are **predicates**, not operations (the algebra's only operations are `∨` and kline formation). They classify the fit of an arbitrary `(value, sequence)` pair — a kline is one such pair, and the pairwise relationship below is another, so one classification function serves both.

**Definition 4 (coverage).** Node `n` is **covered** by head `v` :≡ `n ∧ v ≠ 0`.

**Definition 5 (fit classification).** For `v:ν`:

- `ν = []` — **Unknown** (a terminal).
- `ν = [v]` — **Identity** (a terminal).
- non-terminal, `v = σ(ν)` — **Canon** (the section property).
- non-terminal, `v ≠ σ(ν)` — **Misfit** (genus), with gap `g = v ∧ ¬σ(ν)` and excess `e = σ(ν) ∧ ¬v`:
  - **Underfit** — `g ≠ 0, e = 0`: the signature claims beyond its nodes.
  - **Overfit** — `e ≠ 0, g = 0`: the nodes carry beyond the signature.
  - **Under+over** — `g ≠ 0, e ≠ 0`.
  - **No-fit** — no node covered (`∀n ∈ ν: n ∧ v = 0`). (The disjoint shape; renamed from the draft's row label "Misfit", which is reserved for the genus.)
  - single-node misfits carry the KScript names **Denotation** (`ab:[b]` — a covered single-node underfit) and **Connotation** (`a:[b]` — an uncovered single-node shape). These are names of convenience for scripting; algebraically they are species instances. (A single-node overfit, e.g. `a:[ab]`, exists and is unnamed.)

**Definition 6 (pairwise relationship).** For klines `A = (s, ν_A)`, `B = (t, ν_B)`, the **relationship kline** is `C(A,B) = (σ(ν_A) : ν_B)` — the head is what A's nodes evaluate to, the slots are B's nodes. The fit classification of `C(A,B)` is the **structural relationship of A and B**. The single-kline classification is the special case with the kline's own head against its own nodes.

The canonical table (atoms lowercase; klines roman):

| Structure   | A nodes | B nodes | C(A,B)        | Band | Species condition        |
| ----------- | ------- | ------- | ------------- | ---- | ------------------------ |
| Canon       | [a,b,c] | [a,b,c] | `abc:[a,b,c]` | S1   | `σ(ν_A) = σ(ν_B)`, exact |
| Identity    | [a]     | [a]     | `a:[a]`       | S1   | terminal                 |
| Underfit    | [a,b,c] | [a,c]   | `abc:[a,c]`   | S2   | gap only                 |
| Overfit     | [a,b]   | [a,b,c] | `ab:[a,b,c]`  | S2   | excess only              |
| Under+over  | [a,b,c] | [b,c,d] | `abc:[b,c,d]` | S2   | gap and excess           |
| Denotation  | [a,b]   | [b]     | `ab:[b]`      | S2   | single node, covered     |
| Connotation | [a]     | [b]     | `a:[b]`       | S3   | single node, uncovered   |
| No-fit      | [a,b]   | [c,d]   | `ab:[c,d]`    | S3   | no node covered          |
| Unknown     | [a]     | []      | `a:[]`        | S4   | terminal, empty          |

---

## 2. The Rewrite System

### 2.1 What is rewritten

A **derivation** rewrites the node sequence of a queued kline `A` toward the node sequence of a held kline `B`. The head of `A` is fixed (the claim); the nodes are the content. States `A = A₀ → A₁ → …` differ only in `ν`; each step is graded by the fit of `(Aᵢ, B)`.

The rule set is not fixed — it is a **fixed family of schemata instantiated by memory**: every schema is parameterised by the B it targets (and, for granularity moves, by held canon witnesses). Read model-theoretically this is rewriting modulo the theory "what is held"; the two readings are the same system (§2.5).

### 2.2 Two kinds of step

The earlier nine-row rewrite table conflated two orthogonal axes. Separated:

**Normalising moves (witness-licensed)** — change granularity, independent of any target:

- **Expand** — replace node `n` by `νₙ`, licensed by a held canon `n:νₙ`. (`abc → [a,b,c]`)
- **Contract** — replace a node subsequence by `σ` of it, licensed by a held canon `σ:[…]`. The inverse of expand.
- **Retain** — the identity move; a terminal stands.

**Targeting moves (B-licensed)** — align `ν_A` toward `ν_B`:

- **Remove** — drop a node of `ν_A ∖ ν_B`.
- **Add** — insert a node of `ν_B ∖ ν_A`.
- **Replace** — remove and add (the composite).

The primitives are add, remove (and halt, below); expand/contract are their witnessed composites; the fit species of `(A,B)` says which direction the work lies in:

| Fit of (A,B)          | Permitted targeting           |
| --------------------- | ----------------------------- |
| Canon, Identity (S1)  | none — the derivation is done |
| Underfit / Denotation | remove                        |
| Overfit               | add                           |
| Under+over            | remove and add                |
| No-fit / Connotation  | replace (full substitution)   |
| Unknown (S4)          | **no rule applies**           |

**Halt** is not a rewrite rule (a rule needs a term on its right-hand side; `abc → ?` has none). Halt is the _absence of an applicable rule_ — a stuck state. The strategy reacts to it as an event: an unknown signature is the ask, under which ungrounded proposals are generated (§3).

### 2.3 Metatheory — claims taken, claims renounced

- **Weak termination (claimed).** Significance-monotone derivations terminate: the value universe `V = 2^|A|` is finite, and targeting moves that raise significance cannot cycle. Non-monotone strategies _can_ cycle (add/remove oscillation); monotonicity is a strategy invariant, not a theorem about arbitrary derivations.
- **Confluence (renounced, deliberately).** The order of derivation changes what is grounded first, and the reachable S1 depends on the path. Path-dependence is not a defect to be repaired — it is the learning phenomenon.
- **Decidability (in principle).** The finite universe makes existence and non-existence of a derivation decidable in principle; the tractability gap between that and any affordable search is exactly where cogitation, study and scaffolding live.

### 2.4 Feedback

By measuring the significance of `(Aᵢ, B)` at each step, Kalvin ascertains whether its effort is increasingly or decreasingly significant, and tracks the rate of change of that measure over steps taken. These are strategy-level metrics (§4) — they steer the derivation; they are not part of the rule set.

### 2.5 Relation to the constraint-solving reading

A derivation _is_ a solver's search: each held kline is a constraint, each licensed step a resolution step, S1 a constructive existence proof within what is held, S4 relative non-existence (nothing shared). The rewrite system of this section is the operational face of that reading; it does not compete with it.

---

## 3. Strategy & Control

**Candidate selection.** Targeting schemata require a `B` to instantiate them. B-candidates are grounded klines (long-term and frame memory), selected when the fit of `(A, B)` is at least S2 — which implies at least one grounded signature covered by a node of `A`. Working-memory (STM) candidates have no coverage (S3); they enter derivations to evolve stepwise toward S2 overlap through progressive connotation — each connotation witness licensed by a held single-node misfit.

**Bounds.** Step budgets and hop ceilings are strategy parameters, not part of the system.

**Reentry.** The output of a derivation state re-enters as the input of another: propose from a proposal, one hop further out. Formally this is self-application of the rewrite search. The earlier space/time-axis language is **deferred** (§7): if it is wanted formally, define the time axis _as_ derivation order — nothing else in the system supplies one.

**Escalation and ratification live outside.** Countersigning (`==`) is a protocol concept — reciprocal connotation pairs held as ratified — not an algebra operation. The algebra provides the shape; the protocol provides the commitment.

---

## 4. Measurement

**Two levels of significance, stated as such:**

- **Band predicate (algebra level).** The classification of Definition 5, into S1 (exact — _I know that I know this_), S2 (relates but diverges — _I infer this, but it does not yet fit_), S3 (connects only indirectly — _I recognise aspects of this, indirectly_), S4 (shares nothing — _I do not understand this at all_). A discrete predicate on structure; every agent assesses independently.
- **Graded distance (strategy level).** A continuous measure of the gap between two klines: per-slot accountedness decays with resolution hops and composes across slots into a value graded across the open interval between the bands. Rate of change is defined **only** at this level — a four-band predicate has no useful derivative.

**Terminology.** _Significance_ is the value; _rationalisation_ is the process that produces and consumes it. (Not: "significance is the value Kalvin directly equates to rationalisation".) Understanding, informally, is high significance attained and held.

---

## 5. Syntax appendix — KScript mapping

Surface syntax is presentation; it does not appear in the formal tables.

| Token            | Structure produced          | Band claim once solved |
| ---------------- | --------------------------- | ---------------------- |
| `a => b c d`     | canon candidate             | S1 (canon) / open S2   |
| `a == b`         | reciprocal connotation pair | S1 on ratification     |
| `a > b`, `a < b` | connotation (`a:[b]`)       | S3                     |
| `a = b`          | denotation (`ab:[b]`)       | S2                     |
| `a` (bare)       | unknown (`a:[]`)            | S4 — the ask           |
| `a = a`, `a > a` | identity (`a:[a]`)          | S1                     |
| ask-annotated    | any signature, ask-marked   | S4                     |

A token declares an _intent_; the fit classification of the produced kline may or may not satisfy it (e.g. `=>` declares composition intent; the result is a canon only if the section property holds).

---

## 6. Worked micro-example

Atoms `m, h, a, l`. Held: identity `m:[m]`, canon `mall:[m,a,l,l]`. Queue `A = mall:[m,a]` (underfit: gap `l`, no excess).

1. Fit `(A, mall:[m,a,l,l])`: head `σ([m,a]) = ma`, slots `[m,a,l,l]` — overfit direction; permitted: add.
2. Add `l`, add `l` (targeting, B-licensed): `mall:[m,a,l,l]` — fit S1, canon. Derivation terminates at a constructive existence proof.

Had nothing been held: fit S4, no rule applies, halt — the ask event, and ungrounded proposals follow under strategy control.
