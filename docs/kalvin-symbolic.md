# Kalvin — A Term Algebra and Rewrite System

Goal: a formal definition of Kalvin as a **term algebra** (what exists) and a working Kalvin system as a **term rewrite system** (what may happen). First pass, updated in place to the ks2 semantics and rules; `ks2.md` (the fourth pass) is normative — for the formalisation and for the KScript surface syntax (§14 there, which absorbs this document's §5). CONTEXT.md is normative for role names.

Status: draft.

---

## 0. Reading guide — four layers

Only the first two layers are the formal system proper; the last two are dynamics over it.

1. **Algebra** — what exists: values, klines, operations, laws (§1).
2. **Rewrite system** — what may happen: licensed steps from a new kline toward held klines (§2).
3. **Strategy & control** — what chooses: candidate selection, slot walks, bounds, reentry (§3).
4. **Measurement** — what is observed: significance bands and the graded distance (§4).

---

## 1. The Algebra

### 1.1 Sorts

Two sorts. The operations of the algebra split across them, so a one-sorted reading ("the domain is klines") does not type-check.

**Definition 1 (values).** A finite set of atoms `A = {a₀ … a₃₀}` — a parameter; only its _finiteness_ is load-bearing. In the engine this is the word-bit space: one bit per distinct word (the u64 packing `(word_bit << 32) | bpe_token_id` is an encoding this algebra deliberately tidies). The **value sort** `V = 2^A` is the set of atom sets — any atom or compound. Write `a` for the singleton and `abc` for `{a,b,c}`; `∅` is the empty value, `A` the full value. There is no ask atom: the ask is structural — the Unknown shape of Definition 6 — not a mark.

**Definition 2 (klines).** The **kline sort** `K` is the set of pairs `s:ν` where `s ∈ V` is nonzero (the head — role name **signature**) and `ν ∈ V*` is a finite sequence of non-empty values (the slots — role name **nodes**). The sequence is free: order and multiplicity are retained at this sort.

The same value may appear as a head in one kline and a slot in another. A kline's slot may hold the signature of another kline — nesting is by reference.

**Definition 3 (memory).** A **memory** `M` is a finite set of klines. Two klines may share a signature — distinct claims, or distinct decompositions of the same value. The reference graph is unrestricted and may cycle — canon self-reference (`a:[a,a]`), countersign pairs (`a:[b]`, `b:[a]`) — for memory holds any kline; cycles carry no decomposition content (§2.2). Which klines are _held_, and in what tier, are relations over `M` defined above the algebra (CONTEXT.md).

### 1.2 The node algebra and its laws

`(V, ∨, ∧, ¬, ∅)` carries the structure of the powerset Boolean algebra over the atoms: `∨` is composition (union), `∧` is overlap (intersection), `¬` complement within `A`, `∅` the empty value.

**Laws.** `∨` is commutative, associative, idempotent, with unit `∅`; `∧` distributes over `∨`; and **atoms are pairwise disjoint**: for distinct atoms `a, b`: `a ∧ b = ∅`. The laws hold by construction — consequences of the set definition, not axioms.

### 1.3 Evaluation — `signature_of` forgets

**Definition 4 (evaluation).** `σ = signature_of : V* → V` is the homomorphism `σ([n₁,…,nₖ]) = n₁ ∨ … ∨ nₖ` (`σ([]) = ∅`).

`σ` is a homomorphism from the free monoid (sequences under concatenation) onto `(V, ∨)`: it **forgets exactly order and multiplicity — nothing else**. `V` identifies precisely what `σ` identifies. The whole architecture lives in the gap it creates — the kline sort is free, the value sort is maximally quotiented, and `σ` is the evaluation map between them. The term-algebra claim is located precisely here: **klines are the terms; values are the quotient those terms evaluate into.**

### 1.4 Klines are decomposition claims

`σ` has no distinguished inverse. For `s = a∨b∨c`, every sequence `ν` with `σ(ν) = s` is a decomposition of `s`: `[a,b,c]`, `[a,bc]`, `[ab,c]`, `[abc]`, `[b,a,c]`, … A kline `s:ν` with `σ(ν) = s` is **exact** — a **witness**: a chosen decomposition, retained. The trivial inverse `s ↦ [s]` always exists; its images are the **identities**, witnesses that carry no decomposition content — every other witness is a real choice. Nothing in `V` reconstructs which witness was chosen — that is what memory is _for_.

> **The central claim.** Kalvin's memory is a set of claims over a forgetting map. A kline claims its signature as the composition of its nodes; **Canon** is the claim exactly kept; **Underfit/Overfit** are the two directions a claim can miss while still being answered; the S3 shapes are claims nothing answers; **Unknown** makes no composition claim — nothing held, or nothing left — and lands S4. Significance grades the claim.

### 1.5 Structural predicates — the nine shapes

These are **predicates**, not operations (the algebra's only operations are `∨`, `∧`, `¬` and kline formation). They classify the fit of an arbitrary `(value, sequence)` pair — one total function; a kline is one such pair, and the pairwise relationship below is another, so one classifier serves both.

**Definition 5 (coverage).** Node `n` is **covered** by head `s` :≡ `n ∧ s ≠ ∅` — they share at least one atom. Coverage is overlap, not containment.

**Definition 6 (fit classification).** `fit : V × V* → Shape`. Cases in order; every pair matches exactly one:

| #   | Condition                 | Shape       | Band |
| --- | ------------------------- | ----------- | ---- |
| 1   | `ν = []` or `s = ∅`       | Unknown     | S4   |
| 2   | `ν = [s]`                 | Identity    | S1   |
| 3   | `s = σ(ν)`                | Canon       | S1   |
| 4   | not covered, `\|ν\| = 1`  | Connotation | S3   |
| 5   | not covered, `\|ν\| > 1`  | No-fit      | S3   |
| 6   | covered, `g ≠ ∅`, `e = ∅` | Underfit    | S2   |
| 7   | covered, `g = ∅`, `e ≠ ∅` | Overfit     | S2   |
| 8   | covered, `g ≠ ∅`, `e ≠ ∅` | Under+over  | S2   |

with gap `g = s ∧ ¬σ(ν)` — atoms the signature claims beyond its nodes — and excess `e = σ(ν) ∧ ¬s` — atoms the nodes carry beyond the signature (`g = ∅` and `e = ∅` together iff `s = σ(ν)`).

The cases are disjoint by construction. Coverage is the primary split: a pair with no covered node forces `g = s ≠ ∅` and `e = σ(ν) ≠ ∅`, so cases 4–5 can never satisfy 6; exactness is caught at case 3, before the covered cases, which require a nonzero gap or excess. Case 1's disjunction is the no-claim case in both directions: `ν = []` — nothing held; `s = ∅` — nothing left. Replace never empties a node sequence (§2.2) — the empty side is an entry condition, not a run outcome.

**Species.** **Denotation** is the single-node Underfit (`ab:[b]` — one covered node, gap only); **Connotation** is case 4. These are names of convenience for KScript (`=`, `>`/`<`); algebraically they are single-node instances of cases 6 and 4. Two single-node shapes are unnamed: the single-node Overfit `a:[ab]`, and the single-node Under+over `ab:[bc]`.

**Bands.** Derived, not asserted: S1 = cases 2–3 (exact), S2 = covered misfits, S3 = uncovered misfits, S4 = Unknown. Readings carry over: S1 — _I know that I know this_; S2 — _I infer this, but it does not yet fit_; S3 — _I recognise aspects of this, indirectly_; S4 — _I do not understand this at all_. The **ask** is structural, not declared: Unknown (`s:[]` — nothing held for the signature) is the ask's shape, the halt signal under which strategy generates ungrounded proposals (§3). No atom, mark, or decree is involved.

**Invariance.** `fit` is insensitive to node order: it depends on `ν` only through `σ(ν)` and the node count. Duplicating a node changes the fit only when it crosses a count boundary (`a:[a]` Identity vs `a:[a,a]` Canon; `a:[b]` Connotation vs `a:[b,b]` No-fit); otherwise the repetition is witness structure invisible to classification.

**Definition 7 (pairwise relationship).** For klines `A = s:ν_A`, `B = t:ν_B`, the **relationship kline** is `C(A,B) = σ(ν_A) : ν_B` — the head is what A's nodes evaluate to (not a claim: _defined_, hence exact against `ν_A` by construction, so all of C's misfit-ness comes from B's side), the slots are B's nodes. The fit classification of `C(A,B)` is the **structural relationship of A and B**. `fit` is one function with two readings — a kline's own `(s, ν)`, or `(σ(ν_A), ν_B)` — and neither reading is a special case of the other; both are arguments to the same classifier.

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

Useful equivalence: `C(A,B)` is Canon iff `σ(ν_A) = σ(ν_B)` — the two klines hold the same value, differently decomposed.

---

## 2. The Rewrite System

### 2.1 What is rewritten

A **derivation** rewrites the node sequence of a queued kline `A = s:ν` against one held goal `B = t:ν_B`, relative to memory: `A ⊢_{M,B} A′`. The signature `s` never changes — the claim is fixed; the content is rewritten. States `A₀ ⊢_{M,B} A₁ ⊢_{M,B} …` differ only in `ν`. Nothing in this section reads `s`: licences, endings, bounds and grades read only the relationship, whose head `σ(ν_A)` is exact against `ν_A` at every state by construction. Within a hop the derivation is the two sides of `C` — `σ(ν_A):ν_A` against `t:ν_B` — and the queued head rides along inert, mattering only beyond the hop (absorb, reentry §3, the claim's grounding §5). Membership and difference on node sequences are multiset-wise; sequence order is used only by contract's contiguity (below) and otherwise retained for witness purposes.

The rule set is not a fixed family of schemata — it is **one rule, instantiated by memory**: every step is licensed by a held correspondence kline. Read model-theoretically this is rewriting modulo the theory "what is held" — canon evidence generates a congruence on sequences (expand and contract its two directions), and the full evidence set generates a **correspondence graph**: held klines as edges between a signature and its witness, traversable in either direction from wherever the derivation has arrived. A derivation is a path in that graph. The two readings are the same system (§2.5).

### 2.2 One rule, two licences

**Replace is the only rule; what differs is what licenses it.**

```text
replace:   a held correspondence kline K = n:ν_K ∈ M and an occurrence in ν_A
           matching one of K's two sides:
  forward:   an occurrence of n  → replaced by ν_K
  reverse:   an occurrence of ν_K (a contiguous block) → replaced by [n]
```

The terminals are inert as evidence: an Unknown (`n:[]`) has no second side; an Identity (`n:[n]`) replaces a node by itself. Every other held kline is a correspondence. K's own fit fixes the **mode**: a canon exacts granularity — forward is **expand**, reverse is **contract**, both `σ(ν_A)`-preserving; a covered misfit moves content by its gap and excess — forward **sheds** K's gap and **adopts** K's excess, reverse the mirror; an uncovered misfit is a **traverse** — disjoint atoms swap, either direction. **Direction is not a property of the kline**: a correspondence is read forward from its signature, reverse from its witness; only the side the derivation stands on occurs, so arrival orients the licence — the same kline read from the other side is the mirror derivation's licence. An edge is **ratified** or unratified — a tier relation over `M`; no rule of this section reads it, measurement does (§4).

Canon evidence must be well-founded (`n ∉ ν_K`). The clause is exactly strong enough: a canon's nodes are atom-subsets of its head, so an expansion cycle forces atom-equality at every step — a canon containing its own signature. Short of that, licensed expansion terminates and depth is well-defined (§4). Identities and self-containing canons are the two inert witness classes.

The two licences:

- **Witnessed** (canon-mode: expand/contract). Preserves `σ(ν_A)` exactly — granularity changes at constant content, band unchanged — licensed by `M` alone, blind to any goal.
- **Evidenced targeting** (denotation-/connotation-mode). Moves `σ(ν_A)` toward `σ(ν_B)`, licensed by a held correspondence and scoped by the misfit region.

A replace may be _exhibited_ as an interleaving of removals and insertions — a presentational device with no algebraic status: the licence is the correspondence, never the band alone. The interleaving never empties `ν_A` — both sides of a correspondence carry content. There is no retain move; an Identity relationship is done, yet the identity kline's own node may still expand under a held well-founded witness (`s:[s] → s:[a,b]`) — terminals are targeting-closed, not rule-closed.

**Licensing.** The relationship scopes; the evidence licenses. The band says where work remains; only a correspondence kline says what may move:

| `fit(C(A,B))`            | Licensed targeting                             |
| ------------------------ | ---------------------------------------------- |
| S1 — Canon, Identity     | none — done                                    |
| S2 — covered misfits     | replace, on the misfit region (scoping clause) |
| S3 — Connotation, No-fit | replace — every node sits wholly in the misfit |
| Unknown — S4             | none — stuck                                   |

**Scoping clause.** A replace is targeting-licensed iff it strictly decreases the **misfit mass** `|σ(ν_A) Δ σ(ν_B)|` — the node replaced carries a gap atom, or the witness carries excess atoms, or both. Band-blind alignment is thereby unlicensable: a replace that touches only shared content, or grows the misfit, is not a targeting move however well evidenced. At S3 no node is covered, so replaces there move whole content and the route to overlap runs through the progressive path (§3).

**Endings.** A derivation ends at **done** or **stuck**, or is **abandoned** by strategy:

- **Done** — `fit(C(A,B)) ∈ S1`: the relationship holds. The goal is **value-equality**, `σ(ν_A) = σ(ν_B)`, not node-equality — done may arrive early, an Identity relationship done with `ν_A ≠ ν_B`, pending nodes as witness structure.
- **Stuck** — not done, and no licensed targeting move. Two reachable conditions, both the ask: **no goal** — nothing to scope against (the ask at entry); or **no connection** — a goal is held, the misfit region is non-empty, and no held correspondence licenses a replace into it, nor does any slot walk arrive (§3): nothing in `M` connects A's misfit to the goal's content. Stuck is the absence of an applicable rule — a rule needs two sides to stand on — and the strategy reacts to it as an event: the ask, under which ungrounded proposals are generated (§3).
- **Abandoned** — not an ending the rules produce: strategy halts or re-targets a run mid-derivation (§3), e.g. when graded effort falls (§4).

Licences are permissive in one sense: a correspondence may itself be an ungrounded claim — S3 evidence is a promise, not a fact — and the derivation follows it faithfully. Weighing promises is protocol (§3, §5), not the rule system's.

### 2.3 Metatheory — claims taken, claims renounced

- **Targeting termination (claimed).** Any run of targeting replaces from `A₀` terminates in at most `Δ₀ = |σ(ν_{A₀}) Δ σ(ν_B)|` steps: each licensed replace strictly decreases the misfit mass, and a replace may move several atoms at once — steps are evidence-sized, the bound atom-wise, both computable, the natural unit for step budgets. Regressive and circular targeting is not merely unlikely but _unlicensable_.
- **Witnessed termination (strategy property, not a theorem).** Witnessed replaces preserve `σ(ν_A)` and can cycle — `bc` expands to `[b,c]` and contracts back against the same held canon, at constant band, forever — and slot-wise traversals (§3) can wander the correspondence graph at constant misfit mass. Termination of mixed derivations is a strategy invariant: bound witnessed runs (no expand-after-contract of the same witness) and bound traversals by no-revisit — each held signature consumed at most once per slot run; `M` is finite.
- **Confluence (renounced, deliberately).** The order of derivation changes what is grounded first, and the reachable S1 depends on the path. Path-dependence is not a defect to be repaired — it is the learning phenomenon.
- **Decidability (in principle).** The finite universe makes existence and non-existence of a derivation decidable in principle; the tractability gap between that and any affordable search is exactly where cogitation, study and scaffolding live.

### 2.4 Feedback

By measuring the fit of `C(Aᵢ, B)` at each step, Kalvin ascertains whether its effort is increasingly or decreasingly significant, and tracks the rate of change of that measure over steps taken. These are strategy-level metrics (§4) — they steer the derivation; they are not part of the rule set.

### 2.5 Relation to the constraint-solving reading

A derivation _is_ a solver's search: each held kline is a constraint, each correspondence an edge, each licensed replace a resolution step along one, S1 a constructive existence proof within what is held, and stuck relative non-existence — nothing in `M` connects. Done proves `σ(ν_A) = σ(ν_B)`, with the final node sequence as the witness and every step licensed by a correspondence, so the path itself is carried as evidence; it does **not** prove A's own head-claim, nor the truth of the correspondences followed. The rewrite system of this section is the operational face of that reading; it does not compete with it.

---

## 3. Strategy & Control

The loop is **cogitation** (CONTEXT.md): **select** a hop, **derive** to an ending, **absorb** the result into memory, **reenter** with the output as the next queue's input. Each phase is strategy — the rule system of §2 constrains what any of it may do, never what it must.

**Candidate selection.** Selection chooses the next hop, never the final goal. A held kline `K = t:ν_K` is **selectable** for queued `A` when `t ∈ ν_A` — K's signature occurs as a node of A: A already references what K is. That clause is the forward half of replace's licence: to select K is to be licensed to replace its signature by its witness. The guard compounds into a ratchet: each replace's arrival puts new nodes into `ν_A`, making their klines selectable next — the path is the guard, not the point. The derivation's goal `B` is never selected: it is declared (KScript `=>`, §5) or supplied by reentry, it scopes the misfit region, and it is _checked_ at done. S3 connotations are selectable as evidence; an Unknown has no second side to offer; weighing claims is protocol. The two-overlap caveat stands: **content overlap** `σ(ν_A) ∧ σ(ν_B) ≠ ∅` — exactly what relationship-S2 asserts — and **signature-in-node** `t ∈ ν_A` — the selection clause — imply neither the other. `A = abc:[a]` against `B = x:[c,a]` stands in an Overfit relationship (S2) while `x` neither occurs in nor overlaps A's nodes; `B = x:[y]` against `A = abc:[x]` occurs in A's nodes yet yields S3. Selection requires the second; the band routes by the first.

**Slot derivation.** A misfit decomposes per node: each node of `ν_A` carrying a gap atom is a **slot**, seeking the goal's excess `e = σ(ν_B) ∖ σ(ν_A)`. A slot with a licensed replace fires it. A slot with none — the misfit _asks_ at that node — may be **walked**: queue `n:[n]` and derive with no goal. The walk's licence is occurrence: any held correspondence, either side occurring in the walk's nodes, replaces it; the scoping clause does not apply — the walk travels the correspondence graph. A goal-less derivation has no done; its endings are **arrival** — `σ(ν) ∧ e ≠ ∅`, canon-mode replaces then setting granularity freely — or **stuck** — no unvisited correspondence occurs: the ask localised to one slot. Each hop writes its output to STM, and the walk's end state is absorbed as the **composed correspondence**: signature `n`, witness the arrival, acquisition depth the edges crossed (§4). The main line replaces `n ⇉ ν_arrived` on it — targeting-licensed because the arrival's content is excess and the misfit mass strictly decreases. The walk is how a missing licence is built: evidence is literally the work undertaken by the **progressive path** — inserted nodes as single-node misfits, connotation witnesses — and when the relationship reaches S2, ordinary evidenced targeting takes over.

**Bounds.** Three numbers, all strategy parameters: the **targeting budget** — targeting runs are bounded by `Δ₀` (§2.3), so a budget at or above it never binds mid-run; the **witnessed-run and traversal bounds** — no expand-after-contract of the same witness, no revisit of consumed signatures; and the **hop ceiling** — the reentry depth, below.

**Reentry.** Derivations compose. Hop `k` runs under parameters `(M_k, B_k)` — the goal `B_k`, or none for a slot walk; its end state — done, arrival, or stuck — queues as hop `k+1`'s input, and memory may grow between hops (`M_{k+1} ⊇ M_k`, by STM writes — the growth is the evidence accumulating), so successive hops are not derivations of one fixed system. Propose from a proposal, one hop further out, bounded by the hop ceiling. Hop order is the only time the system has; if a time axis is wanted, it is this order and nothing else.

**Escalation and ratification live outside.** Countersigning (`==`) is a protocol concept — reciprocal connotation pairs held as ratified — not an algebra operation. The algebra provides the shape; the protocol provides the commitment. Ratifying a traversed pair promotes it to a standing one-hop licence — memory compounds its evidence. The queue itself — which klines are admitted for cogitation, and in what order — belongs to the harness, not the system.

---

## 4. Measurement

**The band order.** The bands are derived from shape (Definition 6); measurement adds one axiom: they are **ordered by significance**, `S1 > S2 > S3 > S4`, the shapes within a band unordered. The predicate is observer-independent — given the same held memory, every agent classifies alike — so a band never needs to be exchanged. Two band attachments are in play: a kline's **own band** — `fit(s, ν)` on itself, the claim it makes standing alone — and a **relationship band** — `fit(C(A,B))`, what the pair achieves. The first is what a kline asserts; the second is what a derivation establishes or fails to.

**Two levels of significance, stated as such:**

- **Band predicate (algebra level).** The classification of Definition 6, into S1 (exact — _I know that I know this_), S2 (relates but diverges — _I infer this, but it does not yet fit_), S3 (connects only indirectly — _I recognise aspects of this, indirectly_), S4 (shares nothing — _I do not understand this at all_). A discrete predicate on structure; every agent assesses independently.
- **Graded distance (strategy level).** A continuous measure, fixed rather than chosen — four requirements force one form:

  `γ(A, B) = J(σ(ν_A), σ(ν_B)) · δ^(D̄ + Ĥ)` where `J(x, y) = |x ∧ y| / |x ∨ y|`

  `J` is the depth-free core — Jaccard overlap: symmetric, 0 exactly at content-disjointness, 1 exactly at value-equality. It is forced: per-slot accountedness `α(n) = |n ∧ σ(ν_B)| / |n|`, composed atom-weighted, yields the A-side coverage fraction — which reads 1 on pairs whose content sits wholly inside the goal's (still S2), so B's excess must be weighed too, and Jaccard is the result. `D̄` is the **mean resolution depth** of A's content — granularity, 0 for content held as itself, well-defined because licensed expansion terminates. `Ĥ` is the **mean acquisition depth** — provenance, counting the unratified correspondence edges crossed to bring content in; it is **carried, not computed**: entry content is 0; content entering by a replace on evidence `K` records `Ĥ(K) + 1` through an unratified edge, `Ĥ(K)` through a ratified one — grounded correspondences cost nothing; shed enters nothing; canon-mode replaces move granularity, not content. `δ ∈ (0,1)` is the strategy's knob, the only one. `J` says _how close_; `δ^(D̄ + Ĥ)` says _how hard-won_. γ is directional and path-dependent by design: won and given knowledge with the same `(s, ν)` grade differently — the record is part of what memory holds.

  Four properties, each doing a job: **band-consistency** — γ is 0 exactly at disjointness, maximal only at value-equality; **granularity-invariance** — atom-weighted composition is blind to how content is sliced into slots; **granularity-monotonicity** — expand increases `D̄`, so decreases γ, making gratuitous expansion detectable; **provenance-monotonicity** — unratified acquisition increases `Ĥ` and nothing in a derivation lowers it, making promise-stacking detectable. Consuming unratified evidence can lower γ even as `J` rises — the step that wins the answer dips: the price signal steering strategy toward ratified standing licences. Both monotonicities are strategy invariants, not theorems about arbitrary derivations.

  Rate of change is defined **only** at this level — a four-band predicate has no useful derivative — and is the signal cogitation's feedback acts on.

**Exchange.** The graded value travels in a KValue (CONTEXT.md) as the sender's assessment, and the acquisition record travels with the kline that earned it — depths are part of what memory holds. Bands need not travel, for they are recomputable from structure.

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

A token declares an _intent_; the fit classification of the produced kline may or may not satisfy it. In particular `=>` declares composition intent _and supplies the goal for the done-check_ — it is not a licence; licences are correspondence klines (§2.2), and the result is a canon only if case 3 of Definition 6 fires. The ask row is structural: a bare signature is the ask — stuck at S4 — and no atom or mark is involved.

---

## 6. Worked micro-example

Atoms `m, a, l`. Held: identity `m:[m]`; canon `mall:[m,a,l,l]` — the answer; `a:[a,l]` — declared `a => a l` (composition intent), structurally an Overfit: its nodes carry `l` beyond its head, an unratified correspondence. Queue `A = mall:[m,a]` — the question, itself an Underfit (gap `l`). Declared goal `B = mall` (the KScript `MALL` line).

Relationship `C = ma:[m,a,l,l]` — head `σ([m,a]) = ma`, slots B's — Overfit (S2), misfit mass `Δ₀ = 1`.

One licensed replace finishes it: `a ⇉ [a,l]` forward on `a:[a,l]` — a covered misfit, forward adopts its excess, so `l` enters. State `mall:[m,a,l]`: `σ(ν_A) = mall`, relationship Canon — **done**, one targeting replace under a bound of one, its own fit Canon as well. The witness carries the licence: that is Kalvin _knowing_ the rhyme, not copying it.

Had `a:[a,l]` not been held, no correspondence licenses a move into the excess, and expand/contract cannot help — they preserve content: **stuck**, the misfit asks, and ungrounded proposals follow under strategy control. Had `mall` itself not been held, there is no goal to check done against: the ask from the other side.

Priced (§4): `l` enters through an unratified edge — acquisition depth 1, with `m, a` at 0 — so `Ĥ = 1/3` over `[m,a,l]` and done grades `δ^{1/3} < 1`: known through a promise. Once `a:[a,l]` is ratified, a re-derivation through it costs nothing and γ reaches 1: consolidated.
