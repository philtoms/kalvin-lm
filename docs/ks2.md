# Kalvin — Symbolic II

Status: draft. Third pass at the formalisation of `kalvin-symbolic.md` §§1–4; the KScript surface syntax remains normative there (§5). CONTEXT.md remains normative for role names.

Four tracts: **what exists** — the algebra (§1–5); **what may happen** — the rewrite system (§6–9); **what chooses** — strategy (§10); **what is observed** — measurement (§11–12). The first two are the formal system proper; the last two are dynamics over it. §12 fixes terminology; §13 lists what stays outside.

---

## 1. Atoms and values

**Definition 1 (atoms).** `A = {a₀ … a₃₀}` — a finite parameter set; only its finiteness is load-bearing. In the engine this is the word-bit space: one bit per distinct word. The u64 packing `(word_bit << 32) | bpe_token_id` is not literally a set of atoms; this algebra is a deliberate tidying of that encoding, and the tidying is what the formalisation builds on.

**Definition 2 (values).** A **value** is a set of atoms: `V = 2^A`. Write `a` for the singleton value `{a}` and `abc` for `{a,b,c}`. The **empty value** is `∅`; the **full value** is `A`. Operations:

- composition `v ∨ w = v ∪ w` — the whole is the sum of its parts;
- overlap `v ∧ w = v ∩ w` — what two values share;
- complement `¬v = A ∖ v`.

The Boolean laws (commutativity, associativity, idempotence, distribution, `a ∧ b = ∅` for distinct atoms) hold by construction. They are consequences of the set definition, not axioms.

## 2. Node sequences — the terms

**Definition 3 (node sequence).** A node sequence `ν = [n₁ … nₖ]` is a member of `(V ∖ {∅})*`: order and multiplicity retained, no empty nodes. `V*` is the free monoid on `V` — the only free object in the system.

**Definition 4 (evaluation).** `signature_of : V* → V`:

> `signature_of([n₁ … nₖ]) = n₁ ∨ … ∨ nₖ` `signature_of([]) = ∅`

It is a monoid homomorphism that forgets exactly order and multiplicity — nothing else. `V` identifies precisely what `signature_of` identifies. The architecture lives in the gap: **node sequences are the terms; values are what they evaluate to.**

## 3. Klines — claims and witnesses

**Definition 5 (kline).** A **kline** `s:ν` pairs a nonzero **signature** `s ∈ V` with a node sequence `ν`. A kline claims its signature as the composition of its nodes.

**Definition 6 (exactness).** `s:ν` is **exact** when `s = signature_of(ν)`. An exact, non-empty kline is a **witness**: a chosen decomposition of `s`. `signature_of` has no distinguished inverse. The trivial one, `s ↦ [s]`, always exists — its images are the identities, witnesses that carry no decomposition content; every other witness is a real choice. Nothing in `V` reconstructs which choice was made — that is what memory is for.

> **The central claim.** Kalvin's memory is a set of claims over a forgetting map. A kline claims its signature as the composition of its nodes; **Canon** is the claim exactly kept; **Underfit/Overfit** are the two directions a claim can miss while still being answered; the S3 shapes are claims nothing answers; **Unknown** makes no composition claim — nothing held, or nothing left — and lands S4. Significance grades the claim.

**Definition 7 (memory).** A **memory** `M` is a finite set of klines. Two klines may share a signature — distinct claims, or distinct decompositions of the same value. A node may be the signature of another kline: nesting is by reference, and the reference graph may cycle — canon self-reference (`a:[a,a]`), countersign pairs (`a:[b]`, `b:[a]`) — for memory is unrestricted: any kline may be held. Cycles carry no decomposition content (§6). Which klines are _held_, and in what tier, are relations over `M` defined above the algebra (CONTEXT.md).

A witness may repeat a node (`[l,l]`); the repetition is invisible to `signature_of` but retained — it is part of the chosen decomposition.

## 4. Coverage and the fit classifier

**Definition 8 (coverage).** A node `n` is **covered** by a value `s` when `n ∧ s ≠ ∅` — they share at least one atom. Coverage is overlap, not containment: a covered node may also carry atoms outside `s`.

No new constructors appear beyond §3. The nine shapes are derived predicates — one total function on `(value, sequence)` pairs. A kline is one such pair (its own signature against its own nodes); the relationship kline of §5 is another; one classifier serves both.

**Definition 9 (gap and excess).** For `(s, ν)` with `ν ≠ []`: the **gap** `g = s ∧ ¬signature_of(ν)` — atoms the signature claims beyond its nodes; the **excess** `e = signature_of(ν) ∧ ¬s` — atoms the nodes carry beyond the signature. Note `g = ∅` and `e = ∅` together hold iff `s = signature_of(ν)`.

**Definition 10 (fit).** `fit : V × V* → Shape`. Cases in order; every pair matches exactly one:

| #   | Condition                 | Shape       | Band |
| --- | ------------------------- | ----------- | ---- |
| 1   | `ν = []` or `s = ∅`       | Unknown     | S4   |
| 2   | `ν = [s]`                 | Identity    | S1   |
| 3   | `s = signature_of(ν)`     | Canon       | S1   |
| 4   | not covered, `\|ν\| = 1`  | Connotation | S3   |
| 5   | not covered, `\|ν\| > 1`  | No-fit      | S3   |
| 6   | covered, `g ≠ ∅`, `e = ∅` | Underfit    | S2   |
| 7   | covered, `g = ∅`, `e ≠ ∅` | Overfit     | S2   |
| 8   | covered, `g ≠ ∅`, `e ≠ ∅` | Under+over  | S2   |

The cases are disjoint by construction. Coverage is the primary split: a pair with no covered node forces `g = s ≠ ∅` and `e = signature_of(ν) ≠ ∅`, so cases 4–5 can never satisfy 6; exactness is caught at case 3, before the covered cases, which require a nonzero gap or excess. Case 1's disjunction is the no-claim case in both directions: `ν = []` — nothing held; `s = ∅` — nothing left. A derivation that empties `ν_A` acquires an Unknown relationship whatever its target holds (§8).

**Species.** **Denotation** is the single-node Underfit (`ab:[b]` — one covered node, gap only); **Connotation** is case 4. These are names of convenience for KScript (`=`, `>`/`<`); algebraically they are single-node instances of cases 6 and 4. Two single-node shapes are unnamed: the single-node Overfit `a:[ab]`, and the single-node Under+over `ab:[bc]` (covered on `b`, gap `a`, excess `c`).

**Bands.** Derived, not asserted: S1 = cases 2–3 (exact), S2 = covered misfits, S3 = uncovered misfits, S4 = Unknown. Readings carry over: S1 — _I know that I know this_; S2 — _I infer this, but it does not yet fit_; S3 — _I recognise aspects of this, indirectly_; S4 — _I do not understand this at all_. The **ask** is structural, not declared: Unknown (`s:[]` — nothing held for the signature) is the ask's shape, the halt signal under which strategy generates ungrounded proposals (§10). No atom, mark, or decree is involved.

**Invariance.** `fit` is insensitive to node order: it depends on `ν` only through `signature_of(ν)` and the node count. Duplicating a node changes the fit only when it crosses a count boundary (`a:[a]` Identity vs `a:[a,a]` Canon; `a:[b]` Connotation vs `a:[b,b]` No-fit); otherwise the repetition is witness structure invisible to classification.

**Canonical table** (illustration, not definition):

| Structure   | `s:ν`         | gap  | excess | Band |
| ----------- | ------------- | ---- | ------ | ---- |
| Canon       | `abc:[a,b,c]` | `∅`  | `∅`    | S1   |
| Identity    | `a:[a]`       | `∅`  | `∅`    | S1   |
| Underfit    | `abc:[a,c]`   | `b`  | `∅`    | S2   |
| Overfit     | `ab:[a,b,c]`  | `∅`  | `c`    | S2   |
| Under+over  | `abc:[b,c,d]` | `a`  | `d`    | S2   |
| Denotation  | `ab:[b]`      | `a`  | `∅`    | S2   |
| Connotation | `a:[b]`       | `a`  | `b`    | S3   |
| No-fit      | `ab:[c,d]`    | `ab` | `cd`   | S3   |
| Unknown     | `a:[]`        | —    | —      | S4   |

Note the S3 rows: their gap _and_ excess are both nonzero, yet they are not Under+over — coverage decides first. That precedence is what keeps the partition disjoint.

## 5. The relationship kline

**Definition 11 (pairwise).** For klines `A = s:ν_A` and `B = t:ν_B`, the **relationship kline** is `C(A,B) = signature_of(ν_A) : ν_B`. Its head is not claimed — it is _defined_ as what A's nodes evaluate to — so all of C's misfit-ness comes from B's side. `fit(C(A,B))` is the **structural relationship of A and B**.

`fit` is one function with two readings. Applied to `(s, ν)` it grades a kline's own claim; applied to `(signature_of(ν_A), ν_B)` it grades two klines against each other. Neither reading is a special case of the other construction — both are arguments to the same classifier.

Useful equivalence: `C(A,B)` is Canon iff `signature_of(ν_A) = signature_of(ν_B)` — the two klines hold the same value, differently decomposed.

Canonical relationships (atoms lowercase):

| Relationship | A nodes | B nodes | C(A,B)        | Band |
| ------------ | ------- | ------- | ------------- | ---- |
| Canon        | [a,b,c] | [a,b,c] | `abc:[a,b,c]` | S1   |
| Identity     | [a]     | [a]     | `a:[a]`       | S1   |
| Underfit     | [a,b,c] | [a,c]   | `abc:[a,c]`   | S2   |
| Overfit      | [a,b]   | [a,b,c] | `ab:[a,b,c]`  | S2   |
| Under+over   | [a,b,c] | [b,c,d] | `abc:[b,c,d]` | S2   |
| Denotation   | [a,b]   | [b]     | `ab:[b]`      | S2   |
| Connotation  | [a]     | [b]     | `a:[b]`       | S3   |
| No-fit       | [a,b]   | [c,d]   | `ab:[c,d]`    | S3   |
| Unknown      | [a]     | []      | `a:[]`        | S4   |

## 6. Derivations

**Definition 12 (derivation).** A **derivation** rewrites the node sequence of a queued kline `A = s:ν` against one held target `B = t:ν_B`. The relation is memory-relative: `A ⊢_{M,B} A′`, with the memory `M` (Def 7), the target `B`, and the queue `A` as parameters. The signature `s` never changes — the claim is fixed; the content is rewritten. States `A₀ ⊢_{M,B} A₁ ⊢_{M,B} …` differ only in `ν`. Nothing in §§6–9 reads `s` — licenses, endings, bounds and grades read only the relationship, whose head `σ(ν_A)` is exact against `ν_A` at every state by construction. Within a hop, then, the derivation is the two sides of `C` — `σ(ν_A):ν_A` against `t:ν_B` — and the queued head rides along inert, mattering only beyond the hop: absorb, reentry (§10), the claim's grounding (§13).

Membership and difference on node sequences are **multiset-wise**; sequence order is used only by contract's pattern match and otherwise retained for witness purposes. No rule reads a kline's own fit — licensing reads only the relationship `C(A,B)` (Def 11).

**Definition 13 (one step).** `A = s:ν_A ⊢_{M,B} A′ = s:ν′`:

```text
expand:    an occurrence of n in ν_A, the well-founded canon n:νₙ ∈ M
               → ν′ replaces that occurrence of n by νₙ
contract:  w a contiguous block of ν_A, the well-founded canon σ(w):w ∈ M
               → ν′ replaces the block w by [σ(w)]
remove:    a ∈ ν_A ∖ ν_B, licensed by Def 14
               → ν′ = ν_A ∖ [a]
add:       b ∈ ν_B ∖ ν_A, licensed by Def 14
               → ν′ = ν_A with b inserted
replace:   remove; add — the one composite; §7 constrains its interleaving
```

Canons are exact and non-trivial by Def 10 (case 3 fires after the terminals), so expand is never a no-op. A canon `n:νₙ` with `n ∉ νₙ` is a **well-founded witness**; witnessed moves license only well-founded witnesses. The clause is exactly strong enough: a canon's nodes are atom-subsets of its head (`σ(νₙ) = n`), so an expansion cycle forces atom-equality at every step — a canon containing its own signature. Short of that, every licensed expand replaces a node by strictly atom-smaller nodes; the multiset of node atom-counts descends, and descent is well-founded — expand-only runs terminate per derivation, with no invariant on `M`. Identities (case 2, never canons) and self-containing canons (`a:[a,a]` — classifiable, holdable, selectable as a target) are the two witness classes inert for witnessed moves. The insertion position of `add` is free: invisible to `σ`, retained for witness purposes, consequential only for later contract contiguity — a strategy degree of freedom, like ordering generally (Def 12).

**Definition 14 (licensing).** The relationship's fit at the current state licenses the targeting moves:

| `fit(C(A,B))`               | Licensed targeting |
| --------------------------- | ------------------ |
| S1 — Canon, Identity        | none — done        |
| Underfit (incl. Denotation) | remove             |
| Overfit                     | add                |
| Under+over                  | remove and add     |
| S3 — Connotation, No-fit    | replace            |
| Unknown — S4                | none — stuck       |

For S3 the substitution is forced to be total: no node of `ν_A` is covered, so node-disjointness makes both difference sets everything. The S4 row is entered through A, not B: selection never yields an empty target (Def 16), so `ν_B = []` is unreachable — but permissive removes can empty `ν_A` (case 1's empty head, `σ(ν_A) = ∅`), and the ask is then acquired mid-run. Witnessed moves need no license from this table — a held well-founded witness anywhere in `M` suffices, whatever the relationship.

## 7. The two move families

The families are orthogonal in invariant and in license source; neither reduces to the other.

- **Witnessed moves** (expand, contract) apply a held witness. They **preserve `σ(ν_A)` exactly** — decomposition granularity changes at constant content, so the relationship's band is unchanged. Licensed by `M`, independent of `B`.
- **Targeting moves** (add, remove) align content toward the target. They **change `σ(ν_A)` toward `σ(ν_B)`** at whatever granularity `ν_A` currently has, and may touch only nodes from the difference sets. Licensed by `B`.

Replace is the only composite, and its interior is licensed under one constraint: it executes as an interleaving of its removes and adds in which `ν_A` never empties. Under that constraint every interior state is S3, S2, or done — never S4: removes shrink `σ(ν_A)`, so S3 persists; the first add covers its node, giving S2 or better; from S2 a remove may drop coverage and return S3, still licensed. Done may arrive early — value-equality can outrun node-equality (`ν_B = [y,y,z]` is done at interior `[y,z]`, an add still pending) — and is a legitimate ending; the pending nodes are witness structure. Per-step grading (§9) sees the interior. The composite earns its place where the primitives cannot finish the job alone: targeting is value-complete only modulo the granularities memory supplies. Shedding one atom of a compound node, or adopting one atom of a compound node of `B`, takes a witnessed move to expose.

Read model-theoretically: held well-founded witnesses generate a congruence on sequences — expand and contract are its two directions — and targeting moves operate on representatives. A derivation is rewriting relativised to what is held.

Terminals are **targeting-closed, not rule-closed**: an Identity relationship is done, yet the identity kline's own node may still expand under a held well-founded witness (`s:[s]` → `s:[a,b]`) — the claim made explicit, the identity turned canon. There is no retain move; targeting-closure is the S1 row of the licensing table, not a rule.

## 8. Endings, progress, termination

**Definition 15 (endings).** A derivation ends at **done** or **stuck**, or is **abandoned** by strategy:

- **Done** — `fit(C(A,B)) ∈ S1`: the relationship holds. The goal is **value-equality**, `σ(ν_A) = σ(ν_B)`, not node-equality — an Identity relationship is done with `ν_A ≠ ν_B`, B holding A's content as one node.
- **Stuck** — not done, and no licensed targeting move. Witnessed moves never end a derivation: they preserve the band and cannot reach done; their only use is granularity exposure, and spending them is strategy (the witnessed-run bound, T2). With a target selected and `ν_A` non-empty, a not-done state always licenses a targeting move — the S2 rows license remove or add, S3 licenses replace — so stuck has two reachable conditions, both the ask: **no target was selected** (candidate selection is §10) — the ask at entry, nothing grounded answers, §9's relative non-existence; or **`ν_A` has been emptied** by permissive removes (case 1's empty head, `σ(ν_A) = ∅`, §4) — the ask acquired mid-run. The `ν_B = []` route into the S4 row remains unreachable: selection never yields an empty target (Def 16).
- **Abandoned** — not an ending the rules produce: strategy halts or re-targets a run mid-derivation (§10), e.g. when graded effort falls (§11).

Licenses are permissive, not safe. A licensed remove can strand the derivation: `A = ab:[ab]` against `B = a:[a]` is underfit (remove licensed); removing `ab` empties `ν_A` → Unknown — stuck at the ask. Pruning such dead ends is band feedback's job (§9), not the rule system's.

**Termination.** Two statements:

- **(T1)** Any run of targeting moves from `A₀` terminates in at most `D₀ = |ν_{A₀} ∖ ν_B| + |ν_B ∖ ν_{A₀}|` steps. Each targeting move decreases `D` by exactly one: remove requires `count_{ν_A}(a) > count_{ν_B}(a)` and shrinks the left difference; add is symmetric. No significance-monotonicity is required, and the bound is computable — the natural unit for step budgets. In particular, add/remove oscillation is not merely unlikely but unlicensable.
- **(T2)** Witnessed moves preserve `σ(ν_A)` and can cycle — `bc` expands to `[b,c]` and contracts back against the same held canon, at constant band, forever. Termination of mixed derivations is therefore a **strategy property**: bound witnessed-move runs (for instance, no expand-after-contract of the same witness). Monotonicity of the graded measure is a strategy invariant, not a theorem about arbitrary derivations.

**Confluence — renounced, deliberately.** The order of derivation changes what is grounded first, and the reachable S1 depends on the path. Path-dependence is not a defect to be repaired; it is the learning phenomenon.

**Decidability — in principle.** `V` is finite and licensed expansion terminates (§6), so the states reachable from `A₀` are finite in number, and existence and non-existence of a derivation are decidable in principle; the tractability gap between that and any affordable search is exactly where cogitation, study and scaffolding live.

## 9. What a derivation proves

Done proves `σ(ν_A) = σ(ν_B)`: the queued claim's content is (value-)equal to held content, with the final node sequence as the witness — a constructive existence proof **within what is held**. The solver reading is §§6–9 restated: each held kline is a constraint, each licensed step a resolution step, S1 a constructive existence proof, and a stuck S4 state relative non-existence — nothing in `M` answers. Done does **not** prove A's own head-claim: the end state's own fit may still be a misfit; grounding the claim itself is protocol and strategy (§10, §13).

**Feedback.** `fit(C(Aᵢ, B))` is graded at each state and its rate of change tracked over steps — telling Kalvin whether its effort is increasingly or decreasingly significant. These are strategy-level metrics: they steer the derivation; they are not part of the rule set.

**Worked micro-example.** Atoms `m, h, a, l`. Held: identity `m:[m]`, canon `mall:[m,a,l,l]`. Queue `A₀ = mall:[m,a]`. Relationship `C = ma:[m,a,l,l]` — overfit (excess `l`); licensed: add; `D₀ = 2`. Add `l` → `mall:[m,a,l]` — still overfit. Add `l` → `mall:[m,a,l,l]` — canon: done, in exactly `D₀` steps, a constructive existence proof of `mall` within what is held. Had nothing been held, no target exists: stuck before any step — the ask, and ungrounded proposals follow under strategy control.

## 10. Strategy — the cogitation loop

§§6–9 fixed the parameters of a derivation; this section chooses them, step after step. The loop is **cogitation** (CONTEXT.md): **select** a target, **derive** to an ending, **absorb** the result into memory, **reenter** with the output as the next queue's input. Each phase is strategy — the rule system of §§6–9 constrains what any of it may do, never what it must.

**Definition 16 (selection).** A kline `B = t:ν_B ∈ M` is **selectable** as target for queued `A` when:

- `grounded(B)` — B is held as counted-on knowledge (a tier relation over `M`; CONTEXT.md); grounded excludes the Unknown shape — an empty kline grounds nothing — so a selectable target holds content (`ν_B ≠ []`), and
- `t ∈ ν_A` — B's signature occurs as a node of A: A already references what B is.

The relationship's band then routes the derivation: S2 → ordinary targeting (§§6–8); S3 → the progressive path, below. Two overlap conditions are easy to conflate and imply neither the other: **content overlap** `σ(ν_A) ∧ σ(ν_B) ≠ ∅` — exactly what relationship-S2 asserts — and **signature-in-node** `t ∈ ν_A` — the selection clause, which is what lets grounding propagate from B into A's nodes. `A = abc:[a]` against `B = x:[c,a]` stands in an Overfit relationship (S2) while `x` neither occurs in nor overlaps A's nodes; `B = x:[y]` against `A = abc:[x]` occurs in A's nodes yet yields S3. Selection requires the second; the band routes by the first.

**Progressive path.** An S3 relationship licenses replace — total by node-disjointness (Def 14) — and the path executes that composite incrementally: each inserted node is held in STM as a single-node misfit (a connotation witness), and the next step is licensed against the accumulated overlap. When the relationship reaches S2, ordinary targeting takes over. The stepwise-ness is the composite's licensed interior (§7), spaced out by memory writes.

**Bounds.** Three numbers, all strategy parameters, each with a natural unit: the **targeting budget** — T1 bounds any targeting run by `D₀`, so a budget at or above `D₀` never binds mid-run; the **witnessed-run bound** — T2's requirement, for instance no expand-after-contract of the same witness; and the **hop ceiling** — the reentry depth, below.

**Reentry.** Derivations compose. Hop `k` runs under parameters `(M_k, B_k)`; its end state — done or stuck — queues as hop `k+1`'s input, and memory may grow between hops (`M_{k+1} ⊇ M_k`, by STM writes), so successive hops are not derivations of one fixed system. Propose from a proposal, one hop further out, bounded by the hop ceiling. Hop order is the only time the system has; if a time axis is wanted, it is this order and nothing else. Re-targeting mid-derivation — abandoning a run whose graded effort is falling (§11) and selecting anew — is likewise a strategy move, not a rule.

**Outside the system.** Escalation and ratification are protocol: countersigning (`==`) holds reciprocal connotation pairs as ratified — the algebra provides the shape, the protocol the commitment. The queue itself — which klines are admitted for cogitation, and in what order — belongs to the harness, not the system.

## 11. Measurement

**The band order.** The bands are derived from shape (Def 10); this section adds one axiom: they are **ordered by significance**, `S1 > S2 > S3 > S4`, the shapes within a band unordered. The predicate is observer-independent — given the same held memory, every agent classifies alike — so a band never needs to be exchanged.

Two band attachments are in play: a kline's **own band** — `fit(s, ν)` on itself, the claim it makes standing alone — and a **relationship band** — `fit(C(A,B))`, what the pair achieves. The first is what a kline asserts; the second is what a derivation establishes or fails to.

**Graded distance.** `γ(A, B)` is fixed, not free — three requirements force one form:

> `γ(A, B) = J(σ(ν_A), σ(ν_B)) · δ^D̄` where `J(x, y) = |x ∧ y| / |x ∨ y|`

`J` is the **depth-free core**: symmetric, 0 exactly at content-disjointness, 1 exactly at value-equality. It is forced: per-slot accountedness `α(n) = |n ∧ σ(ν_B)| / |n|`, composed atom-weighted (each slot weighed by `|n|`), yields the A-side coverage fraction `|σ(ν_A) ∧ σ(ν_B)| / |σ(ν_A)|` — which fails band-consistency's second clause (A's content may sit wholly inside B's — underfit, still S2 — at full coverage), so B's excess must be weighed too, and Jaccard is the result. `D̄` is the **mean witness depth of A's content**: the atom-weighted mean of the resolution depths at which A's atoms are held — 0 for content held as itself — well-defined because licensed expansion terminates (§6). `δ ∈ (0,1)` is the strategy's knob, the only one. `J` says _how close_; `δ^D̄` says _how hard-won_ — γ is directional by design, grading this derivation's effort toward its target; B's depth is B's own derivation's problem.

- **Band-consistency.** `γ` is 0 exactly at content-disjointness and maximal only at value-equality. Both ends are `J`'s; depth only scales down.
- **Granularity-invariance.** Witnessed moves move `γ` only through `D̄`, never through recomposition: atom-weighted composition is blind to how A's content is sliced into slots. An unweighted per-slot mean violates this — expansion alone can raise it at constant content and constant depth.
- **Depth-monotonicity.** Expand strictly increases `D̄`, so strictly decreases `γ`; contract strictly decreases `D̄`, so increases `γ`. This is what makes gratuitous expansion detectable — and why monotonicity of the graded measure is a strategy invariant (T2), not a structural fact: targeting moves shift `J` in either direction, and a derivation may wander against the gradient; the strategy declines to.

**Rate of change** per step is defined only at this level — a four-band predicate has no useful derivative — and is the signal cogitation's feedback acts on.

**Exchange.** The graded value travels in a KValue (CONTEXT.md) as the sender's assessment; bands need not travel, for they are recomputable from structure.

## 12. Terminology

_Significance_ is the value; _rationalisation_ is the process that produces and consumes it. (Not: "significance is the value Kalvin directly equates to rationalisation".) Understanding, informally, is high significance attained and held.

## 13. What this document does not cover

- **Tier mechanics** — what writes STM, what promotes LTM, how Frame attention shifts: relations over `M` defined in CONTEXT.md and consumed by selection (Def 16).
- **The multi-agent loop** — trainer, trainee, supervisor; escalation when cogitation yields no reply. Protocol above the system.
- **KScript tokens** — surface syntax declaring intent; `fit` may or may not satisfy the declared intent (`=>` declares composition; the result is a Canon only if Def 10 case 3 fires; a bare signature is the ask — stuck at S4). See `kalvin-symbolic.md` §5.
- **Countersigning** — a protocol commitment (reciprocal connotation pairs held as ratified); the algebra provides the shape, the protocol the commitment.
