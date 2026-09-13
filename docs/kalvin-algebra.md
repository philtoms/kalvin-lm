# Kalvin — A Term Algebra and Rewrite System

Status: draft. Fourth pass at the formalisation of `kalvin-symbolic.md` §§1–4 (that first pass is deleted), with its KScript surface syntax (§5 there) absorbed as §13 — the normative document for both. CONTEXT.md remains normative for role names.

Four tracts: **what exists** — the algebra (§1–5); **what may happen** — the rewrite system (§6–9); **what chooses** — strategy (§10); **what is observed** — measurement (§11–12). The first two are the formal system proper; the last two are dynamics over it. §12 fixes terminology; §14 lists what stays outside; §13 fixes the KScript surface syntax.

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

The cases are disjoint by construction. Coverage is the primary split: a pair with no covered node forces `g = s ≠ ∅` and `e = signature_of(ν) ≠ ∅`, so cases 4–5 can never satisfy 6; exactness is caught at case 3, before the covered cases, which require a nonzero gap or excess. Case 1's disjunction is the no-claim case in both directions: `ν = []` — nothing held; `s = ∅` — nothing left. Replace never empties a node sequence (§6) — the empty head is an entry condition (the ask, §10), not a run outcome.

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

**Definition 12 (derivation).** A **derivation** rewrites the node sequence of a queued kline `A = s:ν` against one held goal `B = t:ν_B`. The relation is memory-relative: `A ⊢_{M,B} A′`, with the memory `M` (Def 7), the goal `B`, and the queue `A` as parameters. The signature `s` never changes — the claim is fixed; the content is rewritten. States `A₀ ⊢_{M,B} A₁ ⊢_{M,B} …` differ only in `ν`. Nothing in §§6–9 reads `s` — licenses, endings, bounds and grades read only the relationship, whose head `σ(ν_A)` is exact against `ν_A` at every state by construction. Within a hop, then, the derivation is the two sides of `C` — `σ(ν_A):ν_A` against `t:ν_B` — and the queued head rides along inert, mattering only beyond the hop: absorb, reentry (§10), the claim's grounding (§14).

Membership, difference, and occurrence on node sequences are **multiset-wise**; no rule reads sequence order — arrangement is witness structure, retained for witness purposes alone. No rule reads the _queued_ kline's own fit — scoping reads the relationship `C(A,B)` (Def 11); mode and direction read the _evidence_ kline's own fit (Def 13).

**Definition 13 (one step).** `A = s:ν_A ⊢_{M,B} A′ = s:ν′` — one rule, instantiated by memory:

```text
replace:   a held correspondence kline K = n:ν_K ∈ M and an occurrence in ν_A
           matching one of K's two sides:
  forward:   an occurrence of n  → replaced by ν_K
  reverse:   an occurrence of ν_K, multiset-wise — an unordered
             configuration of ν_A's nodes → replaced by [n]
```

The terminals are inert as evidence: an Unknown (`n:[]`) has no second side; an Identity (`n:[n]`) replaces a node by itself. Every other held kline is a correspondence. K's own fit fixes the **mode**: a canon exacts granularity — forward is expand, reverse is contract, both σ(ν_A)-preserving; a covered misfit moves content by its gap and excess — forward sheds K's gap and adopts K's excess, reverse the mirror; an uncovered misfit is a traverse — disjoint atoms swap, either direction. **Direction is not a property of the kline**: a correspondence is read forward from its signature, reverse from its witness; only the side the derivation stands on occurs, so arrival orients the licence — the same kline read from the other side is the mirror derivation's licence. An edge is **ratified** or unratified — a tier relation over `M` (§10, §14). No rule of this section reads it; measurement does (§11).

**Canonicalisation.** Because occurrence is multiset-wise, a derivation may **canonicalise** its nodes: survey their unordered configurations against held witnesses and contract the correctly witnessed ones. A configuration is **correctly witnessed** when its nodes witness the compound and the compound counter-witnesses them — each node covered by the candidate head (Def 8: recognition from below), and a held canon `n:ν_K` whose witness is exactly that configuration (the composition claim from above). Exactness implies the coverage — a canon's nodes are atom-subsets of its head — so the licence is formally just `ν_K`'s multiset occurrence, and the survey is memory-bounded: held witnesses propose the configurations; nothing unwitnessed contracts. This is not a second rule but the reverse replace engaged position-free: both sides of a correspondence are selectable by occurrence, forward on a node wherever it sits, reverse on a configuration wherever its nodes sit — a phrase's discontinuity in the sequence is invisible to the licence.

Canon evidence must be well-founded (`n ∉ ν_K`). The clause is exactly strong enough: a canon's nodes are atom-subsets of its head (`σ(ν_K) = n`), so an expansion cycle forces atom-equality at every step — a canon containing its own signature. Short of that, licensed expansion terminates and depth is well-defined (§11). Identities and self-containing canons are the two inert witness classes.

A replace may be _exhibited_ as an interleaving of removals and insertions — a presentational device with no algebraic status: the licence is the correspondence, never the band alone. The interleaving never empties `ν_A` — both sides of a correspondence carry content. Node order and insertion position are retained for witness purposes alone — strategy degrees of freedom over the arrangement, which no rule reads (Def 12).

**Definition 14 (licensing).** The relationship scopes; the evidence licenses. The band says where work remains; only a correspondence kline (Def 13) says what may move:

| `fit(C(A,B))`            | Licensed targeting                             |
| ------------------------ | ---------------------------------------------- |
| S1 — Canon, Identity     | none — done                                    |
| S2 — covered misfits     | replace, on the misfit region (scoping clause) |
| S3 — Connotation, No-fit | replace — every node sits wholly in the misfit |
| Unknown — S4             | none — stuck                                   |

**Scoping clause.** A replace is targeting-licensed iff it strictly decreases the misfit mass `|σ(ν_A) Δ σ(ν_B)|` — the node replaced carries a gap atom, or the witness carries excess atoms, or both. Band-blind alignment is thereby unlicensable: a replace that touches only shared content, or grows the misfit, is not a targeting move however well evidenced. Canon-mode replaces (expand, contract) are witnessed moves — σ(ν*A)-preserving, band-preserving, licensed by `M` alone, needing nothing from this table. At S3 no node is covered, so replaces there move whole content and the route to overlap runs through the progressive path (§10). The S4 row is entered through A, not B: with no goal there is nothing to scope against — the ask at entry; with a goal and no licensed replace, the misfit \_asks* — nothing in `M` connects it (§8).

## 7. One rule, two licences

Replace is the only rule; what differs is what licenses it. Canon-mode replaces (expand, contract) are **witnessed**: they preserve `σ(ν_A)` exactly — granularity changes at constant content, band unchanged — licensed by `M` alone, blind to any goal. Denotation- and connotation-mode replaces are **evidenced targeting**: they move `σ(ν_A)` toward `σ(ν_B)`, licensed by a held correspondence and scoped by the misfit region (Def 14). The two licences are orthogonal in invariant and source — two licences on one rule, not two rules.

Done may arrive early — value-equality can outrun node-equality — and is a legitimate ending; pending nodes are witness structure.

Every content move is a claim read off a held kline: shedding one atom of a compound node is a denotation's claim — its gap is exactly what drops; adopting goal content is a connotation traversed or a canon expanded at an arrived node. Targeting is value-complete only modulo the correspondences memory supplies — without a correspondence there is no move, and the misfit asks (§8).

Read model-theoretically: canon evidence generates a congruence on sequences — expand and contract its two directions — and the full evidence set generates a **correspondence graph**: held klines as edges between a signature and its witness, traversable in either direction from wherever the derivation has arrived. A derivation is a path in that graph, relativised to what is held. The path is the semantics: done by blind alignment is unreachable by construction.

Terminals are **targeting-closed, not rule-closed**: an Identity relationship is done, yet the identity kline's own node may still expand under a held well-founded witness (`s:[s]` → `s:[a,b]`) — the claim made explicit, the identity turned canon. There is no retain move; targeting-closure is the S1 row of the licensing table, not a rule.

## 8. Endings, progress, termination

**Definition 15 (endings).** A derivation ends at **done** or **stuck**, or is **abandoned** by strategy:

- **Done** — `fit(C(A,B)) ∈ S1`: the relationship holds. The goal is **value-equality**, `σ(ν_A) = σ(ν_B)`, not node-equality — an Identity relationship is done with `ν_A ≠ ν_B`, B holding A's content as one node.
- **Stuck** — not done, and no licensed targeting move. Witnessed moves never end a derivation: they preserve the band and cannot reach done; their only use is granularity exposure, and spending them is strategy (the witnessed-run bound, T2). Stuck has two reachable conditions, both the ask: **no goal** — nothing to scope against, the ask at entry (candidate selection is §10); or **no connection** — a goal is held, the misfit region is non-empty, and no held correspondence licenses a replace into it, nor does any slot walk arrive (Def 17): nothing in `M` connects A's misfit to the goal's content, directly or through the correspondence graph. Relative non-existence (§9), reachable at entry and mid-run alike — the honest outcome when the semantic bridge is missing. Replace cannot strand a derivation: both sides of a correspondence carry content, so `ν_A` never empties mid-run.
- **Abandoned** — not an ending the rules produce: strategy halts or re-targets a run mid-derivation (§10), e.g. when graded effort falls (§11).

Licences are permissive in one sense: a correspondence may itself be an ungrounded claim — S3 evidence is a promise, not a fact — and the derivation follows it faithfully. Weighing promises is protocol (ratification, §10, §14), not the rule system's.

**Termination.** Two statements:

- **(T1)** Any run of targeting replaces from `A₀` terminates in at most `Δ₀ = |σ(ν_{A₀}) Δ σ(ν_B)|` steps. Each licensed replace strictly decreases the misfit mass (the scoping clause), and a replace may move several atoms at once: steps are evidence-sized, the bound atom-wise, both computable — the natural unit for step budgets. Regressive and circular targeting is not merely unlikely but unlicensable: a replace that does not shrink the misfit mass is not a targeting move.
- **(T2)** Witnessed replaces preserve `σ(ν_A)` and can cycle — `bc` expands to `[b,c]` and contracts back against the same held canon, at constant band, forever — and slot-wise traversals (§10) can wander the correspondence graph at constant misfit mass. Termination of mixed derivations is therefore a **strategy property**: bound witnessed runs (for instance, no expand-after-contract of the same witness) and bound traversals by no-revisit — each held signature consumed at most once per slot run; `M` is finite. Monotonicity of the graded measure is a strategy invariant, not a theorem about arbitrary derivations.

**Confluence — renounced, deliberately.** The order of derivation changes what is grounded first, and the reachable S1 depends on the path. Path-dependence is not a defect to be repaired; it is the learning phenomenon.

**Decidability — in principle.** `V` is finite and licensed expansion terminates (§6), so the states reachable from `A₀` are finite in number, and existence and non-existence of a derivation are decidable in principle; the tractability gap between that and any affordable search is exactly where cogitation, study and scaffolding live.

## 9. What a derivation proves

Done proves `σ(ν_A) = σ(ν_B)`: the queued claim's content is (value-)equal to held content, with the final node sequence as the witness — a constructive existence proof **within and through what is held**: every step of the witness was licensed by a correspondence, so the path itself is carried as evidence. The solver reading is §§6–9 restated: each held kline is a constraint, each correspondence an edge, each licensed replace a resolution step along one, S1 a constructive existence proof, and a stuck state relative non-existence — nothing in `M` connects. Done does **not** prove A's own head-claim, nor the truth of the correspondences followed — S3 evidence is a promise; weighing promises is protocol and strategy (§10, §14).

**Feedback.** `fit(C(Aᵢ, B))` is graded at each state and its rate of change tracked over steps — telling Kalvin whether its effort is increasingly or decreasingly significant. These are strategy-level metrics: they steer the derivation; they are not part of the rule set.

**Worked micro-example.** (what did Mary have)WDMH => MHALL
Atoms `w, d, m, h, a, l` (word bits; role values such as `o` live outside them, disjoint). Held: canon `mhall:[m,h,a,l,l]` — the rhyme; canon `dh:[d,h]`; canon `all:[a,l,l]` — the object phrase; denotation `dh:[h]` — "did have" → "had", its gap `{d}` naming exactly what drops; connotations `w:[o]` and `all:[o]` — the question word and the object phrase, each claiming the object role; identity `m:[m]`. Queue `A₀ = wdmh:[w,d,m,h]` — the question as it enters: four bare word-bit nodes, the verb phrase not yet composed; itself a canon. Declared goal `B = mhall` (the KScript `WDMH => MHALL`, §13). Relationship `C = wdmh:[m,h,a,l,l]` — under+over: gap `{w,d}`, excess `{a,l}`, misfit mass `Δ₀ = 4`.

One canonicalisation and two evidenced replaces finish it. **Verb, first the canonicalisation:** the question enters holding `did` and `have` discontinuously — nodes `d` and `h` with `m` between — and the denotation `dh:[h]` licenses a replace only on a `dh` node, which does not yet exist; so the derivation canonicalises — surveying the unordered configurations of `[w,d,m,h]` against held witnesses, `{d,h}` is correctly witnessed (`d` and `h` each cover `dh`; the canon `dh:[d,h]` counter-witnesses exactly them) and the reverse replace contracts `[d,h] ⇉ [dh]`, position-free, the discontinuity invisible to the licence; every other configuration answers nothing held, so the survey proposes exactly one composition. Witnessed, `σ(ν_A)`-preserving, spending no targeting budget: granularity exposure, witnessed moves' only use (§8), here load-bearing. **Verb, then the shed:** replace `dh ⇉ [h]`, forward on the denotation `dh:[h]` — shed mode, `d` leaves the gap. **Object:** no single kline connects `w` to `a,l`; the connection is composed by a slot walk (Def 17): `w:[w] ⊢ w:[o]` (forward on `w:[o]`, traverse), `⊢ w:[all]` (reverse on `all:[o]` — arrived at `o`, the answer-side kline read backwards to find the filler), `⊢ w:[a,l,l]` (forward on the canon `all:[a,l,l]`, expand — granularity set freely at arrival); its absorbed end state **is** the composed evidence, and the main line replaces `w ⇉ [a,l,l]` forward on it — traverse mode, `{w}` out, `{a,l}` in. **Subject:** `m:[m]`, identity — inert; Mary carries over untouched.

End state `wdmh:[h,m,a,l,l]`: `σ(ν_A) = mhall`, relationship Canon — done, in two targeting replaces under a bound of four (the canonicalisation is witnessed and spends none of it), every replace licensed by a held correspondence. The witness carries the chain: that is Kalvin _knowing_ what Mary had, not copying it. Had the connotations not been held, no replace reaches the object gap: stuck — the misfit asks, and ungrounded proposals follow under strategy control. Had `mhall` itself not been held, there is no goal to check done against: the ask from the other side. The end state's own fit remains under+over (gap `{w,d}`): the claim is answered, not grounded — grounding is protocol (§14), where the traversed pair `w:[o]`, `all:[o]` countersigns into a standing one-hop licence. Priced (§11): the slot walk crosses two unratified edges and the consume one more — `a,l,l` enter at acquisition depth 3 — so `Ĥ_A = 9/5` and done grades `δ^{9/5}`, below 1: known through promises. The consuming step dips γ — `J` reaches 1 as `Ĥ` bites — the price signal; after the countersign, a re-derivation through the standing licence costs nothing and γ reaches 1: consolidated. Read from the answer side (`MHALL => WDMH`), the same correspondences license the mirror derivation: the klines are direction-free; arrival orients them.

## 10. Strategy — the cogitation loop

§§6–9 fixed the parameters of a derivation; this section chooses them, step after step. The loop is **cogitation** (CONTEXT.md): **select** a hop, **derive** to an ending, **absorb** the result into memory, **reenter** with the output as the next queue's input. Each phase is strategy — the rule system of §§6–9 constrains what any of it may do, never what it must.

**Definition 16 (selection).** Selection chooses the next hop, never the final goal. A held kline `K = t:ν_K` is **selectable** for queued `A` when `t ∈ ν_A` — K's signature occurs as a node of A: A already references what K is. That clause is the forward half of Def 13's licence: to select K is to be licensed to replace its signature by its witness. The guard compounds into a ratchet: each replace's arrival puts new nodes into `ν_A`, making their klines selectable next — the path is the guard, not the point. The derivation's goal `B` is never selected: it is declared (KScript `=>`, §13) or supplied by reentry, it scopes the misfit region (Def 14), and it is _checked_ at done. S3 connotations are selectable as evidence, and an Unknown has no second side to offer; weighing claims is protocol (§14). The two-overlap caveat stands: **content overlap** `σ(ν_A) ∧ σ(ν_B) ≠ ∅` — exactly what relationship-S2 asserts — and **signature-in-node** `t ∈ ν_A` — the selection clause, which is what lets correspondence propagate into A's nodes — imply neither the other. `A = abc:[a]` against `B = x:[c,a]` stands in an Overfit relationship (S2) while `x` neither occurs in nor overlaps A's nodes; `B = x:[y]` against `A = abc:[x]` occurs in A's nodes yet yields S3. Selection requires the second; the band routes by the first.

**Definition 17 (slot derivation).** A misfit decomposes per node: each node of `ν_A` carrying a gap atom is a **slot**, seeking the goal's excess `e = σ(ν_B) ∖ σ(ν_A)`. A slot with a licensed replace fires it (Def 13, Def 14). A slot with none — the misfit _asks_ at that node — may be **walked**: queue `n:[n]` and derive `⊢_M` with no goal. The walk's licence is occurrence: any held correspondence, either side occurring in the walk's nodes, replaces it (Def 16's clause is the forward half). The scoping clause does not apply — the walk travels the correspondence graph, whose role nodes are orthogonal to both contents. A goal-less derivation has no done; its endings are **arrival** — `σ(ν) ∧ e ≠ ∅`, canon-mode replaces then setting granularity freely — or **stuck** — no unvisited correspondence occurs: nothing in `M` bridges the node to the excess, the ask localised to one slot. Each hop writes its output to STM, and the walk's end state is absorbed as the **composed correspondence**: signature `n`, witness the arrival, acquisition depth the edges crossed (§11). The main line replaces `n ⇉ ν_arrived` on it — evidence by Def 13, targeting-licensed because the arrival's content is excess and the misfit mass strictly decreases. The walk is how a missing licence is built: no single kline connects `w` to `a,l`, two connotations through `o` do, and the absorbed `w:[a,l,l]` is that connection, earned.

**Progressive path.** Def 17's walk is the evidence-builder: each hop's output is written to STM — inserted nodes as single-node misfits, connotation witnesses — and the absorbed end state is the composed correspondence the main line consumes (the worked example, §9). Evidence is literally the work undertaken by the progressive path; hops matter because `M` grows between them. When the relationship reaches S2, ordinary evidenced targeting takes over.

**Bounds.** Three numbers, all strategy parameters, each with a natural unit: the **targeting budget** — T1 bounds any targeting run by `Δ₀`, so a budget at or above it never binds mid-run; the **witnessed-run and traversal bounds** — T2's requirements, no expand-after-contract of the same witness and no-revisit of consumed signatures; and the **hop ceiling** — the reentry depth, below.

**Reentry.** Derivations compose. Hop `k` runs under parameters `(M_k, B_k)` — the goal `B_k`, or none for a slot walk (Def 17); its end state — done, arrival, or stuck — queues as hop `k+1`'s input, and memory may grow between hops (`M_{k+1} ⊇ M_k`, by STM writes — the growth is the evidence accumulating), so successive hops are not derivations of one fixed system. Propose from a proposal, one hop further out, bounded by the hop ceiling. Hop order is the only time the system has; if a time axis is wanted, it is this order and nothing else. Re-targeting mid-derivation — abandoning a run whose graded effort is falling (§11) and selecting anew — is likewise a strategy move, not a rule.

**Outside the system.** Escalation and ratification are protocol: countersigning (`==`) holds reciprocal connotation pairs as ratified — the algebra provides the shape, the protocol the commitment. The queue itself — which klines are admitted for cogitation, and in what order — belongs to the harness, not the system.

## 11. Measurement

**The band order.** The bands are derived from shape (Def 10); this section adds one axiom: they are **ordered by significance**, `S1 > S2 > S3 > S4`, the shapes within a band unordered. The predicate is observer-independent — given the same held memory, every agent classifies alike — so a band never needs to be exchanged.

Two band attachments are in play: a kline's **own band** — `fit(s, ν)` on itself, the claim it makes standing alone — and a **relationship band** — `fit(C(A,B))`, what the pair achieves. The first is what a kline asserts; the second is what a derivation establishes or fails to.

**Graded distance.** `γ(A, B)` is fixed, not free — four requirements force one form:

> `γ(A, B) = J(σ(ν_A), σ(ν_B)) · δ^(D̄ + Ĥ)` where `J(x, y) = |x ∧ y| / |x ∨ y|`

`J` is the **depth-free core**: symmetric, 0 exactly at content-disjointness, 1 exactly at value-equality. It is forced: per-slot accountedness `α(n) = |n ∧ σ(ν_B)| / |n|`, composed atom-weighted (each slot weighed by `|n|`), yields the A-side coverage fraction `|σ(ν_A) ∧ σ(ν_B)| / |σ(ν_A)|` — which fails band-consistency's second clause (A's content may sit wholly inside B's — underfit, still S2 — at full coverage), so B's excess must be weighed too, and Jaccard is the result.

`D̄` is the **mean resolution depth** of A's content — granularity: the atom-weighted mean of the resolution depths at which A's atoms are held, 0 for content held as itself, well-defined because licensed expansion terminates (§6). `Ĥ` is the **mean acquisition depth** — provenance: the atom-weighted mean of the recorded acquisition depths of A's atoms, each counting the unratified correspondence edges crossed to bring it in. Acquisition depth is **carried, not computed**: content present at entry is 0; content entering by a replace on evidence `K` records `Ĥ(K) + 1` through an unratified edge, `Ĥ(K)` through a ratified one — grounded correspondences cost nothing; shed enters nothing; canon-mode replaces move granularity, not content. Absorbed klines carry their recorded depths, and consuming one composes with its own. No `(ν_A, M)`-computable distance could price this: the grounded goal holds the arrived atoms, so a structural route to them is always short, and consumption leaves no trace in `ν_A`. As with the chosen witness (§1), the acquisition chain is a fact the algebra forgets and memory carries.

`δ ∈ (0,1)` is the strategy's knob, the only one — the two depths share it deliberately, both denominated in edges. `J` says _how close_; `δ^(D̄ + Ĥ)` says _how hard-won_ — granularity and promises priced alike. γ is directional by design, grading this derivation's effort toward its goal; B's depths are B's own derivation's problem. γ is **path-dependent by design**: won and given knowledge with the same `(s, ν)` grade differently — the record is part of what memory holds.

- **Band-consistency.** `γ` is 0 exactly at content-disjointness and maximal only at value-equality. Both ends are `J`'s; the depths only scale down.
- **Granularity-invariance.** Witnessed moves move `γ` only through `D̄`, never through recomposition: atom-weighted composition is blind to how A's content is sliced into slots. An unweighted per-slot mean violates this — expansion alone can raise it at constant content and constant depth.
- **Granularity-monotonicity.** Expand strictly increases `D̄`, so strictly decreases `γ`; contract strictly decreases `D̄`, so increases `γ`. This is what makes gratuitous expansion detectable.
- **Provenance-monotonicity.** Unratified acquisition strictly increases `Ĥ`; nothing in a derivation lowers it — only ratification (protocol, §14) or re-derivation through ratified licences does. This is what makes promise-stacking detectable, the provenance counterpart of gratuitous expansion. Consuming unratified evidence can lower `γ` even as `J` rises — the step that wins the answer dips — the price signal steering strategy toward ratified standing licences.

Both monotonicities are strategy invariants, not theorems about arbitrary derivations (T2): a mixed derivation may wander against the gradient, and the strategy declines to.

**Rate of change** per step is defined only at this level — a four-band predicate has no useful derivative — and is the signal cogitation's feedback acts on.

**Exchange.** The graded value travels in a KValue (CONTEXT.md) as the sender's assessment, and the acquisition record travels with the kline that earned it — depths are part of what memory holds. Bands need not travel, for they are recomputable from structure.

## 12. Terminology

_Significance_ is the value; _rationalisation_ is the process that produces and consumes it. Understanding, informally, is high significance attained and held.

## 13. KScript surface syntax

The written tokens and the structure each produces. Surface syntax is presentation; it does not appear in the formal tables of §§1–12.

| Token            | Structure produced          | Band claim once solved |
| ---------------- | --------------------------- | ---------------------- |
| `a => b c d`     | canon candidate             | S1 (canon) / open S2   |
| `a == b`         | reciprocal connotation pair | S1 on ratification     |
| `a > b`, `a < b` | connotation (`a:[b]`)       | S3                     |
| `a = b`          | denotation (`ab:[b]`)       | S2                     |
| `a` (bare)       | unknown (`a:[]`)            | S4 — the ask           |
| `a = a`, `a > a` | identity (`a:[a]`)          | S1                     |
| ask-annotated    | any signature, ask-marked   | S4                     |

A token declares an _intent_; the fit classification of the produced kline may or may not satisfy the declared intent (`=>` declares composition and supplies the goal for the done-check — it is not a licence; licences are correspondence klines, Def 13; the result is a Canon only if Def 10 case 3 fires; a bare signature is the ask — stuck at S4). The ask is structural (§4): the Unknown shape is the ask's shape, and no atom, mark, or decree is involved — an ask annotation is surface presentation of it, not an algebraic object.

## 14. What this document does not cover

- **Tier mechanics** — what writes STM, what promotes LTM, how Frame attention shifts: relations over `M` defined in CONTEXT.md and consumed by selection (Def 16).
- **The multi-agent loop** — trainer, trainee, supervisor; escalation when cogitation yields no reply. Protocol above the system.
- **KScript compilation mechanics** — MTS expansion, word binding, annotations: CONTEXT.md's. The tokens themselves, and the intent principle, are §13.
- **Countersigning** — a protocol commitment (reciprocal connotation pairs held as ratified); the algebra provides the shape, the protocol the commitment. Ratifying a traversed pair promotes it to a standing one-hop licence — memory compounds its evidence.
