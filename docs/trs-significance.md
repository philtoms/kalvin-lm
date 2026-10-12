# Kalvin — A Significance-Routed Term Rewriting System

**Status:** Draft — normative. This document is the single normative definition of the significance-routed model: an alternative to the goal-directed algebra of `docs/kalvin-algebra.md`. That document remains normative for the engine; this one is normative for this model.

Both models share one centre: the significance valuation of the pairing of one side's evaluated content against the other side's witness (§6) — graded there (`J`), banded here. They differ in what drives rewriting. The engine derives toward a goal kline under a strict-decrease licence; this model has no goal — the band alone routes the rationalisation.

---

## 0. Notation

### 0.1 Operators

| Symbol         | Meaning                            | Reference realisation |
| -------------- | ---------------------------------- | --------------------- |
| `v ∨ w`        | composition                        | `v \| w`              |
| `v ∧ w`        | overlap                            | `v & w`               |
| `v ∖ w`        | residue — v beyond w               | `v & ~w`              |
| `⊥`            | the empty node, composition's unit | `0`                   |
| `x ⪯ y`        | containment: `x ∧ y = x`           | `x & y == x`          |
| `can(N)`       | canonical signature of adjacency N | fold `∨`              |
| `witness_D(s)` | the held witness of signature s    | domain lookup         |
| `V(C)`         | the band of candidate C            | §4                    |

### 0.2 Terminology mapping

Terms are shared with `kalvin-algebra.md`; the mapping below gives both documents' readings side by side.

| Term (this document)      | This document                                           | `kalvin-algebra.md`                         |
| ------------------------- | ------------------------------------------------------- | ------------------------------------------- |
| underfit `U = σ ∖ can(N)` | claimed by the signature, not supplied by the adjacency | `u = s ∖ σ(ν)` — the same                   |
| overfit `O = can(N) ∖ σ`  | supplied by the adjacency, not claimed by the signature | `o = σ(ν) ∖ s` — the same                   |
| kline `⟨σ, N⟩`            | signature + adjacency set                               | `s:ν` — head + node **sequence**            |
| canonical signature       | `can(N)`, the S1 head                                   | exact witness / Canon band                  |
| candidate                 | `⟨c_E, N_i⟩` (§3)                                       | relationship kline `C(A,B) = σ(ν_A):ν_B`    |
| entry kline               | new information; bands belong to its candidates         | queued kline A                              |
| neighbour                 | held kline found by entry lookup                        | goal-side material                          |
| S1 / S2 / S3 / S4         | §4 bands                                                | same bands, §4 (S3 boundary equivalent, §6) |

---

## 1. Sorts and the Domain

### Definition 1 — Node algebra

`Node` is equipped with `(⊥, ∨, ∧, ∖)`:

- `∨` — commutative, associative, idempotent; unit `⊥`;
- `∧` — commutative; `v ∧ ⊥ = ⊥`;
- `∖` — `(v ∖ w) ∨ (v ∧ w) = v` and `(v ∖ w) ∧ w = ⊥`;
- `μ : Node → ℕ` — a measure: `μ(⊥) = 0`, and `v ∧ w = ⊥ ⟹ μ(v ∨ w) = μ(v) + μ(w)`;
- decidable equality.

Write `x ⪯ y` for `x ∧ y = x`. These are the laws of `kalvin-algebra.md` Definition 1, measure included — `μ` grades significance (Definition 13). The reference realisation is the same word-bit space, `μ` = popcount. A node is opaque; the algebra never looks inside one — `μ` weighs, it does not decompose.

### Definition 2 — Kline

A kline is a pair

```text
K = ⟨σ, N⟩
```

where `σ ∈ Node ∖ {⊥}` is its signature and `N ⊆_fin Node` is its adjacency, possibly empty.

`N` is a **set**: arrangement carries no information. Decomposition structure lives in held witnesses, never in arrangement.

### Definition 3 — Domain

The domain `D ⊆_fin Kline` is a finite set of klines. Write `sig(K)` and `adj(K)` for the accessors.

No referential invariant is claimed over `D`: references need not be held for the system to behave consistently. An unheld node has no witness (Definition 6), so a pairing through it values S4 and the branch ends — the no-neighbours route carries the consistency (Definition 9).

---

## 2. Canonicalisation and Witnessing

### Definition 4 — Canonical signature

```text
can : P_fin(Node) → Node
can(N) = ⋁_{n ∈ N} n        can(∅) = ⊥
```

`can` is total. It corresponds to the engine's evaluation `σ(ν)` restricted to sets.

### Definition 5 — Canonical kline

`K` is **canonical** when `sig(K) = can(adj(K))`.

### Definition 6 — Witness

```text
witness_D : Node → P_fin(Node)            total
witness_D(s) = the first-stored N with ⟨s, N⟩ ∈ D and can(N) = s    when D holds one
witness_D(s) = ∅                                                     otherwise
```

A signature may hold several canonical witnesses — `C:[AB]` alongside `C:[A,B]` — witness structure the system keeps rather than collapses. The lookup returns the first-stored; selection among held witnesses at use is a deferred choice point. Fundamental law:

```text
can(witness_D(s)) = s        for every s held canonically
```

Witness is the converse of canonicalisation under strict conditions — the held witness of a signature, as the engine's held witness (`kalvin-algebra.md` Definition 6): the known adjacency is the meaning of a held signature. A signature not held has no witness — no meaning yet — and a kline carrying no adjacency values S4 (Definition 11, case 1).

### Definition 7 — Three lookups

```text
exact(s, D)    = { K ∈ D : sig(K) = s }          serves witness
overlap(s, D)  = { K ∈ D : sig(K) ∧ s ≠ ⊥ }      serves the entry (Definition 9)
covered(c, D)  = { K ∈ D : sig(K) ⪯ c }          serves R3
```

They are distinct operations and remain distinct.

---

## 3. Entry and Candidates

### Definition 8 — Entry kline

New information enters as

```text
E = ⟨σ_E, N_E⟩.
```

Entries arrive from outside presentation and from the re-entering products of R2 and R3 (§5). The pipeline makes no distinction between them, but re-entry is distinguishable from entry: a presented entry is one that arrives, and a re-entry product is an entry that the rationalisation itself produced.

An entry's content reaches the domain through the R1 stores earned by its candidates (R1). Presentation admits any kline — the system never seeds and checks nothing at the boundary. `can(N_E)` may still be a signature the domain does not hold. The entry locates neighbours; the candidates constructed from it are what get valued (Definition 9, Definition 11).

### Definition 9 — Entry pipeline

```text
neighbours = overlap(σ_E, D) = { K_1 … K_m }
c_E = can(N_E)

for each neighbour K_i = ⟨σ_i, N_i⟩ :
    candidate  C_i = ⟨c_E, N_i⟩
```

Empty lookup: the entry terminates without candidates.

### Definition 10 — Candidate pairing

Every candidate carries a signature rationalised from one context — the entry's adjacency, canonicalised — and an adjacency inherited from another — a held kline. The band therefore always compares two independently rationalised representations of information.

---

## 4. The Band

Significance has two species, distinguished by prefix. **Graded significance** is the measure of Definition 13 — the entropy of a pairing. **Banded significance** is the classifier below: it assigns each candidate exactly one band, S1–S4. The grade quantifies the band, as `J` quantizes the engine's bands (`kalvin-algebra.md` §10); the band routes the rewrite system.

The band is the routing signal: each band selects one of the four routes, R1–R4, and the tree grows by band alone.

### Definition 11 — Band classifier

Cases are evaluated in order; the first match decides.

| Order | Condition                       | Band |
| ----: | ------------------------------- | ---- |
|     1 | `N = ∅`                         | S4   |
|     2 | `σ = can(N)`                    | S1   |
|     3 | `σ ∧ can(N) ≠ ⊥  ∧  σ ≠ can(N)` | S2   |
|     4 | `σ ∧ can(N) = ⊥`                | S3   |

Order matters: S1 implies non-empty overlap. Informally:

- **S1** — complete correspondence; the candidate is canonical. A normal form, not merely a high score.
- **S2** — partial correspondence; information flow is possible through the shared part.
- **S3** — no correspondence; no flow path through this pairing.
- **S4** — no adjacency; terminal.

### Definition 12 — Decomposition and flow

For a candidate `C = ⟨σ, N⟩`, let `â = can(N)` and

```text
F = σ ∧ â          fit / common
U = σ ∖ â          underfit — claimed by the signature, not supplied by the adjacency
O = â ∖ σ          overfit  — supplied by the adjacency, not claimed by the signature
```

Then

```text
σ = F ∨ U      â = F ∨ O      F ∧ O = F ∧ U = O ∧ U = ⊥
```

`S2 ⟺ F ≠ ⊥ ∧ (U ≠ ∥ ∨ O ≠ ⊥)`. Flow directions: `U ≠ ⊥` is underfit — the unresolved content sits in the signature, and information must flow signature → adjacency; `O ≠ ⊥` is overfit — the unresolved content sits in the adjacency, and information must flow adjacency → signature; both is bidirectional. The directions are orientations of flow, not size comparisons — neither side need contain the other.

### Definition 13 — Entropy and graded significance

For a candidate `C = ⟨σ, N⟩` with `â = can(N)` and decomposition `F, U, O` (Definition 12), the **entropy** of the pairing is

```text
H(C) = 1 − μ(F) / μ(σ ∨ â)
```

and the **graded significance** is its complement `g(C) = μ(F) / μ(σ ∨ â)`. The more significant, the less entropy.

Entropy is surprise: the fraction of the pairing's mass that is unexplained — the cognitive effort remaining before the pairing is fully understood. The pairing's own correspondence decides it; `D` enters only through the witnesses the rules fetch.

- `S1` — `F = σ = â`: `H = 0`, `g = 1` — no surprise, in canon and identity alike.
- `S2` — `F ≠ ⊥`, `U ∨ O ≠ ⊥`: `0 < g < 1` strictly.
- `S3` — `F = ⊥`: `g = 0` — no correspondence, full surprise.
- `S4` — `â = ⊥`: `μ(F)/μ(σ) = 0` — `H = 1`: 100% surprise.

Entropy does not separate `S3` from `S4`: both carry full surprise. The empty adjacency is structural — an arrival condition (Definition 11, case 1), not a point on the correspondence axis.

Entropy flow under the rules:

- `R1` fires at the arrival: directly at `H = 0`; through a product, an S3 candidate whose substituted re-entry returns with stores stores as a licensed claim — its own `H` may stand above zero. The store merges the earned composition (Rule R2): the minted claim may carry the fit heads.
- `R2` re-pairs each head with content from its own side: `can(N_h) ⪯ h` for every product, so no product arrives at S3 — the misfit mass that drove `H(C)` does not survive the split. A node-clean split (no node of `N` or `N_E` crosses head boundaries) lands every product at S1 and discharges the surprise entirely; interior entropy survives only where a head's atoms cross node boundaries (`H = 1 − μ(can(N_h)) / μ(h)`), and a head whose side supplies no contained node degrades to R4.
- `R3` substitutes and re-enters on the candidate's own signature: the product `⟨σ, N′⟩` keeps the claim, and the substituted adjacency re-pairs — correspondence may emerge from the re-pairing (S3 → S2 or S1), or not (S3 → S4).
- `R4` is the full-surprise arrival: nothing held explains the entry.

### Definition 14 — Effort (biased entropy)

Entropy does not separate `S3` from `S4` — both carry full surprise on the correspondence axis. But `S3` is strictly less costly than `S4`: its adjacency is present and dereferenceable, while `S4` has nothing to work with. The **effort** discounts the adjacency-side residual accordingly:

```text
E(C) = ( μ(U) + β · μ(O) ) / μ(σ ∨ â)      β ∈ (0, 1), a named constant; default ½
```

The `O`-side mass is present but unconnected — resolvable by dereferencing, not acquisition — so it counts at `β`. Consequences:

- `E(S1) = 0` and `E(S4) = 1` exactly — the anchors survive the bias.
- `E(S3) < 1` pointwise whenever `â ≠ ⊥`: `β·μ(â) < μ(â)`. Repairability is now strictly less surprising.
- `0 < E(S2) < 1` in both kinds: `β > 0` keeps pure overfit off zero, `U` keeps pure underfit off zero.
- `β → 1` recovers `H`; `E ≤ H` for `β ≤ 1`. The R2 entropy theorem is stated for `H` and does not transfer to `E`.
- An `S2` may outweigh an `S3` — effort grades within bands; the band still routes.

Effort is the halt measure: when rationalisation ends, `E` of the halted kline is how far the rationalisation did not get — the ranking fuel for the proposals the halt generates.

---

## 5. The Rewrite Rules

The rewrite system operates on candidates. R2 and R3 take a kline and produce klines, each in its own way; the products that carry adjacency re-enter as entries (Definition 8), meeting the same pipeline and the same bands — R2's fit product alone is earned rather than re-entered (Rule R2). A product with no adjacency routes to R4 — the misfit asks, and later-arriving information may complete the path. Reentry is not a rewrite rule. Only S2 and S3 produce further work.

The rules transform the candidate: direct S1 arrival is a full canon (Definition 5) — the candidate's claim and content coincide. Rationalisation is in principle tail-recursive: every re-entry applies the same procedure to the product, and the recursion returns at S1 or S4.

`D` is live throughout the tree. R1 stores enter `D` at the moment of arrival, and every later application — lookup, witness, substitution — reads `D` as it then stands. A run's own writes can close a route a later candidate would have taken, or open one: a witness minted in one round can keep a later expansion from emptying. The tree is a sequence of applications, not an evaluation against a snapshot of `D`.

### Rule R1 — proof of full understanding (S1): store the returning candidate ∨ earned

```text
a candidate : S1 — directly or through its products  ⟹  D := D ∪ { C ∨ earned }
```

Reaching S1 is a proof of full understanding. Directly, the candidate's claim and content coincide — a full canon (Definition 5). Through a product, only R3's: a substituted re-entry that returns with stores — the misfit resolved by the product's own grounding, and the proof licenses the candidate's claim over the adjacency it was valued against — a licensed claim, not necessarily a canon. The kline that enters `D` is the returning candidate merged with the earned composition: `C ∨ earned` composes the signatures and unions the adjacencies, and `earned` is empty for a presented entry. Idempotent. The returning candidate's head `c_E` may be a signature the domain does not yet hold — it is minted over the held adjacency it was valued against, and this is how entry information becomes knowledge. Reinforcement and promotion are outside the algebra (§7).

### Rule R2 — partial fit (S2): project, re-enter

For `C = ⟨σ, N⟩ : S2`, compute `F, U, O` (Definition 12). The heads are special: `F` and `O` are pieces of `can(N)` — a held witness — and `U` is a piece of `σ = can(N_E)`, the entry's canon. Decomposing a witness retains witness status across its components, so each head's adjacency is reconstructed from the witness whose canon covers it — no lookup:

```text
N_F = { n ∈ N   : n ⪯ F }      the candidate's adjacency, F-side
N_O = { n ∈ N   : n ⪯ O }      the candidate's adjacency, O-side
N_U = { n ∈ N_E : n ⪯ U }      the entry's adjacency, U-side
```

R2 produces three klines — for the non-empty heads:

```text
⟨F, N_F⟩      ⟨U, N_U⟩      ⟨O, N_O⟩
```

The pairing is same-side — each head carries the content reconstructed from its own provenance: `F` and `O` from the candidate's adjacency, `U` from the entry's. The misfit does not question the other side; each residual returns to the side that claims or supplies it.

The fit product does not re-enter: it is the understood portion, and it is earned — merged into the composition that rides with the split (accumulation, below). The residual products re-enter as entries (Definition 8) in order, each carrying the earned composition; they ground for their own sake — a product's stores are its own trace's (R1), and no product return stores the candidate that split. The halts live in the products' own traces. A product with no adjacency routes to R4.

The products' bands are fixed by the reconstruction, and the pairing is same-side: `can(N_h) ⪯ h` for every head, so no product arrives at S3. A node-clean split — no node of `N` or `N_E` crosses head boundaries — gives `can(N_h) = h` for every head: each product is itself a canon (Definition 5), and the split discharges the pairing's surprise entirely. A head whose atoms cross node boundaries carries interior entropy (Definition 13); a head whose side supplies no contained node degrades to R4.

### Rule R3 — misfit (S3): substitute, re-enter

For `C = ⟨σ, N⟩ : S3`, substitute the covered references in `N` by their witnesses (Definition 7):

```text
c_N = can(N)
for each K ∈ covered(c_N, D) whose signature occurs in N :
    replace every occurrence of sig(K) in N by adj(K)
N′ = the result
```

R3 produces the kline

```text
⟨σ, N′⟩
```

The claim `σ` is kept; the adjacency is dereferenced through the covered witnesses. Like for like is not a substitution — a replacement whose witness is the occurrence itself is not performed — and a pass that substitutes nothing ends the branch at R4. The product re-enters as an entry (Definition 8).

### Rule R4 — terminal (S4): end the branch

The arrival is no adjacency — a presented entry, a re-entering product, a candidate whose neighbour supplied none (Definition 11, case 1), or an expansion that emptied its adjacency. The branch ends undone — the knowledge boundary: a system knows what it knows (S1–S3) and knows what it does not (S4).

### The rationalisation tree

```text
entry ⟨σ_E, N_E⟩
  │                │
  │ overlap(σ_E)   │ can(N_E) = c_E
  ▼                ▼
neighbours K_i   candidates C_i = ⟨c_E, N_i⟩
  └────────► V(C_i) ◄───────────┘
                │
      ┌─────────┼──────────┬───────────────────┐
      S1        S2         S3                  S4 (no adjacency) → end
      │         │          │
    store C   project     substitute
   ∨ earned      │          │
      │      ⟨F,N_F⟩ ──► earned rides       N′ ──► ⟨σ, N′⟩
      │      ⟨U,N_U⟩ ──┐  with the split        │
      │      ⟨O,N_O⟩ ──┴── re-entries ──► repeat │
      │                                           │
      └── an S3 re-entry returning with stores stores C ∨ earned ──┘
                ▼
           next candidate
```

Across R1–R4, every rewrite resolves against held content and signatures enter `D` at R1 stores: the system composes what it holds, and asks at its boundary.

An entry's candidates are processed in turn, every one, each formed from its neighbour's adjacency alone. R2 accumulates down the recursion: the fit kline merges — signature and adjacency — into an earned composition that rides with the split, and every re-entry the split produces carries it; R3's re-entry carries none — it keeps the candidate's claim. The composition is spent at the store: a candidate that arrives at S1 stores merged with the earned composition (R1), and nested splits compose — each split's fit kline merges into the composition it passes down. The candidate set exhausted, the rationalisation returns with its stores.

### Definition 15 — Proposals

A halt names what is missing. A **proposal** is the halted kline read as a request — kline-shaped, carrying its effort (Definition 14), generated at every end of the tree:

- **acquire** — a witness gap. The halted kline is a head without usable adjacency: an unheld-witness arrival `⟨U, ∅⟩`, a candidate whose neighbour supplied none, or an expansion that emptied its adjacency. The request: present an entry whose content witnesses the head.
- **connect** — an overlap gap. The entry terminated with no neighbours — nothing held overlaps `σ_E`. The request: present content overlapping `σ_E`.
- **continue** — a frontier halt. The bound stopped the run with pending entries; the request names the minimal-effort pending kline: run on.

A presented entry `E` discharges a proposal with target `t` iff `(σ_E ∨ can(N_E)) ∧ t ≠ ⊥` — the answer touches the gap; the answer's own rationalisation does the grounding. Proposals do not enter `D` and are not entries; like the engine's asks, a proposal never grounds itself. Proposals are ordered by graded significance — ascending effort (Definition 14): the least-entropy proposal is the most significant next step; ties keep generation order.

---

## 6. Relationship to the Kline Algebra

| This model                              | `kalvin-algebra.md`                                                                                   |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------- | -------- | --- | -------- | ------------------------------------------------- |
| candidate `⟨c_E, N_i⟩`                  | relationship kline `C(A,B) = σ(ν_A):ν_B` — same construction, independently reached                   |
| `can(N)` = fold `∨` over a set          | evaluation `σ(ν)` = fold `∨` over a sequence                                                          |
| R3 substitution `sig(K) ⇉ adj(K)`       | forward replacement `n ⇉ ν_K` (witnessed expand)                                                      |
| S1 band                                 | S1 band {Identity, Canon} — `⟨s, {s}⟩` is S1 here, collapsing the engine's Identity/Canon distinction |
| S3 boundary `σ ∧ can(N) = ⊥`            | equivalent to the engine's not-covered case (distributivity: `σ ∧ can(N) = ⊥ ⟺ ∀ n ∈ N : σ ∧ n = ⊥`)  |
| residuals `U = σ ∖ â`, `O = â ∖ σ`      | `u = s ∖ σ(ν)`, `o = σ(ν) ∖ s` (Definition 9) — identical                                             |
| graded significance `g = μ(F)/μ(σ ∨ â)` | significance ratio `J =                                                                               | s ∩ σ(ν) | /   | s ∪ σ(ν) | ` (§10) — the same Jaccard, independently reached |
| no goal, no targeting licence           | goal kline B, strict-decrease targeting (Def 14)                                                      |

The architectural difference: the engine is goal-directed — a derivation works toward B under a licence that strictly decreases mismatch; this model is significance-routed — the tree grows by valuation alone, and no kline plays the goal's role. The candidate construction being the same object in both models is the strongest evidence that the two are readings of one system.

---

## 7. Outside the Algebra

- selection order among neighbours, among R3 substitutions, and among re-entering products — strategy;
- arrival — the rules route every band but do not guarantee termination of the full tree: bounds on round length, re-entry depth, and repeated entry canons are strategy, as the engine's T2 bounds witnessed cycles;
- confluence — intentionally not required: branching over neighbours is the model's semantics, and different branches may ground different resolutions;
- reinforcement, tiering, memory-management policy;
- ratification of stored klines.

### The harness boundary

A well-formed script holds an ask, a goal, and all the scaffolding needed to trace from ask to goal. The point of rationalisation is the transformation — the ask's content becomes the goal's form. The model neither knows nor cares: which klines the presenting machinery offers, and whether a run reaches the scripted goal, are harness concerns. The model's own ask==goal configuration is the candidate set (Definition 9, Definition 10): rationalisation pairs the entry's content against everything held, and the scripted pairing is, at most, one of them.
