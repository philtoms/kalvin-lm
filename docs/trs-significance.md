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

Entries arrive from outside presentation and from the products of R2 (§5); the pipeline makes no distinction between them. R3's product does not arrive as an entry — it re-enters within its candidate's chain (R3).

An entry's content reaches the domain through the R1 stores earned by its candidates' traces (R1). Presentation admits any kline — the system never seeds and checks nothing at the boundary. `can(N_E)` may still be a signature the domain does not hold. The entry locates neighbours; the candidates constructed from it are what get valued (Definition 9, Definition 11).

### Definition 9 — Entry pipeline

```text
neighbours = overlap(σ_E, D) = { K_1 … K_m }
c_E = can(N_E)

for each neighbour K_i = ⟨σ_i, N_i⟩ :
    candidate  C_i = ⟨c_E, N_i⟩
```

Empty lookup: the entry terminates without candidates.

Candidates on re-entering products extend earlier chains: a **trace** is a chain `C₀, C₁, …` where each `C_k` (k > 0) is a candidate formed on an entry that is a product of `C_{k−1}` (R2); R3 substitution continues a trace without a new entry. `C₀` is formed on a presented entry and opens the trace.

### Definition 10 — Candidate invariant

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

- `R1` fires exactly at `H = 0` — the proof of full understanding is the discharge of all surprise.
- `R2` partitions the surprise: the misfit products are disjoint or empty and carry `H = 1`; the fit product carries `H = 0` when its head is held. Weighted by head measure, R2 never reduces entropy — `Σ μ(head) · H(product) ≥ μ(σ ∨ â) · H(C)`, with equality iff the fit head is held or `⊥` (an unheld fit head degrades to R4 at full surprise).
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

Effort is the halt measure: when rationalisation ends, `E` of the halted kline is how far the trace did not get — the ranking fuel for the proposals the halt generates.

---

## 5. The Rewrite Rules

The rewrite system operates on candidates. R2 and R3 take a kline and produce klines, each in its own way. R2's products re-enter as entries (Definition 8), meeting the same pipeline and the same bands; R3's product re-enters within its candidate's chain. A product with no adjacency routes to R4 — the misfit asks, and later-arriving information may complete the path. Reentry is not a rewrite rule. Only S2 and S3 produce further work.

`D` is live throughout the tree. R1 stores enter `D` at the moment of arrival, and every later application — lookup, witness, substitution — reads `D` as it then stands. A run's own writes can close a route a later candidate would have taken, or open one: a witness minted in one round can keep a later expansion from emptying. The tree is a sequence of applications, not an evaluation against a snapshot of `D`.

### Rule R1 — proof of full understanding (S1): store the opening candidate

```text
C₀ opens a trace;  a candidate on that trace : S1  ⟹  D := D ∪ { C₀ }
```

Reaching S1 is a proof of full entry understanding — the work entry has been completely rationalised. The kline that enters `D` is the candidate that opened the trace — the work entry's first candidate — not the candidate formed on the final entry, which by this proof is constructable from the former by reapplying the rules. Where the opening candidate is itself the S1 candidate, the two coincide. Idempotent. A kline enters `D` at R1, when a trace it opens reaches S1. The opening candidate's head `c_E` may be a signature the domain does not yet hold — it is minted over the held adjacency the candidate was valued against, and this is how entry information becomes knowledge. Reinforcement and promotion are outside the algebra (§7).

### Rule R2 — partial fit (S2): project

For `C = ⟨σ, N⟩ : S2`, compute `F, U, O` (Definition 12) and witness:

```text
N_F = witness_D(F)    N_O = witness_D(O)    N_U = witness_D(U)
```

R2 produces three klines — for the non-empty heads:

```text
⟨F, N_F⟩      ⟨U, N_O⟩      ⟨O, N_U⟩
```

The pairing is cross-wise — `U` carries `N_O`, `O` carries `N_U` — so `U` and `O` each serve as lookup key in one product and as claim in the other. Each product re-enters as an entry (Definition 8).

### Rule R3 — misfit (S3): substitute, re-enter

R3 applies to an S3 candidate `C = ⟨σ, N⟩`:

```text
c_N = can(N)
for each K ∈ covered(c_N, D) whose signature occurs in N :
    replace every occurrence of sig(K) in N by adj(K)      single pass
N′ = the result
```

Substituting a signature replaces a reference with the witness it names — dereferencing through the hierarchy. A signature is substituted at most once per pass (well-foundedness). Occurrence implies coverage (`n ∈ N ⟹ n ⪯ can(N)`), so the licence is the occurrence: substitution applies where a covered neighbour's signature occurs in `N`. The covered lookup is the finding mechanism; the occurrence is the licence.

R3's product keeps the candidate's signature and carries the substituted adjacency:

```text
C′ = ⟨σ, N′⟩
```

Substitution is the whole of the rule: the signature makes its claim, and the substituted adjacency is the re-pairing. `C′` re-enters within the candidate's chain — Definition 9 applies: neighbours are located by overlap on `σ`, candidates are constructed against their adjacencies, and Definition 11 assigns each a band. An expansion that empties `N′` arrives at R4.

### Rule R4 — terminal (S4): end the branch

The arrival is no adjacency — a re-entering product, a candidate whose neighbour supplied none (Definition 11, case 1), or an expansion that emptied its adjacency. The branch ends undone — the knowledge boundary: a system knows what it knows (S1–S3) and knows what it does not (S4).

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
    store C₀  project    substitute (single pass)
      │         │          │
      end   ⟨F, N_F⟩    N′ ──► ⟨σ, N′⟩ (in-chain)
            ⟨U, N_O⟩              │
            ⟨O, N_U⟩              │
               │                  │
               └── new entries ───┘──► repeat
```

Across R1–R4, every rewrite resolves against held content and signatures enter `D` at R1 stores: the system composes what it holds, and asks at its boundary.

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
- arrival — the rules route every band but do not guarantee termination of the full tree: bounds on round length and re-entry depth are strategy, as the engine's T2 bounds witnessed cycles;
- confluence — intentionally not required: branching over neighbours is the model's semantics, and different branches may ground different resolutions;
- reinforcement, tiering, memory-management policy;
- ratification of stored klines.
