# Kalvin — A Significance-Routed Term Rewriting System

**Status:** Draft — normative. This document is the single normative definition of the significance-routed model: an alternative to the goal-directed algebra of `docs/kalvin-algebra.md`. That document remains normative for the engine; this one is normative for this model. The design record is the dialogue `data/trs.md`; this document consolidates its final layer and supersedes every earlier formulation found there — in particular, bands belong to candidates (§3), and the entry is the mechanism that locates them.

Both models share one centre: the significance valuation of the pairing of one side's evaluated content against the other side's witness (§6) — graded there (`J`), banded here. They differ in what drives rewriting. The engine derives toward a goal kline under a strict-decrease licence; this model has no goal — the band alone routes the rewrite tree.

## Decisions taken in this consolidation

The dialogue left points open. This document takes positions; each is listed here for review. Method: the rules (§5) follow the definitions (§§1–4) and never lead them — a rule uses only terms the definitions supply.

| #   | Decision                                                                                                                                                                                 |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D1  | Overfit/underfit follow `kalvin-algebra.md` Definition 9: underfit `U = σ ∖ can(N)` is signature-side, overfit `O = can(N) ∖ σ` is adjacency-side (§0.2, §4). Node operators unify on `∧ ∨ ∖`.                                       |
| D2  | `can` is the composition fold (§2, Definition 4).                                                                                                                                        |
| D3  | Adjacency is a finite set; order and multiplicity are dropped (§1, Definition 2).                                                                                                        |
| D4  | Entry lookup is overlap; exact lookup serves witness; covered lookup serves R3 (§2, Definition 7).                                                                                        |
| D5  | R2 produces three klines — `⟨F, N_F⟩`, `⟨U, N_O⟩`, `⟨O, N_U⟩` — which re-enter as entries (§5, R2).                                                                       |
| D6  | R3's product is an entry kline (Definition 8): `⟨can(N′), N′⟩`; the prior signature, where there was one, is discarded (§5, R3).                                                    |
| D7  | The decomposition pairs cross-wise: `U` carries `N_O`, `O` carries `N_U` — `U` and `O` each serve as lookup key in one product and as claim in the other (§5, R2). |
| D11 | R3 is single pass, with a single outcome: the reentrant kline `⟨can(N′), N′⟩`. The loop continues until a band routes to S1 or S4 (§5, R3).                                          |
| D8  | `witness_D` yields no adjacency for a signature not held (Definition 6); a product carrying it routes to R4 at reentry (§5). Signatures enter `D` at R1 (D10). |
| D9  | The classifier's value is the **band** (banded significance). Graded significance — an int — is reserved and undefined here (§4). The band routes the TRS; R1–R4 are the four routes.    |
| D10 | A kline enters `D` at R1, iff its candidate reaches S1. Entry content persists through the stores of its candidates.                                                               |

Amended eight times on review: (1) misfit pairing taken to the mirror, then superseded — the misfit products are now the unpaired residual adjacencies (D7); (2) Axiom A-1 deleted — an unheld derived signature is a transition to S4, not a materialisation, and closure is I-1 alone; (3) no R3 splice — occurrence is the licence; band vs graded significance terminology fixed (D9); admission iff S1, entries transient (D10); (4) R3 canonicalises and re-enters as a new entry, prior signature discarded, single pass, loop until S1 or S4 (D6, D11); (5) R2's products are three re-entering klines — `⟨F, N_F⟩`, `⟨U, N_O⟩`, `⟨O, N_U⟩` — replacing the fit-termination and raw-adjacency products; R3 takes an S3 candidate uniformly (D5, D7); (6) the unheld-residual S4 transition moved from R2 to reentry — the empty adjacency routes to R4, now with the single arrival of an entry kline with no adjacency — and R3's single outcome is the reentrant kline (D8, D11); (7) decode made total — `∅` for signatures not held — and `residual`/`empty adjacency` withdrawn from the rules: R2 states the decomposition, decode, and the three products; the no-adjacency routing follows Definitions 2, 6, and 11 (D8); (8) `decode` renamed `witness` — the strict converse of canonicalisation, aligning with the engine's held witness (Definition 6).

---

## 0. Notation

### 0.1 Operators

| Symbol        | Meaning                                  | Reference realisation |
| ------------- | ---------------------------------------- | --------------------- |
| `v ∨ w`       | composition                              | `v \| w`              |
| `v ∧ w`       | overlap                                  | `v & w`               |
| `v ∖ w`       | residue — v beyond w                     | `v & ~w`              |
| `⊥`           | the empty node, composition's unit       | `0`                   |
| `x ⪯ y`       | containment: `x ∧ y = x`                 | `x & y == x`          |
| `can(N)`      | canonical signature of adjacency N       | fold `∨`              |
| `witness_D(s)` | the held witness of signature s             | domain lookup         |
| `V(C)`        | the band of candidate C                  | §4                    |

### 0.2 Terminology mapping

Terms are shared with `kalvin-algebra.md`; the mapping below gives both documents' readings side by side.

| Term (this document)      | This document                             | `kalvin-algebra.md`                         |
| ------------------------- | ----------------------------------------- | ------------------------------------------- |
| underfit `U = σ ∖ can(N)` | claimed by the signature, not supplied by the adjacency | `u = s ∖ σ(ν)` — the same |
| overfit `O = can(N) ∖ σ`  | supplied by the adjacency, not claimed by the signature | `o = σ(ν) ∖ s` — the same  |
| kline `⟨σ, N⟩`            | signature + adjacency set                 | `s:ν` — head + node **sequence**            |
| canonical signature       | `can(N)`, the S1 head                     | exact witness / Canon band                  |
| candidate                 | `⟨c_E, N_i⟩` (§3)                         | relationship kline `C(A,B) = σ(ν_A):ν_B`    |
| entry kline               | new information; bands belong to its candidates | queued kline A                         |
| neighbour                 | held kline found by entry lookup          | goal-side material                          |
| S1 / S2 / S3 / S4         | §4 bands                                  | same bands, §4 (S3 boundary equivalent, §6) |

---

## 1. Sorts and the Domain

### Definition 1 — Node algebra

`Node` is equipped with `(⊥, ∨, ∧, ∖)`:

- `∨` — commutative, associative, idempotent; unit `⊥`;
- `∧` — commutative; `v ∧ ⊥ = ⊥`;
- `∖` — `(v ∖ w) ∨ (v ∧ w) = v` and `(v ∖ w) ∧ w = ⊥`;
- decidable equality.

Write `x ⪯ y` for `x ∧ y = x`. These are the laws of `kalvin-algebra.md` Definition 1 minus the measure `μ` — graded significance is not defined by this model (§4). The reference realisation is the same word-bit space. A node is opaque; the algebra never looks inside one.

### Definition 2 — Kline

A kline is a pair

```text
K = ⟨σ, N⟩
```

where `σ ∈ Node ∖ {⊥}` is its signature and `N ⊆_fin Node` is its adjacency, possibly empty.

`N` is a **set** (D3): arrangement carries no information. Decomposition structure lives in held witnesses, never in arrangement.

### Definition 3 — Domain

The domain `D ⊆_fin Kline` is a finite set of klines. Write `sig(K)` and `adj(K)` for the accessors.

### Invariant I-1 — Referential closure

Every node occurring in an adjacency is the signature of a kline in `D`:

```text
∀ K ∈ D : ∀ n ∈ adj(K) : ∃ K′ ∈ D : sig(K′) = n.
```

The invariant applies to admitted entries as well (Definition 8). The system references only what it holds.

I-1 is the closure property of the design, and it is maintained by construction: presentation (Definition 8) offers only entries whose references are held, and R1 stores pair heads with already-held adjacencies. Every node of every witnessed adjacency is therefore a domain signature — a corollary of I-1.

---

## 2. Canonicalisation and Decoding

### Definition 4 — Canonical signature (D2)

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
witness_D(s) = the N such that ⟨s, N⟩ ∈ D and can(N) = s    when D holds one
witness_D(s) = ∅                                          otherwise
```

Held signatures are unique by I-2. Fundamental law:

```text
can(witness_D(s)) = s        for every s held canonically
```

Witness is the converse of canonicalisation under strict conditions — the held witness of a signature, as the engine's held witness (`kalvin-algebra.md` Definition 6): the known adjacency is the meaning of a held signature. A signature not held has no witness — no meaning yet — and a kline carrying no adjacency values S4 (Definition 11, case 1).

### Invariant I-2 — Canonical keys

At most one canonical kline per signature:

```text
⟨c, N⟩, ⟨c, N′⟩ ∈ D canonical  ⟹  N = N′.
```

### Definition 7 — Three lookups (D4)

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

Entries arrive from outside presentation and from the products of R2 and R3 (§5); the pipeline makes no distinction between them.

An entry's content reaches the domain through the S1 stores of its candidates (R1, D10). Presentation requires I-1: its references are held — `can(N_E)` may still be a signature the domain does not hold. The entry locates neighbours; the candidates constructed from it are what get valued (Definition 9, Definition 11).

### Definition 9 — Entry pipeline

```text
neighbours = overlap(σ_E, D) = { K_1 … K_m }
c_E = can(N_E)

for each neighbour K_i = ⟨σ_i, N_i⟩ :
    candidate  C_i = ⟨c_E, N_i⟩
```

Empty lookup: the entry terminates without candidates.

### Definition 10 — Candidate invariant

Every candidate carries a signature derived from one context — the entry's adjacency, canonicalised — and an adjacency inherited from another — a held kline. The band therefore always compares two independently derived representations of information.

---

## 4. The Band

Significance has two species, distinguished by prefix. **Graded significance** is a measure — an int — and is not defined by this model. **Banded significance** is the classifier below: it assigns each candidate exactly one band, S1–S4. A graded measure, if added later, would quantify the band, as `J` quantizes the engine's bands (`kalvin-algebra.md` §10).

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

`S2 ⟺ F ≠ ⊥ ∧ (U ≠ ⊥ ∨ O ≠ ⊥)`. Flow directions: `U ≠ ⊥` is underfit — the unresolved content sits in the signature, and information must flow signature → adjacency; `O ≠ ⊥` is overfit — the unresolved content sits in the adjacency, and information must flow adjacency → signature; both is bidirectional. The directions are orientations of flow, not size comparisons — neither side need contain the other.

---

## 5. The Rewrite Rules

The rewrite system operates on candidates. R2 and R3 take a kline and produce klines, each in its own way; their products re-enter as entries (Definition 8), so every product meets the same pipeline and the same bands. A product with no adjacency routes to R4 — the misfit asks, and later-arriving information may complete the path. Reentry is not a rewrite rule. Only S2 and S3 produce further work.

### Rule R1 — fit (S1): store

```text
C : S1  ⟹  D := D ∪ { C }
```

Idempotent. The candidate is proven canonical; it is stored if absent. A kline enters `D` at R1, when its candidate reaches S1 (D10). `c_E` may be a signature the domain does not yet hold, and an S1 candidate mints it over a held adjacency — the witness stored is the adjacency the candidate was valued against, and this is how entry information becomes knowledge. Reinforcement and promotion are outside the algebra (§7).

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

### Rule R3 — misfit (S3): substitute, canonicalise, re-enter

R3 applies to an S3 candidate `C = ⟨σ, N⟩`:

```text
c_N = can(N)
for each K ∈ covered(c_N, D) whose signature occurs in N :
    replace every occurrence of sig(K) in N by adj(K)      single pass
N′ = the result
```

Substituting a signature replaces a reference with the witness it names — dereferencing through the hierarchy. A signature is substituted at most once per pass (well-foundedness). Occurrence implies coverage (`n ∈ N ⟹ n ⪯ can(N)`), so the licence is the occurrence: substitution applies where a covered neighbour's signature occurs in `N`. The covered lookup is the finding mechanism; the occurrence is the licence.

R3's product is the entry kline

```text
E′ = ⟨can(N′), N′⟩
```

The prior signature, where there was one, is discarded: S3 means no correspondence, and none is asserted. Definition 9 applies: neighbours are located by overlap on `can(N′)`, candidates are constructed against their adjacencies, and Definition 11 assigns each a band.

The loop is rule composition: each entry's candidates route by band — S1 stores, S4 ends — and S2 and S3 produce work whose products are entries again, until a band routes to S1 or S4.

### Rule R4 — terminal (S4): end the branch

The arrival is an entry kline with no adjacency — a re-entering product, or a candidate whose neighbour supplied none (Definition 11, case 1). The branch ends undone — the knowledge boundary: a system knows what it knows (S1–S3) and knows what it does not (S4).

### The derivation tree

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
    store    project    substitute (single pass)
      │         │          │
      end   ⟨F, N_F⟩    N′ ──► ⟨can(N′), N′⟩
            ⟨U, N_O⟩              │
            ⟨O, N_U⟩              │
               │                  │
               └── new entries ───┘──► repeat
```

Across R1–R4, every rewrite resolves against held content and signatures enter `D` at S1 stores: the system composes what it holds, and asks at its boundary.

---

## 6. Relationship to the Kline Algebra

| This model                        | `kalvin-algebra.md`                                                                                   |
| --------------------------------- | ----------------------------------------------------------------------------------------------------- |
| candidate `⟨c_E, N_i⟩`            | relationship kline `C(A,B) = σ(ν_A):ν_B` — same construction, independently reached                   |
| `can(N)` = fold `∨` over a set    | evaluation `σ(ν)` = fold `∨` over a sequence                                                          |
| R3 substitution `sig(K) ⇉ adj(K)` | forward replacement `n ⇉ ν_K` (witnessed expand)                                                      |
| S1 band                           | S1 band {Identity, Canon} — `⟨s, {s}⟩` is S1 here, collapsing the engine's Identity/Canon distinction |
| S3 boundary `σ ∧ can(N) = ⊥`      | equivalent to the engine's not-covered case (distributivity: `σ ∧ can(N) = ⊥ ⟺ ∀ n ∈ N : σ ∧ n = ⊥`)  |
| residuals `U = σ ∖ â`, `O = â ∖ σ` | `u = s ∖ σ(ν)`, `o = σ(ν) ∖ s` (Definition 9) — identical |
| no goal, no targeting licence     | goal kline B, strict-decrease targeting (Def 14)                                                      |

The architectural difference: the engine is goal-directed — a derivation works toward B under a licence that strictly decreases mismatch; this model is significance-routed — the tree grows by valuation alone, and no kline plays the goal's role. The candidate construction being the same object in both models is the strongest evidence that the two are readings of one system.

---

## 7. Outside the Algebra

- selection order among neighbours, among R3 substitutions, and among re-entering products — strategy;
- arrival — the rules route every band (D11) but do not guarantee termination of the full tree: bounds on round length and re-entry depth are strategy, as the engine's T2 bounds witnessed cycles;
- confluence — intentionally not required: branching over neighbours is the model's semantics, and different branches may ground different resolutions;
- graded significance — the measure species, undefined here; adding it would quantify the band;
- reinforcement, tiering, memory-management policy;
- ratification of stored klines.
