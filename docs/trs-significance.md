# Kalvin — A Significance-Routed Term Rewriting System

**Status:** Draft — normative. This document is the single normative definition of the significance-routed model: an alternative to the goal-directed algebra of `docs/kalvin-algebra.md`. That document remains normative for the engine; this one is normative for this model. The design record is the dialogue `data/trs.md`; this document consolidates its final layer and supersedes every earlier formulation found there — in particular, a band is never assigned to an entry kline, only to candidates (§3).

Both models share one centre: the significance valuation of the pairing of one side's evaluated content against the other side's witness (§7) — graded there (`J`), banded here. They differ in what drives rewriting. The engine derives toward a goal kline under a strict-decrease licence; this model has no goal — the band alone routes the rewrite tree.

## Decisions taken in this consolidation

The dialogue left points open. This document takes positions; each is listed here for review.

| #   | Decision                                                                                                                                                                                 |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D1  | Overfit/underfit follow `kalvin-algebra.md` Definition 9: underfit `U = σ ∖ can(N)` is signature-side, overfit `O = can(N) ∖ σ` is adjacency-side (§0.2, §4). Node operators unify on `∧ ∨ ∖`.                                       |
| D2  | `can` is the composition fold (§2, Definition 4).                                                                                                                                        |
| D3  | Adjacency is a finite set; order and multiplicity are dropped (§1, Definition 2).                                                                                                        |
| D4  | Entry lookup is overlap; exact lookup serves decode; covered lookup serves R3 (§2, Definition 7).                                                                                        |
| D5  | The S2 fit branch selects an already-held kline; it performs no rewrite (§5, R2a).                                                                                                       |
| D6  | R3's product is an entry kline (Definition 8): `⟨can(N′), N′⟩`; the prior signature, where there was one, is discarded — no correspondence is asserted (§5, R3).                    |
| D7  | R2's misfit products are the **residual adjacencies** `N_O`, `N_U` — each expands under R3. No candidate pairing is formed: there is no correspondence with `U`, nor with `O` (§5, R2b). |
| D11 | R3 is single pass: no substitutions → R4; otherwise its product is an entry and §3 applies afresh. The loop continues until a band routes to S1 or S4 (§5, R3).                        |
| D8  | No signature is minted inside the TRS: a derived residual not held terminates its branch at S4 (§5, R2). Creation happens only at R1 stores (D10).                                       |
| D9  | The classifier's value is the **band** (banded significance). Graded significance — an int — is reserved and undefined here (§4). The band routes the TRS; R1–R4 are the four routes.    |
| D10 | The domain grows only through R1: a kline not currently in `D` is added iff its candidate reaches S1. Entries are transient — presented, never stored.                                   |

Amended four times on review: (1) misfit pairing taken to the mirror, then superseded — the misfit products are now the unpaired residual adjacencies (D7); (2) Axiom A-1 deleted — an unheld derived signature is a transition to S4, not a materialisation, and closure is I-1 alone; (3) no R3 splice — occurrence is the licence; band vs graded significance terminology fixed (D9); admission iff S1, entries transient (D10); (4) R3 canonicalises and re-enters as a new entry, prior signature discarded, single pass, loop until S1 or S4 (D6, D11).

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
| `decode_D(c)` | known adjacency of canonical signature c | domain lookup         |
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
| entry kline               | new information; never valued             | queued kline A                              |
| neighbour                 | held kline found by entry lookup          | goal-side material                          |
| S1 / S2 / S3 / S4         | §4 bands                                  | same bands, §4 (S3 boundary equivalent, §7) |

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

I-1 is the closure property of the design, and it is maintained by construction: presentation (Definition 8) offers only entries whose references are held, and R1 stores — the sole admission rule (D10) — pair heads with already-held adjacencies. Every node of every decoded adjacency is therefore a domain signature — as a corollary of I-1, not a separate axiom.

---

## 2. Canonicalisation and Decoding

### Definition 4 — Canonical signature (D2)

```text
can : P_fin(Node) → Node
can(N) = ⋁_{n ∈ N} n        can(∅) = ⊥
```

`can` is total. It corresponds to the engine's evaluation `σ(ν)` restricted to sets.

### Definition 5 — Canonical kline

`K` is **canonical** when `sig(K) = can(adj(K))`. Write

```text
Can_D = { sig(K) : K ∈ D, K canonical }.
```

### Definition 6 — Decode

```text
decode_D : Can_D → P_fin(Node)
decode_D(c) = the N such that ⟨c, N⟩ ∈ D and can(N) = c
```

Well-defined by I-2 and **partial by design**: a signature nothing canonical heads has no decode. Fundamental law:

```text
can(decode_D(c)) = c        for every c ∈ Can_D
```

Decode is not bit decomposition of `c`; it is domain lookup — the known adjacency is the meaning of the signature. An unheld signature has no meaning yet: encountering one during rewriting is a transition to S4 (R2), the knowledge boundary.

### Invariant I-2 — Canonical keys

At most one canonical kline per signature:

```text
⟨c, N⟩, ⟨c, N′⟩ ∈ D canonical  ⟹  N = N′.
```

### Definition 7 — Three lookups (D4)

```text
exact(s, D)    = { K ∈ D : sig(K) = s }          serves decode
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

Entries arrive from two sources — outside presentation, and R3's canonical product (§5) — and the pipeline makes no distinction between them.

An entry is **presented, never stored** (D10): it mints nothing, and its content persists only through the S1 stores of its candidates (R1). Presentation requires I-1: its references are held — `can(N_E)` may still be a signature the domain does not hold. The entry is **never itself valued** — no band is computed for it. It is the mechanism for locating neighbours; the candidate constructed from it is what gets valued.

### Definition 9 — Entry pipeline

```text
neighbours = overlap(σ_E, D) = { K_1 … K_m }
c_E = can(N_E)

for each neighbour K_i = ⟨σ_i, N_i⟩ :
    candidate  C_i = ⟨c_E, N_i⟩
```

Empty lookup: no candidates; the entry derives nothing and terminates.

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

The rewrite system operates on candidates. Only S2 and S3 produce further work.

### Rule R1 — fit (S1): store

```text
C : S1  ⟹  D := D ∪ { C }
```

Idempotent. The candidate is proven canonical; it is stored if absent. R1 is the **sole admission rule** (D10): a kline not currently in `D` is added iff its candidate reaches S1. `c_E` may be a signature the domain does not yet hold, and an S1 candidate mints it over a held adjacency — the witness stored is the adjacency the candidate was valued against, and this is how entry information becomes knowledge. Reinforcement and promotion are outside the algebra (§8).

### Rule R2 — partial fit (S2): project

Compute `F, O, U` (Definition 12). Each is a derived signature; each is resolved by lookup:

```text
N_F = decode_D(F)    N_O = decode_D(O)    N_U = decode_D(U)
```

**A residual not held is a transition to S4.** `decode_D` is undefined for it: the domain does not hold the kline that gives it meaning, and rewriting mints no signatures (D8). The product that consumes it terminates at the knowledge boundary — the misfit asks, and later-arriving information may complete the path. The residuals fail independently: a product proceeds exactly when the residuals it consumes are held.

The single candidate adjacency is projected onto three adjacencies `{N_F, N_O, N_U}`, with two products:

**(a) fit — terminate.** `⟨F, N_F⟩` is held and canonical (I-2). The resolved structure already exists; the branch terminates. No rewrite occurs (D5).

**(b) misfit — the residual adjacencies** (D7). The products are the decoded residual adjacencies `N_O` and `N_U` — whichever exist. Each product enters R3 for expansion.

### Rule R3 — misfit (S3): substitute, canonicalise, re-enter

R3 applies to a misfit adjacency `N` — a residual adjacency from R2 (`N_O` or `N_U`), or the adjacency of an S3 candidate:

```text
c_N = can(N)
for each K ∈ covered(c_N, D) whose signature occurs in N :
    replace every occurrence of sig(K) in N by adj(K)      single pass
N′ = the result
```

Substituting a signature replaces a reference with the witness it names — dereferencing through the hierarchy. A signature is substituted at most once per pass (well-foundedness). Occurrence implies coverage (`n ∈ N ⟹ n ⪯ can(N)`), so the licence is the occurrence itself: eligible neighbours are exactly those whose signatures occur in `N`, and a covered neighbour that does not occur yields no candidate — there is nothing to substitute. There is no splice form. The covered lookup is the finding mechanism; the occurrence is the licence.

Two outcomes (D11):

- **No substitutions** (`N′ = N`): the branch enters R4 — no progress is possible.
- **Otherwise** R3's product is the entry kline

```text
E′ = ⟨can(N′), N′⟩
```

  The prior signature, where there was one, is discarded: S3 means no correspondence, and none is asserted.

  `E′` satisfies Definition 8 exactly. It is presented, never stored — R3 writes nothing to `D` (D10). Its references are held — `N′` results from substituting held witnesses into held references, so I-1 holds for it. And it is never itself valued — its canonicality is never scored, for no band is computed for an entry. Definition 9 therefore applies verbatim: neighbours are located by overlap on `can(N′)`, candidates are constructed against their adjacencies, and Definition 11 assigns each a band.

  There is no re-entry mechanism. R3's output is an entry, and §3 is the only consumer of entries.

The loop is rule composition: each entry's candidates route by band — S1 stores, S4 ends — and S2 and S3 produce work whose products are entries again, until a band routes to S1 or S4.

### Rule R4 — terminal (S4): end the branch

Two arrivals:

```text
N = ∅            — no adjacency from which known structure can be derived
R3: N′ = N       — no substitution, no progress possible
```

Both end the branch undone. This is a knowledge boundary, not a failed lookup: a system knows what it knows (S1–S3) and knows what it does not (S4).

### The derivation tree

```text
entry ⟨σ_E, N_E⟩
  │                │
  │ overlap(σ_E)   │ can(N_E) = c_E
  ▼                ▼
neighbours K_i   candidates C_i = ⟨c_E, N_i⟩
  └────────► V(C_i) ◄───────────┘
                │
      ┌─────────┼──────────┬──────────┐
      S1        S2         S3         S4 → end
      │         │          │
    store    project     substitute (single pass)
      │      │      │         │
      end   fit:  N_O, N_U    no substitutions → end
            end    │
                   ▼
             N′ ──► ⟨can(N′), N′⟩ ──► new entry ──► repeat
```

---

## 6. Properties

### P-1 — Separated misfit

R2's products are the fit selection (terminal) and the residual adjacencies `N_O`, `N_U` — no pairing between them is asserted, because none exists: `O ∧ U = ⊥` by construction. The S2 rewrite separates the resolvable correspondence from residuals that share no content, and R3 resolves each through the graph rather than against the other.

### P-2 — Local bounds

- **R2** weakens content claims: `F, U ⪯ σ` and `F, O ⪯ â`, with `U ∧ â = ⊥` and `O ∧ σ = ⊥`. Metric bounds would require graded significance (§4), which this model does not define.
- **R3** is single pass: at most one substitution per signature, at most `|D|` per pass.
- **R3 progress:** `N′ = N` routes to R4; a pass either expands the adjacency or ends the branch.

### P-3 — Global termination and confluence

The loop's routing is total — every band has a route, and the loop continues until S1 or S4 — but arrival is not guaranteed by the rules alone: a cycle may present an entry whose candidates re-derive earlier states. Termination of the full tree is therefore a **strategy property**, bounded by re-entry depth, as the engine's T2 bounds witnessed cycles. Confluence is **intentionally not required**: branching over neighbours is the model's semantics, and different branches may ground different resolutions.

### P-4 — No fabrication

Every rule resolves against held content: R2 decodes by lookup — an unheld residual terminates at S4 rather than being minted; R3 substitutes held witnesses into held references (I-1); R1 stores only what valuation proved canonical, and is the sole admission rule (D10). The domain grows only through S1 stores, so no-fabrication is structural, not axiomatic.

---

## 7. Relationship to the Kline Algebra

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

## 8. Outside the Algebra

- selection order among neighbours, among R3 substitutions, and between the residual adjacencies R3 expands — strategy;
- bounds on round length and re-entry depth;
- graded significance — the measure species, undefined here; adding it would quantify the band;
- reinforcement, tiering, memory-management policy;
- ratification of stored klines.
