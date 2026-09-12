# Kalvin — Symbolic II, in plain terms

Status: draft. This is `ks2.md` restated for programmers: same content, same section numbers, less mathematical apparatus, more examples. Where wording differs, **ks2.md remains normative**. `kalvin-symbolic.md` §5 remains normative for KScript surface syntax; CONTEXT.md for role names.

**The whole system in six sentences.** Kalvin's memory is a set of records — _klines_ — each claiming that its head value is the composition of its node list. Composition is just set-union, so many different lists evaluate to the same head; _which list you chose_ is the memory. A pure function grades any (head, list) pair into one of nine shapes, grouped into four bands from "exact" to "no connection". Thinking — _cogitation_ — is a rewrite loop over one node list, using held klines as the rewrite rules, each step either re-expressing the same content more or less granularly, or moving it strictly closer to a goal kline's content; a run ends **done** when the contents match or **stuck** when nothing held connects them. A score γ — overlap, discounted by granularity and by unratified promises — grades every step, and its rate of change steers the loop. Strategy chooses what to derive next; measurement decides what it was worth.

Four tracts, as in ks2: **what exists** (§1–5), **what may happen** (§6–9), **what chooses** (§10), **what is observed** (§11–12). §13 lists what stays outside.

## 0. Notation cheat sheet

Everything this document writes in symbols, decoded. Values are sets of atoms, and the set operations are the ones you already know from bitmasks:

**Sets and values**

| symbol      | it means                                         | in code           |
| ----------- | ------------------------------------------------ | ----------------- |
| `a`, `abc`  | the singleton set `{a}`, the set `{a,b,c}`       | bitmasks          |
| `v ∨ w`     | union                                            | `v \| w`         |
| `v ∧ w`     | intersection                                     | `v & w`           |
| `¬v`        | complement, within the full atom set `A`         | `~v & MASK`       |
| `∅`         | the empty value                                  | `0`               |
| `σ(ν)`      | `signature_of(ν)` — the OR-fold of the list (§2) | `fold_or(nu)`     |
| `x Δ y`     | symmetric difference: in one but not both        | `x ^ y`           |
| `v ∖ w`     | set difference                                   | `v & ~w`          |
| `[n₁ … nₖ]` | a list of values, order and duplicates kept      | `[n1, ..., nk]`   |
| `x ∈ y`, `n ∉ ν` | (non-)membership: x is (not) an element of y | `x in y`      |
| `M ⊇ M_k`   | superset: M contains everything M_k does         | `M.issuperset(M_k)` |
| `\|x\|`   | size: how many atoms x has                       | `popcount(x)`     |

(ks2 additionally writes `V*` for "lists of values" and `2^A` for "all subsets of A" — the set of all bitmasks.)

**Klines and rewriting**

| symbol       | it means |
| ------------ | --------- |
| `s:ν`        | the kline with signature `s` and node list `ν` — a `(head, nodes)` record |
| `A ⊢_{M,B} A′` | **one derivation step**: kline A rewrites to A′ by a single licensed replace, running against goal B with memory M. The subscript just names the parameters and is dropped when clear (`A ⊢ A′`). Chained `A₀ ⊢ A₁ ⊢ …` is a multi-step run, left to right in time. (The symbol is the "turnstile" of logic texts — here it simply reads "rewrites to") |
| `n ⇉ ν_K`    | one replace applied to a node: splice `n` out of the node list, splice the witness `ν_K` in — the arrow used in worked examples |
| `x → y`      | a generic arrow: "x becomes y" — rewrite direction or state change |
| `s ↦ [s]`    | "maps to": a function's input to its output (here, the trivial inverse of σ) |
| `A′`         | A-prime: the state after one more step |
| `C(A,B)`     | the relationship kline of A and B (§5) — ordinary function notation |
| `fit(s, ν)`  | the nine-shape classifier (§4) — ordinary function notation |

**Measurement (§11)**

| symbol   | it means |
| -------- | --------- |
| `γ(A, B)` | the graded distance — the significance score of A working toward B |
| `J(x, y)` | Jaccard overlap: shared atoms over combined atoms — in code, `popcount(x & y) / popcount(x \| y)` |
| `δ`       | the discount knob: a number strictly between 0 and 1 (`δ ∈ (0,1)` is interval notation, not a tuple) |
| `D̄`, `Ĥ` | mean resolution / acquisition depth — the bar on D and the hat on H both mean "average" (atom-weighted) |
| `Δ₀`      | the initial misfit mass — a *number*, not an operation; see the reading practice below |
| `α(n)`    | per-slot accountedness: one node's overlap fraction with the goal (appears in the argument for why J is Jaccard) |
| `·`       | ordinary multiplication, as in `J · δ^(D̄ + Ĥ)` |

**Reading conventions.** `iff` = if and only if — equivalence, both directions hold. Subscripts name ownership or time: `ν_A` is kline A's node list, `ν_K` the evidence kline's witness; `A₀` is the state at entry, `Aᵢ` the state at step i; `M_k`, `B_k` are hop k's memory and goal (§10).

**Reading practice.** T1's bound (§8), decoded piece by piece:

```text
σ(ν_{A₀})    fold A's entry node list into one value
σ(ν_B)       fold the goal's node list into one value
x Δ y        atoms on exactly one side of the two
Δ₀           how many atoms that is — the initial misfit mass
```

No run of targeting replaces from `A₀` can take more than `Δ₀` steps, because each licensed step strictly decreases that count (§6's scoping clause).

---

## 1. Atoms and values

**In programmer terms:** an atom is one bit in a fixed, small bit-space; a value is a bitmask. In the engine, the atoms are _word bits_ — one bit per distinct word Kalvin has seen (bits 0–30 of the upper half of a token id; the u64 packing `(word_bit << 32) | bpe_token_id` is the engine's encoding, which this algebra tidies up). Only finiteness matters to the formalism, not the exact count.

Say the atoms are `a`, `b`, `c` (think: three words). Then values are the subsets:

```text
∅, a, b, c, ab, ac, bc, abc
```

with the obvious operations — union (`∨`), intersection (`∧`), complement (`¬`). Example computations:

```text
ab ∨ ac = abc        ab ∧ ac = a        ¬a = bc
```

The Boolean laws (commutativity, associativity, idempotence, distribution, distinct atoms being disjoint) all hold automatically because values _are_ sets. They are consequences of the representation, not rules you have to enforce.

## 2. Node sequences — the terms

**In programmer terms:** a node sequence is a list of non-empty values. Plain list semantics: order retained, duplicates retained, concatenation is the only structure.

**Evaluation.** `signature_of` (shorthand `σ`) folds the list with union:

```python
def signature_of(nu):        # nu: list of non-empty values
    out = ∅
    for n in nu:
        out = out ∨ n
    return out

# signature_of([ab, c])   = abc
# signature_of([c, ab])   = abc     (order forgotten)
# signature_of([a, b, a]) = ab      (multiplicity forgotten)
# signature_of([abc])     = abc     (granularity forgotten too — see below)
# signature_of([])        = ∅
```

`signature_of` forgets _exactly_ order and multiplicity — nothing else. Two lists evaluate to the same value iff they contain the same atoms; and any two such lists are folded to the same head.

The whole architecture lives in the gap this creates:

> **Lists are the terms; values are what the terms evaluate to.**

Like `3 + 4` versus `7`: many expressions, one result. A calculator discards the expression; Kalvin's memory keeps it (§3) — _which_ decomposition you hold is the content of memory. Note one extra forgetting the fold does silently: `[abc]` and `[a, b, c]` evaluate identically. The list remembers granularity; the value does not. That distinction becomes the expand/contract moves of §6 and the `D̄` term of §11.

## 3. Klines — claims and witnesses

**In programmer terms:** a kline is a pair — a head value and a node list — written `s:ν`.

```text
mhall:[m, h, a, l, l]
```

A kline _claims_: "my signature `s` is the composition of my nodes." The claim may be exact or sloppy; that is graded in §4.

- **Exact** means `s = signature_of(ν)`. An exact, non-empty kline is a **witness**: a chosen decomposition of `s`. There is no canonical inverse to `σ` — for `abc`, the witnesses include `[a,b,c]`, `[ab,c]`, `[a,bc]`, `[abc]`, `[c,a,b]`, … — the trivial one, `s ↦ [s]`, is called an **Identity** and carries no decomposition content; every other witness is a real choice. Nothing in the value world reconstructs which choice was made — _that is what memory is for_.
- A witness may repeat a node (`[l, l]` above): `σ` cannot see the repetition, but the list retains it — it is part of the chosen decomposition.

**Memory** `M` is a finite set of klines. Two klines may share a signature (different claims, or different decompositions of the same value). A node may itself be the signature of another kline — nesting is by reference, so memory is a graph, not a tree. The graph may even cycle:

```text
a:[a, a]            # canon self-reference
a:[b]  and  b:[a]   # a countersign pair
```

Anything may be held; cycles simply carry no decomposition content (§6 makes them inert). _Which_ klines are held, and in what tier (STM, Frame, LTM), are relations defined above this algebra — see CONTEXT.md.

> **The central claim.** Kalvin's memory is a set of claims over a forgetting map. A kline claims its signature as the composition of its nodes; **Canon** is the claim exactly kept; **Underfit/Overfit** are the two directions a claim can miss while still being answered; the S3 shapes are claims nothing answers; **Unknown** makes no composition claim — nothing held, or nothing left — and lands S4. Significance grades the claim.

## 4. Coverage and the fit classifier

**In programmer terms:** `fit` is a pure function from a (value, list) pair to one of nine shapes, grouped into four bands. There is no state, no default case — every pair lands in exactly one shape.

Two set notions first:

- **Coverage.** A node `n` is _covered_ by a value `s` when `n ∧ s ≠ ∅` — they share at least one atom. Coverage is overlap, not containment: a covered node may still carry atoms outside `s`.
- **Gap and excess.** For a pair `(s, ν)` with `ν ≠ []`:

```text
gap    g = s ∧ ¬σ(ν)     # atoms the signature claims but the nodes don't deliver
excess e = σ(ν) ∧ ¬s     # atoms the nodes deliver but the signature doesn't claim
```

Both empty together iff `s = σ(ν)` (exact). Example: for `abc:[b, c, d]` — gap `a` (claimed, not delivered), excess `d` (delivered, not claimed).

**The classifier.** Cases in order; first match wins:

```python
def fit(s, nu):                     # s: value, nu: list of non-empty values
    if nu == [] or s == ∅:
        return Unknown              # S4  — no claim in either direction
    if nu == [s]:
        return Identity             # S1  — the trivial witness: claims itself, nothing more
    sigma = signature_of(nu)
    if s == sigma:
        return Canon                # S1  — claim kept exactly
    gap    = s  ∧ ¬sigma            # claimed, not delivered
    excess = sigma ∧ ¬s             # delivered, not claimed
    covered = any(n ∧ s ≠ ∅ for n in nu)
    if not covered:
        return Connotation if len(nu) == 1 else NoFit    # S3
    if gap and not excess:    return Underfit             # S2
    if excess and not gap:    return Overfit              # S2
    return UnderAndOver                                    # S2
```

The check order is load-bearing. Coverage is the primary split: if _no_ node is covered, then every claimed atom is missing and every delivered atom is extra (`g = s`, `e = σ(ν)`), so an uncovered pair can never satisfy the covered cases — S3 is decided before gap/excess are even consulted. Exactness is caught before the covered cases, which all require a nonzero gap or excess. Case 1's disjunction is "no claim" in both directions: `ν = []` — nothing held; `s = ∅` — nothing left. (Replace never empties a node list, §6, so the empty side is an entry condition — the _ask_, §10 — never a run outcome.)

**The canonical table** (illustration, not definition):

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

Look at the S3 rows: gap _and_ excess are both nonzero, yet they are not Under+over — coverage decided first. That precedence is what keeps the nine shapes disjoint.

**Species.** **Denotation** is just the single-node Underfit (`ab:[b]` — one covered node, gap only); **Connotation** is case 4. These are names of convenience for KScript (`=` and `>`/`<`); algebraically they are single-node instances of cases 6 and 4. Two single-node shapes go unnamed: `a:[ab]` (single-node Overfit) and `ab:[bc]` (single-node Under+over — covered on `b`, gap `a`, excess `c`).

**Bands.** Derived from the shapes, not asserted alongside them: S1 = exact (cases 2–3), S2 = covered misfits, S3 = uncovered misfits, S4 = Unknown. The informal readings:

```text
S1 — I know that I know this.
S2 — I infer this, but it does not yet fit.
S3 — I recognise aspects of this, indirectly.
S4 — I do not understand this at all.
```

The **ask** is structural, not a flag: Unknown (`s:[]` — nothing held for the signature) _is_ the ask's shape, the halt signal under which strategy generates ungrounded proposals (§10). No atom or mark is involved.

**Invariance.** `fit` does not care about node order — it reads `ν` only through `σ(ν)` and the node count. Duplicating a node changes the fit only when it crosses a count boundary:

```text
a:[a]     → Identity (S1)      a:[a,a]   → Canon (S1)      # 1 node vs 2 nodes
a:[b]     → Connotation (S3)   a:[b,b]   → No-fit (S3)     # ditto
```

Otherwise repetition is witness structure, invisible to classification.

One function, two uses (this is §5): applied to a kline's own `(s, ν)` it grades the kline's own claim; applied to a pair of klines it grades their relationship. No new constructors anywhere — the nine shapes are derived predicates.

## 5. The relationship kline

**In programmer terms:** to compare two klines, line up _what the first one actually holds_ against _how the second one is decomposed_, and run the same classifier on that.

For klines `A = s:ν_A` and `B = t:ν_B`, the **relationship kline** is:

```text
C(A,B) = σ(ν_A) : ν_B
```

The head of `C` is not a claim — it is _defined_ as what A's nodes evaluate to, so it is exact against `ν_A` by construction. All of C's misfit-ness therefore comes from B's side: `fit(C(A,B))` is the **structural relationship of A and B**.

Worked pair. A holds `abc:[a,b,c]`; B holds `ab:[a,b,c]`:

```text
C(A,B) = abc:[a,b,c]        # head = σ([a,b,c]) = abc; nodes = B's
gap = ∅, excess = c          # Overfit — B's decomposition carries more than A holds
```

A useful equivalence to keep in your pocket:

> `C(A,B)` is Canon iff `σ(ν_A) = σ(ν_B)` — the two klines hold the _same value_, differently decomposed.

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

**In programmer terms:** a derivation is a loop that rewrites one kline's node list, using other klines from memory as the rewrite rules. Three klines are in play at every step:

```text
A   the queued kline   — its claim s is frozen; only its node list ν_A is rewritten
B   the goal kline     — scopes where work remains; checked at done
K   the evidence kline — a held kline that licenses one replace
```

Formally: a derivation rewrites `A = s:ν` against one held goal `B = t:ν_B`, relative to memory `M`. Write it `A ⊢_{M,B} A′`. States differ only in `ν`; nothing in §§6–9 ever reads `s` — the claim rides along inert, mattering only after the hop (absorb, reentry §10, grounding §13). Within a hop, the derivation works on the two sides of `C` — `σ(ν_A):ν_A` against `t:ν_B` — which is why scoping reads the _relationship_ `C(A,B)`, while mode and direction read the _evidence's own_ fit (below).

**The one rule: replace.** A held correspondence kline `K = n:ν_K` and an occurrence in `ν_A` matching one of K's two sides:

```text
forward:  an occurrence of the node n            → replaced by the list ν_K
reverse:  a contiguous occurrence of the list ν_K → replaced by [n]
```

"Correspondence" means: any held kline with two real sides. The two terminals are inert as evidence — an Unknown (`n:[]`) has no second side to offer; an Identity (`n:[n]`) replaces a node by itself.

**Mode — from K's own fit.** What the replace _does to content_ depends on the evidence's shape:

| K's own fit           | forward (`n → ν_K`)                    | reverse (`ν_K → [n]`)                |
| --------------------- | -------------------------------------- | ------------------------------------ |
| Canon (exact)         | **expand** — same content, finer grain | **contract** — coarser grain         |
| covered misfit (S2)   | sheds K's gap, adopts K's excess       | the mirror: adopts gap, sheds excess |
| uncovered misfit (S3) | **traverse** — disjoint atoms swap     | traverse, the other way              |

Examples. Canon `abc:[a,b,c]`: replacing node `abc` with `[a,b,c]` changes nothing in content — pure granularity. Denotation `dh:[h]` (gap `d`): forward on it, `dh → [h]`, drops exactly the atom `d` — the gap is _what drops_. Connotation `w:[o]` (disjoint): forward swaps `w` out, `o` in.

**Direction is not a property of the kline.** A correspondence is read forward from its signature and reverse from its witness; whichever side the derivation is standing on is the side that can occur. Arrival orients the licence — the same kline read from the other side is the mirror derivation's licence. This is why the klines in memory are direction-free.

**Ratified or not.** A correspondence edge may be _ratified_ or unratified — a tier relation over `M` (§10, §13). No rule of this section reads that status; only measurement does (§11).

**Licensing — who may fire.** The relationship scopes; the evidence licenses. The band says _where work remains_; only a correspondence kline says _what may move_:

| `fit(C(A,B))`            | Licensed targeting                             |
| ------------------------ | ---------------------------------------------- |
| S1 — Canon, Identity     | none — done                                    |
| S2 — covered misfits     | replace, on the misfit region (scoping clause) |
| S3 — Connotation, No-fit | replace — every node sits wholly in the misfit |
| Unknown — S4             | none — stuck                                   |

**Scoping clause.** A targeting replace is licensed iff it strictly decreases the **misfit mass** `|σ(ν_A) Δ σ(ν_B)|` — the number of atoms on exactly one side. Concretely: the node replaced must carry a gap atom, or its replacement must carry excess atoms, or both. This one clause outlaws busywork: a replace that only shuffles shared content, or grows the misfit, is not a targeting move however well evidenced. Canon-mode replaces (expand/contract) are a different licence — see §7 — they preserve content, preserve band, and are licensed by `M` alone, needing nothing from this table.

**Well-foundedness.** Canon evidence must satisfy `n ∉ ν_K` — the head does not occur in its own witness. The clause is exactly strong enough: a canon's nodes are atom-subsets of its head, so the only way an expansion chain can fail to terminate is a canon whose witness contains its own signature. Exclude those and every expansion bottoms out, so "resolution depth" is well-defined (§11). Identities and self-containing canons are the two inert witness classes.

**Two mechanical footnotes.** Membership and difference on node sequences are multiset-wise; node order is used only by reverse-replace's contiguity requirement, and otherwise retained — order and insertion position are strategy degrees of freedom, part of the witness. And a replace never empties `ν_A`: both sides of a correspondence carry content, so there is always at least one node left. (A replace may be _exhibited_ as an interleaving of removals and insertions — presentational only; the licence is the correspondence, never the band alone.)

## 7. One rule, two licences

Replace is the only rule; what differs is what licenses it. Two orthogonal licences:

- **Witnessed** (canon-mode: expand/contract). Preserves `σ(ν_A)` exactly — granularity changes at constant content, band unchanged. Licensed by `M` alone; blind to any goal.
- **Evidenced targeting** (denotation/connotation-mode). Moves `σ(ν_A)` toward `σ(ν_B)`. Licensed by a held correspondence _and_ scoped by the misfit region (§6's clause: strictly decrease the misfit mass).

Different invariant (content preserved vs content moved), different source (memory alone vs memory + goal) — two licences on one rule, not two rules.

**Done may arrive early.** Done is _value_-equality, not node-equality (§8), so a run can finish while its node list still looks unlike the goal's — pending nodes are witness structure, not unfinished work.

**The graph picture.** Read memory as a graph: values are vertices; each held kline is an edge between its signature and its witness, traversable in either direction from wherever the derivation has arrived. Canon evidence connects sequences of identical content (expand/contract = the two directions); misfit evidence connects _different_ content (shed/adopt/traverse). A derivation is a path in this graph, relativised to what is held. The path _is_ the semantics — which is why "done by blind alignment" is unreachable by construction: witnessed moves never change content, so no amount of them can close a content gap.

**Terminals are targeting-closed, not rule-closed.** An Identity _relationship_ is done, yet the identity kline's own node may still expand under a held well-founded witness:

```text
s:[s]  →  s:[a,b]        # the claim made explicit; the identity turned canon
```

There is no "retain" move; targeting-closure is just the S1 row of the licensing table, not a rule.

## 8. Endings, progress, termination

A derivation ends at **done** or **stuck**, or is **abandoned** by strategy:

- **Done** — `fit(C(A,B))` is S1: the relationship holds. Remember the goal is value-equality, `σ(ν_A) = σ(ν_B)` — an Identity relationship is done with `ν_A ≠ ν_B`, e.g. B holding A's whole content as one node.
- **Stuck** — not done, and no licensed targeting move. Two reachable conditions, both the _ask_: **no goal** — nothing to scope against (the ask at entry; candidate selection is §10); or **no connection** — a goal is held, the misfit region is non-empty, and no held correspondence licenses a replace into it, nor does any slot walk arrive (§10): nothing in `M` connects A's misfit to the goal's content, directly or through the correspondence graph. This is relative non-existence — the honest outcome when the semantic bridge is missing. Witnessed moves never end a derivation (they preserve the band and cannot reach done); their only use is granularity exposure, and spending them is strategy (bound T2). Replace cannot strand a derivation mid-run: both sides carry content, so `ν_A` never empties.
- **Abandoned** — not an ending the rules produce: strategy halts or re-targets a run mid-derivation (§10), e.g. when graded effort falls (§11).

Licences are permissive in one sense: a correspondence may itself be an ungrounded claim — S3 evidence is a _promise_, not a fact — and the derivation follows it faithfully. Weighing promises is protocol (ratification, §10, §13), not the rule system's.

**Termination, two statements.**

- **(T1)** Any run of targeting replaces from `A₀` terminates in at most `Δ₀ = |σ(ν_{A₀}) Δ σ(ν_B)|` steps. Each licensed replace strictly decreases the misfit mass (the scoping clause), and a single replace may move several atoms at once — steps are evidence-sized, the bound atom-wise, both computable. This is the natural unit for step budgets. Regressive or circular targeting is not merely unlikely, it is _unlicensable_: a replace that does not shrink the misfit mass is not a targeting move.
- **(T2)** Witnessed replaces preserve content and can cycle — `bc` expands to `[b,c]` and contracts back against the same held canon, at constant band, forever — and slot-wise traversals (§10) can wander the correspondence graph at constant misfit mass. Termination of mixed derivations is therefore a **strategy property**: bound witnessed runs (e.g. no expand-after-contract of the same witness) and bound traversals by no-revisit — each held signature consumed at most once per slot run; `M` is finite. Monotonicity of the graded measure is likewise a strategy invariant, not a theorem about arbitrary derivations.

**Confluence — renounced, deliberately.** The order of derivation changes what is grounded first, and the reachable S1 depends on the path. Path-dependence is not a defect to repair; it is the learning phenomenon. (If you come from term rewriting: this is the opposite of the usual desideratum. Kalvin's paths are the point.)

**Decidability — in principle.** The value space is finite and licensed expansion terminates (§6), so the states reachable from `A₀` are finite, and whether a derivation exists is decidable in principle. The gap between that and any affordable search is exactly where cogitation, study and scaffolding live.

## 9. What a derivation proves

**Done proves `σ(ν_A) = σ(ν_B)`** — the queued claim's content equals held content — with the final node sequence as the witness: a constructive existence proof _within and through what is held_. Every step was licensed by a correspondence, so the path itself is carried as evidence. In solver terms: each held kline is a constraint, each correspondence an edge, each licensed replace a resolution step along one; done is a constructive existence proof, and stuck is relative non-existence — nothing in `M` connects. Done does **not** prove A's own head-claim, nor the truth of the correspondences followed — S3 evidence is a promise; weighing promises is protocol and strategy (§10, §13).

**Feedback.** `fit(C(Aᵢ, B))` is graded at each state, and the _rate of change_ of that grade is tracked over steps — telling Kalvin whether its effort is increasingly or decreasingly significant. These steer the derivation; they are not part of the rule set.

### The worked example: `(what did Mary have) WDMH => MHALL`

Atoms `w, d, m, h, a, l` (word bits). Role values such as `o` live outside them, disjoint. Held in memory:

```text
mhall:[m,h,a,l,l]     canon — the rhyme (the answer)
dh:[d,h]              canon — "did have"
all:[a,l,l]           canon — the object phrase
dh:[h]                denotation — "did have" → "had"; its gap {d} is exactly what drops
w:[o]                 connotation — the question word claims the object role
all:[o]               connotation — the object phrase claims the object role
m:[m]                 identity
```

Queue `A₀ = wdmh:[w, dh, m]` — the question, itself a canon. Declared goal `B = mhall` (the KScript `WDMH => MHALL`). The relationship:

```text
C = wdmh:[m,h,a,l,l]          # head = σ([w,dh,m]) = {w,d,h,m}; nodes = B's
gap {w,d}, excess {a,l}       # Under+over; misfit mass Δ₀ = 4
```

Two evidenced replaces finish it:

| step       | move                  | licensed by                                                           | state of `ν_A`    | `σ(ν_A)` | misfit mass  |
| ---------- | --------------------- | --------------------------------------------------------------------- | ----------------- | -------- | ------------ |
| entry      | —                     | —                                                                     | `[w, dh, m]`      | `wdhm`   | 4            |
| 1 — verb   | `dh ⇉ [h]` forward    | denotation `dh:[h]` (shed mode: `d` is its gap)                       | `[w, h, m]`       | `whm`    | 3            |
| 2 — object | `w ⇉ [a,l,l]` forward | composed correspondence `w:[a,l,l]` (traverse: `{w}` out, `{a,l}` in) | `[h, m, a, l, l]` | `mhall`  | 0 — **done** |

No single held kline connects `w` to `a,l`, so step 2's licence had to be _built_ first, by a slot walk (§10): a side derivation queued as `w:[w]`, run with no goal, travelling the correspondence graph by occurrence alone:

```text
w:[w] ⊢ w:[o]        forward on w:[o]        (traverse)
      ⊢ w:[all]      reverse on all:[o]      (arrived at o, read the answer-side kline backwards)
      ⊢ w:[a,l,l]    forward on all:[a,l,l]  (expand — granularity set freely at arrival)
```

The arrival state `w:[a,l,l]` overlaps the goal's excess `{a,l}` — that is the walk's ending — and it is absorbed as the **composed correspondence**, evidence the main line then consumes. The subject `m` never moved: `m:[m]` is identity, inert; Mary carries over untouched.

End state `wdmh:[h,m,a,l,l]`: relationship Canon — done, in two targeting replaces under a bound of four, every step licensed by held or composed evidence. The witness carries the chain — that is Kalvin _knowing_ what Mary had, not copying it.

**The counterfactuals.** Had the connotations not been held, no replace reaches the object gap: stuck — the misfit _asks_, and ungrounded proposals follow under strategy control. Had `mhall` itself not been held, there is no goal to check done against: the ask from the other side.

**The price.** The end state's _own_ fit is still Under+over (gap `{w,d}`): the claim is answered, not grounded — grounding is protocol (§13), where the traversed pair `w:[o]`, `all:[o]` countersigns into a standing one-hop licence. Priced (§11): the slot walk crossed two unratified edges, and the consuming replace crosses one more, so `a,l,l` enter at acquisition depth 3:

```text
h, m entered at depth 0 (held material); a, l, l at depth 3 (two
unratified walk edges + the consuming replace):
Ĥ = (0 + 0 + 3 + 3 + 3) / 5 = 9/5      # atom-weighted over the node list [h, m, a, l, l]
done grade: J = 1, so γ = δ^(9/5) < 1  # known through promises
```

The consuming replace _dips_ γ — J rises from 1/2 to 1 just as `Ĥ` bites (with δ = ½: γ goes 1/2 → ≈ 0.29): the price signal. After the countersign, a re-derivation through the standing licence crosses only ratified edges, costs nothing, and γ reaches 1: consolidated. Read from the answer side (`MHALL => WDMH`), the same correspondences license the mirror derivation — the klines are direction-free; arrival orients them.

## 10. Strategy — the cogitation loop

§§6–9 fixed the parameters of a derivation; this section chooses them, step after step:

```text
        ┌───────────────────────────────────────────────────────┐
        │                                                       │
        ▼                                                       │
   select a hop ──► derive to an ending ──► absorb result into memory
                                                                │
                                  reenter: the output queues as the next hop's input
```

Each phase is strategy. The rule system of §§6–9 constrains what any of it may do, never what it must.

**Selection.** Selection chooses the next hop, never the final goal. A held kline `K = t:ν_K` is **selectable** for queued `A` when `t ∈ ν_A` — K's signature occurs as a node of A: A already references what K is. That clause is the forward half of §6's licence: to select K is to be licensed to replace its signature by its witness. It compounds into a ratchet — each replace's arrival puts new nodes into `ν_A`, making their klines selectable next. _The path is the guard, not the point._ The goal `B` is never selected: it is declared (KScript `=>`, §13) or supplied by reentry, it scopes the misfit region, and it is _checked_ at done. S3 connotations are selectable as evidence; an Unknown has no second side to offer; weighing claims is protocol.

**Two overlaps, neither implying the other.** _Content overlap_ `σ(ν_A) ∧ σ(ν_B) ≠ ∅` — exactly what relationship-S2 asserts — and _signature-in-node_ `t ∈ ν_A` — the selection clause. Examples of the independence:

```text
A = abc:[a],  B = x:[c,a]    → Overfit relationship (S2), yet x neither occurs in nor overlaps A's nodes
B = x:[y],    A = abc:[x]    → x occurs in A's nodes, yet the relationship is S3
```

Selection requires the second; the band routes by the first.

**Slot derivation.** A misfit decomposes per node: each node of `ν_A` carrying a gap atom is a **slot**, seeking the goal's excess `e = σ(ν_B) ∖ σ(ν_A)`. A slot with a licensed replace fires it. A slot with none — the misfit _asks_ at that node — may be **walked**: queue `n:[n]` and derive with no goal. The walk's licence is mere occurrence: any held correspondence, either side occurring in the walk's nodes, replaces it. The scoping clause does not apply — the walk travels the correspondence graph, whose role nodes are orthogonal to both contents. A goal-less derivation has no done; its endings are **arrival** — `σ(ν) ∧ e ≠ ∅`, with canon-mode replaces then setting granularity freely — or **stuck** — no unvisited correspondence occurs: nothing in `M` bridges the node to the excess, the ask localised to one slot. Each hop writes its output to STM, and the walk's end state is absorbed as the **composed correspondence**: signature `n`, witness the arrival, acquisition depth the edges crossed (§11). The main line then replaces `n ⇉ ν_arrived` on it — targeting-licensed because the arrival's content is excess and the misfit mass strictly decreases. The walk is how a missing licence is built: no single kline connects `w` to `a,l`; two connotations through `o` do; the absorbed `w:[a,l,l]` is that connection, earned.

**The progressive path.** The walk is the evidence-builder: each hop's output is written to STM — inserted nodes as single-node misfits, connotation witnesses — and the absorbed end state is the composed correspondence the main line consumes (the worked example, §9). Evidence is literally the work undertaken by the progressive path; hops matter because `M` grows between them. When the relationship reaches S2, ordinary evidenced targeting takes over.

**Bounds.** Three numbers, all strategy parameters: the **targeting budget** — T1 bounds any targeting run by `Δ₀`, so a budget at or above it never binds mid-run; the **witnessed-run and traversal bounds** — T2's requirements: no expand-after-contract of the same witness, no revisit of consumed signatures; and the **hop ceiling** — the reentry depth, below.

**Reentry.** Derivations compose. Hop `k` runs under parameters `(M_k, B_k)` — the goal `B_k`, or none for a slot walk; its end state — done, arrival, or stuck — queues as hop `k+1`'s input, and memory may grow between hops (`M_{k+1} ⊇ M_k`, by STM writes — the growth is the evidence accumulating). So successive hops are not derivations of one fixed system: propose from a proposal, one hop further out, bounded by the hop ceiling. Hop order is the only time the system has; if a time axis is wanted, it is this order and nothing else. Re-targeting mid-derivation — abandoning a run whose graded effort is falling (§11) and selecting anew — is likewise a strategy move, not a rule.

**Outside the system.** Escalation and ratification are protocol: countersigning (`==`) holds reciprocal connotation pairs as ratified — the algebra provides the shape, the protocol the commitment. The queue itself — which klines are admitted for cogitation, and in what order — belongs to the harness, not the system.

## 11. Measurement

**The band order.** The bands are derived from shape (§4); this section adds one axiom: they are **ordered by significance**, `S1 > S2 > S3 > S4`, shapes within a band unordered. The predicate is observer-independent — given the same held memory, every agent classifies alike — so a band never needs to be exchanged.

Two band attachments are in play: a kline's **own band** — `fit(s, ν)` on itself, the claim it makes standing alone — and a **relationship band** — `fit(C(A,B))`, what the pair achieves. The first is what a kline asserts; the second is what a derivation establishes or fails to.

**Graded distance.** `γ(A, B)` is fixed, not free — four requirements force one form:

```text
γ(A, B) = J(σ(ν_A), σ(ν_B)) · δ^(D̄ + Ĥ)       where  J(x, y) = |x ∧ y| / |x ∨ y|
```

Read it as: **overlap × discount^(granularity + promises)**.

- `J` is the **depth-free core**: the Jaccard overlap of the two contents — symmetric, 0 exactly at content-disjointness, 1 exactly at value-equality. Example: `J({a,b,c}, {a,b,c,d,e}) = 3/5`. Why Jaccard and not something simpler? Per-slot accountedness `α(n) = |n ∧ σ(ν_B)| / |n|`, averaged, yields A's coverage fraction — which reads 1 whenever A's content sits wholly inside B's, i.e. on Under/Overfit pairs that are still S2. A measure that scores "perfect" on an S2 pair is band-inconsistent; B's excess must be weighed too, and the symmetric form that weighs both sides is Jaccard.
- `D̄` is the **mean resolution depth** of A's content — granularity: how expanded the content is (0 for content held as itself; `[abc]` → depth 0, `[a,b,c]` → depth 1). An atom-weighted mean over the node list: each node weighed by its size `|n|`. Well-defined because licensed expansion terminates (§6).
- `Ĥ` is the **mean acquisition depth** — provenance: how many unratified correspondence edges were crossed to bring A's content in, averaged the same atom-weighted way. It is **carried, not computed**: content present at entry is 0; content entering by a replace on evidence `K` records `Ĥ(K) + 1` through an unratified edge, `Ĥ(K)` through a ratified one — grounded correspondences cost nothing; shed enters nothing; canon-mode replaces move granularity, not content. Absorbed klines carry their recorded depths, and consuming one composes with its own. No distance computable from `(ν_A, M)` could price this: the grounded goal holds the arrived atoms, so a structural route to them is always short, and consumption leaves no trace in `ν_A`. As with the chosen witness (§2), the acquisition chain is a fact the algebra forgets and memory carries.
- `δ ∈ (0,1)` is the strategy's knob, the only one — the two depths share it deliberately, both denominated in edges.

`J` says _how close_; `δ^(D̄ + Ĥ)` says _how hard-won_ — granularity and promises priced alike. γ is directional by design, grading this derivation's effort toward its goal; B's depths are B's own derivation's problem. γ is **path-dependent by design**: won and given knowledge with the same `(s, ν)` grade differently — the record is part of what memory holds.

Four properties, each doing a job:

- **Band-consistency.** γ is 0 exactly at content-disjointness and maximal only at value-equality — both ends are `J`'s; the depths only scale down.
- **Granularity-invariance.** Witnessed moves move γ only through `D̄`, never through recomposition: the atom-weighted mean is blind to how A's content is sliced into slots (`[abc]` and `[a,b,c]` at the same depths score alike). An unweighted per-slot mean would violate this — expansion alone could raise it at constant content and constant depth.
- **Granularity-monotonicity.** Expand strictly increases `D̄`, so strictly decreases γ; contract the reverse. This is what makes gratuitous expansion detectable.
- **Provenance-monotonicity.** Unratified acquisition strictly increases `Ĥ`; nothing in a derivation lowers it — only ratification (protocol, §13) or re-derivation through ratified licences does. This is what makes promise-stacking detectable. Consuming unratified evidence can lower γ even as `J` rises — the step that wins the answer dips (the worked example: J 1/2 → 1 while γ falls 1/2 → ≈ 0.29 at δ = ½) — the price signal steering strategy toward ratified standing licences.

Both monotonicities are strategy invariants, not theorems about arbitrary derivations (T2): a mixed derivation may wander against the gradient, and the strategy declines to.

**Rate of change** per step is defined only at this level — a four-band predicate has no useful derivative — and is the signal cogitation's feedback acts on.

**Exchange.** The graded value travels in a KValue (CONTEXT.md) as the sender's assessment, and the acquisition record travels with the kline that earned it — depths are part of what memory holds. Bands need not travel: they are recomputable from structure.

## 12. Terminology

_Significance_ is the value; _rationalisation_ is the process that produces and consumes it. (Not: "significance is the value Kalvin directly equates to rationalisation".) Understanding, informally, is high significance attained and held.

## 13. What this document does not cover

- **Tier mechanics** — what writes STM, what promotes LTM, how Frame attention shifts: relations over `M` defined in CONTEXT.md and consumed by selection (§10).
- **The multi-agent loop** — trainer, trainee, supervisor; escalation when cogitation yields no reply. Protocol above the system.
- **KScript tokens** — surface syntax declaring intent; `fit` may or may not satisfy the declared intent. Note `=>` declares composition and supplies the goal for the done-check — it is _not_ a licence; licences are correspondence klines (§6); the result is a Canon only if the §4 classifier says so; a bare signature is the ask — stuck at S4. See `kalvin-symbolic.md` §5.
- **Countersigning** — a protocol commitment (reciprocal connotation pairs held as ratified); the algebra provides the shape, the protocol the commitment. Ratifying a traversed pair promotes it to a standing one-hop licence — memory compounds its evidence.
