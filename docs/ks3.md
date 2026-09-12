# Kalvin — Symbolic II, in plain terms

Status: draft. This is `ks2.md` restated for programmers: same content, same section numbers, less mathematical apparatus, more examples. **Every definition is quoted verbatim from ks2 in a blockquote**; the surrounding prose is framing — what the definition says in programmer terms, and how the pieces behave. ks2.md remains normative: where framing and quote seem to differ, the quote wins. Quoted text keeps ks2's internal cross-references (Def numbers, § numbers), which match this document's structure. `kalvin-symbolic.md` §5 remains normative for KScript surface syntax; CONTEXT.md for role names.

**The whole system in six sentences.** Kalvin's memory is a set of records — *klines* — each claiming that its head value is the composition of its node list. Composition is just set-union, so many different lists evaluate to the same head; *which list you chose* is the memory. A pure function grades any (head, list) pair into one of nine shapes, grouped into four bands from "exact" to "no connection". Thinking — *cogitation* — is a rewrite loop over one node list, using held klines as the rewrite rules, each step either re-expressing the same content more or less granularly, or moving it strictly closer to a goal kline's content; a run ends **done** when the contents match or **stuck** when nothing held connects them. A score γ — overlap, discounted by granularity and by unratified promises — grades every step, and its rate of change steers the loop. Strategy chooses what to derive next; measurement decides what it was worth.

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

**In programmer terms:** an atom is one bit in a fixed, small bit-space; a value is a bitmask over that space.

> **Definition 1 (atoms).** `A = {a₀ … a₃₀}` — a finite parameter set; only its finiteness is load-bearing. In the engine this is the word-bit space: one bit per distinct word. The u64 packing `(word_bit << 32) | bpe_token_id` is not literally a set of atoms; this algebra is a deliberate tidying of that encoding, and the tidying is what the formalisation builds on.

Reading Definition 1: the atom set is a *parameter* — 31 word bits in the engine — and only its finiteness matters to anything that follows. Say the atoms are `a`, `b`, `c` (think: three words). Then the values are all the bitmasks over them:

```text
∅, a, b, c, ab, ac, bc, abc
```

> **Definition 2 (values).** A **value** is a set of atoms: `V = 2^A`. Write `a` for the singleton value `{a}` and `abc` for `{a,b,c}`. The **empty value** is `∅`; the **full value** is `A`. Operations:
>
> - composition `v ∨ w = v ∪ w` — the whole is the sum of its parts;
> - overlap `v ∧ w = v ∩ w` — what two values share;
> - complement `¬v = A ∖ v`.
>
> The Boolean laws (commutativity, associativity, idempotence, distribution, `a ∧ b = ∅` for distinct atoms) hold by construction. They are consequences of the set definition, not axioms.

Which is to say: values are bitmasks, the three operations are OR, AND, and complement-within-mask — and you never have to *enforce* the Boolean laws; they fall out of the representation. Example computations:

```text
ab ∨ ac = abc        ab ∧ ac = a        ¬a = bc
```

## 2. Node sequences — the terms

**In programmer terms:** a node sequence is a list of non-empty values — plain list semantics: order retained, duplicates retained, concatenation the only structure.

> **Definition 3 (node sequence).** A node sequence `ν = [n₁ … nₖ]` is a member of `(V ∖ {∅})*`: order and multiplicity retained, no empty nodes. `V*` is the free monoid on `V` — the only free object in the system.

"Free monoid" is the mathematician's name for exactly the plain-list semantics above: lists with concatenation and no equations between them — `[a,b]` and `[b,a]` are different lists until something folds them.

> **Definition 4 (evaluation).** `signature_of : V* → V`:
>
> `signature_of([n₁ … nₖ]) = n₁ ∨ … ∨ nₖ` `signature_of([]) = ∅`
>
> It is a monoid homomorphism that forgets exactly order and multiplicity — nothing else. `V` identifies precisely what `signature_of` identifies. The architecture lives in the gap: **node sequences are the terms; values are what they evaluate to.**

In code, `signature_of` (shorthand `σ`) is a fold — and "monoid homomorphism" means nothing more than: the fold respects concatenation, `σ(ν₁ · ν₂) = σ(ν₁) ∨ σ(ν₂)`:

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

"Identifies precisely" cuts both ways: two lists evaluate to the same value *iff* they contain the same atoms — and any two such lists are folded to the same head.

Like `3 + 4` versus `7`: many expressions, one result. A calculator discards the expression; Kalvin's memory keeps it (§3) — *which* decomposition you hold is the content of memory. Note one extra forgetting the fold does silently: `[abc]` and `[a, b, c]` evaluate identically. The list remembers granularity; the value does not. That distinction becomes the expand/contract moves of §6 and the `D̄` term of §11.

## 3. Klines — claims and witnesses

**In programmer terms:** a kline is a pair — a head value and a node list — written `s:ν`:

```text
mhall:[m, h, a, l, l]
```

> **Definition 5 (kline).** A **kline** `s:ν` pairs a nonzero **signature** `s ∈ V` with a node sequence `ν`. A kline claims its signature as the composition of its nodes.

The claim may be exact or sloppy; §4 grades it.

> **Definition 6 (exactness).** `s:ν` is **exact** when `s = signature_of(ν)`. An exact, non-empty kline is a **witness**: a chosen decomposition of `s`. `signature_of` has no distinguished inverse. The trivial one, `s ↦ [s]`, always exists — its images are the identities, witnesses that carry no decomposition content; every other witness is a real choice. Nothing in `V` reconstructs which choice was made — that is what memory is for.

For `abc`, the witnesses include `[a,b,c]`, `[ab,c]`, `[a,bc]`, `[abc]`, `[c,a,b]`, … — and which one is held is the memory.

> **Definition 7 (memory).** A **memory** `M` is a finite set of klines. Two klines may share a signature — distinct claims, or distinct decompositions of the same value. A node may be the signature of another kline: nesting is by reference, and the reference graph may cycle — canon self-reference (`a:[a,a]`), countersign pairs (`a:[b]`, `b:[a]`) — for memory is unrestricted: any kline may be held. Cycles carry no decomposition content (§6). Which klines are _held_, and in what tier, are relations over `M` defined above the algebra (CONTEXT.md).
>
> A witness may repeat a node (`[l,l]`); the repetition is invisible to `signature_of` but retained — it is part of the chosen decomposition.

In programmer terms: `M` is a set of records whose fields may point at other records' heads, so memory is a graph, not a tree — and per Definition 7 the pointer graph is unrestricted, cycles included (they are inert, §6). Tier membership (STM, Frame, LTM) lives above this algebra — CONTEXT.md.

> **The central claim.** Kalvin's memory is a set of claims over a forgetting map. A kline claims its signature as the composition of its nodes; **Canon** is the claim exactly kept; **Underfit/Overfit** are the two directions a claim can miss while still being answered; the S3 shapes are claims nothing answers; **Unknown** makes no composition claim — nothing held, or nothing left — and lands S4. Significance grades the claim.

## 4. Coverage and the fit classifier

**In programmer terms:** `fit` is a pure function from a (value, list) pair to one of nine shapes, grouped into four bands — no state, no default case, every pair lands in exactly one shape.

> **Definition 8 (coverage).** A node `n` is **covered** by a value `s` when `n ∧ s ≠ ∅` — they share at least one atom. Coverage is overlap, not containment: a covered node may also carry atoms outside `s`.

> No new constructors appear beyond §3. The nine shapes are derived predicates — one total function on `(value, sequence)` pairs. A kline is one such pair (its own signature against its own nodes); the relationship kline of §5 is another; one classifier serves both.

> **Definition 9 (gap and excess).** For `(s, ν)` with `ν ≠ []`: the **gap** `g = s ∧ ¬signature_of(ν)` — atoms the signature claims beyond its nodes; the **excess** `e = signature_of(ν) ∧ ¬s` — atoms the nodes carry beyond the signature. Note `g = ∅` and `e = ∅` together hold iff `s = signature_of(ν)`.

In code:

```text
gap    g = s ∧ ¬σ(ν)     # atoms the signature claims but the nodes don't deliver
excess e = σ(ν) ∧ ¬s     # atoms the nodes deliver but the signature doesn't claim
```

Example: for `abc:[b, c, d]` — gap `a` (claimed, not delivered), excess `d` (delivered, not claimed).

> **Definition 10 (fit).** `fit : V × V* → Shape`. Cases in order; every pair matches exactly one:
>
> | #   | Condition                 | Shape       | Band |
> | --- | ------------------------- | ----------- | ---- |
> | 1   | `ν = []` or `s = ∅`       | Unknown     | S4   |
> | 2   | `ν = [s]`                 | Identity    | S1   |
> | 3   | `s = signature_of(ν)`     | Canon       | S1   |
> | 4   | not covered, `\|ν\| = 1`  | Connotation | S3   |
> | 5   | not covered, `\|ν\| > 1`  | No-fit      | S3   |
> | 6   | covered, `g ≠ ∅`, `e = ∅` | Underfit    | S2   |
> | 7   | covered, `g = ∅`, `e ≠ ∅` | Overfit     | S2   |
> | 8   | covered, `g ≠ ∅`, `e ≠ ∅` | Under+over  | S2   |
>
> The cases are disjoint by construction. Coverage is the primary split: a pair with no covered node forces `g = s ≠ ∅` and `e = signature_of(ν) ≠ ∅`, so cases 4–5 can never satisfy 6; exactness is caught at case 3, before the covered cases, which require a nonzero gap or excess. Case 1's disjunction is the no-claim case in both directions: `ν = []` — nothing held; `s = ∅` — nothing left. Replace never empties a node sequence (§6) — the empty head is an entry condition (the ask, §10), not a run outcome.

The same classifier as code — same order, first match wins:

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

The check order is the point, as the quote's second paragraph explains: an uncovered pair has `g = s` and `e = σ(ν)` both nonzero, so it can never satisfy the covered cases — S3 is decided before gap and excess are even consulted.

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

> Note the S3 rows: their gap _and_ excess are both nonzero, yet they are not Under+over — coverage decides first. That precedence is what keeps the partition disjoint.

> **Species.** **Denotation** is the single-node Underfit (`ab:[b]` — one covered node, gap only); **Connotation** is case 4. These are names of convenience for KScript (`=`, `>`/`<`); algebraically they are single-node instances of cases 6 and 4. Two single-node shapes are unnamed: the single-node Overfit `a:[ab]`, and the single-node Under+over `ab:[bc]` (covered on `b`, gap `a`, excess `c`).

> **Bands.** Derived, not asserted: S1 = cases 2–3 (exact), S2 = covered misfits, S3 = uncovered misfits, S4 = Unknown. Readings carry over: S1 — _I know that I know this_; S2 — _I infer this, but it does not yet fit_; S3 — _I recognise aspects of this, indirectly_; S4 — _I do not understand this at all_. The **ask** is structural, not declared: Unknown (`s:[]` — nothing held for the signature) is the ask's shape, the halt signal under which strategy generates ungrounded proposals (§10). No atom, mark, or decree is involved.

> **Invariance.** `fit` is insensitive to node order: it depends on `ν` only through `signature_of(ν)` and the node count. Duplicating a node changes the fit only when it crosses a count boundary (`a:[a]` Identity vs `a:[a,a]` Canon; `a:[b]` Connotation vs `a:[b,b]` No-fit); otherwise the repetition is witness structure invisible to classification.

## 5. The relationship kline

**In programmer terms:** to compare two klines, line up *what the first one actually holds* against *how the second one is decomposed*, and run the same classifier on that.

> **Definition 11 (pairwise).** For klines `A = s:ν_A` and `B = t:ν_B`, the **relationship kline** is `C(A,B) = signature_of(ν_A) : ν_B`. Its head is not claimed — it is _defined_ as what A's nodes evaluate to — so all of C's misfit-ness comes from B's side. `fit(C(A,B))` is the **structural relationship of A and B**.
>
> `fit` is one function with two readings. Applied to `(s, ν)` it grades a kline's own claim; applied to `(signature_of(ν_A), ν_B)` it grades two klines against each other. Neither reading is a special case of the other construction — both are arguments to the same classifier.

Worked pair. A holds `abc:[a,b,c]`; B holds `ab:[a,b,c]`:

```text
C(A,B) = abc:[a,b,c]        # head = σ([a,b,c]) = abc; nodes = B's
gap = ∅, excess = c          # Overfit — B's decomposition carries more than A holds
```

> Useful equivalence: `C(A,B)` is Canon iff `signature_of(ν_A) = signature_of(ν_B)` — the two klines hold the same value, differently decomposed.

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

> **Definition 12 (derivation).** A **derivation** rewrites the node sequence of a queued kline `A = s:ν` against one held goal `B = t:ν_B`. The relation is memory-relative: `A ⊢_{M,B} A′`, with the memory `M` (Def 7), the goal `B`, and the queue `A` as parameters. The signature `s` never changes — the claim is fixed; the content is rewritten. States `A₀ ⊢_{M,B} A₁ ⊢_{M,B} …` differ only in `ν`. Nothing in §§6–9 reads `s` — licenses, endings, bounds and grades read only the relationship, whose head `σ(ν_A)` is exact against `ν_A` at every state by construction. Within a hop, then, the derivation is the two sides of `C` — `σ(ν_A):ν_A` against `t:ν_B` — and the queued head rides along inert, mattering only beyond the hop: absorb, reentry (§10), the claim's grounding (§13).
>
> Membership and difference on node sequences are **multiset-wise**; sequence order is used only by contract's contiguity and otherwise retained for witness purposes. No rule reads the _queued_ kline's own fit — scoping reads the relationship `C(A,B)` (Def 11); mode and direction read the _evidence_ kline's own fit (Def 13).

Decode: the derivation's state is `ν_A`; every step is `rewrite(ν_A, K) → ν_A′`; the claim `s` is an immutable field nothing reads until after the hop. "Multiset-wise" means matching treats the list as a bag — except reverse-replace, which needs its witness as a *contiguous* run.

> **Definition 13 (one step).** `A = s:ν_A ⊢_{M,B} A′ = s:ν′` — one rule, instantiated by memory:
>
> ```text
> replace:   a held correspondence kline K = n:ν_K ∈ M and an occurrence in ν_A
>            matching one of K's two sides:
>   forward:   an occurrence of n  → replaced by ν_K
>   reverse:   an occurrence of ν_K (a contiguous block where K is a canon)
>                                    → replaced by [n]
> ```
>
> The terminals are inert as evidence: an Unknown (`n:[]`) has no second side; an Identity (`n:[n]`) replaces a node by itself. Every other held kline is a correspondence. K's own fit fixes the **mode**: a canon exacts granularity — forward is expand, reverse is contract, both σ(ν_A)-preserving; a covered misfit moves content by its gap and excess — forward sheds K's gap and adopts K's excess, reverse the mirror; an uncovered misfit is a traverse — disjoint atoms swap, either direction. **Direction is not a property of the kline**: a correspondence is read forward from its signature, reverse from its witness; only the side the derivation stands on occurs, so arrival orients the licence — the same kline read from the other side is the mirror derivation's licence. An edge is **ratified** or unratified — a tier relation over `M` (§10, §13). No rule of this section reads it; measurement does (§11).
>
> Canon evidence must be well-founded (`n ∉ ν_K`). The clause is exactly strong enough: a canon's nodes are atom-subsets of its head (`σ(ν_K) = n`), so an expansion cycle forces atom-equality at every step — a canon containing its own signature. Short of that, licensed expansion terminates and depth is well-defined (§11). Identities and self-containing canons are the two inert witness classes.
>
> A replace may be _exhibited_ as an interleaving of removals and insertions — a presentational device with no algebraic status: the licence is the correspondence, never the band alone. The interleaving never empties `ν_A` — both sides of a correspondence carry content. Node order and insertion position are retained for witness purposes and for later contract contiguity — strategy degrees of freedom, like ordering generally (Def 12).

The quote's mode sentence, tabulated — what the replace *does to content*, by the evidence's shape:

| K's own fit           | forward (`n → ν_K`)                    | reverse (`ν_K → [n]`)                |
| --------------------- | -------------------------------------- | ------------------------------------ |
| Canon (exact)         | **expand** — same content, finer grain | **contract** — coarser grain         |
| covered misfit (S2)   | sheds K's gap, adopts K's excess       | the mirror: adopts gap, sheds excess |
| uncovered misfit (S3) | **traverse** — disjoint atoms swap     | traverse, the other way              |

Examples. Canon `abc:[a,b,c]`: replacing node `abc` with `[a,b,c]` changes nothing in content — pure granularity. Denotation `dh:[h]` (gap `d`): forward on it, `dh → [h]`, drops exactly the atom `d` — the gap is *what drops*. Connotation `w:[o]` (disjoint): forward swaps `w` out, `o` in. On well-foundedness: expand replaces a node with strictly finer content built from the same atoms, so the only way an expansion chain can fail to terminate is a witness containing its own head — exactly what `n ∉ ν_K` excludes; with it, every expansion bottoms out and "depth" is well-defined (§11).

> **Definition 14 (licensing).** The relationship scopes; the evidence licenses. The band says where work remains; only a correspondence kline (Def 13) says what may move:
>
> | `fit(C(A,B))`            | Licensed targeting                             |
> | ------------------------ | ---------------------------------------------- |
> | S1 — Canon, Identity     | none — done                                    |
> | S2 — covered misfits     | replace, on the misfit region (scoping clause) |
> | S3 — Connotation, No-fit | replace — every node sits wholly in the misfit |
> | Unknown — S4             | none — stuck                                   |
>
> **Scoping clause.** A replace is targeting-licensed iff it strictly decreases the misfit mass `|σ(ν_A) Δ σ(ν_B)|` — the node replaced carries a gap atom, or the witness carries excess atoms, or both. Band-blind alignment is thereby unlicensable: a replace that touches only shared content, or grows the misfit, is not a targeting move however well evidenced. Canon-mode replaces (expand, contract) are witnessed moves — σ(ν_A)-preserving, band-preserving, licensed by `M` alone, needing nothing from this table. At S3 no node is covered, so replaces there move whole content and the route to overlap runs through the progressive path (§10). The S4 row is entered through A, not B: with no goal there is nothing to scope against — the ask at entry; with a goal and no licensed replace, the misfit _asks_ — nothing in `M` connects it (§8).

Decode: misfit mass is the number of atoms on exactly one side (§0's reading practice). The scoping clause turns "progress" into a strictly-decreasing counter that only targeting moves may decrement — which is precisely what T1's bound (§8) counts down.

## 7. One rule, two licences

> Replace is the only rule; what differs is what licenses it. Canon-mode replaces (expand, contract) are **witnessed**: they preserve `σ(ν_A)` exactly — granularity changes at constant content, band unchanged — licensed by `M` alone, blind to any goal. Denotation- and connotation-mode replaces are **evidenced targeting**: they move `σ(ν_A)` toward `σ(ν_B)`, licensed by a held correspondence and scoped by the misfit region (Def 14). The two licences are orthogonal in invariant and source — two licences on one rule, not two rules.

Framed as the two licences:

- **Witnessed** (canon-mode: expand/contract). Preserves `σ(ν_A)` exactly — granularity changes at constant content, band unchanged. Licensed by `M` alone; blind to any goal.
- **Evidenced targeting** (denotation/connotation-mode). Moves `σ(ν_A)` toward `σ(ν_B)`. Licensed by a held correspondence *and* scoped by the misfit region (Def 14's clause: strictly decrease the misfit mass).

Different invariant (content preserved vs content moved), different source (memory alone vs memory + goal).

> Done may arrive early — value-equality can outrun node-equality — and is a legitimate ending; pending nodes are witness structure.

> Every content move is a claim read off a held kline: shedding one atom of a compound node is a denotation's claim — its gap is exactly what drops; adopting goal content is a connotation traversed or a canon expanded at an arrived node. Targeting is value-complete only modulo the correspondences memory supplies — without a correspondence there is no move, and the misfit asks (§8).

> Read model-theoretically: canon evidence generates a congruence on sequences — expand and contract its two directions — and the full evidence set generates a **correspondence graph**: held klines as edges between a signature and its witness, traversable in either direction from wherever the derivation has arrived. A derivation is a path in that graph, relativised to what is held. The path is the semantics: done by blind alignment is unreachable by construction.

Decode of the model-theoretic paragraph: memory is a graph walked in both directions — values are vertices, klines are edges. Canon edges join sequences of *identical* content ("congruence" = expand and contract are inverses, making those sequences interchangeable); misfit edges join *different* content (shed/adopt/traverse). A derivation is a path in this graph, and "done by blind alignment is unreachable by construction" because witnessed edges never change content — no amount of them closes a content gap.

> Terminals are **targeting-closed, not rule-closed**: an Identity relationship is done, yet the identity kline's own node may still expand under a held well-founded witness (`s:[s]` → `s:[a,b]`) — the claim made explicit, the identity turned canon. There is no retain move; targeting-closure is the S1 row of the licensing table, not a rule.

## 8. Endings, progress, termination

> **Definition 15 (endings).** A derivation ends at **done** or **stuck**, or is **abandoned** by strategy:
>
> - **Done** — `fit(C(A,B)) ∈ S1`: the relationship holds. The goal is **value-equality**, `σ(ν_A) = σ(ν_B)`, not node-equality — an Identity relationship is done with `ν_A ≠ ν_B`, B holding A's content as one node.
> - **Stuck** — not done, and no licensed targeting move. Witnessed moves never end a derivation: they preserve the band and cannot reach done; their only use is granularity exposure, and spending them is strategy (the witnessed-run bound, T2). Stuck has two reachable conditions, both the ask: **no goal** — nothing to scope against, the ask at entry (candidate selection is §10); or **no connection** — a goal is held, the misfit region is non-empty, and no held correspondence licenses a replace into it, nor does any slot walk arrive (Def 17): nothing in `M` connects A's misfit to the goal's content, directly or through the correspondence graph. Relative non-existence (§9), reachable at entry and mid-run alike — the honest outcome when the semantic bridge is missing. Replace cannot strand a derivation: both sides of a correspondence carry content, so `ν_A` never empties mid-run.
> - **Abandoned** — not an ending the rules produce: strategy halts or re-targets a run mid-derivation (§10), e.g. when graded effort falls (§11).
>
> Licences are permissive in one sense: a correspondence may itself be an ungrounded claim — S3 evidence is a promise, not a fact — and the derivation follows it faithfully. Weighing promises is protocol (ratification, §10, §13), not the rule system's.

> **Termination.** Two statements:
>
> - **(T1)** Any run of targeting replaces from `A₀` terminates in at most `Δ₀ = |σ(ν_{A₀}) Δ σ(ν_B)|` steps. Each licensed replace strictly decreases the misfit mass (the scoping clause), and a replace may move several atoms at once: steps are evidence-sized, the bound atom-wise, both computable — the natural unit for step budgets. Regressive and circular targeting is not merely unlikely but unlicensable: a replace that does not shrink the misfit mass is not a targeting move.
> - **(T2)** Witnessed replaces preserve `σ(ν_A)` and can cycle — `bc` expands to `[b,c]` and contracts back against the same held canon, at constant band, forever — and slot-wise traversals (§10) can wander the correspondence graph at constant misfit mass. Termination of mixed derivations is therefore a **strategy property**: bound witnessed runs (for instance, no expand-after-contract of the same witness) and bound traversals by no-revisit — each held signature consumed at most once per slot run; `M` is finite. Monotonicity of the graded measure is a strategy invariant, not a theorem about arbitrary derivations.

T1's `Δ₀` is decoded in §0's reading practice; T2's cycle example (`bc` ⇄ `[b,c]`, forever) is the reason the strategy layer must bound witnessed runs — the rules alone do not.

> **Confluence — renounced, deliberately.** The order of derivation changes what is grounded first, and the reachable S1 depends on the path. Path-dependence is not a defect to be repaired; it is the learning phenomenon.

> **Decidability — in principle.** `V` is finite and licensed expansion terminates (§6), so the states reachable from `A₀` are finite in number, and existence and non-existence of a derivation are decidable in principle; the tractability gap between that and any affordable search is exactly where cogitation, study and scaffolding live.

(If you come from term rewriting: renounced confluence is the opposite of the usual desideratum — Kalvin's paths are the point.)

## 9. What a derivation proves

> Done proves `σ(ν_A) = σ(ν_B)`: the queued claim's content is (value-)equal to held content, with the final node sequence as the witness — a constructive existence proof **within and through what is held**: every step of the witness was licensed by a correspondence, so the path itself is carried as evidence. The solver reading is §§6–9 restated: each held kline is a constraint, each correspondence an edge, each licensed replace a resolution step along one, S1 a constructive existence proof, and a stuck state relative non-existence — nothing in `M` connects. Done does **not** prove A's own head-claim, nor the truth of the correspondences followed — S3 evidence is a promise; weighing promises is protocol and strategy (§10, §13).

> **Feedback.** `fit(C(Aᵢ, B))` is graded at each state and its rate of change tracked over steps — telling Kalvin whether its effort is increasingly or decreasingly significant. These are strategy-level metrics: they steer the derivation; they are not part of the rule set.

### The worked example: `(what did Mary have) WDMH => MHALL`

*(ks2 §9's worked micro-example, expanded step by step.)* Atoms `w, d, m, h, a, l` (word bits). Role values such as `o` live outside them, disjoint. Held in memory:

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

No single held kline connects `w` to `a,l`, so step 2's licence had to be *built* first, by a slot walk (Def 17): a side derivation queued as `w:[w]`, run with no goal, travelling the correspondence graph by occurrence alone:

```text
w:[w] ⊢ w:[o]        forward on w:[o]        (traverse)
      ⊢ w:[all]      reverse on all:[o]      (arrived at o, read the answer-side kline backwards)
      ⊢ w:[a,l,l]    forward on all:[a,l,l]  (expand — granularity set freely at arrival)
```

The arrival state `w:[a,l,l]` overlaps the goal's excess `{a,l}` — that is the walk's ending — and it is absorbed as the **composed correspondence**, evidence the main line then consumes. The subject `m` never moved: `m:[m]` is identity, inert; Mary carries over untouched.

End state `wdmh:[h,m,a,l,l]`: relationship Canon — done, in two targeting replaces under a bound of four, every step licensed by held or composed evidence. The witness carries the chain — that is Kalvin *knowing* what Mary had, not copying it.

**The counterfactuals.** Had the connotations not been held, no replace reaches the object gap: stuck — the misfit *asks*, and ungrounded proposals follow under strategy control. Had `mhall` itself not been held, there is no goal to check done against: the ask from the other side.

**The price.** The end state's *own* fit is still Under+over (gap `{w,d}`): the claim is answered, not grounded — grounding is protocol (§13), where the traversed pair `w:[o]`, `all:[o]` countersigns into a standing one-hop licence. Priced (§11): the slot walk crossed two unratified edges, and the consuming replace crosses one more, so `a,l,l` enter at acquisition depth 3:

```text
h, m entered at depth 0 (held material); a, l, l at depth 3 (two
unratified walk edges + the consuming replace):
Ĥ = (0 + 0 + 3 + 3 + 3) / 5 = 9/5      # atom-weighted over the node list [h, m, a, l, l]
done grade: J = 1, so γ = δ^(9/5) < 1  # known through promises
```

The consuming replace *dips* γ — J rises from 1/2 to 1 just as `Ĥ` bites (with δ = ½: γ goes 1/2 → ≈ 0.29): the price signal. After the countersign, a re-derivation through the standing licence crosses only ratified edges, costs nothing, and γ reaches 1: consolidated. Read from the answer side (`MHALL => WDMH`), the same correspondences license the mirror derivation — the klines are direction-free; arrival orients them.

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

> **Definition 16 (selection).** Selection chooses the next hop, never the final goal. A held kline `K = t:ν_K` is **selectable** for queued `A` when `t ∈ ν_A` — K's signature occurs as a node of A: A already references what K is. That clause is the forward half of Def 13's licence: to select K is to be licensed to replace its signature by its witness. The guard compounds into a ratchet: each replace's arrival puts new nodes into `ν_A`, making their klines selectable next — the path is the guard, not the point. The derivation's goal `B` is never selected: it is declared (KScript `=>`, §13) or supplied by reentry, it scopes the misfit region (Def 14), and it is _checked_ at done. S3 connotations are selectable as evidence, and an Unknown has no second side to offer; weighing claims is protocol (§13). The two-overlap caveat stands: **content overlap** `σ(ν_A) ∧ σ(ν_B) ≠ ∅` — exactly what relationship-S2 asserts — and **signature-in-node** `t ∈ ν_A` — the selection clause, which is what lets correspondence propagate into A's nodes — imply neither the other. `A = abc:[a]` against `B = x:[c,a]` stands in an Overfit relationship (S2) while `x` neither occurs in nor overlaps A's nodes; `B = x:[y]` against `A = abc:[x]` occurs in A's nodes yet yields S3. Selection requires the second; the band routes by the first.

Decode: the selection clause is a guard, not a heuristic — "K's head occurs as a node of A" is the same occurrence Def 13's forward replace matches, so selecting is being licensed. The ratchet: each replace's arrivals become next-selectable nodes, so reachability grows as the run proceeds. Note the two-overlap caveat's punchline in programmer terms: *content overlap* (what relationship-S2 asserts) and *signature-in-node* (the selection clause) are independent conditions — selection requires the second, the band routes by the first.

> **Definition 17 (slot derivation).** A misfit decomposes per node: each node of `ν_A` carrying a gap atom is a **slot**, seeking the goal's excess `e = σ(ν_B) ∖ σ(ν_A)`. A slot with a licensed replace fires it (Def 13, Def 14). A slot with none — the misfit *asks* at that node — may be **walked**: queue `n:[n]` and derive `⊢_M` with no goal. The walk's licence is occurrence: any held correspondence, either side occurring in the walk's nodes, replaces it (Def 16's clause is the forward half). The scoping clause does not apply — the walk travels the correspondence graph, whose role nodes are orthogonal to both contents. A goal-less derivation has no done; its endings are **arrival** — `σ(ν) ∧ e ≠ ∅`, canon-mode replaces then setting granularity freely — or **stuck** — no unvisited correspondence occurs: nothing in `M` bridges the node to the excess, the ask localised to one slot. Each hop writes its output to STM, and the walk's end state is absorbed as the **composed correspondence**: signature `n`, witness the arrival, acquisition depth the edges crossed (§11). The main line replaces `n ⇉ ν_arrived` on it — evidence by Def 13, targeting-licensed because the arrival's content is excess and the misfit mass strictly decreases. The walk is how a missing licence is built: no single kline connects `w` to `a,l`, two connotations through `o` do, and the absorbed `w:[a,l,l]` is that connection, earned.

Decode: a walk is a goal-less side derivation whose only licence is occurrence — any held kline whose either side occurs in the walk's nodes can fire. It ends at *arrival* (content touching the goal's excess — the `w:[w] ⊢ … ⊢ w:[a,l,l]` chain in §9's worked example) or *stuck*, and its arrival is absorbed as the composed correspondence the main line consumes.

> **Progressive path.** Def 17's walk is the evidence-builder: each hop's output is written to STM — inserted nodes as single-node misfits, connotation witnesses — and the absorbed end state is the composed correspondence the main line consumes (the worked example, §9). Evidence is literally the work undertaken by the progressive path; hops matter because `M` grows between them. When the relationship reaches S2, ordinary evidenced targeting takes over.

> **Bounds.** Three numbers, all strategy parameters, each with a natural unit: the **targeting budget** — T1 bounds any targeting run by `Δ₀`, so a budget at or above it never binds mid-run; the **witnessed-run and traversal bounds** — T2's requirements, no expand-after-contract of the same witness and no-revisit of consumed signatures; and the **hop ceiling** — the reentry depth, below.

> **Reentry.** Derivations compose. Hop `k` runs under parameters `(M_k, B_k)` — the goal `B_k`, or none for a slot walk (Def 17); its end state — done, arrival, or stuck — queues as hop `k+1`'s input, and memory may grow between hops (`M_{k+1} ⊇ M_k`, by STM writes — the growth is the evidence accumulating), so successive hops are not derivations of one fixed system. Propose from a proposal, one hop further out, bounded by the hop ceiling. Hop order is the only time the system has; if a time axis is wanted, it is this order and nothing else. Re-targeting mid-derivation — abandoning a run whose graded effort is falling (§11) and selecting anew — is likewise a strategy move, not a rule.

Decode: hop order is the only clock — each hop's output is the next hop's input, and memory grows in between (`M_{k+1} ⊇ M_k`), so hops are not steps of one fixed derivation.

> **Outside the system.** Escalation and ratification are protocol: countersigning (`==`) holds reciprocal connotation pairs as ratified — the algebra provides the shape, the protocol the commitment. The queue itself — which klines are admitted for cogitation, and in what order — belongs to the harness, not the system.

## 11. Measurement

> **The band order.** The bands are derived from shape (Def 10); this section adds one axiom: they are **ordered by significance**, `S1 > S2 > S3 > S4`, the shapes within a band unordered. The predicate is observer-independent — given the same held memory, every agent classifies alike — so a band never needs to be exchanged.
>
> Two band attachments are in play: a kline's **own band** — `fit(s, ν)` on itself, the claim it makes standing alone — and a **relationship band** — `fit(C(A,B))`, what the pair achieves. The first is what a kline asserts; the second is what a derivation establishes or fails to.

> **Graded distance.** `γ(A, B)` is fixed, not free — four requirements force one form:
>
> `γ(A, B) = J(σ(ν_A), σ(ν_B)) · δ^(D̄ + Ĥ)` where `J(x, y) = |x ∧ y| / |x ∨ y|`

Read it as: **overlap × discount^(granularity + promises)**. The quote's development, then the decode of each term:

> `J` is the **depth-free core**: symmetric, 0 exactly at content-disjointness, 1 exactly at value-equality. It is forced: per-slot accountedness `α(n) = |n ∧ σ(ν_B)| / |n|`, composed atom-weighted (each slot weighed by `|n|`), yields the A-side coverage fraction `|σ(ν_A) ∧ σ(ν_B)| / |σ(ν_A)|` — which fails band-consistency's second clause (A's content may sit wholly inside B's — underfit, still S2 — at full coverage), so B's excess must be weighed too, and Jaccard is the result.

Decode: `J` is the Jaccard overlap of the two contents (`J({a,b,c}, {a,b,c,d,e}) = 3/5`). The forcing argument in plain steps: score each node by its overlap fraction with the goal and average, weighted by node size — you get A's coverage, which reads 1 ("perfect") whenever A's content sits wholly inside B's, i.e. on pairs that are still S2. A measure that scores perfect on a misfit is band-inconsistent; B's excess must be weighed too, and the symmetric form that weighs both sides is Jaccard.

> `D̄` is the **mean resolution depth** of A's content — granularity: the atom-weighted mean of the resolution depths at which A's atoms are held, 0 for content held as itself, well-defined because licensed expansion terminates (§6). `Ĥ` is the **mean acquisition depth** — provenance: the atom-weighted mean of the recorded acquisition depths of A's atoms, each counting the unratified correspondence edges crossed to bring it in. Acquisition depth is **carried, not computed**: content present at entry is 0; content entering by a replace on evidence `K` records `Ĥ(K) + 1` through an unratified edge, `Ĥ(K)` through a ratified one — grounded correspondences cost nothing; shed enters nothing; canon-mode replaces move granularity, not content. Absorbed klines carry their recorded depths, and consuming one composes with its own. No `(ν_A, M)`-computable distance could price this: the grounded goal holds the arrived atoms, so a structural route to them is always short, and consumption leaves no trace in `ν_A`. As with the chosen witness (§1), the acquisition chain is a fact the algebra forgets and memory carries.

Decode: `D̄` is how expanded A's content is — `[abc]` → depth 0, `[a,b,c]` → depth 1; "atom-weighted" = each node weighed by its size `|n|` (§9's Ĥ arithmetic works this way). `Ĥ` is bookkeeping attached to the kline, not recomputable from structure: entry content 0; entering by replace on evidence `K` costs `Ĥ(K)+1` unratified, `Ĥ(K)` ratified; shed enters nothing; canon-mode moves granularity only. That is why γ is path-dependent: won and given knowledge with the same `(s, ν)` grade differently.

> `δ ∈ (0,1)` is the strategy's knob, the only one — the two depths share it deliberately, both denominated in edges. `J` says _how close_; `δ^(D̄ + Ĥ)` says _how hard-won_ — granularity and promises priced alike. γ is directional by design, grading this derivation's effort toward its goal; B's depths are B's own derivation's problem. γ is **path-dependent by design**: won and given knowledge with the same `(s, ν)` grade differently — the record is part of what memory holds.

> - **Band-consistency.** `γ` is 0 exactly at content-disjointness and maximal only at value-equality. Both ends are `J`'s; the depths only scale down.
> - **Granularity-invariance.** Witnessed moves move `γ` only through `D̄`, never through recomposition: atom-weighted composition is blind to how A's content is sliced into slots. An unweighted per-slot mean violates this — expansion alone can raise it at constant content and constant depth.
> - **Granularity-monotonicity.** Expand strictly increases `D̄`, so strictly decreases `γ`; contract strictly decreases `D̄`, so increases `γ`. This is what makes gratuitous expansion detectable.
> - **Provenance-monotonicity.** Unratified acquisition strictly increases `Ĥ`; nothing in a derivation lowers it — only ratification (protocol, §13) or re-derivation through ratified licences does. This is what makes promise-stacking detectable, the provenance counterpart of gratuitous expansion. Consuming unratified evidence can lower `γ` even as `J` rises — the step that wins the answer dips — the price signal steering strategy toward ratified standing licences.

The last clause in numbers (§9's worked example, δ = ½): J rises 1/2 → 1 while γ falls 1/2 → ≈ 0.29 — the answer is won and the score *drops*: the price signal.

> Both monotonicities are strategy invariants, not theorems about arbitrary derivations (T2): a mixed derivation may wander against the gradient, and the strategy declines to.

> **Rate of change** per step is defined only at this level — a four-band predicate has no useful derivative — and is the signal cogitation's feedback acts on.
>
> **Exchange.** The graded value travels in a KValue (CONTEXT.md) as the sender's assessment, and the acquisition record travels with the kline that earned it — depths are part of what memory holds. Bands need not travel, for they are recomputable from structure.

## 12. Terminology

> _Significance_ is the value; _rationalisation_ is the process that produces and consumes it. (Not: "significance is the value Kalvin directly equates to rationalisation".) Understanding, informally, is high significance attained and held.

## 13. What this document does not cover

> - **Tier mechanics** — what writes STM, what promotes LTM, how Frame attention shifts: relations over `M` defined in CONTEXT.md and consumed by selection (Def 16).
> - **The multi-agent loop** — trainer, trainee, supervisor; escalation when cogitation yields no reply. Protocol above the system.
> - **KScript tokens** — surface syntax declaring intent; `fit` may or may not satisfy the declared intent (`=>` declares composition and supplies the goal for the done-check — it is not a licence; licences are correspondence klines, Def 13; the result is a Canon only if Def 10 case 3 fires; a bare signature is the ask — stuck at S4). See `kalvin-symbolic.md` §5.
> - **Countersigning** — a protocol commitment (reciprocal connotation pairs held as ratified); the algebra provides the shape, the protocol the commitment. Ratifying a traversed pair promotes it to a standing one-hop licence — memory compounds its evidence.
