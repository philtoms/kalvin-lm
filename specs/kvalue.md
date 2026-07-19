# KValue Specification

## Overview

A KValue is the unit of exchange between participants — a **KLine** (objective
structure) paired with a **significance** (the sender's assessment of it).
KLines are what Kalvin stores; KValues are what participants send and what
rationalisation consumes and emits. Storage is objective-only; significance is
re-derived on retrieval and never persisted.

## Dependencies

This spec depends on the following concepts, defined elsewhere:

### KLine (@kline spec)

- The objective structure: a signature and a nodes list. Stored in memory;
  equal by signature + nodes.

### Significance (@CONTEXT.md §Significance)

- A participant's subjective assessment on the S1–S4 scale. Realised as an
  integer: either a **band-representative value** (asserted by a producer)
  or a **computed value** (produced by `expand()`).

### Model (@model spec §Significance Semantics)

- Owns the significance constants `D_MAX` and `MASK64`, the inversion rule,
  the boundaries, and the **band-representative values** this spec consumes.
- Provides the structural predicates (`is_identity`, `is_canon`,
  `is_countersigned`) used to re-derive significance on retrieval. The model
  also exposes `is_s1` — `is_canon OR is_countersigned` — but the retrieval
  cascade does not invoke it directly (see §Retrieval).

### Structural Relationship (@CONTEXT.md §Structural Relationship)

- The closed set of structural relationships (COUNTERSIGNS, CANONIZES, CONNOTES,
  DENOTES, IDENTITY) produced by the written relational token.

## Definition

A KValue consists of:

| Field       | Type    | Description                                                                          |
| ----------- | ------- | ----------------------------------------------------------------------------------- |
| kline       | KLine   | The objective structure (signature + nodes). Immutable.                             |
| significance | uint64 | The sender's assessment of the kline. See §Significance.                            |

The KLine is the data; the significance is an assessment of it. They are
independent: two participants may produce KValues over the **same** KLine with
**different** significance. This is the asymmetry the KValue exists to close.

## Construction

A KValue is constructed from a KLine and a significance:

```
KValue(kline, significance)
```

- `kline` — required, a KLine.
- `significance` — required, a uint64. There is no default and no "unset"
  value: every KValue carries a significance set by its producer.

## Equality and Hashing

KValue **identity is structural**: two KValues are equal, and hash equally,
when their KLines are equal. Significance is an assessment, not identity, and
is excluded from both — a KLine may be assessed differently by different
participants without becoming a different KValue.

This mirrors the KLine's own treatment of non-structural attributes and keeps
structural matching (e.g. auto-countersign, sender maps) unchanged.

## Significance

A KValue's `significance` is one of two kinds of integer, both on the same
inverted scale (higher = more grounded):

1. **Band-representative value** — one of four fixed constants, asserted by a
   producer that declares a band rather than computing a distance. The values
   and their derivation are owned by @model spec §Significance Semantics ›
   Band-representative Values. The producer maps an entry's structural relationship
   (@CONTEXT.md §Structural Relationship) to its band as follows:

   | Structural Relationship       | Band |
   | ---------------------- | ---- |
   | COUNTERSIGNS          | S1   |
   | CANONIZES              | S2   |
   | CONNOTES / DENOTES | S3   |
   | IDENTITY               | S4   |

2. **Computed value** — a full 64-bit inverted distance produced by
   `expand()` (Kalvin's method). Any value within a band, not just the
   representative.

A consumer cannot, from the value alone, tell which producer kind it came
from; both are plain integers on the same scale. How Kalvin consumes the
significance carried on an inbound KValue is **out of scope** for this spec
(see §Out of Scope).

## Producers

Every participant that emits a KValue sets its significance:

| Producer            | KLine source               | significance                                |
| ------------------- | -------------------------- | ------------------------------------------- |
| Compiler            | compiled entry             | band-representative, derived from the entry's structural relationship (COUNTERSIGNS→S1, CANONIZES→S2, CONNOTES/DENOTES→S3, IDENTITY→S4) |
| Countersign         | the reciprocal kline       | `D_MAX` (S1) — the act of countersigning is an S1 ratification             |
| Cogitation / expand | expansion proposal kline   | the computed value yielded by `expand()`    |
| Kalvin fast path    | the inbound kline          | the computed/structural value Kalvin assigns during rationalisation |

### KP-1 — Compiler attaches band-representative significance

The compiler's existing per-entry structural relationship (today carried as unspec'd
debug provenance) becomes the source of the significance: each compiled KValue
receives the band-representative value for its structural relationship.

### KP-2 — Countersign produces an S1 KValue

The reciprocal produced by a countersign carries `D_MAX`. Countersign does not
go through the compiler; its significance is fixed at S1 by definition of the
act.

### KP-3 — Cogitation proposals carry the computed significance

Expansion proposals emitted by the cogitator carry the `expand()`-computed
significance for that proposal, not a band-representative value.

## Storage and Retrieval

### Storage is objective-only

The Model stores KLines. The KValue's significance is **never persisted**:
the codec persists `{signature, nodes}` only. This honours the glossary — the
assessment is re-made, not recorded.

### Retrieval re-derives significance

Materialising a KValue from a stored KLine re-derives its significance from
the KLine's structure via `structural_significance` (@expand), composed from
the structural predicates (@kline spec) and the node count:

| Structure (composed from predicates)                                  | significance |
| --------------------------------------------------------------------- | ------------ |
| identity with empty nodes `{A:[]}` (the identity ask)                 | S4           |
| identity with nodes (self-referential `{A:[A]}` or a compound-word)   | S1           |
| canon `{AB:[A, B]}`                                                    | S1           |
| single-node non-identity relationship `{A:[B]}` (CONNOTES / DENOTES)  | S3           |
| multi-node misfit (underfit / overfit / misfit)                       | S2           |

The integer for each band is the band-representative value owned by @model
spec §Significance Semantics › Band-representative Values.

The sole model-state adjustment to this structural band is the
**countersigned upgrade**: a structurally-S2 misfit whose reciprocal
countersigner is present in the model upgrades to S1. This fork is applied
at the call site that needs model state (e.g. @agent spec §Phase 1b),
keeping `structural_significance` itself model-free. A retrieved KLine's
re-derived significance therefore reflects the model as it currently is —
consistent with significance being re-made rather than recorded.

### KV-1 — Re-derivation never yields "unset"

Every KLine resolves to exactly one band via the cascade above. There is no
KValue without a significance.

## Exchange — RationaliseEvent

The RationaliseEvent carries KValues, not bare KLines, and carries **no
separate significance field** — each KValue supplies its own.

```
RationaliseEvent:
  kind:     str       # "ground", "frame", "done"
  query:    KValue    # the inbound KValue (the sender's declared assessment)
  proposal: KValue    # Kalvin's assessment of it
```

### Two assessments, two KValues

`query` and `proposal` carry **independent** assessments of the same KLine:

- `query.significance` — the significance the **sender** declared
  (e.g. the trainer's band-representative value for a compiled entry).
- `proposal.significance` — the significance **Kalvin** assigns
  (a computed value on the fast/slow path, or a band value for ground/S4).

### KE-1 — Fast path wraps one KLine in two KValues

On the fast path the query and proposal wrap the **same immutable KLine**:
`query.kline is proposal.kline`. The two KValues differ only in significance.
No KLine is copied for the sake of significance — the copy is forbidden by
KLine immutability and unnecessary because the KLine is shared.

### KE-2 — Slow path carries the computed proposal

On the slow path (cogitation), `proposal.significance` is the value computed
by `expand()` for the expansion proposal; `query` is the original inbound
KValue.

### KE-3 — The event has no significance field

`RationaliseEvent` exposes significance only via `query.significance` and
`proposal.significance`. There is no top-level `significance` field.

### KE-4 — Consumers read KValue significance

Any consumer that previously read the event's significance field reads
`event.proposal.significance` (Kalvin's assessment) or
`event.query.significance` (the sender's declared assessment), as appropriate
to what it is deciding.

## What a KValue is Not

The following are explicitly **out of scope** for this spec:

- **Consumption of an inbound KValue's significance.** Rationalisation
  *does* consume the sender's declared significance — see @agent spec
  §Rationalisation (the significance-comparison gate). This spec owns only the
  exchange unit: it guarantees the declared significance is present and
  addressable on the inbound KValue; the consumption contract (how Kalvin
  combines a declared and a computed significance) is out of scope here.
- **KLine internals.** The KLine (signature, nodes, structural predicates,
  equality) is defined by the @kline spec. A KValue references a KLine; it
  does not redefine it.
- **The significance scale itself** (inversion, distance, boundaries). Owned
  by @model spec §Significance Semantics. This spec only consumes its
  constants and re-derivation predicates.
- **Wire format.** A KValue is an in-process exchange unit. Whether and how a
  significance survives a future remote (WebSocket) boundary is out of scope;
  today all participants that exchange KValues are embedded in-process.

## Test Matrix

| ID    | Criterion                                                                  | Origin ref |
| ----- | -------------------------------------------------------------------------- | ---------- |
| KV-1  | Construction requires both kline and significance (no default)             | §Construction |
| KV-2  | Equality ignores significance: same KLine, different significance → equal  | §Equality |
| KV-3  | Hash ignores significance: same KLine, different significance → equal hash | §Equality |
| KV-4  | Compiler attaches S1 for `==`, S2 for `=>`, S3 for `=`/`>`, S4 for identity | §KP-1 |
| KV-5  | Countersign reciprocal KValue carries `D_MAX`                              | §KP-2 |
| KV-6  | Cogitation proposal KValue carries the `expand()`-computed significance    | §KP-3 |
| KV-7  | Codec persists `{signature, nodes}` only; significance not serialised      | §Storage |
| KV-8  | Re-derivation: identity ask (empty nodes) → S4                               | §KV-1 |
| KV-9  | Re-derivation: grounded identity / canon → S1                                | §KV-1 |
| KV-10 | Re-derivation: S2 misfit with reciprocal countersigner present → S1          | §KV-1 |
| KV-11 | Re-derivation: single-node relationship → S3                                 | §KV-1 |
| KV-12 | Re-derivation never yields an unset significance                             | §KV-1 |
| KV-13 | Fast-path event: `query.kline is proposal.kline`, significances independent | §KE-1 |
| KV-14 | RationaliseEvent has no `significance` field                                | §KE-3 |
| KV-15 | Consumer reads `event.proposal.significance` (Kalvin's assessment)         | §KE-4 |

## Referenced By

- **Agent** (@agent spec §Events) — the RationaliseEvent query and proposal
  are KValues; rationalisation consumes KValues.
- **Model** (@model spec §Significance Semantics) — owns the
  band-representative values and the re-derivation predicates this spec uses.
- **KLine** (@kline spec) — the objective component of a KValue.
- **Harness** (@harness-server spec) — the trainer's submissions and
  countersigns are KValues.
