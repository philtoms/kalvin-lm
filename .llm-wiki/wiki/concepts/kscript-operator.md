---
type: concept
title: KScript Operator
created: 2026-09-14
updated: 2026-09-14
---

# KScript Operator

The operator vocabulary of the KScript surface syntax (§13, `docs/kalvin-algebra.md`).

## Definition

| Token          | Structure                     | Band claim once solved |
| -------------- | ----------------------------- | ---------------------- |
| `a => b c d`   | `a:[b,c,d]`                   | S1 (Canon) or open S2  |
| `a == b`       | reciprocal pair `a:[b]`, `b:[a]` | S1 on ratification  |
| `a > b`        | `a:[b]` (CONNOTES)            | S3                     |
| `a < b`        | `b:[a]` (CONNOTES reversed)   | S3                     |
| `a = b`        | `ab:[b]` (DENOTES, compound signature) | S2          |
| `a` bare       | `a:[]`                        | S4 — the ask           |
| ask-annotated  | any signature marked as ask   | S4                     |

The surface token does not override algebraic classification: syntax specifies
an intended structure; the algebra determines the actual fit shape. `=>`
establishes a composition claim and supplies a goal for completion checking —
it is not itself a rewrite licence; licences are correspondence klines (§6).

## Compilation note

Relocating a compound across the sig/node boundary (`AB:[B]` → `A:[AB]`)
requires component plumbing: the string alone cannot be segmented back into
its words, and word values cannot be composed.
`SymbolicEntry.concat` + `TokenEncoder._compose_concat` carry components
explicitly, resolve each as a registered compound or encoded word, OR-reduce,
and register the result. The concatenation never takes a word bit; `A < B`
compiles identically to `B > A`.

## Links

- [[entities/kscript]] — the language
- [[entities/tokenencoder]] — value allocation and composition
- [[concepts/kvalue]] — the value algebra
- [[concepts/mts-multi-token-signature]] — compound signatures
- [[concepts/ask]] — the bare-annotation row
- [[sources/connotation-compound-relocation-encoder-design]] — the relocation plumbing
