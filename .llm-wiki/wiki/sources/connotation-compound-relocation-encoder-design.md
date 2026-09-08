---
type: source
title: Relocating a compound across the sig/node boundary requires component plumbing
status: insight
category: compiler
created: 2026-09-08
updated: 2026-09-08
slug: connotation-compound-relocation-encoder-design
---

# Relocating a compound across the sig/node boundary requires component plumbing

When a compiler-level semantics change moves a synthesized compound (concatenation identifier) from the signature side of a kline to the node side (`AB:[B]` → `A:[AB]`), the compound's component words vanish from the entry: the string "AB" alone cannot be segmented back into A and B, and word values cannot be composed. The fix is to carry components explicitly — added `SymbolicEntry.concat: list[str] | None` (set only by CONNOTES/RCONNOTES emission) and `TokenEncoder._compose_concat(components, label)`, which resolves each component as a registered compound or an encoded word, OR-reduces, and registers the result so later references (e.g. `C > AB` after `AB` is registered) reuse the same value. Key invariants to verify: the concatenation never takes a word bit; `A < B` compiles identically to `B > A`; compound node value equals the registered compound signature (cross-reference with the MTS canon); any engine predicate encoding the old shape (e.g. [[concepts/kline]] containment direction in EngineState.is_connotation) must be mirrored or it silently rejects every kline of the new shape. Related: [[concepts/kvalue]], [[concepts/kscript-operator]].

*Category: compiler*

---
*Captured: 2026-09-08*

## Related

_Add links to related pages._
