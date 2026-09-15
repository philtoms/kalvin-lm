---
type: source
title: "Operator semantics swaps: move shapes or move names — the countersign test decides"
status: insight
category: kscript
created: 2026-09-15
updated: 2026-09-15
slug: shape-swap-vs-name-swap-operator-semantics
---

# Operator semantics swaps: move shapes or move names — the countersign test decides

Third flip of the connotes/denotes structures, and the two prior variants illuminate a design tension worth remembering.

**Timeline:** pre-2026-09-08: `>` CONNOTES → compound AB:[B], `=` DENOTES → plain A:[B] (the shapes now restored). 09-08: CONNOTES → compound-in-node-slot A:[AB]. 09-09 (8d3e9d2^): CONNOTES → plain A:[B], DENOTES → compound AB:[B]. 09-15 (branch `dialogue`, 8d3e9d2): names moved instead — `=` renamed CONNOTES (compound), `>` renamed DENOTES (plain), lexer/parser/token rebound. 09-15 (this change, main): shapes moved with symbol names fixed — `>` CONNOTES → AB:[B], `<` RCONNOTES → BA:[A], `=` DENOTES → A:[B].

**The deciding semantics:** "countersigning is a pairwise denotes". With denotes as the plain shape, the `==` halves A:[B], B:[A] are each a denotes — mutual denotation as ratification — and [[kalvin/model.py]]'s is_countersigned (reciprocal single-node lookup) then correctly ratifies mutual `=` declarations to S1 with no code change. Under the name-swap variant this identity is obscured.

**Two coherent implementation strategies for such a swap:**
1. Move shapes, fix names (this change): emission branches in ast_emitter swap their output; kline.py's is_connotation/is_denotation swap which coverage they assert (names follow the shapes' compiling operator); band_significance stamps swap to keep Target Significance agreeing with structural sig_level; the S3-bridge machinery (cogitator.connotate → denotate, reentry, expand_fit, pivot_fill, expand) renames to follow the plain shape's new name. Scripts/fixtures keep their op labels (they record source forms) but re-interpret.
2. Move names, fix shapes (8d3e9d2): rebinding lexer/parser/token only — everything downstream already keys off op names. Cheaper, but rewrites the written language's vocabulary and desynchronises every existing script and transcript.

Verification that generalises: inline compile of every operator form asserting shape label, word-bit composition (compound = OR of components, no own bit), sig_level vs band_significance agreement, and the `A < B ≡ B > A` value-identity; then harness runs over all data/scripts/*.ks; pytest; mypy count compared pre/post (138 pre-existing, unchanged). dev/dialogue probes remain broken at HEAD for an unrelated reason: they insert `dev/src` into sys.path, but source lives at repo-root `src/`.

See [[sources/obs-2026-09-15-connotes-denotes-shapes-swapped-connotes-is-the-compound-den]], [[concepts/relational-tokens]], [[concepts/kscript-operator]], [[entities/astemitter]], [[concepts/structural-significance]].

*Category: kscript*

---
*Captured: 2026-09-15*

## Related

_Add links to related pages._
