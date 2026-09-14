---
type: source
title: Structural significance is coverage-based — S2 has ≥1 covered node, S3 none
status: insight
category: kscript
created: 2026-09-10
updated: 2026-09-10
slug: structural-significance-coverage-semantics
---

# Structural significance is coverage-based — S2 has ≥1 covered node, S3 none

Structural significance (`sig_level` in kalvin/kline.py) redefined — significance levels now derive purely from the signature–nodes relationship, per user spec table: S1 — signature covers its nodes exactly (canon ABC:[A,B,C], identity A:[A]); S2 — at least one node covered by the signature (underfit ABC:[A,C], overfit AB:[A,B,C], under+over ABC:[B,C,D], denotation AB:[B]); S3 — no node covered (connotation A:[B], misfit/no-fit AB:[C,D]); S4 — no nodes (unknown A:[]). The rules are independent of node count and of the relational token that compiled the shape. Key behavioural shift: no-coverage multi-node misfits (AB:[C,D]) moved S2→S3 (previously ALL multi-node non-canons were S2). The compiled stamp (`_OP_TO_SIG`: CANONICALISES→S2) may now legitimately disagree with the structure — that is by design: the stamp is the Target Significance (authored intent), sig_level is the measurement. Verified: all nine table rows reproduce exactly from their scripted equivalents via dev compile; engine rationalise smoke clean; probes (rationalise, concat, concat2, wdmh_pair, block_filter, scaffold_label, expand_trace) pass; ruff/mypy counts identical to HEAD. Changed: src/kalvin/kline.py (sig_level rewrite + is_misfit docstring — dropped the stale node-count/S2-expansion-path sentence; no caller ever gates on node count), src/kalvin/significance.py (\_OP_TO_SIG comment fixed — it wrongly said CONNOTES and DENOTES both map to SIG_S3; also notes stamp-vs-structure disagreement is legitimate), CONTEXT.md (Structural Significance entry now states the coverage rules; Misfit shape bullets re-ordered with under+over added, no-fit now S3; rational S2 example qualified "misfit with coverage"), wiki concepts/structural-significance.md updated with the coverage table. The "S2 strategy" names in pivot_fill.py/reentry.py are historical labels for the misfit-fill strategies — they serve S2 and S3 misfits alike. [[sources/connote-denote-structure-swap-compound-sig-denotation]]

_Category: kscript_

---

_Captured: 2026-09-10_

## Related

_Add links to related pages._
