---
type: source
title: Identity priming; engine stopped restating knowledge
status: insight
category: bugfix
created: 2026-08-27
updated: 2026-08-27
slug: identity-priming-and-knowledge-restatement
---

# Identity priming; engine stopped restating knowledge

Added identity priming to the harness run() (commit dfd178c): after compiling, all IDENTITY-op entries are submitted at SIG_S1 in one pre-run engine.rationalise call, so dialogues open on questions rather than identity discovery — mhall asks dropped 34→6, wdmh 36→8, zero proposals, STM drains. Priming exposed two latent engine defects the gradual ask/answer flow had masked: (1) [[src/dialogue/expand_fit.py]] `expand` recursion yielded grounded sub-klines at SIG8_MAX (restating knowledge as proposals) and its terminal yield emitted identity candidates that the top-level `propose` identity filter never saw — fixed with a `_sayable` gate (not identity, not grounded) on every yield, plus the same removal in reentry.py's grounded-target short-circuits; (2) [[src/dialogue/engine.py]] cogitate's asked-guard blocked a *canon* from grounding under an asked signature — a canon is the script's own ground truth and the ask's own answer, so it now grounds even when asked (mhall's ALL had always relied on cascade timing to slip in before asked was set; priming removed that window and ALL stalled in STM). Diagnostic pattern: when a change makes behaviour worse, stash and diff the trace before theorising — the 3→40 proposal flood was priming interacting with these bugs, not priming itself being wrong.

*Category: bugfix*

---
*Captured: 2026-08-27*

## Related

_Add links to related pages._
