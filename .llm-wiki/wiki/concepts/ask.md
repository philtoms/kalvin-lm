---
type: concept
title: Ask
created: 2026-09-14
updated: 2026-09-14
---

# Ask

The structural halt condition — no atom, mark, or decree involved.

## Definition

Two conditions, both the ask (Def 15, §9): **no goal held**, or **no connection
across the correspondence graph**. Relative non-existence — the honest outcome
when the bridge is missing. The [[concepts/unknown]] shape (`S: []` — nothing
held) is the ask's shape, and a misfit region no held correspondence reaches
asks. The event under which ungrounded [[concepts/proposal|proposals]] are
generated (§4, §8); asks are grounded externally via
[[concepts/ratify|ratification]], never by fit.

## The ask atom

An **ordinary atom, externally allocated**. Its only peculiarity is allocation
provenance — reserved by KScript (bit 63 / word-word bit 31) so internal word
allocation (bits 0–30) never crosses it: namespacing between two allocators,
not algebraic distinctiveness. Provenance is not an algebraic property; atoms
are unlabelled in V, and the atom plays exactly the role of any other in
∨/∧/¬, selection, and fit.

The engine no longer branches on it (is_ask gates removed 2026-09-12; routing
is structural — `find_canon(entry.signature) or entry`); `is_ask` is read only
for harness display (the `?` prefix). A bare compound or sigless annotation
keeps its original signature with the ASK bit marking it — so any signature
can be an ask.

## Links

- [[concepts/unknown]] — the structural shape of the ask
- [[concepts/proposal]] — ungrounded proposals emitted under the ask
- [[concepts/ratify]] — the external route to grounding
- [[entities/kscript]] — the compiler's reservation of the atom
- [[concepts/underfit-and-overfit]] — the slot walk stranded by the ask
- [[sources/obs-2026-09-12-ask-atom-settled-ordinary-atom-externally-allocated-only-eng]] — the settled position

_Avoid_: ASK_BPE_TOKEN (superseded engine reading), decree (the ask is structural).
