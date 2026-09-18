---
type: source
title: "Observation: Def 13 evidence inertness: head-carrying klines never licensed — signatures never enter nodes"
tags:
  - dialogue-dev
  - kalvin-algebra
  - definition-13
  - evidence-inertness
  - well-foundedness
  - derivation
status: observation
created: 2026-09-18
updated: 2026-09-18
slug: obs-2026-09-18-def-13-evidence-inertness-head-carrying-klines-never-license
relevance: critical
observed_at: 2026-09-18T08:11:07.819Z
source_context: Re-anchoring the identity-proposal fix upstream at evidence licensing
---

# 🔴 Observation: Def 13 evidence inertness: head-carrying klines never licensed — signatures never enter nodes

The well-foundedness rule re-anchored upstream per user: "better not to put signature into nodes in the first place" — evidence carrying the queued head is INERT, rather than filtering replacement outputs. Also corrected a wrong inference: the compiler is NOT at fault — probe_wdmh_ask_sig.py shows the wdmh.ks ask signature mints correctly as the OR-reduction 0x380100006fff (bits Mary|what|did|have + token OR 0x6fff, ASK bit on), NOT 'what's token value; the earlier is_answered tautology fired via a different key than assumed. New rule placement: (1) docs Def 13 evidence-inertness list: "Evidence carrying the queued head s — as its signature or as a node of its witness — is inert for the derivation of s: nothing held licenses writing s into ν_A, so the signature never enters node position in the first place"; the state-level paragraph now states well-foundedness as the CONSEQUENCE (s:[s] unreachable as a derivation result — the identity is the ask's own shape, delivered by grounding). (2) Code: kalvin/derivation.py usable() gained the two head-carries exclusions (signature==s, or s among k.nodes); canonicalisations() filters canons through usable(); _walk's b_starts drops s (the bridge never writes s into a composed witness). The three enumerator output-guards and _keeps_head_out were REMOVED — one rule, one place, upstream where the option is licensed. engine.py's identity/canon ground branch stays reverted (moot — the terminal is unreachable). Regression test renamed test_evidence_carrying_the_queued_head_is_inert: usable() False for a canon headed at s AND for evidence with s as a witness node; the run never ends at [MHALL]. Verified: 70/70 tests; wdmh.ks trace clean (T01 silent, T02 single WDMH:[ALL,had,Mary] S1 proposal, T03 grounds, no re-proposals, no declines).

*Relevance: critical*
*Context: Re-anchoring the identity-proposal fix upstream at evidence licensing*
*Tags: dialogue-dev kalvin-algebra definition-13 evidence-inertness well-foundedness derivation*

---
*Observed: 2026-09-18T08:11:07.819Z*
