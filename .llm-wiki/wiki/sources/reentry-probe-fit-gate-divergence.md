---
type: source
title: "Reentry probe: pseudocode grounds all but the fit gate"
status: insight
category: architecture
created: 2026-11-02
updated: 2026-11-02
slug: reentry-probe-fit-gate-divergence
---

# Reentry probe: pseudocode grounds all but the fit gate

Implemented the user's six-term reentry pseudocode as a probe ([[dev/probes/probe_simple_reentry.py]]) against the reference Cogitator (src/rationalisation/model.py) over mhall.ks, wdmh.ks, and the two-scenarios script. Result: the simple loop reproduces every canonical grounding — **except one rule**. The pseudocode's ground condition `framed(F) and U_resolved and O_resolved` gates on the FIT being framed; the reference's `_satisfied` deliberately does not ("the fit is the agreement between entry and candidate, not something to prove"). In wdmh, the WDH × what:Object row has both sides resolved (DH stalled→framed, Object framed) but the fit `what` — the query word — is held by nothing in the domain, so it can never frame; with the gate, WDMH never grounds; without it (probe `fit_gate="free"`), WDMH:[MHALL] grounds exactly like the reference. A query word is precisely content the domain does not hold — demanding its frame is demanding the ask already be answered. Otherwise the simple rules suffice: stall-self-framing substitutes for the two-stall exchange (same outcome, weaker justification — one witness instead of two mutual); the stack re-check (parent re-runs its for-loop after children drain) substitutes for the frame cascade; pairing a held canon structurally recovers the o-side witness settle. Related: [[sources/obs-2026-11-02-union-fit-history-removed-from-doc-offering-only-pairing-com]]

*Category: architecture*

---
*Captured: 2026-11-02*

## Related

_Add links to related pages._
