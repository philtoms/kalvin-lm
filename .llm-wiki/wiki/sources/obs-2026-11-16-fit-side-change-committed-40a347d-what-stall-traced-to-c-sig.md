---
type: source
title: "Observation: Fit-side change committed (40a347d); what stall traced to c.sig != e bar"
tags:
  - trs
  - cogitator
  - fit
  - like-for-like
  - wdmh
  - commit
  - exploration
status: observation
created: 2026-11-16
updated: 2026-11-16
slug: obs-2026-11-16-fit-side-change-committed-40a347d-what-stall-traced-to-c-sig
relevance: high
observed_at: 2026-11-16T18:57:38.095Z
source_context: Committing the fit-side rule change and the fingerprint-bar exploration
---

# ⭐ Observation: Fit-side change committed (40a347d); what stall traced to c.sig != e bar

Committed 40a347d on branch forward: the fit-F rule change (close gate f_ok∧u_ok∧o_ok, F descends as side "F" after U and O; F==E mint kept), docs/rationalisation.md §2 preamble updated, halt string now "exhausted: no candidate resolved all three sides". The user's commented-out fingerprint bar (c.fingerprint() != e) was committed as-is: measured INERT on the corpus — traces byte-identical with it off. Exploration findings (dev/probes/probe_what_stall.py, gitignored like all probes): `what` has exactly two potential partners, both sig == what — ⟨what:{Object}⟩ (primed DENOTES:what from `(what Object)`, S3, killed by the c.sig != e like-for-like bar; its fingerprint what|Object was never fingerprint-barred) and ⟨what:{what}⟩ (compiler-emitted IDENTITY:what word seed, canon, triple-barred: fingerprint + identity + like-for-like). Operative bar for the stall: c.sig != e. Probe subclass removing it: wdmh GROUNDS immediately (claim WDMH:[MHALL], 122 pairings) via `what × ⟨what:{Object}⟩` closing with F==E mint and O=Object framed — i.e. the fit would settle by the script's own authored assertion witnessing itself. Contrast: `had` has the same authored shape (⟨had:{did,have}⟩, like-for-like barred too) but settles via HV:Verb whose signature strictly contains had; no held signature strictly contains the what-bit. Open fork for the user: keep the 11-01 ruling (stall honest, script must scaffold what inside a strictly-containing signature) vs let primed claims self-witness.

*Relevance: high*
*Context: Committing the fit-side rule change and the fingerprint-bar exploration*
*Tags: trs cogitator fit like-for-like wdmh commit exploration*

---
*Observed: 2026-11-16T18:57:38.095Z*
