---
type: source
title: "Observation: Fix B landed (exhaustion halts unless proven); fix A rejected — broke mhall"
tags:
  - rationalisation
  - soundness
  - fix
  - exhaustion
  - rejected-fix
  - scenario-harness
status: observation
created: 2026-10-31
updated: 2026-10-31
slug: obs-2026-10-31-fix-b-landed-exhaustion-halts-unless-proven-fix-a-rejected-b
relevance: critical
observed_at: 2026-10-31T23:05:53.730Z
source_context: Implementing fixes A+B for the exhaustion soundness defect
---

# 🔴 Observation: Fix B landed (exhaustion halts unless proven); fix A rejected — broke mhall

Both defect fixes implemented; one rejected on evidence. FIX B LANDED (v2): exhaustion is not evidence — in _reenter's candidate loop, the proven-fit test is computed BEFORE the in_path bar; a path-barred candidate that is itself a proven fit counts as proven. A reentry whose every candidate is skipped resolves _known only when all skips were proven; any unproven turn-away halts with reason "exhausted: candidate barred without proof" (maps to an acquire proposal; test_rationalisation_is_honest allow-list extended with "exhausted"). FIX A REJECTED EMPIRICALLY: comparing klines (sig+adj) instead of sigs in _in_path made mhall STOP grounding — the more permissive pairing changed cascade order and the canonical script lost its claim. Reverted; the sig-level comparison stands, now documented as by-design in docs/rationalisation.md with the exhaustion rule carrying the honesty instead. The regression test test_exhaustion_does_not_ground pins the minimal repro (identities + 9:[1], entry 9:[32,1], goal 1:[1] — must NOT ground; before the fix it grounded with region 40 unbacked). KEY DISCOVERY mid-fix: the user's commit 8e582f5 evolved the harness to run EVERY ask==goal pair as a scenario (ask order, one live domain, earlier stores persist) — my certificate tests assumed the old two-step shape and paired wdmh's last ask with the FIRST scenario's goal. Tests reworked: RecordingRunner captures Records(frames, frame_re, stalls, rows) per step (rationalise rebinds arrays per run, so references survive); _scenarios yields (result, records, d0-with-prior-stores, entry, goal) per pair. verify() signature changed from cog to Records (src/rationalisation/verify.py). wdmh now certifies BOTH scenarios (mhall's MHALL→SVO and wdmh's WDMH→MHALL); mhall certifies with identical leaves to pre-fix (empty 7, witness-held 3, proven 2, witness 1). Suite: 171 passed, 16 xfailed (the xfail un-marked — the property passes live). Doc §2.1 synced to the multi-scenario harness. Lesson banked: fix A's failure mode — permissiveness changes cascade order; mhall's grounding is order-sensitive to candidate admission (a T4/completeness-relevant fragility worth revisiting).

*Relevance: critical*
*Context: Implementing fixes A+B for the exhaustion soundness defect*
*Tags: rationalisation soundness fix exhaustion rejected-fix scenario-harness*

---
*Observed: 2026-10-31T23:05:53.730Z*
