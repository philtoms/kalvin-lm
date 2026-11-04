---
type: source
title: "Observation: Claim grounds via ask×goal now; DH framed by joint.append — the exchange is the record"
tags:
  - trs
  - cogitator
  - s3-exchange
  - joint-frame
  - claim-rule
status: observation
created: 2026-11-03
updated: 2026-11-03
slug: obs-2026-11-03-claim-grounds-via-ask-goal-now-dh-framed-by-joint-append-the
relevance: high
observed_at: 2026-11-03T23:59:04.022Z
source_context: Verifying claim-rule fix; explaining where the exchange frames DH
---

# ⭐ Observation: Claim grounds via ask×goal now; DH framed by joint.append — the exchange is the record

User's claim-rule fix verified: the depth-1 row now takes [goal] as its ONLY candidates (cands = [goal] if depth == 1 — in every drain), so the claim can only ground from the ask×goal pairing. Both scripts now ground through the real opening story: mhall `| 1 | MHALL × SVO | | MHALL | SVO | grounded |` (both sides earned via the registry), wdmh `| 1 | WDMH × MHALL | Mary | WDH | HLAL | grounded |`. Also landed silently by the user: the `continue` statements after side forks are gone (U fork then O fork both run in one candidate iteration), and depth is now len(stack) stored on the row. ANSWER to 'where does DH get framed': the exchange branch's `self.joint.append(E)` IS the framing — the first DH fork parks (stall, parent=WDH row), the second DH fork (different parent row, depth 5) finds the peer, appends DH to self.joint, and from then on framed() consults self.joint on every settle check — DH never enters any path frame. Per the 2026-10-31 ruling (S3 exchange stores nothing — the frame is the record), the exchange row's frame column is the only trace record. Consumption point in the trace: `| 2 | WDH × what:Object | … | WDH | resolved |` (U=DH settled by the joint). Rendering gap noted: closes never show WHICH frame settled their sides — a candidate enhancement for the table.

*Relevance: high*
*Context: Verifying claim-rule fix; explaining where the exchange frames DH*
*Tags: trs cogitator s3-exchange joint-frame claim-rule*

---
*Observed: 2026-11-03T23:59:04.022Z*
