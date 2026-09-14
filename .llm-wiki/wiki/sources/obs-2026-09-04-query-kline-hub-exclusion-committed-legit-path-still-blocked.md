---
type: source
title: "Observation: Query-kline hub exclusion committed; legit path still blocked"
tags:
  - dialogue
  - cogitator
  - engine
  - commit
status: observation
created: 2026-09-04
updated: 2026-09-04
slug: obs-2026-09-04-query-kline-hub-exclusion-committed-legit-path-still-blocked
relevance: high
observed_at: 2026-09-04T10:13:35.605Z
source_context: "Dialogue-dev: hub exclusion commit and re-add check"
---

# ⭐ Observation: Query-kline hub exclusion committed; legit path still blocked

Committed aca3618 (dialogue branch): connotateY now takes exclude= (the query signature, threaded from cogitate through expand) and neither yields nor traverses klines whose own or nodes-derived signature equals it. Rationale: the reverse bit-kinship edge let 'what' enter the query's own kline WDMH and exit via Mary's bit to MHALL/ALL, producing the spurious MDHALL proposal. Re-add check with (Object)W > O primed before the query (data/scripts/wdmh-underfit-o.ks, gitignored) showed the legitimate path STILL cannot fire, for two structural reasons: (1) cogitate uses the S2 ask entry WDMH:[DH] as the query — nodes [DH] only, 'what' not in the query node set at all; (2) connotateY erases bridge signatures — hadDH:[DH] (sig had|DH) is re-derived as DH:[DH], so the DH<->had cross is invisible to both matching arms, and a/little/lamb cannot drain because the second loop only consumes from remainder/fit, never leftover overfit. Next levers: signature preservation in reached klines + coverage rule.

*Relevance: high*
*Context: Dialogue-dev: hub exclusion commit and re-add check*
*Tags: dialogue cogitator engine commit*

---
*Observed: 2026-09-04T10:13:35.605Z*
