---
type: source
title: "Observation: using_resolver now installed by dev callers, not rationaliser/cogitator"
tags:
  - refactor
  - resolver
  - dev-concern
status: observation
created: 2026-09-21
updated: 2026-09-21
slug: obs-2026-09-21-using-resolver-now-installed-by-dev-callers-not-rationaliser
relevance: medium
observed_at: 2026-09-21T17:27:56.655Z
source_context: Moving using_resolver from library to dev callers
---

# 🔍 Observation: using_resolver now installed by dev callers, not rationaliser/cogitator

Moved the decode-resolver install out of library code: `Rationaliser.rationalise` and `cogitator.cogitate` no longer wrap themselves in `using_resolver` — the resolver (which populates KDbg.decoded on freshly minted klines for trace debuggability) is now a development concern installed by the callers. `dev/dialogue/harness.py` wraps its turn (`rationalise(feeds)` + `cogitate(self.state)`) in `with using_resolver(self.state.find):`, and the six probes with direct calls (probe_ask_fate, probe_block_filter, probe_hop_wdmh, probe_rationalise, probe_wdmh_decode, probe_wdmh_s3_recip) wrap their rationalise+cogitate pairs the same way. `cogitate` also absorbed `_pass` (the split existed only to hold the resolver wrapper). The compiler's own dev-gated resolver in `src/ks/token_encoder.py` was intentionally left as-is. Verified: 95 tests pass, mhall.ks output byte-identical, mypy/ruff unchanged (probe_block_filter StopIteration is pre-existing).

*Relevance: medium*
*Context: Moving using_resolver from library to dev callers*
*Tags: refactor resolver dev-concern*

---
*Observed: 2026-09-21T17:27:56.655Z*
