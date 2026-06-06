# Auto-Tune Session State

## Goal
Scope `promote_participating` to only promote klines that structurally participated in the ratification event, not all STM entries sharing a signature.

## Done Criteria
Open session — observe what `promote_participating` currently promotes, identify over-promotion, iterate until it only promotes structurally relevant klines.

## Session
- **Name:** promote-scope
- **Curriculum:** curricula/mhall-svo-single.md
- **Branch:** auto-tune/promote-scope
- **Started:** 2026-06-06

## Current Phase
complete

## Next Action
session complete — ready for cascade documentation if desired

## Run Log

### Run 3 (latest)
- **Code changes:** Scoped `promote_participating` to promote: identity frames (nodes=[]), single-node entries (countersign/undersign), and canonical compositions (canonization). Does NOT promote multi-node non-canonical cogitator expansion proposals.
- **Observation:** 17/18 satisfied, 1 promote_participating call promoting 11 structural + 2 = 13 klines. Same satisfaction as baseline. 87% reduction in promotions (13 vs baseline's 101). Zero regressions.
- **Verdict:** improved — goal met

### Run 2
- **Code changes:** Identity-only promotion (nodes=[])
- **Observation:** 15/18 — regression. Entries 17 (L>M) and 18 (L>O) failed because single-node countersign/undersign entries stayed in STM and cogitator proposals polluted the event routing.
- **Verdict:** regressed — over-scoped

### Run 1
- **Code changes:** baseline with logging
- **Observation:** 17/18, promote_participating called once promoting 99+2=101 klines and once promoting 61+2=63. Severe over-promotion.
- **Verdict:** baseline

## Patterns & Notes
- The original code promoted ALL STM entries whose signature appeared in the union of query+candidate nodes. This swept up cogitator expansion proposals.
- Identity-only promotion is too aggressive — single-node entries (countersign/undersign) also need promotion as structural dependencies.
- Adding canonical compositions (canonization entries) to the promotion set provides the right coverage without over-promoting.
- The filtering criteria: promote if signature in node_sigs AND (nodes empty OR single non-literal node OR canonical)
- Cogitator expansion proposals are multi-node non-canonical klines — they fail all three criteria and are correctly excluded.

## Files Modified
- `src/kalvin/expand.py` — scoped `promote_participating` to structural participants only
