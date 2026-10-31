---
type: source
title: "Observation: Compound KNode labels switched to MTS uppercase-initial notation"
tags:
  - knode
  - labels
  - mts
  - notation
  - compiler
status: observation
created: 2026-10-30
updated: 2026-10-30
slug: obs-2026-10-30-compound-knode-labels-switched-to-mts-uppercase-initial-nota
relevance: high
observed_at: 2026-10-30T14:56:51.886Z
source_context: Switching compound KNode labels to MTS uppercase-character notation
---

# ⭐ Observation: Compound KNode labels switched to MTS uppercase-initial notation

Compound KNode labels now use the MTS uppercase-character notation (one uppercased initial per component signature — MarySubject → MS, hadVerb → HV, ALLQuery → AQ). Implemented at the two label-mint sites: NLPSignifier.signature_of multi-label join (src/kalvin/signifier.py, which now matches its docstring: first char of each node's label uppercased and concatenated), and TokenEncoder._compose_concat (src/ks/token_encoder.py) which no longer overrides with the joined identifier; dbg.label now takes the signature node's label when present so decoded display shows MS. KEY DESIGN POINT: the full joined identifier ('MarySubject') remains the _compound_sigs registry key — initials collide across different component sets (Subject+M vs S+M → both 'SM'), so only labels switched notation, not keys. Words and authored ALL-UPPER identifiers (MHALL, SVO) unchanged — notation is idempotent for them. Updated test_ks_case_rule.py mhall expectations + added TestCompoundLabelNotation (5 tests). 183 tests pass.

*Relevance: high*
*Context: Switching compound KNode labels to MTS uppercase-character notation*
*Tags: knode labels mts notation compiler*

---
*Observed: 2026-10-30T14:56:51.886Z*
