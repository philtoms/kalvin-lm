---
type: source
title: "Observation: Speaking fast path + supervisor escalation + only-misfits-proposed"
tags:
  - dialogue
  - engine
  - fast-path
  - ratification
  - supervision
status: observation
created: 2026-08-21
updated: 2026-08-21
slug: obs-2026-08-21-speaking-fast-path-supervisor-escalation-only-misfits-propos
relevance: critical
observed_at: 2026-08-21T08:56:08.093Z
source_context: Fast path answers from LTM; supervisor ratification loop
---

# 🔴 Observation: Speaking fast path + supervisor escalation + only-misfits-proposed

Landed the speaking fast path + supervisor escalation + only-misfits-proposed rule. Engine route now returns emissions: (1) S1-stamped queries are ratifications grounded on receipt (before answering, so a ratified misfit grounds); (2) a question (unknown or misfit — never statements) whose signature holds grounded klines is answered from LTM immediately (\_answers_from_ltm excludes the query's own shape, identities — tautological answers — and canons — ground truth, not earned knowledge); (3) old fast/slow routes. Harness: off-script asks escalate to a supervisor callback (escalate= on Harness; CLI -s flag prompts 1=S1 ratify/2,3=grade+decline/4=decline; EOF declines) — ratifying at S1 grounds the proposal in LTM so the next ask takes the fast path. Proven on revised mhall.ks (WDMH asked twice): Step 6 proposes WDMH:[had,Mary,a,little,lamb] S3 101 (no harness look-ahead), supervisor ratifies, Step 8 answers from LTM in one turn at S1 255. Supporting fixes: harness answering pools grow only as authored groups open (scope-0 annotation runs form groups; MTS set allocated to first occurrence; answered set resets per group); authored CANONICALISES repeats both emit (temporal distinctness) while authored-subscript-canon vs MTS twin still collapse (skip when either side is MTS; registry now (idx, is_mts) tuples); harness skips replies for already-grounded emissions (statements, not asks) and answers X:[X] terminal identity proposals from the tokenizer; fill and pivot arms skip identity and canon klines — only misfits are proposed. The whole run now shows exactly one proposal (WDMH word form, twice: S3 cogitated then S1 recalled). Tests 1221 green.

_Relevance: critical_
_Context: Fast path answers from LTM; supervisor ratification loop_
_Tags: dialogue engine fast-path ratification supervision_

---

_Observed: 2026-08-21T08:56:08.093Z_
