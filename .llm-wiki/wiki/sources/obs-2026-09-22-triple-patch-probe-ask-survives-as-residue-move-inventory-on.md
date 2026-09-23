---
type: source
title: "Observation: Triple-patch probe: ask survives as residue; move inventory = one targeting then stuck"
tags:
  - mhall
  - ask
  - derivation
  - moves
  - connotation
  - licence
  - descent
status: observation
created: 2026-09-22
updated: 2026-09-22
slug: obs-2026-09-22-triple-patch-probe-ask-survives-as-residue-move-inventory-on
relevance: high
observed_at: 2026-09-22T15:39:13.438Z
source_context: Round-2 ask-attends probe with is_answered disabled
---

# ⭐ Observation: Triple-patch probe: ask survives as residue; move inventory = one targeting then stuck

Round-2 probe (dev/dialogue/probe_mhall_ask_survives.py, probe_mhall_ask_vs_svo.py, probe_mhall_ask_moves.py): triple patch — AskAttends(Rationaliser) _fast_route ignores ask-marked feeds + AskNeverAnswered(Memory) is_answered ignores asks. Results: (1) Exact spec (canon MHALL S1 + empty ask MHALL|ASK:[] S4): ask attends and SURVIVES as permanent residue — goals=[] (Def 22 coverage vacuous over empty nodes), zero emissions, refused empty. The grounded canon is invisible to it as a goal. (2) Riding ask + canon only: same — goals=[] (own-canon exclusion). (3) Riding ask + full scaffolds + canon: one goal (ALL:[a,little,lamb]); derivation stuck at entry, tail unmoved, j1=0.600. (4) SVO canon forced as goal directly (bypassing Def 22 — KDbg.goal carries "SVO" but selection never reads it): the derivation moves EXACTLY ONCE — forward targeting lamb⇉[Object] (misfit 8→6, licence: lamb:[Object] plain head occurs as node) — then stuck. Full move inventory isolated the gaps: [a] canonicalisation {a,little,lamb}⇉ALL offered but exposes=False (§9 survey condition correct: nothing applies at [Mary,had,ALL] granularity) and after lamb→Object the group is destroyed anyway; [b] S2 connotations (MarySubject:[Subject] head=OR(Mary,Subject)) can apply in NEITHER targeting direction — compound heads never occur as nodes, nodes don't occur in A — the "node covered by underfit head ⇉ connotation nodes" licence (Mary⇉[Subject]) does not exist; [c] walk A-side can only depart from values that head non-terminal klines: Mary/had head nothing (identities terminal), and the Query 2-hop chain (Object<Query<ALL) compiles to compound heads ALLQuery/QueryObject so neither ALL nor Query heads an edge — the descent graph at those values is empty; only a→Det departs (via a:[Det]) and B never reaches Det. Def 13 usable-evidence exclusion worked correctly (canon MHALL + ask both excluded). Rationaliser work now sharply framed: mhall needs a member-granularity connotation licence, or plain heads on the chain, or descents departing from covered nodes.

*Relevance: high*
*Context: Round-2 ask-attends probe with is_answered disabled*
*Tags: mhall ask derivation moves connotation licence descent*

---
*Observed: 2026-09-22T15:39:13.438Z*
