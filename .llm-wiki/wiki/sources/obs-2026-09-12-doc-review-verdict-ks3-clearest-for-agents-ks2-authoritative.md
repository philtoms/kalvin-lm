---
type: source
title: "Observation: Doc review verdict: ks3 clearest for agents, ks2 authoritative"
tags:
  - docs
  - formalisation
  - ks2
  - ks3
  - review
status: observation
created: 2026-09-12
updated: 2026-09-12
slug: obs-2026-09-12-doc-review-verdict-ks3-clearest-for-agents-ks2-authoritative
relevance: medium
observed_at: 2026-09-12T10:42:32.852Z
source_context: Reviewing the three formal documents in docs/ for agentic clarity
---

# 🔍 Observation: Doc review verdict: ks3 clearest for agents, ks2 authoritative

Compared docs/ks2.md, ks3.md, kalvin-symbolic.md for agentic usability. Verdict: ks3.md clearest/easiest — only doc with a notation decoder (§0 cheat sheet mapping every symbol to code), executable anchors (fit as first-match-wins pseudocode, worked example as step table with per-step misfit mass, γ arithmetic spelled out), and quote+decode redundancy with explicit precedence ("where framing and quote seem to differ, the quote wins"). ks2 is authoritative but parse-expensive; kalvin-symbolic is cheapest but self-subordinates to ks2 (can't be trusted alone), uses incompatible Def numbering (its Def 6 = ks2's Def 10), and packs definition+commentary+invariance into single blocks. Optimal agentic setup: ks3 as working copy, ks2 as arbiter. Defect found: markdown escaping artifacts in ks2 Def 14 scoping clause (line 141: `σ(ν*A)`, `\_asks*`) propagate verbatim into ks3's quote (line 317); kalvin-symbolic's rendering of the same clause is clean. Both need the same two-character repair; fixing ks2 does not auto-propagate to ks3's frozen quote.

*Relevance: medium*
*Context: Reviewing the three formal documents in docs/ for agentic clarity*
*Tags: docs formalisation ks2 ks3 review*

---
*Observed: 2026-09-12T10:42:32.852Z*
