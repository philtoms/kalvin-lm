---
type: source
title: "Observation: Terminal-unknown own-set = all entry heads; connotation head is the specific side"
tags:
  - ks
  - compiler
  - heads
  - terminal
  - unknown
  - s4
  - ruling
status: observation
created: 2026-10-26
updated: 2026-10-26
slug: obs-2026-10-26-terminal-unknown-own-set-all-entry-heads-connotation-head-is
relevance: high
observed_at: 2026-10-26T15:19:57.101Z
source_context: Refining the terminal-unknown own-set after user ruling on heads
---

# ⭐ Observation: Terminal-unknown own-set = all entry heads; connotation head is the specific side

User ruling completing the terminal-unknown rule (2026-10-26): "All heads count as resolved. In mhall statement `Object < Query`, Object is not the head, Query is. So Object still generates S4." Implemented in ASTEmitter._emit_terminal_unknowns: the resolved/own set is now the HEAD of every entry — concat[0] for connotations (whose sig slot is a compound: `A < B` heads B — the specific side; `A > B` heads A), the sig for every other op (DENOTES/CANONICALISES/IDENTITY/ASK/UNKNOWN). Consequence: `Mood = Happy` then `A = Mood` mints no Mood:[] (Mood heads the denotation), while Object in mhall still mints Object:[] (it heads nothing — DENOTES lamb:[Object] heads lamb; CONNOTES QueryObject:[Object] heads Query). mhall/wdmh outputs unchanged by this refinement.

*Relevance: high*
*Context: Refining the terminal-unknown own-set after user ruling on heads*
*Tags: ks compiler heads terminal unknown s4 ruling*

---
*Observed: 2026-10-26T15:19:57.101Z*
