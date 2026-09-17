---
type: source
title: "Observation: ν_B walk bridge mechanics: Det:[Object] written with zero overlap"
tags:
  - dialogue
  - engine
  - derivation
  - slot-walk
  - traverse
status: observation
created: 2026-09-17
updated: 2026-09-17
slug: obs-2026-09-17-b-walk-bridge-mechanics-det-object-written-with-zero-overlap
relevance: high
observed_at: 2026-09-17T12:27:11.868Z
source_context: "dialogue-dev: examining finding 1, the Det:[Object] slot-walk bridge"
---

# ⭐ Observation: ν_B walk bridge mechanics: Det:[Object] written with zero overlap

Verified how the walk writes Det:[Object] despite Det/Object sharing zero word bits. It is the ν_B (goal-side) walk in Derivation._walk_b: queued a:[Det], goal QueryObject:[Object], gap=Det, excess=Object. The walk departs the goal slot Object, is licensed by occurrence alone, and ends on OVERLAP with σ(ν_A) (end_mask=Det): [Object] -rev ALL:[Object]-> [ALL] -fwd ALL:[a,little,lamb]-> [a,little,lamb] -fwd a:[Det]-> [Det,little,lamb]. The bridge is then written head=anchor (Det, the arrived node in A's nodes), witness = arrived nodes sharing a word bit with the GOAL content (none do — no-fit region) + the slot: Det:[Object], acq=3. Two masks do different jobs (arrival mask = A's content, witness filter mask = goal's content); in any no-fit region their yields are structurally disjoint, so every ν_B bridge degenerates to head:[slot] — a denotation-shaped traverse licence. This is Def 15's letter faithfully implemented. Downstream: Det:[Object] then served as forward-replace evidence ([Det]->[Object], misfit 2->0) for the done proposing a:[Object]. Det:[a,Mod,lamb] arises the same way (walk from slot Mod, arrival drag-along a,lamb leaks into the witness because it DOES overlap the goal). Probe: dev/dialogue/probe_wdmh_walk.py.

*Relevance: high*
*Context: dialogue-dev: examining finding 1, the Det:[Object] slot-walk bridge*
*Tags: dialogue engine derivation slot-walk traverse*

---
*Observed: 2026-09-17T12:27:11.868Z*
