---
type: source
title: "Observation: Sketch defects: push-without-descent kills the claim path"
tags:
  - cogitator
  - debugging
  - stack
status: observation
created: 2026-11-18
updated: 2026-11-18
slug: obs-2026-11-18-sketch-defects-push-without-descent-kills-the-claim-path
relevance: high
observed_at: 2026-11-18T15:22:52.878Z
source_context: Diagnosing the user's Cogitator sketch that produced no claims
---

# ⭐ Observation: Sketch defects: push-without-descent kills the claim path

Diagnosis of why the user's real-stack sketch (staged, never committed) offered no claims — three defects. (1) FATAL: push without descent. HEAD's `entry = push(...); continue` switches control to the child; the sketch's `push(...); continue` left the parent running, so it pushed U then O then F for one candidate and CONSUMED the candidate (cands.pop(), at.clear()) before any child ran. The re-evaluation-on-new-frames loop never fires, S2 alignment only possible with pre-framed components → no claims; traversal also inverts to reverse-candidate/F-O-U order (mhall: 187-line event divergence). (2) Dead branch: `st = entry.st if entry.st else _State()` never instantiates — _State defines no __bool__/__len__, so every entry in every run shares ENone.st's class-default instance; frame and the entries counter leak across scenarios and the 4096 cap is spent process-wide. (3) chain stored on the shared state and rewritten per push — accumulates monotonically (sibling subtrees bar later branches) and common_ancestor becomes S.chain & S.chain, trivially true (free exchange witnesses). Fix discipline: peek-top; push appends child, parent suspends beneath; per-run _State; per-entry chain.

*Relevance: high*
*Context: Diagnosing the user's Cogitator sketch that produced no claims*
*Tags: cogitator debugging stack*

---
*Observed: 2026-11-18T15:22:52.878Z*
