---
type: source
title: "Observation: '<' RCONNOTES operator committed (fe845f5)"
tags:
  - kscript
  - compiler
  - relational-tokens
status: observation
created: 2026-08-27
updated: 2026-08-27
slug: obs-2026-08-27-rconnotes-operator-committed-fe845f5
relevance: medium
observed_at: 2026-08-27T16:35:13.386Z
source_context: Adding reversed CONNOTES operator to KScript
---

# 🔍 Observation: '<' RCONNOTES operator committed (fe845f5)

Committed fe845f5 on dialogue: new '<' relational token (TokenType.RCONNOTES) in KScript. A < B compiles to B:[A] with dbg.op CONNOTES — DENOTES direction, CONNOTES provenance. Touched token.py, lexer.py, parser.py (_OPERATOR_TYPES), ast_emitter.py (own op string "RCONNOTES" dispatching to node:[sig] with CONNOTES op), CONTEXT.md glossary, and two obsolete "'<' is a lexer error" tests. Suite back to baseline.

*Relevance: medium*
*Context: Adding reversed CONNOTES operator to KScript*
*Tags: kscript compiler relational-tokens*

---
*Observed: 2026-08-27T16:35:13.386Z*
