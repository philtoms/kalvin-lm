---
type: entity
title: CONTEXT.md
description: The project's domain glossary — the source of truth for terminology. Source is the truth document; CONTEXT.md's glossary maps the terms the source uses.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-001
    resource: /sources/SRC-2026-08-11-001.md
---

# CONTEXT.md

The project's domain glossary.

## Overview

CONTEXT.md defines the precise meaning of terms used across the code. Per
AGENTS.md: _source is the truth document for behaviour; CONTEXT.md's glossary
maps the terms the source uses._ When behaviour changes, the source and any
affected glossary terms are updated in the same change.

It is organised into four sections: **Structure** (the kline's objective shape
and its claimed significance), **Rationalisation** (how a participant tests that
claim), **KScript** (the language that authors training material), and
**Training and Runtime** (the multi-agent loop).

CONTEXT.md is the canonical source for resolving terminology disputes —
including the wiki synthesis errors it was used to correct (see the
`wiki-duplicate-consolidation` retro).

## Links

- [SRC-2026-08-11-001](/sources/SRC-2026-08-11-001.md) — itself, captured as a wiki source
- [[concepts/kline]], [[concepts/significance]], [[concepts/cogitation]] — representative glossary terms
