---
type: concept
title: Curriculum
description: "A living structured document owned by the Harness — the source of truth for training, never rolled back. Three sections: objective, approach, lessons."
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-001
    resource: /sources/SRC-2026-08-11-001.md
---

# Curriculum

A living structured document owned by the [[concepts/harness]].

## Definition

The curriculum is the source of truth for training — never rolled back, only
evolved forward (see [[concepts/monotonic-growth]]). It is accessible to all
participants. Three sections:

- **objective** — what it teaches
- **approach** — the pedagogical strategy
- **lessons** — ordered [[entities/kscript|KScript]] entries with human-readable
  context

The CurriculumDocument parser supports amendments that mutate the document and
write it back to the source file, so a curriculum evolves as the trainee learns.
The CurriculumGenerator can produce one from a natural-language goal via an LLM.

_Avoid_: lesson plan (too narrow — the curriculum is more than its lessons).

## Links

- [SRC-2026-08-11-001](/sources/SRC-2026-08-11-001.md) — canonical definition (CONTEXT.md)
- [[concepts/harness]] — who owns the curriculum
- [[concepts/target-significance]] — what each lesson's kline is labelled with
- [[concepts/monotonic-growth]] — curricula only evolve forward
- [[entities/kscript]] — the language lessons are authored in
