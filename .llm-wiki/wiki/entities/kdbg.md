---
type: entity
title: KDbg
description: Debug annotation carried on each compiled kline — holds the owning scope's annotation (parens stripped) and a scope level (0 for source, 1 for MTS).
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-002
    resource: /sources/SRC-2026-08-11-002.md
---

# KDbg

Debug annotation carried on each compiled kline.

## Overview

`KDbg.annotation` carries the owning scope's annotation (parens stripped);
`KDbg.scope` is `0` for source entries and `1` for
[[concepts/mts-multi-token-signature|MTS]] entries. Each kline owns its own
annotation; an MTS spawned by a signature inherits the owning scope's
annotation.

KDbg is the bridge between authored KScript (labels, annotations) and the
compiled [[concepts/kline|klines]] the engine rationalises — it carries the
provenance that makes traces readable in the
[[concepts/non-judging-harness|non-judging harness]].

## Links

- [SRC-2026-08-11-002](/sources/SRC-2026-08-11-002.md) — compilation section
- [[concepts/mts-multi-token-signature]] — scope-1 entries
- [[entities/tokenencoder]] — where KDbg annotations attach
- [[entities/kscript]] — the authored source KDbg points back to
- [[concepts/non-judging-harness]] — what KDbg makes readable
