---
type: source
title: "Observation: ASK_BPE_TOKEN to be removed; ks2 drops the ask atom — ask is structural S4 only"
tags:
  - ask
  - formalisation
  - ks2
  - engine
status: observation
created: 2026-09-10
updated: 2026-09-10
slug: obs-2026-09-10-ask-bpe-token-to-be-removed-ks2-drops-the-ask-atom-ask-is-st
relevance: high
observed_at: 2026-09-10T15:52:04.812Z
source_context: ks2 third pass G5 — ask atom removal
---

# ⭐ Observation: ASK_BPE_TOKEN to be removed; ks2 drops the ask atom — ask is structural S4 only

User decided ASK_BPE_TOKEN (bit 31/1<<63 ask flag, introduced 2026-08-27 per obs-2026-08-27) is a hack that routes canons that would otherwise be grounded; it will be removed from the engine. ks2.md Def 1 no longer contains an ask atom — A = {a₀…a₃₀}, no mark, no decree layer. The ask survives purely as the structural S4 event: Unknown (s:[]) is the ask's shape, the halt signal under which strategy generates ungrounded proposals (stated in §4 Bands). All ks2 ask references now read as that event. Engine reconciliation pending: remove ASK_BPE_TOKEN from src/kalvin/signifier.py, src/ks/token_encoder.py, ast_emitter _emit_ask paths; CONTEXT.md ASK/Token ID glossary entries need back-porting (bit-31-reserved clause) with that engine change per AGENTS.md same-change rule.

*Relevance: high*
*Context: ks2 third pass G5 — ask atom removal*
*Tags: ask formalisation ks2 engine*

---
*Observed: 2026-09-10T15:52:04.812Z*
