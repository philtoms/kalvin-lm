---
type: source
title: "Observation: Ask content form never fed; WDMH:[] at γ is the whole question"
tags:
  - harness
  - ask
  - feed
  - scaffolding
status: observation
created: 2026-09-16
updated: 2026-09-16
slug: obs-2026-09-16-ask-content-form-never-fed-wdmh-at-is-the-whole-question
relevance: high
observed_at: 2026-09-16T14:17:50.557Z
source_context: Harness ask-feed contract refinement
---

# ⭐ Observation: Ask content form never fed; WDMH:[] at γ is the whole question

User refined the harness feed contract (T01/T02): the ask's content form — the MTS canon WDMH:[what,did,Mary,have] — is structurally an S1 canon but must NEVER be fed; it is the answer to the question, held in the harness answering pools (heads/exact) and released only when the engine asks. The feed sees exactly: scaffolding (goal MHALL:[had,ALL] S2, word identities, goal MTS canon) + the empty ask WDMH:[] graded at γ(WDMH, MHALL)=S3. Implemented in harness.run: graded() applies only to the empty ask form; is_ask_content() drops same-signature non-empty entries from build_batch (pools unaffected — the canon still joins heads/exact via the group loop). Consequences verified: the spurious is_groundable grounding of WDMH:[] disappears (signature never framed); the ask now sits in the end-of-run work_list (genuine residue: engine attending to the question, holding nothing under WDMH — the structural ask per Def 16). Remaining seam: the engine drops stuck derivations silently, so it never EMITS WDMH:[] as an ask; the _answer identity-ask path (which would release the pooled canon) never fires. Next fork: when the engine asks WDMH:[] and the harness answers with the canon, should the reply be graded at γ too — so the canon slow-routes into cogitation as A₀ (cogitate runs _propose before the grounded-check removes it) — or fed at S1 (grounds and leaves, no derivation)?

*Relevance: high*
*Context: Harness ask-feed contract refinement*
*Tags: harness ask feed scaffolding*

---
*Observed: 2026-09-16T14:17:50.557Z*
