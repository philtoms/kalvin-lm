---
type: source
title: "Observation: Derived heads labelled by content; phantom-edit corruption cleaned"
tags:
  - trs
  - labels
  - residue
  - meet
  - content
  - corruption
  - incident
status: observation
created: 2026-10-06
updated: 2026-10-06
slug: obs-2026-10-06-derived-heads-labelled-by-content-phantom-edit-corruption-cl
relevance: high
observed_at: 2026-10-06T16:58:00.153Z
source_context: Content-composed labels for derived heads; phantom-edit corruption incident
---

# ⭐ Observation: Derived heads labelled by content; phantom-edit corruption cleaned

LABEL RULE (user ruling: 'I was expecting Verb|Object:[Verb, Object], not Subject|Verb|Object∖Subject:...'): residue/meet now label derived heads by CONTENT, not expression — when both operand labels are pure compositions (pipe-joined tokens, no ∖/∧) AND the subtrahend's tokens ⊆ the minuend's, residue's label = remaining tokens (SVO∖Subject → 'Verb|Object', empty → '⊥'); meet's dual: shared tokens (SVO∧Subject → 'Subject'). Fallback to 'A∖B'/'A∧B' expression when containment fails (mixed provenance, e.g. ask trace's 'had|lamb|a|little∖had|Det'). Values were always correct; labels are presentation. Goal trace now reads Object:[Object], Verb|Object:[Verb,Object], Subject|Verb|Object:[Object,Subject,Verb]. Rounds counter now 8 for the goal (resolve() also counts rounds since the three-points round — counter semantics, not behaviour). INCIDENT: three files corrupted by PHANTOM EDITS from earlier failed non-atomic batches — rationalise renamed to rationalise2 in test_trs_model.py (all old test bodies + max_rounds kwarg), test_trs_properties.py (import + 3 property bodies), src/trs/__init__.py (import + __all__ duplicated), and model.py's old-runner def line. Restored all from git HEAD + re-applied legitimate uncommitted edits (connect-test single-connect, witness docstring, rationalise2 import/test). Working-tree diff audited clean; 159 tests passing. LESSON REINFORCED: after every heredoc edit batch, grep-verify the file contains BOTH the intended change AND that no other region changed — phantom edits from failed batches can land LATER.

*Relevance: high*
*Context: Content-composed labels for derived heads; phantom-edit corruption incident*
*Tags: trs labels residue meet content corruption incident*

---
*Observed: 2026-10-06T16:58:00.153Z*
