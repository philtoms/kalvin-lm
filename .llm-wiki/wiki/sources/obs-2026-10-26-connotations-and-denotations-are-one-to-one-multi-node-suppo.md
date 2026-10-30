---
type: source
title: "Observation: Connotations and denotations are one-to-one — multi-node support removed"
tags:
  - ks
  - kscript
  - language-design
  - parser
  - one-to-one
status: observation
created: 2026-10-26
updated: 2026-10-26
slug: obs-2026-10-26-connotations-and-denotations-are-one-to-one-multi-node-suppo
relevance: critical
observed_at: 2026-10-26T18:11:59.959Z
source_context: Removing multi-node connotation/denotation support after data/ks.md review
---

# 🔴 Observation: Connotations and denotations are one-to-one — multi-node support removed

Phil decided connotations and denotations (`=`, `>`, `<`) are ALWAYS one-to-one relationships; multi-node RHS support removed. Implemented in src/ks/parser.py: _ONE_TO_ONE_OPS frozenset; _parse_operator_scope_rest raises ParseError ("a connotation or denotation relates one node; chain operators (A < B < C) or scaffold a `=>` block for more") when node items (Signature + nested OperatorScope) plus block constructs (_count_block_nodes, mirroring _collect_block_node_ids) exceed one — covers same-line `A < B C`, block `A <` + B/C, and mixed item+block forms. Chains (A < B < C → BA:[A], CB:[B]) and => scaffolding/canon (S:[P,Q], composition op) remain the multi-relation forms. ast_emitter.py comments updated (RCONNOTES multi-node note removed; DENOTES singular). Golden corpus check: mhall.ks and wdmh.ks compile byte-identically. Tests: tests/test_ks_one_to_one.py (11 cases); full suite 187 passed. Doc consequence: the suggested Chains entry for data/ks.md drops its `a < b c` discriminator row — one-to-one is now the stated truth.

*Relevance: critical*
*Context: Removing multi-node connotation/denotation support after data/ks.md review*
*Tags: ks kscript language-design parser one-to-one*

---
*Observed: 2026-10-26T18:11:59.959Z*
