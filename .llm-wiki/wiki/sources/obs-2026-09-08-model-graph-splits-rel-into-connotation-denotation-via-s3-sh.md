---
type: source
title: "Observation: Model graph splits rel into connotation/denotation via S3 shape + node_in(sig, node)"
tags:
  - dialogue
  - harness
  - graph
  - connotation
  - denotation
  - node_in
status: observation
created: 2026-09-08
updated: 2026-09-08
slug: obs-2026-09-08-model-graph-splits-rel-into-connotation-denotation-via-s3-sh
relevance: medium
observed_at: 2026-09-08T10:24:48.285Z
source_context: Refining the model-graph relationship labels
---

# 🔍 Observation: Model graph splits rel into connotation/denotation via S3 shape + node_in(sig, node)

Replaced the harness model-graph "rel" edge label with connotation|denotation, using the structural route (option b): the free is_connotation shape gate (single-node, non-terminal, non-identity = the S3 relationship claim) plus signifier.node_in(kline.signature, kline.nodes[0]) — sig sits inside its node (A:[AB]) → "connotation", disjoint → "denotation". This mirrors the committed EngineState.is_connotation predicate (which was flipped to node_in(signature, nodes[0]) direction between sessions), not KDbg.op — dbg is provenance/intent, dropped on save, so it can't label reloaded states. mhall.ks: compound-slot forms Mary:[MarySubject], had:[hadVerb], ALL:[ALLQuery], Query:[QueryObject] → connotation; authored `=` forms a:[Det], little:[Mod], lamb:[Object] and the MHALL==SVO countersign pair → denotation. All three renderers (ascii/dot/mermaid) carry the split; ASCII class column widened to 11 chars.

*Relevance: medium*
*Context: Refining the model-graph relationship labels*
*Tags: dialogue harness graph connotation denotation node_in*

---
*Observed: 2026-09-08T10:24:48.285Z*
