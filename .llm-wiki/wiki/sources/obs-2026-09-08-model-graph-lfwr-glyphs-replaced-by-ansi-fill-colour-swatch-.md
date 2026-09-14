---
type: source
title: "Observation: Model graph: LFWR glyphs replaced by ANSI/fill colour + swatch key"
tags:
  - dialogue
  - harness
  - graph
  - colour
  - ansi
  - render
status: observation
created: 2026-09-08
updated: 2026-09-08
slug: obs-2026-09-08-model-graph-lfwr-glyphs-replaced-by-ansi-fill-colour-swatch-
relevance: low
observed_at: 2026-09-08T13:20:24.212Z
source_context: Graph render switch from glyph indicators to colour
---

# 📝 Observation: Model graph: LFWR glyphs replaced by ANSI/fill colour + swatch key

Removed all LFWR memory-layer glyphs from the harness model graph renderers; layer membership is now conveyed by colour only, using the same strongest-layer logic (min over L>F>W>R) as the DOT fills. ASCII: value names, edge arrows (--->), member arrows, id-fallback names, and leaf names are painted with ANSI codes standing in for the fills (L green #32, F yellow #33, W blue #34, R red #31); colour is enabled when stdout isatty and NO_COLOR is unset, so piped output stays plain. The header key is now coloured swatches (`● ltm ● frame ● work ● refused`) instead of `L=ltm F=frame W=work R=refused`. DOT: node labels dropped layer text (fills retained), graph legend replaced with `green=ltm yellow=frame blue=work red=refused`. Mermaid: node labels dropped layer text (classDef colours retained). _layer_str helper deleted; _graph_heads identities now store raw glyph strings consumed directly by _paint.

*Relevance: low*
*Context: Graph render switch from glyph indicators to colour*
*Tags: dialogue harness graph colour ansi render*

---
*Observed: 2026-09-08T13:20:24.212Z*
