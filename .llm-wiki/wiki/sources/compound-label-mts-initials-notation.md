---
type: source
title: "Compound KNode labels: MTS initials, joined identifier stays registry key"
status: insight
category: compiler
created: 2026-10-30
updated: 2026-10-30
slug: compound-label-mts-initials-notation
---

# Compound KNode labels: MTS initials, joined identifier stays registry key

When a [[KNode]] label is a compound of 2+ signatures, it should render in the MTS uppercase-character notation: one uppercased initial per component signature (MarySubject → MS, hadVerb → HV). The notation rule is `"".join(l[:1].upper() for l in labels)` — the same rule the parser's sentence form and the sigless-annotation synthesis use ((Mary had a little lamb) → MHALL), so a lowercase component still contributes its uppercased initial (had → H).

**Critical separation: label vs registry key.** The full joined identifier ('MarySubject') must stay the `_compound_sigs` registry key in ks.TokenEncoder — initials are not unique (Subject+M and S+M both give 'SM', but different signature values). Switching keys to initials would make a later compound def resolve as a reference to an unrelated compound and clobber its signature. Only the KNode label and dbg label switch notation.

Mint sites: `NLPSignifier.signature_of` multi-label join (kalvin/signifier.py — its docstring already specified this; the code had drifted to full joins), and `TokenEncoder._compose_concat` (drop the `.with_label(joined_identifier)` override). Single-node signature_of keeps the full word label (not a compound). Authored ALL-UPPER identifiers are already the notation — idempotent.

Related: [[sources/obs-2026-10-30-compound-knode-labels-switched-to-mts-uppercase-initial-nota]], [[sources/obs-2026-08-26-knode-became-a-labelled-int-subclass-labels-wired-at-tokenen]]

*Category: compiler*

---
*Captured: 2026-10-30*

## Related

_Add links to related pages._
