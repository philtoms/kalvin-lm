# Undersign is S3 (inverse of connotate), not S1

Undersign (`=`) and connotate (`>`) compile to inverse kline structures — `A = B` produces `{B: A}` while `A > B` produces `{A: B}`. Both are S3. The only S1 operator is countersign (`==`), which produces reciprocal pairs detected structurally by `is_countersigned()`. Treating undersign as S1 required a `sig_level` runtime guard on the agent fast-path, which was the only consumer of that field. Reclassifying undersign as S3 eliminates the need for `sig_level` as a first-class KLine slot and removes the guard.

**Consequences:** The first half of a countersign pair no longer receives immediate S1 treatment when both sides are grounded — it lands as S4 and gets promoted when the reciprocal arrives. This is correct: a countersign is only S1 once the reciprocal exists.
