# Kalvin — Domain Glossary

## KLine
A node-like structure with a **signature** (head) and **nodes** (value). The fundamental unit of Kalvin's knowledge graph. Two kinds: **identity** (no nodes) and **relationship** (one or more nodes).

## Signature
The head of a KLine. A bit-packed integer representing the identity of the KLine. Constructed via `make_signature(nodes)`.

## Compiled Entry
A KLine produced by the KScript compiler. Extends KLine with compilation metadata.

## Expectation
A compiled entry that enters the slow path (S2/S3) during rationalisation and requires a matching proposal to be satisfied.

## Proposal
A KLine emitted by the Agent (via event) as a candidate response during rationalisation.

## Countersign
The reciprocal kline of a proposal. For `{Q: [V]}`, the countersignature is `{V: [Q]}`. Countersigning ratifies a proposal, promoting it toward S1.

## Significance
A 64-bit inverted distance representing how well a KLine relates to the model's current knowledge. Higher values indicate greater understanding. Ranges from S1 (fully grounded) to S4 (completely novel). Displayed as both a raw hex value and a normalised value between 0 (S4) and 1 (S1).

### Significance Levels
- **S1**: Fully grounded — canonical or countersigned.
- **S2**: Partially understood — some nodes match. Misfitting (overfit or underfit).
- **S3**: Recognised aspects — no node match but signature matches previously worked signatures.
- **S4**: Completely novel — no candidates found.

## Fast Path
Rationalisation that resolves immediately (S1 ground, S4 identity, S1 canonical, S1 countersigned). Returns `True` from `agent.rationalise()`. Auto-satisfied by the harness.

## Slow Path
Rationalisation that requires cogitation (S2/S3). Returns `False` from `agent.rationalise()`. Tracked by the harness as pending expectations.

## Harness
The repurposed KScript TUI that acts as a training loop supervisor. Submits compiled entries to the Agent, tracks submission/satisfaction state, and provides ratification controls.

## Submitted Set
The set of compiled entries that have been fed to `agent.rationalise()`. Monotonic — only reset by the Clear action.

## Satisfied Set
The set of compiled entries whose proposals matched expectations and were countersigned. Monotonic — only reset by the Clear action.

## Pending Set
Compiled entries not yet in the submitted set. New entries are submitted; already-submitted entries are skipped.

## Ratify
The action of countersigning a selected proposal. Performed automatically in Run mode on match, or manually by the user in Step mode.

## MCS Entry
A multi-character signature entry where `signature == make_signature(nodes)`. Always S1 (canonical) — no misfit possible.

## Structural Match
Comparison of two KLines by signature and nodes: `a.signature == b.signature and a.nodes == b.nodes`. Used for proposal-expectation matching and event correlation.
