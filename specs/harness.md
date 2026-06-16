# Test Harness Specification

## Overview

The test harness is a repurposed KScript TUI that acts as a training loop supervisor. It submits compiled KScript entries to a Kalvin Agent, tracks which entries have been submitted and satisfied, displays proposals with significance values, and provides ratification controls. The harness serves both as a human-in-the-loop training tool and as a test environment for Kalvin's rationalisation behaviour.

## Dependencies

- `specs/agent.md` — rationalise API, event bus, significance levels
- `specs/kscript.md` — compilation, CompiledEntry, decompiler
- `specs/kline.md` — KLine structure, equality
- `specs/signature.md` — make_signature

## Definitions

### Tracking State

The harness maintains three monotonic sets, reset only by the Clear action:

| Set | Contents | Growth rule |
|-----|----------|-------------|
| **Submitted** | Entries that have been fed to `agent.rationalise()` | Append-only |
| **Satisfied** | Entries whose proposals matched expectations and were countersigned | Append-only |
| **Pending** | `compiled − submitted` — entries not yet sent to the Agent | Grows on recompile, shrinks on submission |

### Satisfaction

An entry is satisfied when either:

1. **Fast path** — `agent.rationalise()` returns `True` (S1 ground, S1 canonical, S1 countersigned, S4 identity). Auto-satisfied immediately.
2. **Slow path match** — a proposal event's kline structurally matches the expectation AND the user (Step) or harness (Run) ratifies it.

Structural match: `proposal.signature == expectation.signature AND proposal.nodes == expectation.nodes`.

### Significance Display

Every response is displayed with two values:

| Format | Description | Range |
|--------|-------------|-------|
| **Raw** | Hex representation of the 64-bit significance integer | `0x0000…FFFF` |
| **Normalised** | `significance / D_MAX` | `0.0` (S4) to `1.0` (S1) |

### Response Status

Each response item carries one of three statuses:

| Status | Meaning | Symbol |
|--------|---------|--------|
| **Pass** | Proposal matched expectation, countersigned | ✓ |
| **Pending** | Slow-path entry awaiting match or ratification | ◌ |
| **Mismatch** | Proposal received but does not match expectation | ✗ |

## API

### Agent.countersign(kline)

Generates the reciprocal kline and rationalises it through the Agent.

**Precondition:** `kline.nodes` is not empty.
**Postcondition:** A reciprocal kline `{nodes_sig: [kline.signature]}` is rationalised. Returns the result of `rationalise(reciprocal)`.

### Harness Compilation

`_compile_script()` returns entries or raises an error. On error, a response item of status ✗ is appended with the error message.

## Behavioural Rules

### Entry Submission

1. Both Run and Step recompile the editor content before submitting.
2. Only entries not in the Submitted set are submitted.
3. Submitted set is monotonic — entries are never re-submitted.
4. Identity entries (`{X: None}`) and MTS entries (canonical) are auto-satisfied via the fast path.

### Run Mode

5. All pending entries are submitted sequentially without pausing.
6. Proposals that structurally match an expectation are auto-countersigned.
7. Mismatches are flagged as pending for human review.

### Step Mode

8. One pending entry is submitted. Execution halts.
9. The user may ratify any selected slow-path proposal by clicking the Ratify button.
10. Ratify is contextual — the button is enabled only when a response item is selected.

### Clear

11. The Clear action resets Submitted, Satisfied, and Pending sets. It also clears the responses panel.

### Event Correlation

12. Events are correlated to compiled entries by structural match: `event.query.signature == entry.signature AND event.query.nodes == entry.nodes`.

### State Persistence

13. The Submitted and Satisfied sets are persisted through hot-reload cycles via the existing JSON state file alongside Agent state and editor content.

### Progress Display

14. The toolbar displays a satisfaction count: `{satisfied}/{total} pending/{pending_count}` alongside the execution state indicator.

## Test Matrix

| ID | Criterion | Origin ref |
|----|-----------|------------|
| HRN-1 | Recompile produces fresh compiled entries; only new entries are submitted | Origin §The Loop |
| HRN-2 | Submitted set is monotonic; Clear is the only reset | Origin §The Loop |
| HRN-3 | Fast-path entries (rationalise returns True) are auto-satisfied | Origin §The Loop |
| HRN-4 | Slow-path match requires structural equality of signature and nodes | Origin §The Loop |
| HRN-5 | Run mode submits all pending entries without pausing | Origin §The Loop |
| HRN-6 | Run mode auto-countersigns matching proposals | Origin §Ratification |
| HRN-7 | Step mode submits one entry and halts | Origin §The Loop |
| HRN-8 | Ratify button enabled only when a response item is selected | Origin §Ratification |
| HRN-9 | Ratify calls agent.countersign(proposal) with the proposal as-is | Origin §Ratification |
| HRN-10 | Clear resets Submitted, Satisfied, Pending sets and responses panel | — |
| HRN-11 | Events correlated to entries by structural match | Origin §The Loop §Event Correlation |
| HRN-12 | Tracking state persisted through hot-reload cycles | — |
| HRN-13 | Each response shows status symbol, decompiled source, raw significance (hex), normalised significance (0.0–1.0) | Origin §Significance |
| HRN-14 | Compilation errors displayed as ✗ response items in the responses panel | — |
| HRN-15 | Progress count displayed in toolbar alongside execution state | — |
| HRN-16 | agent.countersign generates reciprocal kline and rationalises it | Origin §Ratification |
| HRN-17 | Mismatches in Run mode flagged as pending; execution continues | Origin §The Loop |
| HRN-18 | Multiple proposals for one expectation are all displayed; first match accepted in Run, user chooses in Step | Origin §Proposals |

## Out of Scope

- Auto-scaffolding (generating new KScript when a proposal mismatches)
- Temperature adjustment during execution
- Multi-agent cross-training
- Per-query timeout configuration
- Frame stack management
