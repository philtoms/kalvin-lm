# Plan: KValue — Subjective+Objective Exchange Unit

**Parent:** [`plans/implement-kalvin.md`](../implement-kalvin.md)
**Spec:** [`@specs/kvalue.md`](../../specs/kvalue.md) (primary), with edits to
[`@specs/kline.md`](../../specs/kline.md), [`@specs/model.md`](../../specs/model.md),
[`@specs/agent.md`](../../specs/agent.md). Glossary: `@CONTEXT.md` §KLine, §Significance, §KValue.
**Estimate:** 3–4 days
**Depends on:** existing KLine, expand/significance, compiler, harness adapter.

---

## 0. Goal

Close the training-loop asymmetry: the trainer currently emits objective-only
klines (countersign/scaffolding), while Kalvin emits significance-laden events.
Introduce the **KValue** (KLine + significance) as the unit of exchange so every
participant's assessment rides on the klines it produces. **This plan delivers
plumbing only** — it guarantees a sender's declared significance is present and
addressable on every exchanged kline and on rationalisation events. What Kalvin
*does* with an inbound declared significance is explicitly **deferred** (out of
scope per @kvalue spec §What a KValue is Not).

Non-goals: the combination rule for declared-vs-computed significance; any new
behaviour in rationalisation driven by the declared significance; remote
(WireSocket) serialization of KValues.

---

## 1. Spec References (links only)

- **KValue definition, equality, producers, storage, exchange, test matrix KV-1..KV-15** → `@specs/kvalue.md`.
- **Band-representative values (S1=`D_MAX`, S2=`D_MAX−1`, S3=`D_MAX−101`, S4=`0`)** → `@specs/model.md` §Significance Semantics › Band-representative Values (single owner of the integers).
- **RationaliseEvent carries KValues, no significance field; fast path wraps one KLine in two KValues** → `@specs/agent.md` §Events, §Rationalisation.
- **KLine is objective-only; significance re-derived on retrieval** → `@specs/kline.md` §What a Kline is Not.

---

## 2. Design Decisions

### D1 — `compile_source` returns `list[KValue]`, not `list[KLine]`

The compiler is the producer of compiled KValues (@kvalue spec KP-1). The
structural `op` is *always* known during emission (it is a field on
`SymbolicEntry`, and `TokenEncoder` already documents "compile-time intent"
significance levels with a `_SIG_LEVELS` map). The band significance is captured
from `op` at encode time — **before storage** — because re-derivation *cannot*
recover the compiler's intent for a `==` entry before its reciprocal is in the
model (countersigned requires model state).

Therefore the natural home for KValue production is the `TokenEncoder`, and
`compile_source` returns `list[KValue]`.

**Rejected:** keep `compile_source` returning KLines and wrap into KValues at the
adapter. Rejected because a freshly-compiled KLine carries no spec'd structural
state in production (dbg is unspec'd), so the adapter could not recover the
correct band for a countersign entry without re-running the emitter.

**Cost:** mechanical test churn in `tests/test_ks_compiler.py` (insert `.kline`
before `.signature`/`.nodes`/`.dbg`). Contained and non-behavioural.

### D2 — The Model API stays KLine-based

`add_to_stm`/`add_to_frame`/`add_to_ltm`/`find`/`where`/`grounded` all keep
their `KLine` signatures. Storage is objective-only (@kvalue spec §Storage;
KV-7). `rationalise` and other KValue-aware callers extract `.kline` at the
model boundary. This keeps the storage-critical surfaces untouched (the
"minimum flak" goal) and the codec unchanged.

### D3 — Significance is derived from the always-computed `op`, not from `dbg`

`dbg.op` is today set unconditionally (`KDbg(op=entry.op)` in
`token_encoder.py`), but `dbg` is spec'd as unspec'd debug provenance. The
production semantics MUST NOT depend on `dbg`. The encoder computes the band
from the `SymbolicEntry.op` directly (which is production data) and stamps it on
the KValue. `dbg` remains dev-only enrichment; significance is a first-class
production field.

### D4 — Re-derivation function lives in `kalvin/expand.py`

`derive_significance(kline, model, signifier) -> int` implements the
re-derivation cascade (identity→S4, is_countersigned→S1, is_canon→S2, else→S3) per @kvalue
spec §Retrieval. It belongs with the significance semantics and band constants
(already owned by `expand.py`), and reuses the existing structural predicates
(`is_identity`, `is_canon` from `kline.py`; `is_countersigned` from
`expand.py`). Used whenever a stored KLine is materialised as a KValue without a
producer-supplied significance.

### D5 — `RationaliseEvent` loses its `significance` field

Per @kvalue spec KE-3. `query` and `proposal` become KValues carrying their own
significance. Every reader of the removed field refactors to
`event.proposal.significance` (Kalvin's assessment) or
`event.query.significance` (sender's declared assessment) — see §6 consumer map.

### D6 — KLine is immutable; fast path shares, never copies

On the fast path `query.kline is proposal.kline` — the same immutable KLine is
wrapped in two KValues that differ only in significance (@kvalue spec KE-1).
No KLine copy is made for significance's sake. (KLine immutability is an
implementation property, noted in code/docs; not a glossary term.)

---

## 3. Component Tasks

### Task A — `KValue` type (`src/kalvin/kvalue.py`, new)

New module. Spec ref: @kvalue spec §Definition, §Equality, §Construction.

```python
@dataclass(frozen=True)
class KValue:
    kline: KLine        # objective structure (immutable)
    significance: int   # sender's assessment (uint64)

    # Structural identity: equality/hash ignore significance.
    def __eq__(self, other): ...   # kline == other.kline
    def __hash__(self): ...        # hash(self.kline)
```

- Both fields required; no default (KV-1).
- `__eq__`/`__hash__` delegate to `kline` only (KV-2, KV-3). Keeps auto-countersign
  structural matching and the adapter sender-map unchanged.
- Re-export from `kalvin/__init__.py` and `kalvin.kline` if convenient for imports.

**Tests:** `tests/test_kvalue.py` (new) — KV-1, KV-2, KV-3.

### Task B — Band-representative values + re-derivation (`src/kalvin/expand.py`)

Spec ref: @model spec §Band-representative Values; @kvalue spec §Retrieval.

```python
# Band-representative significance (maximal value of each band)
SIG_S1 = D_MAX
SIG_S2 = D_MAX - 1
SIG_S3 = D_MAX - 101
SIG_S4 = 0

_OP_TO_SIG: dict[str, int] = {
    "COUNTERSIGNED": SIG_S1,
    "CANONIZED":     SIG_S2,
    "CONNOTED":      SIG_S3,
    "UNDERSIGNED":   SIG_S3,
    "IDENTITY":      SIG_S4,
}

def band_significance(op: str) -> int:
    """Compile-time op → band-representative significance."""

def derive_significance(kline, model, signifier) -> int:
    """Re-derive significance from a stored KLine's current structural
    relationship to the model (KV-1 cascade)."""
```

- `derive_significance` cascade order: `is_identity`→S4; `is_countersigned`→S1; `is_canon`→S2; else→S3.
  Never returns an unset value (KV-12).
- Note: `token_encoder.py` already has a `_SIG_LEVELS` op→"S1" string map; this
  task replaces/promotes it to the integer map above (single source of truth).

**Tests:** `tests/test_expand.py` (extend) — re-derivation for identity/canon/countersigned/connoted (KV-8..KV-12); `tests/test_ks_compiler.py` — op→band (KV-4).

### Task C — Compiler produces KValues (`src/ks/token_encoder.py`, `src/ks/compiler.py`)

Spec ref: @kvalue spec KP-1; D1, D3.

- `TokenEncoder.encode_entries(symbolic) -> list[KValue]`: build each KLine as
  today, then wrap in `KValue(kline, band_significance(entry.op))`. The op is
  taken from `SymbolicEntry.op` (production), never from `dbg`.
- `Compiler.compile` and `compile_source` return `list[KValue]`.
- `dbg` handling unchanged (still dev-enriched); significance is independent of `dbg`.

**Test churn:** `tests/test_ks_compiler.py` — mechanical: `entries[i].signature` → `entries[i].kline.signature`, etc. Behaviour unchanged.

### Task D — `KAgent.rationalise` / `countersign` consume KValues (`src/kalvin/agent.py`)

Spec ref: @agent spec §Rationalisation, §Events; @kvalue spec KP-2, KP-3, KE-1.

```python
def rationalise(self, value: KValue) -> bool:
    kline = value.kline          # model operations use the KLine
    ...
    # All add_to_*/where/grounded calls pass `kline`, unchanged.
    # _publish wraps results in KValues (see Task E).

def countersign(self, value: KValue) -> bool:
    kline = value.kline
    reciprocal = KLine(make_signature(kline.nodes), [kline.signature])
    reciprocal_value = KValue(reciprocal, SIG_S1)   # KP-2: countersign is S1
    return self.rationalise(reciprocal_value)
```

- `rationalise` operates on `value.kline`; `value.significance` is carried
  through and surfaced on the *query* of published events (KE-1). Consumption of
  the declared significance by the pipeline is **deferred** (no behavioural change).

### Task E — Event refactor + fast-path two-KValue wrapping (`src/kalvin/agent.py`, `src/kalvin/cogitator.py`, `src/kalvin/events.py`)

Spec ref: @kvalue spec §Exchange; D5, D6.

- `RationaliseEvent.__init__(kind, query: KValue, proposal: KValue)` — drop the
  `significance` param and the `candidate` positional (candidate becomes
  `proposal.kline` where needed). `__repr__` uses `proposal.significance`.
- Agent `_publish(kind, query_value, proposal_value)`: both are KValues.
- **Fast path** — the five publish sites that today alias `kline` as both query
  and proposal now wrap the *same* KLine in two KValues:
  ```python
  # ground / frame S1 / frame S4 etc.
  q = value                              # inbound KValue (declared significance)
  p = KValue(kline, kalvin_sig)          # Kalvin's assessment; q.kline is p.kline
  self._publish("frame", q, p)
  ```
  where `kalvin_sig` is the band value for the known outcome (`SIG_S1` or `SIG_S4`).
- **Cogitator** — `on_s1` publishes a proposal KValue at `SIG_S1`;
  `on_expansion(query_value, proposal_kline, significance, ...)` builds the
  proposal KValue at the `expand()`-computed significance (KP-3). The synthetic
  `"done"` idle event wraps its kline in KValues at `SIG_S4`.
- `WorkItem.query` becomes a `KValue` (carries the declared significance into the
  slow path); `candidate` stays a KLine (from the model).

**Tests:** `tests/test_agent.py` (extend) — KV-13 (query.kline is proposal.kline, independent significances), KV-14 (no significance field), KV-5 (countersign S1), KV-6 (cogitation computed sig).

### Task F — Harness adapter boundary (`src/training/harness/adapter.py`)

Spec ref: @agent spec §Kalvin Adapter; @kvalue spec KP-1, KP-2.

- `_handle_submit`: `entries = compile_source(...)` now `list[KValue]`;
  `add_to_stm(entry.kline)`; sender-map key from `entry.kline`; `rationalise(entry)`.
- `_handle_countersign`: materialise the bus payload to a KValue. Extend
  `_materialise_kline` (rename/extend to `_materialise_kvalue`) to accept a live
  KValue, a wire dict `{"signature","nodes","significance"}`, or a legacy KLine
  (wrapped at `SIG_S1`). Then `kagent.countersign(kvalue)`.
- `on_event`: sender-map lookup from `event.query.kline.signature`/`.nodes`.

### Task G — Trainer / Reactor on KValues (`src/training/trainer/trainer.py`, `reactor.py`)

Spec ref: @harness-server spec; D5 consumer map.

- `reactor.load_lesson(entries: list[KValue])`; `_current_entries: list[KValue]`.
- `_entry_key(value)` reads `value.kline.signature`/`value.kline.nodes`.
- Auto-countersign: `entry == proposal` (both KValues; structural equality holds).
- The `countersign` bus message carries the proposal KValue (in-process; no wire format change needed).
- **Consumer refactor (D5):**
  - `Trainer._is_s1`: `event.proposal.significance >= _S1_FRAME_THRESHOLD` (was `event.significance`).
  - distance/normalise logging: `event.proposal.significance`.
  - `ratify_request` payload `"significance"`: `event.proposal.significance` (payload dict field unchanged; value source moves).
  - `trainer/cogitation.py` `_classify_significance` and prompt builder: read `event.proposal.significance`.

### Task H — Codec: confirm objective-only (no code change expected)

Spec ref: @kvalue spec §Storage; KV-7.

- Verify `agent_codec.py` persists `{signature, nodes}` only and reconstructs
  `KLine` only. No change required — this *is* the objective-only contract.
- Add an assertion-level test that a round-tripped KLine has no significance on
  the KLine object (significance lives on KValue, re-derived at use).

**Tests:** `tests/test_agent_codec.py` (extend) — KV-7.

---

## 4. Test Mapping

| Spec ID | Criterion                                         | Test file                  | Component |
| ------- | ------------------------------------------------- | -------------------------- | --------- |
| KV-1    | Construction requires both fields                 | test_kvalue.py             | A         |
| KV-2    | Equality ignores significance                     | test_kvalue.py             | A         |
| KV-3    | Hash ignores significance                         | test_kvalue.py             | A         |
| KV-4    | Compiler attaches op→band (==,=>,=,>,identity)    | test_ks_compiler.py        | B, C      |
| KV-5    | Countersign reciprocal carries `D_MAX`            | test_agent.py              | D, E      |
| KV-6    | Cogitation proposal carries computed significance | test_agent.py / test_cogitator | E      |
| KV-7    | Codec persists `{signature,nodes}` only           | test_agent_codec.py        | H         |
| KV-8    | Re-derive identity → S4                           | test_expand.py             | B         |
| KV-9    | Re-derive canonical → S2                          | test_expand.py             | B         |
| KV-10   | Re-derive countersigned (reciprocal present) → S1 | test_expand.py             | B         |
| KV-11   | Re-derive connoted/undersigned → S3               | test_expand.py             | B         |
| KV-12   | Re-derivation never unset                         | test_expand.py             | B         |
| KV-13   | Fast path: shared kline, independent significances| test_agent.py              | E         |
| KV-14   | RationaliseEvent has no significance field        | test_agent.py              | E         |
| KV-15   | Consumer reads event.proposal.significance        | test_reactor.py / test_trainer | G     |

Existing tests that read `RationaliseEvent.significance` or assume
`compile_source` returns KLines must be updated (mechanical) — primarily
`test_reactor.py`, `test_trainer*.py`, `test_agent.py`, `test_ks_compiler.py`.

---

## 5. Build Order / Phasing

1. **A** (KValue type) — no dependencies; pure type.
2. **B** (band values + re-derivation) — depends on A (returns ints used by KValue producers).
3. **C** (compiler → KValues) — depends on A, B. Breaks `compile_source` callers; do before D/G.
4. **E** (event refactor + fast-path wrapping) — depends on A. Decouple query/proposal.
5. **D** (rationalise/countersign) — depends on A, E. Wires KValue through the pipeline.
6. **F** (adapter) — depends on C, D. Bus boundary.
7. **G** (trainer/reactor) — depends on E (event fields). Consumer refactor.
8. **H** (codec verification) — independent; confirms no regression.

Recommended checkpoint: after **C+E+D** the agent pipeline compiles KValues
end-to-end and publishes two-assessment events; **F+G** restores the trainer
loop; **H** locks the storage contract.

---

## 6. Consumer Map — `event.significance` → `event.<kvalue>.significance`

| Site (file)                              | Was                              | Becomes                              | Which voice |
| ---------------------------------------- | -------------------------------- | ------------------------------------ | ----------- |
| `trainer.py _is_s1`                      | `event.significance`             | `event.proposal.significance`        | Kalvin's    |
| `trainer.py` distance/normalise logging  | `event.significance`             | `event.proposal.significance`        | Kalvin's    |
| `trainer.py` ratify_request payload      | `event.significance`             | `event.proposal.significance`        | Kalvin's    |
| `trainer/cogitation.py` classify + prompt| `event.significance`             | `event.proposal.significance`        | Kalvin's    |
| `auto_tune/events.py` (payload reader)   | `message.significance` (payload) | unchanged (payload dict field)       | n/a         |

All current consumers want **Kalvin's** assessment → `proposal.significance`.
`query.significance` (the sender's declared assessment) is carried but **not yet
consumed** — that is the deferred intentional behaviour.

---

## 7. Status

- **Spec:** complete (`specs/kvalue.md` new; `kline.md`, `model.md`, `agent.md` edited; `CONTEXT.md` glossary updated).
- **Plan:** this document.
- **Implementation:** in progress across KB-351..KB-361 (exchange unit: Tasks
  A–H). The KValue type, compiler, agent, event refactor (D5), and harness
  adapter boundary (Task F) landed in KB-353/KB-354/KB-355.
  **KB-361** landed **Task G** — the trainer/reactor/cogitation consumer
  refactor (every `event.significance` read moved to
  `event.proposal.significance` per the §6 Consumer Map; `_compute_misfit`
  operates on `event.proposal.kline`; `candidate` references removed) — *plus*
  the protocol wire-encoding layer (`protocol.py::_domain_json_default` encodes
  `KValue` → the KLine objective shape and sources event significance from
  `proposal.significance`) and the reactor auto-countersign fix (parameter
  typed `KValue`; structural comparison via `proposal.kline`). This restores
  the training loop end-to-end after the KValue exchange-unit landings.
  Remaining: `tests/test_countersign_resolution.py:74` (one mechanical
  `ev.significance` → `ev.proposal.significance` consumer read, tracked as
  KB-363; outside this task's scope).
- **Deferred (explicitly out of scope):** consumption of an inbound KValue's
  declared significance by rationalisation (the intentional combination rule).
  The plumbing guarantees the value is present and addressable; a future grill
  will specify what Kalvin does with it.
