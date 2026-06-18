# Curriculum Implementation Plan

**Parent:** `docs/roadmap.md` Phase B
**Status:** not started
**Spec refs:** `specs/curriculum.md`

## Spec References

- `@specs/curriculum.md` — CRS-1 through CRS-52
- `@specs/harness-server.md` — HRNS-IDs for existing Trainer integration
- `@specs/agent.md` — rationalise API, event bus

## Implementation Tasks

### Task 1: CurriculumDocument Parser (`src/training/trainer/curriculum_document.py`)

- **Spec ref:** @specs/curriculum §CurriculumDocument, CRS-1..CRS-23
- **Test mapping:** CRS-1..CRS-23
- **Details:**
  - `Lesson` dataclass with `label: str`, `prose: str`, `kscript: list[str]`
  - `CurriculumDocument` class with `objective`, `approach`, `lessons`, `source_path`
  - `from_file(path)`: read file, parse, set `source_path = path`
  - `from_string(text)`: parse from string, set `source_path = None`
  - Parser splits on `## ` headings to find Objective, Approach, Lessons sections
  - Under Lessons section, split on `### <label>` to create Lesson objects
  - Within each lesson's body, extract fenced code blocks (```...```) as kscript entries
  - Remaining text (stripped of code blocks) is prose
  - Validation function checks: required sections present, at least one lesson, no duplicate labels, label format `\d+[a-z]?`
  - `find_lesson(label)` → linear scan of lessons list
  - `all_labels()` → `[l.label for l in self.lessons]`
  - `amend(action, **kwargs)`:
    - `"insert"`: find lesson with `after_label`, insert new lesson after it
    - `"append"`: append to lessons list
    - `"modify"`: replace lesson at given label
    - After mutation: validate, serialize to markdown, write to `source_path`
  - Serialization: reconstruct markdown from sections and lessons

### Task 2: Curriculum/CurriculumState Refactor (`src/training/trainer/curriculum.py`)

- **Spec ref:** @specs/curriculum §CurriculumState, CRS-24..CRS-31
- **Test mapping:** CRS-24..CRS-31
- **Details:**
  - `Curriculum` constructor accepts `CurriculumDocument` (new) or `list[str]` (legacy)
  - When given `list[str]`, creates a synthetic `CurriculumDocument` with empty objective/approach and numbered labels
  - Add `current_label: str | None` to `CurriculumState`
  - Add `lesson_submitted: set[str]` and `lesson_satisfied: set[str]`
  - `mark_lesson_submitted(label)`, `mark_lesson_satisfied(label)` (also advances current_label)
  - `is_lesson_submitted(label)`, `is_lesson_satisfied(label)`
  - `current_label` property: scan labels in order, return first not in `lesson_satisfied`
  - `Curriculum.current()` returns kscript from the lesson matching `current_label`
  - `Curriculum.advance()` moves to next label in order
  - `save()`: add `curriculum_file`, `current_label`, `lesson_submitted`, `lesson_satisfied` to JSON
  - `load()`: detect format — if `curriculum_file` key present, use new format; otherwise use legacy (create synthetic document from `lessons` list + `position`)
  - Retain all existing `EntryKey`-based sets unchanged

### Task 3: CurriculumGenerator (`src/training/trainer/curriculum_generator.py`)

- **Spec ref:** @specs/curriculum §CurriculumGeneration, CRS-32..CRS-37
- **Test mapping:** CRS-32..CRS-37
- **Details:**
  - `CurriculumGenerationError` exception class
  - `CurriculumGenerator.__init__(client: LLMClient, curricula_dir: Path)`
  - System prompt: describes the three-section markdown format with lessons
  - `generate(goal: str) → Path`:
    1. Build messages: system prompt + user message with goal
    2. Call `client.complete(messages)`
    3. Extract text content from response
    4. Parse via `CurriculumDocument.from_string(text)`
    5. On parse failure: retry with error message appended to user message
    6. On second failure: raise `CurriculumGenerationError`
    7. Derive slug from goal: lowercase, replace non-alphanumeric with hyphens, strip leading/trailing hyphens, truncate to 60 chars
    8. Write markdown to `curricula_dir / f"{slug}.md"`
    9. Return the path
  - `curricula_dir` created with `mkdir(parents=True, exist_ok=True)` on write

### Task 4: Trainer Updates (`src/training/trainer/trainer.py`)

- **Spec ref:** @specs/curriculum §Session Startup, §File Polling, §Amendment Flow, §Progress Events, CRS-38..CRS-49
- **Test mapping:** CRS-38..CRS-49
- **Details:**
  - Constructor accepts optional `curriculum_file: Path | None`, `curricula_dir: Path | None`
  - `_resolve_session_start()` implements three-path resolution:
    1. `curriculum_file` provided → `CurriculumDocument.from_file()`, create Curriculum, start
    2. No file, saved state has `curriculum_file` → load from saved path
    3. No file, no saved state → set flag to poll on next `input` message
  - `_submit_next_lesson()`:
    1. If `curriculum_file` exists, re-read via `CurriculumDocument.from_file()`
    2. Update `self._state.curriculum` with re-read document
    3. Get current lesson kscript from `self._state.curriculum.current()`
    4. Compile and submit as before
  - Amendment handler: `_request_amendment(action, **kwargs)`:
    1. Call `document.amend(action, **kwargs)` on the current document
    2. Re-read document from disk
    3. Update curriculum state
    4. Emit progress event with status `"amended"`
    5. Resume from first unsatisfied lesson
  - Progress events via `_emit_progress(status)`:
    1. Build message dict with `lesson_label`, `lessons_total`, `lessons_completed`, `status`
    2. Send `Message(address="ui", action="progress", message={...})`
    3. Called from: `start_session`, `_check_lesson_complete`, `_end_session`, amendment handler
  - Cogitation context: build from `document.objective`, `document.approach`, current `lesson.prose`

### Task 5: Cogitation Context Update (`src/training/trainer/cogitation.py`)

- **Spec ref:** @specs/curriculum §CogitationRequest, CRS-50..CRS-52
- **Test mapping:** CRS-50..CRS-52
- **Details:**
  - Add `objective: str = ""`, `approach: str = ""`, `lesson_prose: str = ""` to `CogitationRequest`
  - Keep existing `curriculum_context: str = ""`
  - Update `build_prompt()`:
    1. If `objective` or `approach` or `lesson_prose` is non-empty: build context from them
    2. Otherwise fall back to `curriculum_context` for backward compat

### Task 6: Harness Wiring (`src/training/harness/__main__.py`)

- **Spec ref:** @specs/curriculum §Session Startup, CRS-38
- **Test mapping:** CRS-38
- **Details:**
  - In `trainer_factory`: extract `curriculum_file` and `curricula_dir` from `trainer_cfg`
  - Pass both to `Trainer` constructor as keyword arguments
  - Create `curricula/` directory if `curricula_dir` is specified and doesn't exist

### Task 7: Harness Config (`training.harness.yaml`)

- **Spec ref:** @specs/curriculum §Curricula Directory
- **Details:**
  - Rename `curriculum_path` to `curriculum_file`
  - Add `curricula_dir: "curricula"` under trainer section

## Test Mapping

| Spec ID | Test file | Test function | Status |
|---------|-----------|---------------|--------|
| CRS-1 | test_curriculum_document.py | test_from_file_parses_valid_document | ✅ |
| CRS-2 | test_curriculum_document.py | test_from_string_parses_valid_document | ✅ |
| CRS-3 | test_curriculum_document.py | test_rejects_missing_objective | ✅ |
| CRS-4 | test_curriculum_document.py | test_rejects_missing_approach | ✅ |
| CRS-5 | test_curriculum_document.py | test_rejects_missing_lessons_section | ✅ |
| CRS-6 | test_curriculum_document.py | test_rejects_no_lessons | ✅ |
| CRS-7 | test_curriculum_document.py | test_rejects_duplicate_labels | ✅ |
| CRS-8 | test_curriculum_document.py | test_rejects_invalid_label_format | ✅ |
| CRS-9 | test_curriculum_document.py | test_objective_property | ✅ |
| CRS-10 | test_curriculum_document.py | test_approach_property | ✅ |
| CRS-11 | test_curriculum_document.py | test_lessons_ordered | ✅ |
| CRS-12 | test_curriculum_document.py | test_lesson_label | ✅ |
| CRS-13 | test_curriculum_document.py | test_lesson_prose | ✅ |
| CRS-14 | test_curriculum_document.py | test_lesson_kscript | ✅ |
| CRS-15 | test_curriculum_document.py | test_find_lesson | ✅ |
| CRS-16 | test_curriculum_document.py | test_all_labels | ✅ |
| CRS-17 | test_curriculum_document.py | test_amend_insert | ✅ |
| CRS-18 | test_curriculum_document.py | test_amend_append | ✅ |
| CRS-19 | test_curriculum_document.py | test_amend_modify | ✅ |
| CRS-20 | test_curriculum_document.py | test_amend_raises_for_missing_target | ✅ |
| CRS-21 | test_curriculum_document.py | test_amend_raises_for_duplicate_label | ✅ |
| CRS-22 | test_curriculum_document.py | test_amend_raises_when_no_source_path | ✅ |
| CRS-23 | test_curriculum_document.py | test_amend_writes_to_file | ✅ |
| CRS-24 | test_curriculum.py | test_current_label_first_unsatisfied | ✅ |
| CRS-25 | test_curriculum.py | test_mark_lesson_submitted | ✅ |
| CRS-26 | test_curriculum.py | test_mark_lesson_satisfied_advances | ✅ |
| CRS-27 | test_curriculum.py | test_is_lesson_submitted_and_satisfied | ✅ |
| CRS-28 | test_curriculum.py | test_label_and_entry_tracking_coexist | ✅ |
| CRS-29 | test_curriculum.py | test_save_includes_label_state | ✅ |
| CRS-30 | test_curriculum.py | test_load_new_format_with_file | ✅ |
| CRS-31 | test_curriculum.py | test_load_legacy_format | ✅ |
| CRS-32 | test_curriculum_generator.py | test_generate_makes_llm_call | ✅ |
| CRS-33 | test_curriculum_generator.py | test_generate_parses_response | ✅ |
| CRS-34 | test_curriculum_generator.py | test_generate_retries_on_parse_failure | ✅ |
| CRS-35 | test_curriculum_generator.py | test_generate_raises_on_second_failure | ✅ |
| CRS-36 | test_curriculum_generator.py | test_generate_writes_to_file | ✅ |
| CRS-37 | test_curriculum_generator.py | test_generate_slug_from_goal | ✅ |
| CRS-38 | test_trainer.py | test_session_startup_from_file_param | ✅ |
| CRS-39 | test_trainer.py | test_session_startup_from_saved_state | ✅ |
| CRS-40 | test_trainer.py | test_session_startup_polls_for_goal | ✅ |
| CRS-41 | test_trainer.py | test_goal_prefix_triggers_generation | ✅ |
| CRS-42 | test_trainer.py | test_goal_file_path_triggers_load | ✅ |
| CRS-43 | test_trainer.py | test_trainer_rereads_file_before_lesson | ✅ |
| CRS-44 | test_trainer.py | test_new_lessons_submitted_after_reread | ✅ |
| CRS-45 | test_trainer.py | test_monotonic_set_prevents_duplicates | ✅ |
| CRS-46 | test_trainer.py | test_progress_event_session_start | ✅ |
| CRS-47 | test_trainer.py | test_progress_event_lesson_complete | ✅ |
| CRS-48 | test_trainer.py | test_progress_event_curriculum_complete | ✅ |
| CRS-49 | test_trainer.py | test_progress_event_amendment | ✅ |
| CRS-50 | test_cogitation.py | test_cogitation_request_new_fields | ✅ |
| CRS-51 | test_cogitation.py | test_build_prompt_prefers_new_fields | ✅ |
| CRS-52 | test_cogitation.py | test_build_prompt_falls_back_to_context | ✅ |
| CRS-53–58 | test_nlp_curriculum_compat.py | `TestCurriculumCompilation` (annotated curricula compile + rationalise; bare-sig compat preserved) | ✅ |

### Task 8: Curriculum Annotation Conventions (`curricula/*.md`)

Annotate all curriculum files so NLPTokenizer produces semantically rich
nodes. Per-curriculum changes live in the curriculum source files; no code
changes are required (the KScript compiler, binding resolver, and NLPTokenizer
already implement the mechanics, @kscript spec §9–10).

| Curriculum | Word theme |
| ---------- | ---------- |
| `first-steps.md`, `first-steps-s2.md` | identity names (Mark, Halo, Alpha) |
| `mhall-svo-single.md`, `mhall-svo-equivalence.md` | MHALL nursery rhyme + grammatical roles |
| `cascade-pressure.md`, `conflict-drill.md` | NATO phonetic alphabet |
| `s3-auto-countersign.md` | mixed (NATO + identity) |

Standalone KScript files under `data/` are already annotated.

## Design Decisions

### DD-1: Parser approach — regex vs line-by-line

**Decision:** Line-by-line parser splitting on `##` and `###` headings.
**Rationale:** Heading detection is straightforward with string matching. Regex
is used only for label validation (`\d+[a-z]?`). Line-by-line gives clear
control over section boundaries and avoids complex multi-line regex patterns.

### DD-2: Amendment serialization — round-trip fidelity

**Decision:** Serialize the full document from the in-memory model rather than
trying to splice text into the original file.
**Rationale:** Preserving exact formatting through text manipulation is fragile.
Reconstructing from the parsed model ensures structural correctness. Any
comments or extra whitespace outside lesson bodies is acceptable to lose.

### DD-3: Legacy compatibility — synthetic CurriculumDocument

**Decision:** When `Curriculum` receives a `list[str]`, create a synthetic
`CurriculumDocument` with empty sections and auto-generated numeric labels.
**Rationale:** Allows existing code that constructs `Curriculum(lessons=[...])`
to continue working without changes. The synthetic document has valid structure
and passes all validation rules.

### DD-4: Slug generation — deterministic, not unique

**Decision:** Slug is derived from goal text without collision detection.
**Rationale:** Same goal produces same slug → idempotent generation. If the user
wants a different curriculum for the same goal, they can rename the file.
Collisions are acceptable because overwriting is intentional in this workflow.

### DD-5: File polling vs file watching

**Decision:** Poll the curriculum file (re-read before each lesson
submission) rather than install a filesystem watcher.
**Rationale:** Polling is portable, needs no extra dependencies, and the
re-read cadence (once per lesson) is infrequent. Watchers add platform
fragility for no real latency gain at this loop speed.

### DD-6: Curriculum annotation word themes

Each curriculum uses a single word theme — identity names, MHALL nursery
rhyme, or NATO phonetic — rather than mixed themes, so curricula stay
self-documenting. NATO phonetic (Alpha, Beta, Charlie…) is used for
abstract-letter curricula because the words are semantically meaningful
nouns that produce clean NLP type bits and `A` is always `Alpha` (standard,
unambiguous). Block comments are placed only on the first appearance of a
multi-character sig in a code block; subsequent uses inherit the binding
via upward scope traversal.

**Decision:** Re-read file before each lesson submission (poll), not file system watches.
**Rationale:** Simpler, no platform dependencies, sufficient for the expected
amendment frequency (human or LLM-driven, not high-throughput).

### DD-6 note: `first-steps-s2.md` lesson-5 misfit operator

`first-steps-s2.md` lesson 5 (the S2-routing lesson) uses the `>` CONNOTED
operator, not `=>` CANONIZED: a CANONIZED compound kline is canonical by
construction (@kscript §11.4) and resolves S1 via AGT-14 before candidate
retrieval, so it can never be the S2 misfit the lesson intends. With
`(Mark Halo Alpha)\nMH > H A`, CONNOTED yields two misfit klines
`{MH:[Halo]}` (underfit, routes S2 against the lesson-3 `{Mark:[Halo]}`
countersign) and `{MH:[Alpha]}` (dual, escapes the auto-countersign backstop
and requests ratification). See @curriculum §Curriculum Annotation Conventions
rule 47.

### DD-6 note: `cascade-pressure.md` + `conflict-drill.md` misfit operators (KB-334)

The misfit lessons in `cascade-pressure.md` (L3/L4/L5) and `conflict-drill.md`
(L3/L4) use the `>` CONNOTED operator, not `=>` CANONIZED — the same rule-47
convention as `first-steps-s2.md` lesson 5. KB-334 audited all six other
curricula for the canonical-implies-not-a-misfit gap KB-320 found: these five
lessons previously used `=>` whose compound RHS is discarded (the §11.4 MTS
compound-sig resolution override) and whose kline is canonical by construction,
so they resolved S1 via AGT-14 and never reached candidate retrieval. The fix
rewrites each to `>` CONNOTED + block word lists; because a genuine misfit is
always single-node under the settled compiler contracts, multi-node `=>` intents
decompose into single-node misfits (e.g. `ABCD > E F` → `{ABCD:[Echo]}` +
`{ABCD:[Foxtrot]}`). `cascade-pressure.md` L5 (`ABCDEFGHIJ > A C E G`) uses four
nodes rather than ten: the "maximum candidate overlap" comes from the ten-atom
signature signifying every established kline, not the node count, and a full
ten-node decomposition would generate a pathological number of expansion events.
`s3-auto-countersign.md`, `mhall-svo-single.md`, `mhall-svo-equivalence.md`, and
`first-steps.md` were audited clean (rule-47-compliant or no S2/S3 intent). See
@curriculum rule 47 and `KB-334-notes.md`.

## Build Order

```
Task 1 (Parser) → Task 2 (State refactor) → Task 3 (Generator)
                                            ↓
Task 5 (Cogitation) ← ─────────────────────┘
     ↓
Task 4 (Trainer)
     ↓
Task 6 (Harness wiring) + Task 7 (Config)
```

Parser must be done first (Tasks 2 and 3 depend on it). State refactor must
precede Trainer updates. Generator can proceed in parallel with Cogitation
after the parser is done. Harness wiring is last.

## Status

| Task | Status   | Notes                          |
| ---- | -------- | ------------------------------ |
| 1    | ✅ done  | Parser                         |
| 2    | ✅ done  | State refactor                 |
| 3    | ✅ done  | Generator                      |
| 4    | ✅ done  | Trainer updates                |
| 5    | ✅ done  | Cogitation context             |
| 6    | ✅ done  | Harness wiring                 |
| 7    | ✅ done  | Config                         |
| 8    | ✅ done  | Curriculum annotation conventions |
