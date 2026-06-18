# Curriculum Specification

## Overview

The curriculum system manages structured training documents that drive Kalvin's
learning. A curriculum is a markdown document with three sections — objective,
approach, and lessons — where each lesson contains KScript source and
human-readable prose. The system parses these documents, tracks progress by
stable lesson labels, generates new curricula via LLM, and supports in-place
amendment without session interruption.

## Dependencies

This spec depends on the following concepts, defined elsewhere:

### KScript (@kscript spec)

- Provides `compile_source(source) → list[CompiledEntry]`.
- CompiledEntry has `signature` and `nodes` attributes used for structural matching.

### Trainer (@specs/harness-server)

- HRNS-IDs define the Trainer's existing harness integration.
- The Trainer drives the training loop, submits lessons, and manages state persistence.

### Cogitation (@specs/cogitation — internal to trainer)

- `CogitationRequest` carries context to the LLM for reactive scaffolding.
- `LLMClient` protocol abstracts the LLM API for testability.

## Definition

### CurriculumDocument

A parsed markdown curriculum with three required sections and ordered lessons.

| Field     | Type            | Description                                       |
| ------ | ------------------------------------------------------------------------------------ | ------------------------------------------- |
| objective | `str`           | Content of the `## Objective` section.            |
| approach  | `str`           | Content of the `## Approach` section.             |
| lessons   | `list[Lesson]`  | Ordered lessons parsed from `## Lessons` section.  |
| source_path | `Path \| None` | File path the document was loaded from (for amendment write-back). |

### Lesson

A single lesson within a curriculum, bounded by an `### <label>` heading.

| Field   | Type           | Description                                            |
| ------ | ------------------------------------------------------------------------------------ | ------------------------------------------- |
| label   | `str`          | Stable label from the heading (e.g. `"1"`, `"2a"`).     |
| prose   | `str`          | Non-code-block text — human-readable context.           |
| kscript | `list[str]`    | Contents of fenced code blocks (KScript source lines).  |

### CurriculumState (updated)

Per-session tracking state extended with label-based lesson tracking.

| Field          | Type                    | Description                                          |
| ------ | ------------------------------------------------------------------------------------ | ------------------------------------------- |
| curriculum     | `Curriculum`            | The curriculum being tracked.                        |
| current_label  | `str \| None`           | Label of the next unsatisfied lesson.                |
| lesson_submitted | `set[str]`            | Labels of lessons whose entries have been submitted.  |
| lesson_satisfied | `set[str]`            | Labels of lessons whose entries are satisfied.        |
| curriculum_file | `str \| None`           | Path to the curriculum file for persistence.          |

Existing `submitted`, `satisfied`, and `pending` sets (keyed by `EntryKey`) are
retained for per-entry tracking within a lesson.

### CurriculumGenerator

LLM-based generator that produces a curriculum from a goal string.

| Field         | Type        | Description                                  |
| ------ | ------------------------------------------------------------------------------------ | ------------------------------------------- |
| client        | `LLMClient` | LLM client for chat completions.             |
| curricula_dir | `Path`      | Directory to write generated curriculum files. |

### CogitationRequest (updated)

Extended with structured curriculum context fields.

| Field              | Type                      | Description                                  |
| ------ | ------------------------------------------------------------------------------------ | ------------------------------------------- |
| objective          | `str`                     | Curriculum objective section content.        |
| approach           | `str`                     | Curriculum approach section content.         |
| lesson_prose       | `str`                     | Current lesson's prose text.                 |
| curriculum_context | `str`                     | Legacy flat context (backward compatibility). |

## API

### CurriculumDocument

| Operation                                      | Description                                               |
| ------ | ------------------------------------------------------------------------------------ | ------------------------------------------- |
| `CurriculumDocument.from_file(path) → CurriculumDocument` | Parse and validate a markdown file.          |
| `CurriculumDocument.from_string(text) → CurriculumDocument` | Parse and validate from string.            |
| `document.objective → str`                     | Objective section content.                                |
| `document.approach → str`                      | Approach section content.                                 |
| `document.lessons → list[Lesson]`              | Ordered lesson list.                                      |
| `document.find_lesson(label) → Lesson \| None` | Find lesson by label.                                     |
| `document.all_labels() → list[str]`            | All lesson labels in order.                               |
| `document.amend(action, **kwargs) → None`      | Mutate and write back: insert, append, or modify lesson.  |

### Lesson

| Operation                  | Description                                        |
| ------ | ------------------------------------------------------------------------------------ | ------------------------------------------- |
| `lesson.label → str`       | Stable label string.                               |
| `lesson.prose → str`       | Non-code-block text.                               |
| `lesson.kscript → list[str]` | KScript source strings from code blocks.         |

### CurriculumState (label-based operations)

| Operation                                       | Description                                         |
| ------ | ------------------------------------------------------------------------------------ | ------------------------------------------- |
| `state.current_label → str \| None`             | Next unsatisfied lesson label.                      |
| `state.mark_lesson_submitted(label) → None`     | Mark a lesson's entries as submitted.               |
| `state.mark_lesson_satisfied(label) → None`     | Mark a lesson's entries as satisfied.               |
| `state.is_lesson_submitted(label) → bool`       | Check if a lesson has been submitted.               |
| `state.is_lesson_satisfied(label) → bool`       | Check if a lesson has been satisfied.               |

### CurriculumGenerator

| Operation                              | Description                                        |
| ------ | ------------------------------------------------------------------------------------ | ------------------------------------------- |
| `generator.generate(goal) → Path`     | Generate curriculum from goal, write to file, return path. |

### Session Startup Resolution

| Step | Condition                                       | Action                                     |
| ------ | ------------------------------------------------------------------------------------ | ------------------------------------------- |
| 1    | `curriculum_file` parameter provided            | Load file, create Curriculum, start.       |
| 2    | No param, saved state has `curriculum_file`     | Resume from persisted path.                |
| 3    | No param, no saved state                        | Poll for goal via bus input.               |

### Goal Resolution

| Input Format         | Action                                           |
| ------ | ------------------------------------------------------------------------------------ | ------------------------------------------- |
| Text starting with `goal:` | Extract goal text, generate curriculum via LLM, write to the curricula directory. |
| File path (existing file)  | Load curriculum directly from file.              |

### Progress Events

| Event                | Message Shape                                                              |
| ------ | ------------------------------------------------------------------------------------ | ------------------------------------------- |
| Session start        | `{address: "ui", action: "progress", message: {lesson_label, lessons_total, lessons_completed, status: "started"}}` |
| Lesson complete      | `{address: "ui", action: "progress", message: {lesson_label, lessons_total, lessons_completed, status: "lesson_complete"}}` |
| Curriculum complete  | `{address: "ui", action: "progress", message: {lesson_label: null, lessons_total, lessons_completed, status: "complete"}}` |
| Amendment applied    | `{address: "ui", action: "progress", message: {lesson_label, lessons_total, lessons_completed, status: "amended"}}` |

### Amendment Actions

| Action   | Parameters                        | Description                                    |
| ------ | ------------------------------------------------------------------------------------ | ------------------------------------------- |
| `insert` | `after_label`, `lesson`           | Insert a new lesson after the specified label. |
| `append` | `lesson`                          | Append a new lesson at the end.                |
| `modify` | `label`, `lesson`                 | Replace the lesson at the given label.         |

## Behavioural Rules

### CurriculumDocument Parsing

1. The document must contain exactly the three required `##` headings:
   `Objective`, `Approach`, and `Lessons`, in any order.
2. At least one lesson must exist under `## Lessons`.
3. Each lesson is an `### <label>` heading. The label must match the
   convention: one or more digits, optionally followed by a single
   lowercase letter (e.g. `"1"`, `"2a"`, `"12"`).
4. No two lessons may share the same label.
5. Lesson content between two consecutive `###` headings (or between the
   last heading and end of document) belongs to the earlier lesson.
6. Fenced code blocks (triple backticks) within a lesson are extracted as
   KScript source. All other text is prose.
7. `from_file` records the source path for later amendment write-back.
8. `from_string` sets `source_path` to `None`.

### CurriculumDocument Amendment

9. `amend("insert", after_label=<str>, lesson=Lesson(...))` inserts the
   new lesson immediately after the specified label and rewrites the file.
10. `amend("append", lesson=Lesson(...))` appends the lesson at the end
    and rewrites the file.
11. `amend("modify", label=<str>, lesson=Lesson(...))` replaces the
    lesson with the given label and rewrites the file.
12. Amendment raises `ValueError` if `after_label` or `label` does not
    exist, or if the new lesson's label duplicates an existing one.
13. Amendment requires `source_path` to be set; raises `ValueError`
    otherwise.
14. After amendment, the serialized markdown preserves section structure
    and formatting.

### CurriculumState Label Tracking

15. `current_label` returns the first label in document order that is not
    in `lesson_satisfied`.
16. `mark_lesson_submitted(label)` adds the label to `lesson_submitted`.
17. `mark_lesson_satisfied(label)` adds the label to `lesson_satisfied`
    and advances `current_label` to the next unsatisfied label.
18. Label-based tracking is independent of entry-level `EntryKey` sets.
    Both coexist: labels track lesson-level progress; entries track
    individual kline status within a lesson.

### CurriculumState Persistence

19. `save()` serializes label-based state (`lesson_submitted`,
    `lesson_satisfied`, `current_label`, `curriculum_file`) alongside
    existing entry-level sets and event log.
20. `load()` reconstructs state from JSON, supporting both new format
    (with `curriculum_file` and label fields) and legacy format (flat
    `lessons` list with positional `position`). Legacy format creates a
    synthetic document with default sections and numeric labels.

### Curriculum Generation

21. The generator makes one LLM call with a system prompt describing the
    curriculum markdown format and a user prompt containing the goal.
22. The LLM response is parsed via `CurriculumDocument.from_string()`.
    If parsing fails, a single retry is attempted with the error message
    included in the prompt.
23. If the second attempt fails, `CurriculumGenerationError` is raised.
24. The validated curriculum is written as markdown into the generator's
    `curricula_dir`, named by a slug derived from the goal (lowercase, hyphens,
    non-alphanumeric characters stripped, truncated to 60 characters).
25. `generate()` returns the `Path` of the written file.

### Session Startup

26. Resolution follows the three-path order: runtime parameter → saved
    state → bus poll.
27. When polling, the Trainer waits for an `input` message. If the text
    starts with `goal:`, the Trainer generates a curriculum. If the text
    is a path to an existing file, the Trainer loads it directly.
28. A session is started only after a valid curriculum is resolved.

### File Polling

29. Before submitting each lesson, the Trainer re-reads the curriculum
    file from disk via `CurriculumDocument.from_file()`.
30. If the re-read document has new lessons after the current label that
    are not in the submitted set, they are queued for submission.
31. If the current lesson's KScript content changed, the updated version
    is submitted. The monotonic entry-level submitted set prevents
    duplicate submissions.

### Amendment Flow

32. Any participant may request an amendment via the Trainer.
33. The Trainer calls `CurriculumDocument.amend()` to modify the file
    in place.
34. After amendment, the Trainer re-reads the document and restarts
    lesson processing from the first unsatisfied lesson.
35. Kalvin's monotonic submitted set ensures only new klines are
    compiled and submitted on replay.

### Progress Events

36. Progress events are emitted on: session start, lesson complete,
    curriculum complete, amendment applied.
37. `lessons_completed` is the count of satisfied lesson labels.
38. `lessons_total` is the total count of lessons in the current document.

### Lesson Completion

39. Lesson completion is determined by satisfaction count:
    `len(state.satisfied) >= len(state.submitted)` and
    `len(state.submitted) > 0`. This replaces event counting because
    cogitation generates multiple events per entry (initial rationalise
    + expansion proposals from S3 candidates).
40. Lesson completion does not re-fire: the check returns early if the
    current lesson is already in `lesson_satisfied` or if there is no
    current lesson (curriculum position past the end). This prevents
    duplicate completion when post-completion cogitation events arrive.

### Curriculum Annotation Conventions

Curriculum KScript signatures carry NLP annotations so the NLPTokenizer
produces semantically rich graph nodes rather than abstract-letter
fallbacks. The annotation *mechanics* (block/inline forms, binding
resolution, claiming rules) are defined in the @kscript spec (§9 BPE
Annotations, §10 NLP Binding Resolution); this section fixes the
*conventions* every curriculum follows.

41. Every single-character signature in a KScript code block MUST carry an
    inline parenthetical comment (`M(ark)`, `H(alo)`, `S(ubject)`). The
    comment text combined with the signature character forms the full NLP
    word.
42. Every multi-token signature MUST be preceded by a block-comment word
    list whose word count matches the character count (e.g.
    `(Mary Had A Little Lamb)` before `MHALL`).
43. Node-position single-character signatures receive an inline comment
    ONLY when the binding would introduce new information. If the character
    is already bound to the same word in the scope chain, the inline
    annotation is redundant and MUST be omitted — the node inherits the
    binding from the parent scope. Inline annotations that intentionally
    shadow a parent binding with a different word are required.
44. Word choices SHOULD be semantically meaningful (proper nouns, common
    nouns, or NATO phonetic alphabet) so NLP-BPE tokens carry grammatically
    useful type bits.
45. NLP annotations appear ONLY inside fenced KScript code blocks. The
    curriculum prose (objective, approach, lesson descriptions) is not
    annotated.
46. The BindingResolver detects and skips inline annotations that would
    redundantly re-bind a character to the same word already present in the
    scope chain (via `NLPSymbolTable.is_bound_to(char, word)`). Redundant
    bindings are silently skipped with a debug-level log message.
47. A lesson that intends a kline to reach the S2/S3 expansion band (a
    *misfit* — its signature does not match its nodes; see @kalvin-vision §S2
    Expansion) MUST NOT use the `=>` CANONIZED operator. A CANONIZED compound
    definition sets its signature to `make_signature(nodes)` by construction
    (@kscript §11.4 canonical encoding), so the kline is canonical, never a
    misfit, and resolves S1 via the self-grounded short-circuit (@agent AGT-14)
    before any candidate is retrieved. Express an S2/S3 misfit with the `>`
    CONNOTED operator (or any operator) using a signature whose value differs
    from `make_signature(nodes)`. Note also that for a *compound* (multi-char)
    signature, the CANONIZED/CONNOTED right-hand nodes are taken from the
    signature's own MTS resolution; bind the intended node words with a block
    word list rather than relying on the right-hand side.

## Test Matrix

| ID     | Criterion                                                                            | Spec ref                                    |
| ------ | ------------------------------------------------------------------------------------ | ------------------------------------------- |
| CRS-1  | `from_file` parses a valid curriculum file with all three sections                   | @curriculum §CurriculumDocument Parsing     |
| CRS-2  | `from_string` parses a valid curriculum from a string                                | @curriculum §CurriculumDocument Parsing     |
| CRS-3  | Parsing rejects a document missing `## Objective`                                    | @curriculum §CurriculumDocument Parsing     |
| CRS-4  | Parsing rejects a document missing `## Approach`                                     | @curriculum §CurriculumDocument Parsing     |
| CRS-5  | Parsing rejects a document missing `## Lessons`                                      | @curriculum §CurriculumDocument Parsing     |
| CRS-6  | Parsing rejects a document with no lessons                                           | @curriculum §CurriculumDocument Parsing     |
| CRS-7  | Parsing rejects a document with duplicate lesson labels                              | @curriculum §CurriculumDocument Parsing     |
| CRS-8  | Parsing rejects a lesson label that does not match the convention                    | @curriculum §CurriculumDocument Parsing     |
| CRS-9  | `document.objective` returns the Objective section content                           | @curriculum §CurriculumDocument             |
| CRS-10 | `document.approach` returns the Approach section content                             | @curriculum §CurriculumDocument             |
| CRS-11 | `document.lessons` returns ordered Lesson objects                                    | @curriculum §CurriculumDocument             |
| CRS-12 | `Lesson.label` returns the stable heading label                                      | @curriculum §Lesson                         |
| CRS-13 | `Lesson.prose` returns non-code-block text                                           | @curriculum §Lesson                         |
| CRS-14 | `Lesson.kscript` returns contents of fenced code blocks                              | @curriculum §Lesson                         |
| CRS-15 | `find_lesson(label)` returns the matching Lesson or None                             | @curriculum §CurriculumDocument             |
| CRS-16 | `all_labels()` returns labels in document order                                      | @curriculum §CurriculumDocument             |
| CRS-17 | `amend("insert", ...)` inserts a lesson after the specified label                    | @curriculum §CurriculumDocument Amendment   |
| CRS-18 | `amend("append", ...)` appends a lesson at the end                                   | @curriculum §CurriculumDocument Amendment   |
| CRS-19 | `amend("modify", ...)` replaces the lesson at the given label                        | @curriculum §CurriculumDocument Amendment   |
| CRS-20 | Amendment raises ValueError for nonexistent target label                             | @curriculum §CurriculumDocument Amendment   |
| CRS-21 | Amendment raises ValueError for duplicate label                                      | @curriculum §CurriculumDocument Amendment   |
| CRS-22 | Amendment raises ValueError when source_path is None                                 | @curriculum §CurriculumDocument Amendment   |
| CRS-23 | Amendment writes updated markdown to the source file                                 | @curriculum §CurriculumDocument Amendment   |
| CRS-24 | `current_label` returns first unsatisfied label in document order                    | @curriculum §CurriculumState Label Tracking |
| CRS-25 | `mark_lesson_submitted` adds label to submitted set                                  | @curriculum §CurriculumState Label Tracking |
| CRS-26 | `mark_lesson_satisfied` adds label to satisfied set and advances current_label       | @curriculum §CurriculumState Label Tracking |
| CRS-27 | `is_lesson_submitted` and `is_lesson_satisfied` check label membership               | @curriculum §CurriculumState Label Tracking |
| CRS-28 | Label-based tracking coexists with entry-level EntryKey tracking                     | @curriculum §CurriculumState Label Tracking |
| CRS-29 | `save()` includes curriculum_file, labels, and label-based state in JSON             | @curriculum §CurriculumState Persistence    |
| CRS-30 | `load()` reconstructs state from new format JSON                                     | @curriculum §CurriculumState Persistence    |
| CRS-31 | `load()` handles legacy format (flat lessons + position) with backward compat        | @curriculum §CurriculumState Persistence    |
| CRS-32 | Generator makes one LLM call with curriculum format system prompt                    | @curriculum §Curriculum Generation          |
| CRS-33 | Generator parses LLM response via `from_string` and validates                        | @curriculum §Curriculum Generation          |
| CRS-34 | Generator retries once on parse failure with error feedback                          | @curriculum §Curriculum Generation          |
| CRS-35 | Generator raises `CurriculumGenerationError` on second failure                       | @curriculum §Curriculum Generation          |
| CRS-36 | Generator writes validated markdown to the curriculum file                            | @curriculum §Curriculum Generation          |
| CRS-37 | Generator derives slug from goal (lowercase, hyphens, non-alphanumeric stripped)     | @curriculum §Curriculum Generation          |
| CRS-38 | Session startup loads curriculum from runtime parameter                              | @curriculum §Session Startup                |
| CRS-39 | Session startup resumes from saved state with curriculum_file                        | @curriculum §Session Startup                |
| CRS-40 | Session startup polls for goal when no param and no saved state                      | @curriculum §Session Startup                |
| CRS-41 | Goal starting with `goal:` triggers curriculum generation                            | @curriculum §Goal Resolution                |
| CRS-42 | Goal that is a file path triggers direct loading                                     | @curriculum §Goal Resolution                |
| CRS-43 | Trainer re-reads curriculum file before each lesson submission                       | @curriculum §File Polling                   |
| CRS-44 | New lessons after current label are submitted after re-read                          | @curriculum §File Polling                   |
| CRS-45 | Monotonic submitted set prevents duplicate kline submissions on replay               | @curriculum §File Polling                   |
| CRS-46 | Progress event emitted on session start                                              | @curriculum §Progress Events                |
| CRS-47 | Progress event emitted on lesson complete                                            | @curriculum §Progress Events                |
| CRS-48 | Progress event emitted on curriculum complete                                        | @curriculum §Progress Events                |
| CRS-49 | Progress event emitted on amendment applied                                          | @curriculum §Progress Events                |
| CRS-50 | `CogitationRequest` accepts objective, approach, and lesson_prose fields             | @curriculum §CogitationRequest              |
| CRS-51 | `build_prompt()` prefers objective + approach + lesson_prose over curriculum_context | @curriculum §CogitationRequest              |
| CRS-52 | `build_prompt()` falls back to curriculum_context when new fields are empty          | @curriculum §CogitationRequest              |
| CRS-53 | Every single-character sig in a code block has an inline parenthetical comment        | @curriculum §Curriculum Annotation Conventions |
| CRS-54 | Every multi-token sig is preceded by a block word list whose count matches            | @curriculum §Curriculum Annotation Conventions |
| CRS-55 | Redundant node-position inline annotations are omitted (parent-scope binding inherited) | @curriculum §Curriculum Annotation Conventions |
| CRS-56 | Intentional shadowing node-position inline annotations are required                   | @curriculum §Curriculum Annotation Conventions |
| CRS-57 | Annotations appear only inside fenced KScript code blocks, never in prose             | @curriculum §Curriculum Annotation Conventions |
| CRS-58 | BindingResolver silently skips redundant re-binds (is_bound_to) with a debug log      | @curriculum §Curriculum Annotation Conventions |

## Out of Scope

- Curriculum versioning or revision history — the curriculum only moves forward.
- File watchers or bus-based file change notifications — the Trainer polls
  before each lesson.
- Multi-session support — one session at a time.
- Human ratification of amendments — amendments are applied immediately.
- Curriculum document signing or integrity verification.
- Nested lessons or multi-level lesson hierarchies beyond `###` headings.
- LLM prompt optimisation or few-shot examples for generation.
- Integration with the Slack participant beyond existing notify/input actions.
