# Text Analyzer

Extract named entities and grammatical tags from text files using spaCy NLP.

## Features

- Named Entity Recognition (NER)
- Part-of-speech (POS) tagging
- Dependency parsing
- Morphological analysis
- Noun chunk extraction
- Verb lemma extraction
- Word frequency counting with percentage
- **Triple NLP type encoding**:
  - `nlp_type32`: 32-bit coarse-grained bit patterns
  - `nlp_type48`: 48-bit finer-grained bit patterns
  - `nlp_fine_type`: Dynamic fine-grained bit patterns
- GPU acceleration support (Apple Silicon MPS / NVIDIA CUDA)

## Setup

1. Install dependencies with GPU support:

**Apple Silicon (M1/M2/M3):**

```bash
uv add "spacy[apple]"
```

**NVIDIA CUDA:**

```bash
uv add "spacy[cuda11x]"  # Replace 11x with your CUDA version
```

**CPU only:**

```bash
uv add spacy
```

2. spaCy models are **auto-downloaded** on first use. No manual installation needed!

   Supported models: `en_core_web_sm`, `en_core_web_md`, `en_core_web_lg`, `en_core_web_trf`

## Usage

Basic usage (CPU):

```bash
uv run python dev/nlp_analyzer/nlp_analyzer.py -v
```

Enable GPU acceleration:

```bash
uv run python dev/nlp_analyzer/nlp_analyzer.py -v --gpu
```

Use transformer model (better GPU utilization):

```bash
uv run python dev/nlp_analyzer/nlp_analyzer.py -v --gpu -m en_core_web_trf
```

Custom input file:

```bash
uv run python dev/nlp_analyzer/nlp_analyzer.py -i /path/to/file.json -v
```

Extend existing dictionaries:

```bash
uv run python dev/nlp_analyzer/nlp_analyzer.py -i input.json -e existing.json -v
```

Use high-bit mode for NLP types (useful for combining with other bit patterns):

```bash
uv run python dev/nlp_analyzer/nlp_analyzer.py -v --high-bits
```

### CLI Options

| Flag               | Description                                                           |
| ------------------ | --------------------------------------------------------------------- |
| `-i, --input`      | Input JSON or text file                                               |
| `-o, --output`     | Output directory (default: same as input)                             |
| `-m, --model`      | spaCy model (default: `en_core_web_sm`)                               |
| `-g, --gpu`        | Enable GPU acceleration                                               |
| `-b, --batch-size` | Batch size for processing (default: 50)                               |
| `-v, --verbose`    | Print detailed output                                                 |
| `-e, --existing`   | Extend existing dictionary files                                      |
| `--high-bits`      | Use high bits for NLP types (bits 32-63 for 32-bit, 16-63 for 48-bit) |

## Input Format

Supports:

- JSON files with array of objects containing `summary` field
- JSON files with single object containing `summary` field
- Plain text files

## Output

Creates separate JSON files for each analysis type:

### `{stem}_grammar.json` - Combined grammatical table

Keys are BPE token IDs (first token of the word's BPE encoding):

```json
{
  "1234": {
    "text": "Tea",
    "pos": "PROPN",
    "pos_fine": "NNP",
    "dep": "nsubj",
    "morph": "Number=Sing",
    "count": 2,
    "tokens": [1234],
    "frequency_pct": 0.0001,
    "nlp_type32": 33685632,
    "nlp_type48": 33685632,
    "nlp_fine_type": 67641376
  }
}
```

Three type encodings are provided:

| Field           | Bits      | Description                           |
| --------------- | --------- | ------------------------------------- |
| `nlp_type32`    | 32        | Coarse-grained, fits in 32-bit int    |
| `nlp_type48`    | 48        | Finer-grained, more DEP/MORPH detail  |
| `nlp_fine_type` | unlimited | All fine-grained features from corpus |

### `{stem}_nlp_type32.json` - 32-bit coarse NLP type legend

Hardcoded flags that always fit in 32 bits:

```json
{
  "POS_ADJ": 1,
  "POS_NOUN": 128,
  "POS_VERB": 32768,
  "DEP_SUBJ": 131072,
  "DEP_OBJ": 262144,
  "MORPH_SING": 33554432,
  "MORPH_PAST": 134217728
}
```

**32-bit bit layout (default low-bit mode):**

- Bits 0-16: Coarse POS tags (17 Universal POS tags)
- Bits 17-24: Simplified dependency groups (8 groups)
- Bits 25-31: Simplified morph features (7 features)

**32-bit bit layout (high-bit mode with `--high-bits`):**

- Bits 32-48: Coarse POS tags (17 Universal POS tags)
- Bits 49-56: Simplified dependency groups (8 groups)
- Bits 57-63: Simplified morph features (7 features)

### `{stem}_nlp_type48.json` - 48-bit finer NLP type legend

Hardcoded flags with more granular DEP and MORPH:

```json
{
  "POS_ADJ": 1,
  "POS_VERB": 32768,
  "DEP_CCOMP": 2097152,
  "DEP_ADVCL": 8388608,
  "DEP_AMOD": 33554432,
  "MORPH_SING": 4294967296,
  "MORPH_PROG": 2199023255552,
  "MORPH_INF": 35184372088832
}
```

**48-bit bit layout (default low-bit mode):**

- Bits 0-16: Coarse POS tags (17 Universal POS tags, same as 32-bit)
- Bits 17-31: Finer dependency groups (15 groups)
- Bits 32-47: Finer morph features (16 features)

**48-bit bit layout (high-bit mode with `--high-bits`):**

- Bits 16-32: Coarse POS tags (17 Universal POS tags)
- Bits 33-47: Finer dependency groups (15 groups)
- Bits 48-63: Finer morph features (16 features)

**Finer DEP groups (48-bit):**
| Group | Dependencies | Description |
|-------|--------------|-------------|
| `DEP_SUBJ` | nsubj, nsubjpass, csubj, csubjpass, agent | Clause subjects |
| `DEP_OBJ` | obj, iobj, dobj | Direct/indirect objects |
| `DEP_OBL` | obl, obl:_ | Oblique nominals (adjuncts) |
| `DEP_NMOD` | nmod, nmod:_ | Nominal modifiers |
| `DEP_CCOMP` | ccomp | Clausal complements |
| `DEP_XCOMP` | xcomp | Open clausal complements |
| `DEP_ADVCL` | advcl | Adverbial clause modifiers |
| `DEP_ACL` | acl, acl:relcl | Adnominal clause modifiers |
| `DEP_AMOD` | amod | Adjectival modifiers |
| `DEP_ADVMOD` | advmod | Adverbial modifiers |
| `DEP_NUMMOD` | nummod, nummod:\* | Numeral modifiers |
| `DEP_APPOS` | appos | Appositional modifiers |
| `DEP_FUNC` | det, case, mark, aux, auxpass, cop, expl, neg | Function words |
| `DEP_STRUCT` | root, conj, cc, compound, flat, fixed, list, parataxis, discourse | Structural relations |
| `DEP_PUNCT` | punct, goeswith, reparandum, orphan | Punctuation and repairs |

**Finer MORPH features (48-bit):**
| Flag | Feature | Description |
|------|---------|-------------|
| `MORPH_SING` | Number=Sing | Singular number |
| `MORPH_PLUR` | Number=Plur | Plural number |
| `MORPH_PAST` | Tense=Past | Past tense |
| `MORPH_PRES` | Tense=Pres | Present tense |
| `MORPH_FUT` | Tense=Fut | Future tense |
| `MORPH_PASS` | Voice=Pass | Passive voice |
| `MORPH_PERSON_1` | Person=1 | First person |
| `MORPH_PERSON_2` | Person=2 | Second person |
| `MORPH_PERSON_3` | Person=3 | Third person |
| `MORPH_PERF` | Aspect=Perf | Perfective aspect |
| `MORPH_PROG` | Aspect=Prog | Progressive aspect |
| `MORPH_IND` | Mood=Ind | Indicative mood |
| `MORPH_IMP` | Mood=Imp | Imperative mood |
| `MORPH_INF` | VerbForm=Inf | Infinitive verb form |
| `MORPH_PART` | VerbForm=Part | Participle verb form |
| `MORPH_GER` | VerbForm=Ger | Gerund verb form |

### `{stem}_nlp_fine_types.json` - Fine-grained NLP type legend

Dynamic flags discovered during processing (unlimited bits):

```json
{
  "POS_ADJ": 1,
  "POS_NOUN": 2,
  "POS_FINE_NNP": 1024,
  "POS_FINE_NN": 2048,
  "DEP_NSUBJ": 4096,
  "MORPH_NUMBER_SING": 32768
}
```

### Decoding type values

```python
import json

# Load legends
with open("output_nlp_type32.json") as f:
    type32_legend = json.load(f)
with open("output_nlp_type48.json") as f:
    type48_legend = json.load(f)

# Load grammar
with open("output_grammar.json") as f:
    grammar = json.load(f)

# Check 32-bit coarse features for a word
entry = next(e for e in grammar.values() if e["text"] == "ran")
nlp_type32 = entry["nlp_type32"]
if nlp_type32 & type32_legend["POS_VERB"]:
    print("'ran' is a verb (32-bit)")

# Check 48-bit finer features
nlp_type48 = entry["nlp_type48"]
if nlp_type48 & type48_legend["MORPH_PROG"]:
    print("'ran' has progressive aspect (48-bit)")

# Find all verbs using 32-bit type
VERB_FLAG = type32_legend["POS_VERB"]
verbs = {data["text"]: data for data in grammar.values()
         if data["nlp_type32"] & VERB_FLAG}
```

### `{stem}_ner.json` - Named entities

```json
{
  "Kim": "PERSON",
  "Tuesday": "DATE",
  "New York": "GPE"
}
```

### `{stem}_verbs.json` - Verb lemma counts

```json
{
  "run": 150,
  "eat": 89,
  "sleep": 45
}
```

### `{stem}_noun_chunks.json` - Noun chunk counts

```json
{
  "the dog": 23,
  "a big house": 12
}
```

## Grammar Expansion

The `expand_grammar.py` script expands coverage of an existing grammar dict without
re-running the full NLP pipeline. It closes gaps in BPE token coverage using three
mechanisms:

1. **Special-token annotation** — Assigns NLP tags to whitespace, punctuation, digits,
   and control characters using deterministic pattern rules (no spaCy needed).
2. **Subword token inheritance** — Resolves BPE subword fragments (e.g., "ning", "ing")
   by finding parent words in the grammar dict and inheriting their POS/DEP/MORPH tags.
3. **Multi-source merge** — Combines grammar dicts from multiple corpora without
   overwriting existing entries.
4. **Manual annotation** — Hardcoded annotations for BPE artifacts (contraction stems,
   negation clitic, newline-composite punctuation) that no automated strategy can
   resolve. Brings coverage to 100%.

### Usage

Expand the default grammar dict:

```bash
uv run python dev/nlp/expand_grammar.py \
    --grammar data/tokenizer/simplestories-1_grammar.json \
    --output data/tokenizer/simplestories-1_grammar.json
```

Dry run (print stats without writing):

```bash
uv run python dev/nlp/expand_grammar.py \
    --grammar data/tokenizer/simplestories-1_grammar.json \
    --dry-run
```

Merge an additional pre-computed grammar dict:

```bash
uv run python dev/nlp/expand_grammar.py \
    --grammar data/tokenizer/simplestories-1_grammar.json \
    --extra-grammar data/tokenizer/openwebtext_grammar.json \
    --output data/tokenizer/simplestories-1_grammar.json
```

### CLI Options

| Flag               | Description                                                           |
| ------------------ | --------------------------------------------------------------------- |
| `--grammar`        | (Required) Path to base grammar JSON file                             |
| `--tokenizer-dir`  | Directory containing tokenizer files (default: `data/tokenizer`)      |
| `--tokenizer-name` | Tokenizer variant name (default: `tokenizer-32768`)                   |
| `--extra-grammar`  | Pre-computed grammar JSON to merge. Can be specified multiple times   |
| `--output`         | Path to write expanded grammar (default: same as `--grammar`)         |
| `--dry-run`        | Print before/after stats without writing                              |

### Expansion strategies

**Special-token annotation** covers tokens that never appear as words in natural text:

| Token type      | POS        | pos_fine | dep       |
| --------------- | ---------- | -------- | --------- |
| Whitespace      | `SPACE`    | `_SP`    | (empty)   |
| Punctuation     | `PUNCT`    | varies   | `punct`   |
| Digits          | `NUM`      | `CD`     | `nummod`  |
| Control chars   | `X`        | (empty)  | (empty)   |

**Subword inheritance** resolves BPE fragments by searching the grammar dict for parent
words containing the fragment as a substring. When multiple parents match, the most
frequent parent (highest `count`) is preferred. Inherited entries get `count=0` and
`frequency_pct=0.0` since they weren't observed directly in the corpus.

**Multi-source merge** adds entries from other grammar dicts (e.g., produced by running
`nlp_analyzer.py` on a different corpus) without overwriting existing entries. Counts
and frequencies from merged sources are reset to zero.

**Manual annotation** handles the final set of BPE tokens that no automated strategy
can resolve. These are BPE artifacts that never appear as standalone words in any corpus:

- **Contraction stems** — BPE fragments like "didn", "couldn", "shouldn" that are the
  first part of contractions ("didn't", "couldn't", "shouldn't"). Annotated as `AUX`
  with the appropriate fine POS tag (VBD for past-tense, MD for modals, VBZ for
  present-tense).
- **Negation clitic** — The `"'t"` token (from don't, isn't, etc.) annotated as
  `PART/RB/neg` with `Polarity=Neg`.
- **Newline-composite punctuation** — BPE tokens like `"\\n\\n"`, `"---\\n\\n"`, `":\\n\\n"`
  that combine punctuation with paragraph breaks. Annotated as `PUNCT` with the
  appropriate fine tag.
- **Special symbols** — The underscore `_` and the full word `cannot`.

These 23 tokens are hardcoded in `annotate_manual_tokens()` and applied as the final
step of the expansion pipeline. This brings coverage to 100% of the 17,392-token
BPE vocabulary.

## Performance Tips

1. **Use GPU with transformer model** for large datasets:

   ```bash
   uv run python dev/nlp_analyzer/nlp_analyzer.py --gpu -m en_core_web_trf
   ```

2. **Increase batch size** for faster processing:

   ```bash
   uv run python dev/nlp_analyzer/nlp_analyzer.py -b 100
   ```

3. **Standard model (`en_core_web_sm`)** is CPU-optimized and already fast without GPU

4. **Transformer model (`en_core_web_trf`)** benefits significantly from GPU acceleration

## Corpus Runner

The `run_corpus.py` script runs the NLP analysis pipeline on external HuggingFace
datasets. It downloads texts, runs spaCy analysis, and saves all outputs (grammar,
NER, verbs, noun chunks, fine-type legends).

### Setup

Install the optional corpus dependency:

```bash
uv pip install -e '.[corpus]'
```

### Usage

Run on OpenWebText (10K streamed samples):

```bash
uv run python dev/nlp/run_corpus.py \
    --dataset Skylion007/openwebtext \
    --max-samples 10000 \
    --stem openwebtext \
    --output data/tokenizer \
    --verbose
```

Run on a custom dataset:

```bash
uv run python dev/nlp/run_corpus.py \
    --dataset wikipedia \
    --text-field content \
    --stem wiki \
    --max-samples 5000
```

### CLI Options

| Flag               | Description                                                           |
| ------------------ | --------------------------------------------------------------------- |
| `--dataset`        | HuggingFace dataset name (default: `stas/openwebtext-10k`)           |
| `--split`          | Dataset split to use (default: `train`)                               |
| `--text-field`     | Field name containing text (default: `text`)                          |
| `--stem`           | Output file stem, e.g. `{stem}_grammar.json` (default: `openwebtext`) |
| `--output`         | Output directory (default: `data/tokenizer`)                          |
| `--model`          | spaCy model (default: `en_core_web_trf`)                              |
| `--gpu`            | Enable GPU acceleration                                               |
| `--batch-size`     | spaCy pipe batch size (default: `100`)                                |
| `--max-samples`    | Limit number of text samples (for testing)                            |
| `--verbose`        | Print progress output                                                 |

### Streaming mode

When `--max-samples` is set, the runner streams data from HuggingFace instead of
downloading the full dataset. This avoids out-of-memory issues with large corpora
like OpenWebText (8M+ documents). Datasets using legacy loading scripts are
automatically handled via streaming fallback.

### Merge workflow

After generating a grammar dict from an external corpus, merge it into the base
grammar using `expand_grammar.py`:

```bash
uv run python dev/nlp/expand_grammar.py \
    --grammar data/tokenizer/simplestories-1_grammar.json \
    --extra-grammar data/tokenizer/openwebtext_grammar.json \
    --output data/tokenizer/simplestories-1_grammar.json
```

The merge adds new entries without overwriting existing ones. Counts and frequencies
from the external corpus are reset to zero.
