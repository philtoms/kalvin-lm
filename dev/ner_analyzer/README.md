# NER Analyzer

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
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl
```

**NVIDIA CUDA:**
```bash
uv add "spacy[cuda11x]"  # Replace 11x with your CUDA version
# Install models as above
```

## Usage

Basic usage (CPU):
```bash
uv run python dev/ner_analyzer/ner_analyzer.py -v
```

Enable GPU acceleration:
```bash
uv run python dev/ner_analyzer/ner_analyzer.py -v --gpu
```

Use transformer model (better GPU utilization):
```bash
uv run python dev/ner_analyzer/ner_analyzer.py -v --gpu -m en_core_web_trf
```

Custom input file:
```bash
uv run python dev/ner_analyzer/ner_analyzer.py -i /path/to/file.json -v
```

Extend existing dictionaries:
```bash
uv run python dev/ner_analyzer/ner_analyzer.py -i input.json -e existing.json -v
```

### CLI Options

| Flag | Description |
|------|-------------|
| `-i, --input` | Input JSON or text file |
| `-o, --output` | Output directory (default: same as input) |
| `-m, --model` | spaCy model (default: `en_core_web_sm`) |
| `-g, --gpu` | Enable GPU acceleration |
| `-b, --batch-size` | Batch size for processing (default: 50) |
| `-v, --verbose` | Print detailed output |
| `-e, --existing` | Extend existing dictionary files |

## Input Format

Supports:
- JSON files with array of objects containing `summary` field
- JSON files with single object containing `summary` field
- Plain text files

## Output

Creates separate JSON files for each analysis type:

### `{stem}_grammar.json` - Combined grammatical table
```json
{
  "Tea": {
    "pos": "PROPN",
    "pos_fine": "NNP",
    "dep": "nsubj",
    "morph": "Number=Sing",
    "count": 2,
    "frequency_pct": 0.0001,
    "nlp_type32": 33685632,
    "nlp_type48": 33685632,
    "nlp_fine_type": 67641376
  }
}
```

Three type encodings are provided:

| Field | Bits | Description |
|-------|------|-------------|
| `nlp_type32` | 32 | Coarse-grained, fits in 32-bit int |
| `nlp_type48` | 48 | Finer-grained, more DEP/MORPH detail |
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

**32-bit bit layout:**
- Bits 0-16: Coarse POS tags (17 Universal POS tags)
- Bits 17-24: Simplified dependency groups (8 groups)
- Bits 25-31: Simplified morph features (7 features)

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

**48-bit bit layout:**
- Bits 0-16: Coarse POS tags (17 Universal POS tags, same as 32-bit)
- Bits 17-31: Finer dependency groups (15 groups)
- Bits 32-47: Finer morph features (16 features)

**Finer DEP groups (48-bit):**
| Group | Dependencies |
|-------|--------------|
| `DEP_SUBJ` | nsubj, nsubjpass, csubj, csubjpass, agent |
| `DEP_OBJ` | obj, iobj, dobj |
| `DEP_OBL` | obl, obl:* |
| `DEP_NMOD` | nmod, nmod:* |
| `DEP_CCOMP` | ccomp |
| `DEP_XCOMP` | xcomp |
| `DEP_ADVCL` | advcl |
| `DEP_ACL` | acl, acl:relcl |
| `DEP_AMOD` | amod |
| `DEP_ADVMOD` | advmod |
| `DEP_NUMMOD` | nummod, nummod:* |
| `DEP_APPOS` | appos |
| `DEP_FUNC` | det, case, mark, aux, auxpass, cop, expl, neg |
| `DEP_STRUCT` | root, conj, cc, compound, flat, fixed, list, parataxis, discourse |
| `DEP_PUNCT` | punct, goeswith, reparandum, orphan |

**Finer MORPH features (48-bit):**
| Flag | Feature |
|------|---------|
| `MORPH_SING` | Number=Sing |
| `MORPH_PLUR` | Number=Plur |
| `MORPH_PAST` | Tense=Past |
| `MORPH_PRES` | Tense=Pres |
| `MORPH_FUT` | Tense=Fut |
| `MORPH_PASS` | Voice=Pass |
| `MORPH_PERSON_1` | Person=1 |
| `MORPH_PERSON_2` | Person=2 |
| `MORPH_PERSON_3` | Person=3 |
| `MORPH_PERF` | Aspect=Perf |
| `MORPH_PROG` | Aspect=Prog |
| `MORPH_IND` | Mood=Ind |
| `MORPH_IMP` | Mood=Imp |
| `MORPH_INF` | VerbForm=Inf |
| `MORPH_PART` | VerbForm=Part |
| `MORPH_GER` | VerbForm=Ger |

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

# Check 32-bit coarse features
nlp_type32 = grammar["ran"]["nlp_type32"]
if nlp_type32 & type32_legend["POS_VERB"]:
    print("'ran' is a verb (32-bit)")

# Check 48-bit finer features
nlp_type48 = grammar["ran"]["nlp_type48"]
if nlp_type48 & type48_legend["MORPH_PROG"]:
    print("'ran' has progressive aspect (48-bit)")

# Find all verbs using 32-bit type
VERB_FLAG = type32_legend["POS_VERB"]
verbs = {word: data for word, data in grammar.items()
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

## Performance Tips

1. **Use GPU with transformer model** for large datasets:
   ```bash
   uv run python dev/ner_analyzer/ner_analyzer.py --gpu -m en_core_web_trf
   ```

2. **Increase batch size** for faster processing:
   ```bash
   uv run python dev/ner_analyzer/ner_analyzer.py -b 100
   ```

3. **Standard model (`en_core_web_sm`)** is CPU-optimized and already fast without GPU

4. **Transformer model (`en_core_web_trf`)** benefits significantly from GPU acceleration
