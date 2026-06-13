# Tokenizer Data Provisioning (CI Cache Strategy)

CI provisions the `data/tokenizer/` assets that the NLP tokenizer subsystem
depends on. Forty-six tests are gated behind the `requires_nlp_data` skip
marker (see `tests/conftest.py`); without these assets they skip silently,
leaving the subsystem with near-zero coverage. The CI workflow
(`../ci.yml`) restores the assets from a GitHub Actions cache and falls back
to a full rebuild only on cache miss.

This document records the investigation that shaped that strategy and serves
as a contributor guide for busting the cache when the rebuild pipeline
changes.

## The rebuild pipeline

`scripts/rebuild-tokenizer-data.sh` regenerates `data/tokenizer/` from scratch
in four stages:

| Stage | What it does | Dependencies | Est. runtime |
|-------|--------------|--------------|--------------|
| 1. Corpus download | Streams 20 000 stories from `SimpleStories/SimpleStories` on HuggingFace → `simplestories-1.json` (~25 MB) | `datasets` package (the `corpus` extra), network access to HuggingFace | ~30–60 s (network-bound) |
| 2. BPE training | Trains a 32 768-token BPE tokenizer over the corpus → `tokenizer-32768.{json,bin}` | `dev/nlp/train_tokenizer.py`, `rustbpe` | ~1–3 min (CPU-bound) |
| 3. spaCy analysis | Runs `en_core_web_sm` over the corpus → grammar, NER, noun-chunk and verb legends | `dev/nlp/nlp_analyzer.py`, `en_core_web_sm` model download, significant CPU | ~5–10 min (CPU + model download) |
| 4. Vocab tagging | Tags the full BPE vocab with NLP types → `tokenizer-32768_tagged_grammar.json` (~5.3 MB) | `dev/nlp/tag_vocab.py` | ~10–30 s (CPU) |

**Total on a cold rebuild: ~7–15 min**, dominated by the spaCy stage.

Prerequisites for a rebuild: the `corpus` extra (`datasets`), a `spacy download
en_core_web_sm`, and network access. See the script's own header for details.

## What the test suite actually needs at runtime

Only **three files** are read when the test suite (or the application) loads an
`NLPTokenizer` via `NLPTokenizer.from_files()`:

| File | Size | Role |
|------|------|------|
| `tokenizer-32768.json` | ~168 B | BPE pattern metadata (read by `Tokenizer.from_directory()`) |
| `tokenizer-32768.bin` | ~925 KB | BPE merge ranks |
| `tokenizer-32768_tagged_grammar.json` | ~5.3 MB | Tagged grammar dictionary |

**~6.2 MB total.** The remaining files (`simplestories-1.json` corpus,
`simplestories-1_grammar.json`, `*_ner.json`, `*_noun_chunks.json`, `*_verbs.json`,
`*_nlp_type*.json`) are intermediate pipeline artifacts not required at test
time.

## Cache decision: cache the full directory

The full `data/tokenizer/` directory is **34 MB** and is cached in its entirety.

**Why not cache only the three runtime files?** It is technically possible, but
adds fragility. `NLPTokenizer.from_files()` uses a fallback glob for grammar
discovery: if the tagged grammar is absent it picks up the first
`*_grammar.json` (e.g. `simplestories-1_grammar.json`), which carries a
different — and for tests, wrong — grammar shape. Restoring the full directory
keeps the on-disk layout identical between a cache hit and a fresh rebuild, so
the auto-discovery logic resolves the same way in both cases.

**Cost ceiling.** GitHub Actions allows up to **10 GB of cache per repository**.
34 MB is well within that budget and is restored in seconds on a cache hit,
versus the 7–15 minute rebuild on a miss. Over thousands of runs the cache-first
strategy saves hours of CI time.

### Cost summary

| Path | Time | Notes |
|------|------|-------|
| Cache hit | ~seconds | 34 MB restore |
| Cache miss | ~7–15 min | Full rebuild (network + CPU), then cache populated for subsequent runs |

## Cache key

```
tokenizer-data-${{ hashFiles(
  'scripts/rebuild-tokenizer-data.sh',
  'dev/nlp/train_tokenizer.py',
  'dev/nlp/nlp_analyzer.py',
  'dev/nlp/tag_vocab.py',
  'data/tokenizer/.cache-version'
) }}
```

The key is the hash of every file that influences the build output:

- `scripts/rebuild-tokenizer-data.sh` — the pipeline orchestrator (samples
  count, vocab size, model choice, stage ordering).
- `dev/nlp/train_tokenizer.py`, `dev/nlp/nlp_analyzer.py`,
  `dev/nlp/tag_vocab.py` — the stage implementations. Any change to BPE
  training, spaCy analysis, or tagging logic produces different artifacts.
- `data/tokenizer/.cache-version` — a manual version stamp for cache busts that
  are not driven by code (see below).

When any of these change, the hash changes, the old cache no longer matches, and
CI falls back to a full rebuild that then writes a fresh cache entry under the
new key.

## Busting the cache

To force CI to rebuild the data assets — e.g. after bumping the corpus sample
count out-of-band, or to re-run the pipeline against the same source files:

1. Edit `data/tokenizer/.cache-version` (e.g. `v1` → `v2`).
2. Commit the change.

The next CI run hashes the new stamp, misses the cache, runs the full rebuild,
and stores the result under the new key. Old cache entries age out
automatically (GitHub evicts least-recently-used entries when the 10 GB repo
limit is approached).

Because `data/` is gitignored, `.cache-version` is force-added:
`git add -f data/tokenizer/.cache-version`.

## Cache-miss behaviour

On a miss the `test` job (in `../ci.yml`):

1. Installs both `dev` and `corpus` extras
   (`uv sync --extra dev --extra corpus`) — never `--extra corpus` alone, which
   resolves to main+corpus and drops pytest/ruff.
2. Makes pip available in the project venv (`uv pip install pip`) — uv-managed
   venvs ship without pip, and spaCy's `download` command invokes
   `python -m pip` internally.
3. Downloads the spaCy model: `uv run python -m spacy download en_core_web_sm`.
4. Runs `bash scripts/rebuild-tokenizer-data.sh` (default arguments:
   `--samples 20000 --vocab-size 32768`), which regenerates every file in
   `data/tokenizer/`.
5. `actions/cache@v4` then saves the freshly built directory under the current
   key for future runs.

The `data/` directory remains gitignored at all times — assets are never
committed; they are either cached or rebuilt.
