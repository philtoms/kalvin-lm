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
| 1. Corpus download | Downloads 20 000 stories from `SimpleStories/SimpleStories` on HuggingFace (non-streaming, cached as local arrow files) → `simplestories-1.json` (~25 MB) | `datasets` package (the `corpus` extra), network access to HuggingFace on a cold download only | ~30–60 s cold (network-bound); ~1–2 s with a warm HF Hub cache (disk-only) |
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
The tokenizer artifact cache (~34 MB) is well within that budget and is restored
in seconds on a cache hit, versus the 7–15 minute rebuild on a miss. On
cache-miss rebuilds, the separate HF Hub cache layer adds ~4.6 GB (see the "Two
separate caches" callout), bringing the theoretical combined footprint to ~4.7 GB
— still under the 10 GB limit. Over thousands of runs the cache-first strategy
saves hours of CI time.

### Cost summary

| Path | Time | Notes |
|------|------|-------|
| Cache hit | ~seconds | 34 MB restore |
| Cache miss | ~7–15 min | Full rebuild (network + CPU), then cache populated for subsequent runs |

## Cache key

```
tokenizer-data-${{ hashFiles(
  'scripts/rebuild-tokenizer-data.sh',
  'dev/nlp/download_corpus.py',
  'dev/nlp/train_tokenizer.py',
  'dev/nlp/nlp_analyzer.py',
  'dev/nlp/tag_vocab.py',
  'data/tokenizer/.cache-version'
) }}
```

The key is the hash of every file that influences the build output:

- `scripts/rebuild-tokenizer-data.sh` — the pipeline orchestrator (samples
  count, vocab size, model choice, stage ordering).
- `dev/nlp/download_corpus.py` — the corpus-download stage (Stage 1). A change
  here — e.g. the KB-218 switch from streaming to non-streaming mode, or a
  revision pin update — produces different output and busts the cache.
- `dev/nlp/train_tokenizer.py`, `dev/nlp/nlp_analyzer.py`,
  `dev/nlp/tag_vocab.py` — the stage implementations. Any change to BPE
  training, spaCy analysis, or tagging logic produces different artifacts.
- `data/tokenizer/.cache-version` — a manual version stamp for cache busts that
  are not driven by code (see below).

> **Two separate caches.** This key governs the `data/tokenizer/` **artifact**
> cache (the finished output, ~34 MB). It is distinct from the
> `~/.cache/huggingface` **dataset** cache (HF Hub cache), which stores the raw
> SimpleStories dataset as local arrow files. The dataset cache has its own key
> (`hf-hub-cache-${{ runner.os }}`) and a broad restore key (`hf-hub-cache-`)
> that keeps it warm across corpus-revision bumps. A warm dataset cache lets
> Stage 1 load the corpus from disk with zero network requests.

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

> **Stage 1 and the HF Hub cache.** Step 4 runs the rebuild via
> `scripts/rebuild-tokenizer-data.sh`. Stage 1 of that script calls
> `dev/nlp/download_corpus` in **non-streaming** mode, which downloads the full
> SimpleStories dataset and caches it as local arrow files under
> `~/.cache/huggingface/datasets`. On a cache-miss rebuild that *also* has a
> warm HF Hub cache (restored via the `hf-hub-cache-` key in the workflow),
> Stage 1 loads the corpus from disk with zero network requests — eliminating
> HTTP 429 rate-limit exposure on the corpus download. On the first-ever run
> (cold HF cache), Stage 1 makes one network download, after which the dataset
> cache is warm for all subsequent rebuilds regardless of `data/tokenizer`
> cache key changes.

The `data/` directory remains gitignored at all times — assets are never
committed; they are either cached or rebuilt.

## Rebuild-path resilience

The rebuild path depends on three external services that can fail
transiently. KB-216 hardened each:

| Dependency | Failure mode | Hardening |
|------------|--------------|-----------|
| HuggingFace `datasets` non-streaming download | Rate limits (HTTP 429), transient outages | Retry with exponential backoff (`dev/nlp/download_corpus.py`, 3 attempts), dataset revision pinned to a commit SHA, and an `actions/cache@v4` layer for `~/.cache/huggingface` that caches the **full dataset as arrow files** (zero network on a warm cache) |
| spaCy model host (`en_core_web_sm` wheel) | Host outages, slow downloads | 3-attempt retry loop with linear backoff in the CI workflow |
| CPU-bound pipeline (BPE + spaCy) | Slow-runner timeout | No change — the ~7–15 min rebuild leaves ample headroom under the 20-min `timeout-minutes` |

The HuggingFace dataset is pinned to revision
`e63b8adc3b1a1bdc7cac5b500d150b71346b0628` (the `HF_REVISION` default in
`scripts/rebuild-tokenizer-data.sh` and `DEFAULT_REVISION` in
`dev/nlp/download_corpus.py`). To pick up new dataset data, look up the
current SHA and update both constants — the change to the rebuild script
busts the CI cache automatically.

## Fallback strategy

If the rebuild path is persistently broken (a sustained HuggingFace outage,
a breaking dataset schema change, or repeated CI timeouts), the assets can
be provisioned from a pre-built release artifact instead of rebuilding
online:

1. **Build locally** on a machine with the corpus extra and spaCy:
   ```bash
   uv sync --extra dev --extra corpus
   uv run python -m spacy download en_core_web_sm
   bash scripts/rebuild-tokenizer-data.sh
   ```
2. **Bundle and publish** the `data/tokenizer/` directory as a GitHub
   release artifact:
   ```bash
   tar czf tokenizer-data.tar.gz -C data tokenizer
   gh release create tokenizer-data-v1 tokenizer-data.tar.gz \
       --title "Tokenizer data v1" \
       --notes "Pre-built data/tokenizer assets for CI provisioning."
   ```
3. **Consume in CI** by replacing the rebuild step with an artifact download
   (in `.github/workflows/ci.yml`, inside the cache-miss branch):
   ```yaml
   - name: Download tokenizer data (fallback)
     if: steps.cache-tokenizer.outputs.cache-hit != 'true'
     run: |
       gh release download tokenizer-data-v1 \
         --pattern tokenizer-data.tar.gz
       tar xzf tokenizer-data.tar.gz -C data
     env:
       GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
   ```

This trades a runtime rebuild for a manual publish step, but removes all
external-service dependencies from the critical CI path. Bump the release
tag (`tokenizer-data-v2`, …) whenever the pipeline changes.

## Reducing rebuild time

The `--samples` flag controls how many SimpleStories stories feed the
pipeline. The default is `20000`; reducing it for a CI-specific rebuild
(e.g. `--samples 5000`) shortens both the corpus download and the spaCy
analysis stage, the two slowest steps.

**Tradeoff.** A smaller corpus produces slightly different BPE merge ranks
and a slightly different tagged grammar. This is **acceptable for CI test
coverage** because the test suite validates encode/decode *behaviour*
(tok_length round-trips, vocab lookups, grammar tag presence) rather than
the *exact* vocabulary. The artefacts would differ from a full 20K-sample
build, so a reduced-sample cache key should be distinct (bump
`.cache-version`).

Do **not** change the default `--samples 20000` — that is the production
quality-of-output setting. CI-specific reduction is opt-in via the CLI flag
or a dedicated workflow invocation.

## Monitoring & troubleshooting

When a CI run fails on the rebuild path, the failure signature identifies
the stage:

| Failure signature | Likely cause | Diagnostic steps |
|-------------------|--------------|------------------|
| `ConnectionError` / `HTTP 429` in the Step 1 (corpus download) log, retried 3× then failed | HuggingFace rate limit or outage | Check the [HF status page](https://status.huggingface.co). The retry-with-backoff absorbs short blips; a sustained 429 indicates the dataset revision pin or a longer backoff is needed. A warm HF Hub cache (non-streaming arrow files) makes a Stage-1 429 unlikely — it only occurs on a cold download. Unauthenticated requests are rate-limited — consider setting `HF_TOKEN` in CI secrets for higher limits. |
| `spacy download` fails, retries 3×, then `spacy.load('en_core_web_sm')` raises `OSError: [E050]` | spaCy model host (GitHub Releases wheel) unavailable | The retry loop absorbs transient errors. If persistent, the model can be installed from the spaCy GitHub release directly, or use the release-artifact fallback above. |
| Job killed at the 20-minute `timeout-minutes` ceiling | Slow runner + full rebuild exceeded budget | Inspect which stage was mid-flight when the job was cancelled. If the spaCy stage (Step 3) is the bottleneck, consider `--samples 5000` or bump `timeout-minutes` to `25`. |
| Cache always misses (rebuild every run) | Cache key includes a file that changes every commit | Verify none of the hashed files (`scripts/rebuild-tokenizer-data.sh`, `dev/nlp/*.py`, `.cache-version`) are being modified unintentionally. Check `actions/cache` post-step for save errors. |

**Note on download mode + HF Hub cache (KB-218).** The corpus download uses
**non-streaming** mode (`streaming=False`). The `~/.cache/huggingface` cache
layer caches the **full dataset as local arrow files**
(`~/.cache/huggingface/datasets`, ~3 GB for SimpleStories), so a warm cache
lets Stage 1 load the corpus from disk with zero network requests — no HTTP
429 exposure. This was an intentional switch from streaming mode, which cached
only dataset metadata (~12 KB) and re-fetched every data row over the network
on each run.

The KB-218 investigation confirmed the difference empirically (datasets 5.0):
streaming left `~/.cache/huggingface/datasets` at 0 bytes and failed offline
with `ConnectionError`; non-streaming cached ~3 GB of arrow files and loaded
fully offline in ~0.13 s. Because CI run history was unavailable for review
(`gh` unauthenticated), the switch was made defensively — non-streaming
strictly improves cache effectiveness with no downside for output format or
correctness, and KB-216's retry-with-backoff and revision pinning are
unchanged.

### Post-switch monitoring (KB-222 / KB-224)

KB-222 attempted to validate KB-218's non-streaming switch against real
cache-miss CI runs. **No CI runs exist to review.** Investigation with
authenticated `gh` access (`gh run list`, `gh api …/actions/runs`) revealed:

- **0 workflow definitions** and **0 total runs** on GitHub Actions. The CI
  workflow (`.github/workflows/ci.yml`, created in KB-214) was committed to
  local `main` but has not been pushed to `origin/main` — local `main` is 160
  commits ahead of the remote at the time of this review.
- **0 cache entries** in the GitHub Actions cache API
  (`gh api …/actions/caches`).

Consequently, neither monitoring concern could be empirically assessed:

| Concern | Status | Detail |
|---------|--------|--------|
| HTTP 429 / transient errors at Stage 1 | **Unverified** — no data | Zero cache-miss runs exist to inspect. The non-streaming mode's theoretical benefit (warm HF cache → zero network requests → no 429 exposure) remains sound but is unconfirmed empirically. KB-216's retry-with-backoff and revision pinning remain in place as cold-download resilience. |
| Cache budget (~4.6 GB HF cache + ~34 MB tokenizer cache vs 10 GB limit) | **Unverified** — no data | No `actions/cache@v4` post-step output is available. The theoretical combined footprint (~4.7 GB) is well under the 10 GB limit. The `hub/` blob redundancy (~1.6 GB of raw parquet blobs duplicating ~3 GB of arrow files) remains a pruning lever should budget pressure materialise — see the "Conditional remediation" note below. |
| No remediation applied | — | Step 5 remediation was not triggered: no empirical issues were found because no CI runs exist. The CI workflow (`ci.yml`) was not modified. |

**Re-evaluation (KB-224).** KB-224 re-verified the same prerequisites with
authenticated `gh` access (`gh auth status`, `gh api …/actions/workflows`,
`gh api …/actions/runs`, `gh api …/actions/caches`). The blockage persists
unchanged: the CI workflow is still **not deployed** — 0 workflow definitions,
0 CI runs, 0 cache entries on GitHub Actions. Local `main` is now **163 commits
ahead** of `origin/main` (KB-222 measured 160; commits have accumulated since).
Because the CI workflow has never been pushed, no cache-miss run history exists
to review — Steps 1–5 of the re-evaluation could not execute. The two
monitoring concerns remain **Unverified — no data**, and no remediation was
applied (`ci.yml` and `download_corpus.py` are unchanged). Follow-up task
**KB-225** tracks pushing `local main → origin/main` and re-running the
monitoring procedure once ≥5 cache-miss runs accumulate.

**Monitoring is pending CI deployment.** Once the CI workflow is pushed to
`origin/main` and cache-miss runs accumulate, re-evaluate both concerns against
≥5 cache-miss runs (per the KB-222 acceptance criteria). Follow-up task **KB-225**
tracks this re-assessment.

**Conditional remediation (for future re-evaluation).** If a future
re-assessment finds cache budget pressure (total approaching the 10 GB limit
or eviction warnings from `actions/cache@v4`), prune the redundant `hub/` blobs
after the rebuild step and before the cache save:

```yaml
- name: Prune redundant HF Hub blobs
  if: steps.cache-tokenizer.outputs.cache-hit != 'true'
  run: rm -rf ~/.cache/huggingface/hub
```

This removes the ~1.6 GB of raw parquet blobs (duplicated by the ~3 GB arrow
files) before the cache is saved. The arrow files alone are sufficient for
non-streaming `load_dataset()` reuse on subsequent runs. If 429s recur on
warm-cache rebuilds despite the non-streaming cache, investigate whether the
`hf-hub-cache-` restore key missed due to LRU eviction, and consider adding
`HF_TOKEN` to CI secrets for higher rate limits.
