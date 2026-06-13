#!/usr/bin/env bash
# rebuild-tokenizer-data.sh — regenerate data/tokenizer from scratch
#
# Pipeline:
#   1. Download SimpleStories corpus from HuggingFace → simplestories-1.json
#   2. Train BPE tokenizer                           → tokenizer-32768.{json,bin}
#   3. Run NLP analysis with spaCy                   → grammar, NER, nlp_type legends
#   4. Tag full BPE vocab with NLP types             → tokenizer-32768_tagged_grammar.json
#
# Prerequisites:
#   uv run pip install datasets spacy
#   uv run python -m spacy download en_core_web_sm
#
# Usage:
#   bash scripts/rebuild-tokenizer-data.sh [--samples N] [--spacy-model MODEL] [--vocab-size SIZE]

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# ── Defaults ──────────────────────────────────────────────────────────────────
SAMPLES=20000
VOCAB_SIZE=32768
SPACY_MODEL="en_core_web_sm"
CORPUS_STEM="simplestories-1"
TOKENIZER_NAME="tokenizer-${VOCAB_SIZE}"
OUTPUT_DIR="data/tokenizer"

# Pinned HuggingFace dataset revision for reproducible corpus downloads.
# Commit SHA looked up from the HF Hub API (see dev/nlp/download_corpus.py).
HF_REVISION="e63b8adc3b1a1bdc7cac5b500d150b71346b0628"

# ── Args ──────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --samples)       SAMPLES="$2";      shift 2 ;;
    --vocab-size)    VOCAB_SIZE="$2";   shift 2 ;;
    --spacy-model)   SPACY_MODEL="$2";  shift 2 ;;
    --output)        OUTPUT_DIR="$2";   shift 2 ;;
    --stem)          CORPUS_STEM="$2";  shift 2 ;;
    --hf-revision)   HF_REVISION="$2";  shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--samples N] [--vocab-size SIZE] [--spacy-model MODEL]"
      echo "       [--output DIR] [--stem NAME] [--hf-revision SHA]"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

TOKENIZER_NAME="tokenizer-${VOCAB_SIZE}"
CORPUS_JSON="${OUTPUT_DIR}/${CORPUS_STEM}.json"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Rebuild data/tokenizer                                     ║"
echo "║  Samples: ${SAMPLES}  Vocab: ${VOCAB_SIZE}  Model: ${SPACY_MODEL}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Download corpus ──────────────────────────────────────────────────
echo "▶ Step 1/4: Downloading ${SAMPLES} stories from SimpleStories/SimpleStories …"
mkdir -p "${OUTPUT_DIR}"

uv run python -m dev.nlp.download_corpus \
    --output "${CORPUS_JSON}" \
    --samples "${SAMPLES}" \
    --revision "${HF_REVISION}"

# ── Step 2: Train BPE tokenizer ─────────────────────────────────────────────
echo ""
echo "▶ Step 2/4: Training BPE tokenizer (vocab_size=${VOCAB_SIZE}) …"
uv run python dev/nlp/train_tokenizer.py \
    "${OUTPUT_DIR}" \
    "${VOCAB_SIZE}" \
    --glob-pattern "${CORPUS_STEM}.json" \
    --json-field summary \
    --output "${OUTPUT_DIR}" \
    --name tokenizer

# ── Step 3: NLP analysis ─────────────────────────────────────────────────────
echo ""
echo "▶ Step 3/4: Running NLP analysis (${SPACY_MODEL}) …"
uv run python dev/nlp/nlp_analyzer.py \
    -i "${CORPUS_JSON}" \
    -o "${OUTPUT_DIR}" \
    -m "${SPACY_MODEL}" \
    --batch-size 100

# ── Step 4: Tag vocab ────────────────────────────────────────────────────────
GRAMMAR="${OUTPUT_DIR}/${CORPUS_STEM}_grammar.json"
TAGGED="${OUTPUT_DIR}/${TOKENIZER_NAME}_tagged_grammar.json"

echo ""
echo "▶ Step 4/4: Tagging BPE vocab with NLP types …"
uv run python dev/nlp/tag_vocab.py \
    --grammar "${GRAMMAR}" \
    --tokenizer-dir "${OUTPUT_DIR}" \
    --tokenizer-name "${TOKENIZER_NAME}" \
    --output "${TAGGED}"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Done. Files in ${OUTPUT_DIR}/"
ls -lh "${OUTPUT_DIR}/" | tail -n +2 | while read -r line; do
  echo "║  ${line}"
done
echo "╚══════════════════════════════════════════════════════════════╝"
