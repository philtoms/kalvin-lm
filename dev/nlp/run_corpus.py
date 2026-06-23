#!/usr/bin/env python3
"""
Corpus Runner - Run NLP analysis on external HuggingFace datasets.

Downloads a dataset from HuggingFace, extracts text fields, runs spaCy
analysis via ``nlp_analyzer``, and saves all outputs (grammar, NER, verbs,
noun chunks, fine-type legends).

Usage:
    # Run on default OpenWebText-10k subset
    uv run python dev/nlp/run_corpus.py --verbose

    # Custom dataset with limits
    uv run python dev/nlp/run_corpus.py \\
        --dataset stas/openwebtext-10k \\
        --max-samples 500 \\
        --stem openwebtext \\
        --output data/tokenizer

    # With GPU acceleration
    uv run python dev/nlp/run_corpus.py --gpu --batch-size 200 --verbose
"""

import argparse
import sys
from pathlib import Path

# Ensure project root and dev/nlp are on sys.path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import spacy
from nlp_analyzer import (
    analyze_texts,
    download_spacy_model,
    filter_empty_texts,
    print_summary,
    save_analysis,
    setup_gpu,
)


def load_dataset_texts(
    dataset_name: str,
    split: str = "train",
    text_field: str = "text",
    max_samples: int | None = None,
    verbose: bool = False,
) -> list[str]:
    """Load text samples from a HuggingFace dataset.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g., ``stas/openwebtext-10k``).
        split: Dataset split to use (default: ``train``).
        text_field: Field name containing text in each row.
        max_samples: If set, limit to this many samples.
        verbose: Print progress.

    Returns:
        List of text strings extracted from the dataset.

    Raises:
        ImportError: If the ``datasets`` library is not installed.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "Error: The 'datasets' library is required. Install it with:\n"
            "  uv pip install datasets\n"
            "Or add the 'corpus' optional dependency:\n"
            "  uv pip install -e '.[corpus]'"
        )
        raise

    if verbose:
        print(f"Loading dataset: {dataset_name} (split={split})")

    # Decide whether to stream.  Streaming is used when:
    #  - max_samples is set (avoid downloading the full dataset), or
    #  - the normal load fails due to a legacy script format.
    streaming = max_samples is not None
    ds = None

    try:
        ds = load_dataset(
            dataset_name,
            split=split,
            streaming=streaming,
            trust_remote_code=True,
        )
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" in str(exc):
            if verbose:
                print("  Dataset uses legacy script — switching to streaming mode")
            ds = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True)
            streaming = True
        else:
            raise

    if streaming:
        # IterableDataset — pull rows one at a time
        texts: list[str] = []
        limit = max_samples
        for row in ds:
            texts.append(row[text_field])
            if limit is not None and len(texts) >= limit:
                break
        if verbose:
            print(f"  Streamed {len(texts):,} rows")
    else:
        if verbose:
            print(f"  Dataset size: {len(ds):,} rows")

        # Extract text field
        texts = [row[text_field] for row in ds]

        # Limit samples if requested
        if max_samples is not None and max_samples < len(texts):
            texts = texts[:max_samples]
            if verbose:
                print(f"  Limited to {max_samples:,} samples")

    if verbose:
        total_chars = sum(len(t) for t in texts)
        print(f"  Extracted {len(texts):,} text segments ({total_chars:,} characters)")

    return texts


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Run NLP analysis on a HuggingFace dataset corpus")

    parser.add_argument(
        "--dataset",
        type=str,
        default="stas/openwebtext-10k",
        help="HuggingFace dataset name (default: %(default)s)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: %(default)s)",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Field name containing text in each row (default: %(default)s)",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default="openwebtext",
        help="Output file stem, e.g. produces {stem}_grammar.json (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/tokenizer"),
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="en_core_web_trf",
        help="spaCy model to use (default: %(default)s)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU acceleration",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="spaCy pipe batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of text samples (for testing)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress output",
    )
    parser.add_argument(
        "--fine-order",
        choices=["canonical", "count-weighted"],
        default="canonical",
        help=(
            "NLPFineType bit-ordering scheme (default: %(default)s). "
            "'canonical' emits families in fixed order POS, POS_FINE, DEP, MORPH "
            "(stable across re-runs); 'count-weighted' emits families by total "
            "per-family count descending (tracks corpus frequency). Within a "
            "family, types are always ranked by count descending (frequent -> "
            "low bit, rare -> high bit) and round-robin interleaved."
        ),
    )

    return parser


def main() -> None:
    """Main entry point for the corpus runner."""
    parser = build_parser()
    args = parser.parse_args()

    # Ensure output directory exists
    args.output.mkdir(parents=True, exist_ok=True)

    # Setup GPU if requested
    if args.gpu:
        gpu_enabled = setup_gpu(use_gpu=True, verbose=args.verbose)
        if not gpu_enabled and args.verbose:
            print("Warning: GPU requested but not available, falling back to CPU")

    # Load spaCy model
    if args.verbose:
        print(f"Loading spaCy model: {args.model}")

    try:
        nlp = spacy.load(args.model)
    except OSError:
        if not download_spacy_model(args.model, verbose=args.verbose):
            print(f"Error: Could not load spaCy model '{args.model}'")
            sys.exit(1)
        nlp = spacy.load(args.model)

    # Load dataset texts
    texts = load_dataset_texts(
        dataset_name=args.dataset,
        split=args.split,
        text_field=args.text_field,
        max_samples=args.max_samples,
        verbose=args.verbose,
    )

    # Filter empty texts (can crash transformer models)
    texts = filter_empty_texts(texts, verbose=args.verbose)

    if args.verbose:
        print(f"\nRunning NLP analysis on {len(texts):,} text segments...")

    # Run analysis
    analysis = analyze_texts(
        nlp=nlp,
        texts=texts,
        batch_size=args.batch_size,
        verbose=args.verbose,
        fine_order=args.fine_order,
    )

    # Save results
    saved_files = save_analysis(analysis, args.output, args.stem)
    print(f"\nSaved {len(saved_files)} analysis files:")
    for name, path in saved_files.items():
        print(f"  {name}: {path}")

    # Print summary
    print_summary(analysis)


if __name__ == "__main__":
    main()
