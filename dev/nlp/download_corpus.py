#!/usr/bin/env python3
"""Download the SimpleStories corpus from HuggingFace with retry and revision pinning.

This module extracts the corpus-download stage (Stage 1 of
``scripts/rebuild-tokenizer-data.sh``) into a reusable, testable module.  It
adds two resilience features over the previous inline ``python -c`` snippet:

* **Revision pinning** — the HuggingFace dataset commit can be pinned via
  ``--revision`` so a rebuild is reproducible even if the dataset author pushes
  new data.
* **Retry with exponential backoff** — transient network errors or HuggingFace
  rate limits (HTTP 429) are retried up to ``--max-retries`` times before the
  download is declared failed.
* **Non-streaming download** — ``streaming=False`` caches the full dataset
  locally as arrow files under ``~/.cache/huggingface/datasets``, so a rebuild
  with a warm HuggingFace Hub cache loads the corpus from disk with zero
  network requests (no HTTP 429 rate-limit exposure). Streaming mode caches
  only metadata and re-fetches every row on each run. See KB-218.

The output format is unchanged: a JSON file containing a list of
``{"summary": story_text}`` dicts, truncated to ``--samples`` rows.

Usage::

    uv run python -m dev.nlp.download_corpus \\
        --output data/tokenizer/simplestories-1.json \\
        --samples 20000 \\
        --revision e63b8adc3b1a1bdc7cac5b500d150b71346b0628
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from itertools import islice
from pathlib import Path
from typing import Any, TypeVar

# Import at module scope so the name ``load_dataset`` can be patched in tests
# (``unittest.mock.patch("dev.nlp.download_corpus.load_dataset")``).  The guard
# keeps the module importable when the optional ``corpus`` extra is absent —
# e.g. on a CI cache-hit run where only ``--extra dev`` is installed.
try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - corpus extra not installed
    load_dataset = None  # type: ignore[assignment,unused-ignore]

_T = TypeVar("_T")

#: HuggingFace dataset identifier.
DATASET_NAME = "SimpleStories/SimpleStories"
#: Row field that holds the story text.
TEXT_FIELD = "story"
#: Pinned dataset revision (commit SHA) for reproducible rebuilds.  Looked up
#: from the HF Hub API (``HfApi().dataset_info(...).sha``).
DEFAULT_REVISION = "e63b8adc3b1a1bdc7cac5b500d150b71346b0628"


def _retry(
    func: Callable[[], _T],
    *,
    max_retries: int,
    backoff_factor: float,
    base_delay: float,
) -> _T:
    """Call *func* with exponential backoff, retrying on failure.

    Up to ``max_retries`` *total attempts* are made.  After each failure the
    function sleeps ``base_delay * (backoff_factor ** attempt)`` seconds before
    the next attempt.  When all attempts are exhausted the last exception is
    re-raised.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            remaining = max_retries - attempt - 1
            if remaining > 0:
                delay = base_delay * (backoff_factor**attempt)
                print(
                    f"  Attempt {attempt + 1}/{max_retries} failed: {exc}. "
                    f"Retrying in {delay:.1f}s …"
                )
                time.sleep(delay)
            else:
                print(f"  Attempt {attempt + 1}/{max_retries} failed: {exc}. Retries exhausted.")
    assert last_exc is not None  # max_retries >= 1 invariant
    raise last_exc


def download_simplestories(
    output_path: str,
    num_samples: int = 20000,
    revision: str | None = None,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    base_delay: float = 2.0,
) -> None:
    """Download SimpleStories stories and write them as a JSON corpus.

    Args:
        output_path: Destination ``.json`` file path.
        num_samples: Maximum number of stories to download.
        revision: HuggingFace dataset commit SHA to pin.  ``None`` uses the
            default branch (backward-compatible with the original inline code).
        max_retries: Total number of download attempts before giving up.
        backoff_factor: Multiplier for the exponential backoff delay.
        base_delay: Base delay in seconds for the first retry backoff.
    """

    def _fetch():
        kwargs: dict[str, Any] = {"split": "train", "streaming": False}
        if revision is not None:
            kwargs["revision"] = revision
        return load_dataset(DATASET_NAME, **kwargs)

    ds = _retry(
        _fetch,
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        base_delay=base_delay,
    )

    # Truncate to ``num_samples`` rows.  ``itertools.islice`` reads only the
    # requested rows lazily from the cached arrow file (≈6 ms for 100 rows),
    # whereas ``list(ds)[:num_samples]`` would materialise the full 2.1M-row
    # corpus into memory (~105 s).  ``islice`` also works unchanged on the
    # plain-list doubles used in the unit tests.
    stories: list[dict[str, str]] = []
    for i, row in enumerate(islice(ds, num_samples)):
        stories.append({"summary": row[TEXT_FIELD]})
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1:,} stories loaded …")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(stories, f)

    chars = sum(len(s["summary"]) for s in stories)
    print(f"  ✓ {len(stories):,} stories ({chars:,} chars) → {output_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the SimpleStories corpus from HuggingFace"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/tokenizer/simplestories-1.json",
        help="Output JSON file path (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20000,
        help="Maximum number of stories (default: %(default)s)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=DEFAULT_REVISION,
        help="HuggingFace dataset commit SHA to pin (default: %(default)s)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Total download attempts (default: %(default)s)",
    )
    parser.add_argument(
        "--backoff-factor",
        type=float,
        default=2.0,
        help="Exponential backoff multiplier (default: %(default)s)",
    )
    parser.add_argument(
        "--base-delay",
        type=float,
        default=2.0,
        help="Base retry delay in seconds (default: %(default)s)",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()
    download_simplestories(
        output_path=args.output,
        num_samples=args.samples,
        revision=args.revision,
        max_retries=args.max_retries,
        backoff_factor=args.backoff_factor,
        base_delay=args.base_delay,
    )


if __name__ == "__main__":
    main()
