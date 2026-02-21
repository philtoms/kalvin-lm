#!/usr/bin/env python3
"""Encode a text file into Kalvin embeddings.

This script loads a text file, splits it into sentences, encodes them
using Kalvin's batch encoding, and reports model statistics.

Can optionally load and extend an existing model.
"""

import argparse
import re
import sys
from pathlib import Path
from tqdm import tqdm
import pyarrow.parquet as pq
from typing import Iterator


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kalvin import Kalvin


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using simple regex-based splitting.

    Args:
        text: Input text to split

    Returns:
        List of sentences
    """
    # Split on sentence-ending punctuation followed by space or end of string
    # This is a simple approach; for production use consider nltk.sent_tokenize
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter out empty sentences
    return [s.strip() for s in sentences if s.strip()]

def stream_text(text_file: str):

    text_path = Path(text_file)

    if not text_path.exists():
        print(f"Error: File not found: {text_path}")
        sys.exit(1)

    # Resolve file paths
    if text_path.is_dir():
        glob_suffix = "*.parquet" if text_path.suffix == ".parquet" else "*.json"
        files = sorted(text_path.rglob(glob_suffix))
    else:
        files = [text_path]

    def text_iterator() -> Iterator[str]:
        for file_path in files:
            print(f"Loading {file_path}")
            if text_path.suffix == ".parquet":
                table = pq.read_table(file_path, columns=["text"])
                column = []
                for text in table["text"].to_pylist():
                    if text is not None:
                        column.append(text)
                text = "\n".join(column)
            else:
                text = text_path.read_text(encoding="utf-8")

            yield text
    
    return text_iterator()

def main():
    parser = argparse.ArgumentParser(
        description="Encode text files into Kalvin embeddings"
    )
    parser.add_argument("text_file", help="Path to the text file to encode")
    parser.add_argument(
        "model",
        nargs="?",
        help="Path to load/save the model (default: data/kalvin.bin)",
    )

    parser.add_argument(
        "-f", "--format",
        choices=["binary", "json"],
        default="binary",
        help="Model format (default: binary)",
    )
    args = parser.parse_args()

    # Load or initialize Kalvin
    model_path = Path(args.model)
    model_size = 0
    if model_path:
        print(f"\nLoading model from: {model_path}")
        kalvin = Kalvin.load(model_path, format=args.format)
        model_size = kalvin.model_size()
        print(f"Loaded model size: {model_size:,} KLines")
    else:
        print("\nInitializing new model...")
        model_path = "data/kalvin.bin"
        kalvin = Kalvin()
        print(f"Initial model size: {kalvin.model_size():,} KLines")


    for text in stream_text(args.text_file):
        # Split into sentences
        sentences = split_into_sentences(text)

        # Encode all sentences in batch
        for sentence in tqdm(sentences, desc=f"Encoding {len(sentences):,} sentences..."):
            kalvin.encode(sentence)

    # Report model size
    final_size = kalvin.model_size()
    print(f"\nFinal model size: {final_size:,} KLines")
    if final_size > model_size:       
        print(f"Extended by: {final_size - model_size:,} KLines")

    # Save model
    print(f"\nSaving model to: {model_path}")
    kalvin.save(model_path, format=args.format)
    actual_size = Path(model_path).stat().st_size
    print(f"Saved: {actual_size / 1024:.1f} KB ({actual_size / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    main()
