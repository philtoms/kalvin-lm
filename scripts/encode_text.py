#!/usr/bin/env python3
"""Encode a text file into Kalvin embeddings.

This script loads a text file, splits it into sentences, encodes them
using Kalvin's batch encoding, and reports model statistics.

Can optionally load and extend an existing model.
"""

import argparse
import re
import sys
import signal
from pathlib import Path
from tqdm import tqdm
import pyarrow.parquet as pq
from typing import Iterator
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kalvin import Kalvin


# Global state for interrupt handling
_interrupted = False
_signalled = False;

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global _interrupted
    global _signalled
    if not _interrupted:
        _interrupted = True
        print("\n\nInterrupt received, saving model...")
    else:
        _interrupted = False
        _signalled = True
        raise Exception("Already signalled - closing")
    
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

def stream_text(input_file: str):

    text_path = Path(input_file)

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
            if text_path.suffix == ".parquet":
                table = pq.read_table(file_path, columns=["text"])
                column = []
                for text in tqdm(table["text"].to_pylist(), f"Loading {file_path}"):
                    if text is not None:
                        column.append(text)
                text = "\n".join(column)
            elif text_path.suffix == ".json":
                print(f"Loading {file_path}")
                with open(text_path,"r") as f:
                    data = json.load(f)
                    row = [item["summary"] for item in data]
                    text = "\n".join(row)
            else:
                print(f"Loading {file_path}")
                text = text_path.read_text(encoding="utf-8")

            yield text
    
    return text_iterator()

def main():
    # Set up signal handler for graceful interrupt
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(
        description="Encode text files into Kalvin embeddings"
    )
    parser.add_argument("-i", "--input_file", default="/Volumes/USB-Backup/ai/data/tidy-ts/simplestories-1.json", help="Path to the text file to encode")
    parser.add_argument(
        "--model",
        default="data/kalvin.bin",
        help="Path to load/save the model (default: data/kalvin.bin)",
    )

    parser.add_argument(
        "-f", "--format",
        choices=["binary", "json"],
        default="binary",
        help="Model format (default: binary)",
    )
    args = parser.parse_args()
    model = args.model

    # Load or initialize Kalvin
    model_path = Path(model)
    model_size = 0
    if model_path and model_path.exists():
        print(f"\nLoading model from: {model_path}")
        kalvin = Kalvin.load(model_path, format=args.format)
        model_size = kalvin.model_size()
        print(f"Loaded model size: {model_size:,} KLines")
    else:
        print("\nInitializing new model...")
        if not model_path:
            model_path = "data/kalvin.bin" 
        kalvin = Kalvin()
        print(f"Initial model size: {kalvin.model_size():,} KLines")


    for text in stream_text(args.input_file):
        if _interrupted:
            break

        # Split into sentences
        sentences = split_into_sentences(text)

        # Encode all sentences in batch
        for sentence in tqdm(sentences, desc=f"Encoding {len(sentences):,} sentences..."):
            if _interrupted:
                break
            kalvin.encode(sentence)

    # Report model size
    final_size = kalvin.model_size()
    print(f"\nFinal model size: {final_size:,} KLines")
    if final_size > model_size:       
        print(f"Extended by: {final_size - model_size:,} KLines")

    kalvin = kalvin.prune()

    # Report model size
    final_size = kalvin.model_size()
    print(f"\nFinal model size after pruning: {final_size:,} KLines")
    if final_size > model_size:       
        print(f"Extended by: {final_size - model_size:,} KLines")

    # Save model
    print(f"\nSaving model to: {model_path}")
    kalvin.save(model_path, format=args.format)
    actual_size = Path(model_path).stat().st_size
    print(f"Saved: {actual_size / 1024:.1f} KB ({actual_size / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    main()
