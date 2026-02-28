#!/usr/bin/env python3
"""Train a BPE tokenizer on text data.

Accepts a path (file or directory) and target vocabulary size,
trains using rustbpe, and saves the tokenizer for later use.

Supports:
- .txt files (plain text)
- .parquet files (with configurable text column)
- .json files with formats:
  - {"summaries": [{"summary": "..."}]}
  - [{"summary": "..."}]
"""

import argparse
import json
import sys
import pyarrow.parquet as pq
from pathlib import Path
from typing import Iterator

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kalvin.tokenizer import Tokenizer

# Pattern that splits words, punctuation, and suffixes into separate tokens:
# - Contractions: 's, 'd, 'm, 't, 'll, 've, 're, 'cause, 'em, etc.
# - Punctuation: ! ? ; . , (each as separate token)
# - Unlike GPT-2 pattern which creates tokens like " hello", this creates "hello" + " "
SPLIT_WORDS_PATTERN = r"n't|'(?:[sdmt]|ll|ve|re|cause|em|twas|tis|neath|round|cos|cuz)|\p{L}+|\p{N}{1,3}|[!?;.,]|[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s++"



def parquet_texts_iterator(files: list[Path], text_column: str) -> Iterator[str]:
    for file_path in files:
        print(f"  Loading {file_path.name}...")
        table = pq.read_table(file_path, columns=[text_column])
        for text in table[text_column].to_pylist():
            if text is not None:
                yield text


def json_texts_iterator(files: list[Path], text_field: str = "summary") -> Iterator[str]:
    """Load texts from a JSON file.

    Supports two formats:
    1. {"summaries": [{"summary": "..."}]}
    2. [{"summary": "..."}]

    Args:
        file_path: Path to the JSON file
        text_field: Field name containing the text (default: "summary")

    Returns:
        iterator that yields text strings
    """
    for file_path in files:
        print(f"  Loading {file_path.name}...")
        data = json.loads(file_path.read_text(encoding="utf-8"))
        items = []

        # Format 1: {"summaries": [...]}
        if isinstance(data, dict) and "summaries" in data:
            items = data["summaries"]
        # Format 2: [...]
        elif isinstance(data, list):
            items = data
        else:
            print(f"Unsupported JSON format in {file_path}")
            
        for item in items:
            if isinstance(item, dict) and text_field in item:
                text = item[text_field]
                if text is not None:
                    yield text


def txt_texts_iterator(files: list[Path]):
    for file_path in files:
        print(f"  Loading {file_path.name}...")
        yield file_path.read_text(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer on text data"
    )
    parser.add_argument(
        "path",
        help="Path to training data (file or directory with .txt/.json/.parquet files)",
    )
    parser.add_argument(
        "vocab_size",
        type=int,
        nargs="?",
        default=4096,
        help="Target vocabulary size (default: 4096)",
    )
    parser.add_argument(
        "-g","--glob-pattern",
        default="*.parquet",
        help="Glob pattern (default: *.parquet)",
    )
    parser.add_argument(
        "-o", "--output",
        default="data/tokenizer",
        help="Output directory for the trained tokenizer (default: data/tokenizer)",
    )
    parser.add_argument(
        "-n", "--name",
        default="tokenizer",
        help="Base name for saved files (default: tokenizer)",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Column name for text data in parquet files (default: text)",
    )
    parser.add_argument(
        "--json-field",
        default="summary",
        help="Field name for text data in JSON files (default: summary)",
    )
    parser.add_argument(
        "--split-words",
        action="store_true",
        help="Use a pattern that splits words, punctuation, and suffixes into separate tokens",
    )
    args = parser.parse_args()

    # Determine pattern
    pattern = SPLIT_WORDS_PATTERN if args.split_words else None

    data_path = Path(args.path)
    output_path = Path(args.output)

    if not data_path.exists():
        print(f"Error: Path not found: {data_path}")
        sys.exit(1)

    train_files = list(data_path.glob(args.glob_pattern))

    if not train_files:
        print(f"Error: Training data not found: {data_path}/{args.glob_pattern}")
        sys.exit(1)

    # Determine if we're training from parquet, json, or text files
    if args.glob_pattern.endswith(".parquet"):
        print(f"Training from {len(train_files)} parquet file(s)...")
        train_iterator = parquet_texts_iterator(train_files, args.text_column)
    elif args.glob_pattern.endswith(".json"):
        print(f"Training from {len(train_files)} JSON file(s)...")
        train_iterator = json_texts_iterator(train_files, args.json_field)
    elif args.glob_pattern.endswith(".txt"):
        print(f"Training from {len(train_files)} text file(s)...")
        train_iterator = txt_texts_iterator(train_files)
    else:
        print(f"Error: No .txt, .json, or .parquet files found in {data_path}")
        sys.exit(1)

    tokenizer = Tokenizer()
    tokenizer.train(train_iterator, vocab_size=args.vocab_size, pattern=pattern)

    # Save tokenizer
    output_path.mkdir(parents=True, exist_ok=True)
    name = args.name + f"-{args.vocab_size}"
    tokenizer.save_to_directory(output_path, name=name)
    print(f"Saved tokenizer to: {output_path}/{name}.json + {name}.bin")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Test encoding/decoding
    test_text = "Hello, world!"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"\nTest: '{test_text}' -> {encoded} -> '{decoded}'")


if __name__ == "__main__":
    main()
