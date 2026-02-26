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
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kalvin.tokenizer import Tokenizer

# Pattern that does NOT encode leading spaces (spaces are separate tokens)
# Unlike GPT-2 pattern which creates tokens like " hello", this creates "hello" + " "
NO_LEADING_SPACE_PATTERN = r"'(?:[sdmt]|ll|ve|re)|\p{L}+|\p{N}{1,3}|[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s++"


def load_json_texts(file_path: Path, text_field: str = "summary") -> list[str]:
    """Load texts from a JSON file.

    Supports two formats:
    1. {"summaries": [{"summary": "..."}]}
    2. [{"summary": "..."}]

    Args:
        file_path: Path to the JSON file
        text_field: Field name containing the text (default: "summary")

    Returns:
        List of text strings
    """
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
        
    texts = []
    for item in items:
        if isinstance(item, dict) and text_field in item:
            text = item[text_field]
            if text is not None:
                texts.append(text)

    return texts


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
        "--no-leading-space",
        action="store_true",
        help="Use a pattern that keeps spaces as separate tokens (no ' hello' tokens)",
    )
    args = parser.parse_args()

    # Determine pattern
    pattern = NO_LEADING_SPACE_PATTERN if args.no_leading_space else None

    data_path = Path(args.path)
    output_path = Path(args.output)

    if not data_path.exists():
        print(f"Error: Path not found: {data_path}")
        sys.exit(1)

    # Determine if we're training from parquet, json, or text files
    if data_path.is_dir():
        parquet_files = list(data_path.glob("*.parquet"))
        json_files = list(data_path.glob("*.json"))
        txt_files = list(data_path.glob("*.txt"))
        if parquet_files:
            print(f"Training from {len(parquet_files)} parquet file(s)...")
            tokenizer = Tokenizer()
            tokenizer.train_from_parquet(
                data_path,
                text_column=args.text_column,
                vocab_size=args.vocab_size,
                pattern=pattern,
            )
        elif json_files:
            print(f"Training from {len(json_files)} JSON file(s)...")
            texts = []
            for json_file in json_files:
                print(f"  Loading {json_file.name}...")
                text = load_json_texts(json_file, text_field=args.json_field)
                if text:
                    texts.extend(text)
            print(f"  Loaded {len(texts)} texts")
            tokenizer = Tokenizer()
            tokenizer.train(texts, vocab_size=args.vocab_size, pattern=pattern)
        elif txt_files:
            print(f"Training from {len(txt_files)} text file(s)...")
            texts = []
            for txt_file in txt_files:
                print(f"  Loading {txt_file.name}...")
                texts.append(txt_file.read_text(encoding="utf-8"))
            tokenizer = Tokenizer()
            tokenizer.train(texts, vocab_size=args.vocab_size, pattern=pattern)
        else:
            print(f"Error: No .txt, .json, or .parquet files found in {data_path}")
            sys.exit(1)
    elif data_path.suffix == ".parquet":
        print(f"Training from parquet file: {data_path}...")
        tokenizer = Tokenizer()
        tokenizer.train_from_parquet(
            data_path,
            text_column=args.text_column,
            vocab_size=args.vocab_size,
            pattern=pattern,
        )
    elif data_path.suffix == ".json":
        print(f"Training from JSON file: {data_path}...")
        texts = load_json_texts(data_path, text_field=args.json_field)
        print(f"  Loaded {len(texts)} texts")
        tokenizer = Tokenizer()
        tokenizer.train(texts, vocab_size=args.vocab_size, pattern=pattern)
    else:
        print(f"Training from text file: {data_path}...")
        text = data_path.read_text(encoding="utf-8")
        tokenizer = Tokenizer()
        tokenizer.train([text], vocab_size=args.vocab_size, pattern=pattern)

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
