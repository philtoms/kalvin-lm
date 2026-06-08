"""CLI entry point for KScript compiler.

Usage:
    python -m kscript script.ks              # -> script.jsonl
    python -m kscript script.ks -out out.json
    python -m kscript script.ks -out out.bin
    python -m kscript script.ks --tokenizer nlp
    python -m kscript model.bin              # Load and display
"""

import argparse
import json
import sys
from pathlib import Path

from kalvin.abstract import KTokenizer
from kalvin.mod_tokenizer import Mod32Tokenizer

from . import KScript


def _make_tokenizer(name: str) -> KTokenizer:
    """Instantiate tokenizer by name."""
    if name == "nlp":
        from kalvin.nlp_tokenizer import NLPTokenizer
        return NLPTokenizer.from_files()
    return Mod32Tokenizer()


def main() -> None:
    """Run KScript CLI."""
    parser = argparse.ArgumentParser(
        description="KScript compiler - compile .ks files to KLine graph format"
    )
    parser.add_argument("input", help="Input file (.ks, .json, .jsonl, or .bin)")
    parser.add_argument("-out", dest="output", help="Output file path")
    parser.add_argument("-dev", action="store_true", help="Enable dev mode")
    parser.add_argument("--tokenizer", choices=["mod32", "nlp"], default="mod32",
                        help="Tokenizer to use (default: mod32)")
    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        tokenizer = _make_tokenizer(args.tokenizer)

        # Load/compile the input
        model = KScript(input_path, tokenizer=tokenizer, dev=args.dev)

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.with_suffix(".jsonl")

        # Write output
        model.output(output_path)
        print(f"Compiled {len(model.entries)} entries to {output_path}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
