"""CLI entry point for KScript compiler."""

import argparse
import json
import sys
from pathlib import Path

from . import KScript


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="KScript compiler")
    parser.add_argument("input", help="Input file (.ks, .json, .jsonl)")
    parser.add_argument("-out", dest="output", help="Output file (.json or .jsonl)")
    parser.add_argument("-base", dest="base", help="Base model file to extend (.json or .jsonl)")
    args = parser.parse_args()

    input_path = Path(args.input)

    try:
        # Load base model if provided
        base_model = KScript(args.base) if args.base else None

        # Compile or load input
        kscript = KScript(input_path, base=base_model)

        if args.output:
            output_path = Path(args.output)
            kscript.output(output_path)
            print(f"Compiled {input_path} -> {output_path}")
        else:
            # Output formatted model to terminal
            for entry in kscript.entries:
                print(json.dumps({entry.signature: entry.nodes}))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
