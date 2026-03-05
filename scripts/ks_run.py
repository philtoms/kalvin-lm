#!/usr/bin/env python3
"""Run a KScript file and save the resulting model.

Usage:
    python scripts/ks_run.py script.ks [--model path/to/model.bin]

Examples:
    python scripts/ks_run.py my-script.ks
    python scripts/ks_run.py my-script.ks --model existing-model.bin
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kalvin.kalvin import Kalvin
from kscript import interpret_script


def main():
    parser = argparse.ArgumentParser(
        description="Run a KScript file and save the resulting model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s my-script.ks                    # Creates my-script.bin
  %(prog)s my-script.ks --model model.bin  # Loads, updates, and saves model.bin
        """,
    )
    parser.add_argument("script", help="Path to .ks script file")
    parser.add_argument(
        "--model",
        "-m",
        help="Path to existing model file (.bin or .json). "
        "If not provided, creates a new model named after the script.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output path for the model. Defaults to model path or script name.",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["binary", "json"],
        default="binary",
        help="Output format (default: binary)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print verbose output"
    )

    args = parser.parse_args()

    script_path = Path(args.script)
    if not script_path.exists():
        print(f"Error: Script file not found: {script_path}", file=sys.stderr)
        sys.exit(1)

    # Read the script
    source = script_path.read_text()
    if args.verbose:
        print(f"Loaded script: {script_path}")
        print(f"Script length: {len(source)} bytes")

    # Load or create Kalvin agent
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"Error: Model file not found: {model_path}", file=sys.stderr)
            sys.exit(1)

        if args.verbose:
            print(f"Loading model: {model_path}")

        kalvin = Kalvin.load(model_path)
    else:
        # Create new Kalvin instance with default tokenizer
        if args.verbose:
            print("Creating new Kalvin agent with default tokenizer")

        kalvin = Kalvin()  # Uses default tokenizer

    # Interpret the script
    if args.verbose:
        print("Interpreting script...")

    result = interpret_script(source, agent=kalvin)

    if args.verbose:
        print(f"KLines in model: {len(result.model)}")
        print(f"Symbol table entries: {len(result.symbol_table)}")
        if result.load_paths:
            print(f"Load paths: {result.load_paths}")
        if result.save_path:
            print(f"Script save path: {result.save_path}")
        if result.attention_klines:
            print(f"Attention KLines: {len(result.attention_klines)}")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    elif args.model:
        output_path = Path(args.model)
    elif result.save_path:
        output_path = Path(result.save_path)
    else:
        # Use script name with .bin extension
        output_path = script_path.with_suffix(".bin")

    # Save the model (KLines already added via Kalvin agent)
    if args.verbose:
        print(f"Saving model to: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    kalvin.save(output_path, format=args.format)

    print(f"Model saved: {output_path} ({len(result.model)} KLines)")


if __name__ == "__main__":
    main()
