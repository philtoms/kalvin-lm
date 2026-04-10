#!/usr/bin/env python3
"""Run a KScript file and save the resulting agent.

Usage:
    python scripts/ks_run.py script.ks [--agent path/to/model.json]

Examples:
    python scripts/ks_run.py my-script.ks
    python scripts/ks_run.py my-script.ks --agent existing-model.json
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kalvin.agent import Agent
from kscript import interpret_script, Mod32Tokenizer as mod_tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Run a KScript file and (by default) save the rationalised agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s my-script.ks                    # Creates my-script.bin
  %(prog)s my-script.ks --agent model.bin  # Loads, updates, and saves model.bin
        """,
    )
    parser.add_argument("script", help="Path to .ks script file")
    parser.add_argument(
        "--agent",
        "-m",
        help="Path to existing agent file (.bin or .json). "
        "If not provided, creates a new agent named after the script.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output path for the agent. Defaults to agent path or script name.",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["bin", "json"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--save", "-s", action="store_true", help="Save rationalised agent."
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

    # Load or create Agent
    if args.agent:
        agent_path = Path(args.agent)
        if not agent_path.exists():
            print(f"Error: Agent file not found: {agent_path}", file=sys.stderr)
            sys.exit(1)

        if args.verbose:
            print(f"Loading agent: {agent_path}")

        agent = Agent.load(agent_path)
    else:
        # Create new Agent instance
        if args.verbose:
            print("Creating new Agent")

        tokenizer = mod_tokenizer()
        agent = Agent(tokenizer=tokenizer)

    # Interpret the script
    if args.verbose:
        print("Interpreting script...")

    result = interpret_script(source, agent=agent)

    if args.verbose:
        print(f"KLines in agent: {len(result.agent)}")
        print(f"Symbol table entries: {len(result.symbol_table)}")
        if result.load_paths:
            print(f"Load paths: {result.load_paths}")
        if result.save_path:
            print(f"Script save path: {result.save_path}")

    # Determine output path
    if args.save:
        if args.output:
            output_path = Path(args.output)
        elif args.agent:
            output_path = Path(args.agent)
        elif result.save_path:
            output_path = Path(result.save_path)
        else:
            # Use script name with format extension
            output_path = script_path.with_suffix(f".{args.format}")

        # Save the agent (KLines already added via Agent)
        if args.verbose:
            print(f"Saving agent: {output_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        agent.save(output_path, format=args.format)

        print(f"Agent saved: {output_path} ({len(result.agent)} KLines)")


if __name__ == "__main__":
    main()
