#!/usr/bin/env python3
"""Encode a text file into Agent embeddings.

This script loads a text file, splits it into sentences, encodes them
using Agent's batch encoding, and reports agent statistics.

Can optionally load and extend an existing agent.
"""

import argparse
import json
import re
import signal
import sys
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kalvin.agent import KAgent
from kalvin.events import EventBus
from kalvin.kline import KLine

# Global state for interrupt handling
_interrupted = False
_signalled = False


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global _interrupted
    global _signalled
    if not _interrupted:
        _interrupted = True
        print("\n\nInterrupt received, saving agent...")
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
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
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
                with open(text_path) as f:
                    data = json.load(f)
                    if "summaries" in data:
                        data = data["summaries"]
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

    parser = argparse.ArgumentParser(description="Encode text files into Agent embeddings")
    parser.add_argument("-i", "--input_file", default=None, help="Path to the text file to encode")
    parser.add_argument(
        "--agent",
        default=None,
        help="Path to load/save the agent (default: <data>/kalvin.bin)",
    )

    parser.add_argument(
        "-f",
        "--format",
        choices=["binary", "json"],
        default="binary",
        help="Agent format (default: binary)",
    )
    args = parser.parse_args()
    from kalvin.paths import data_dir

    _data = data_dir()
    agent_path = args.agent or str(_data / "kalvin.bin")
    input_file = args.input_file or str(_data / "tokenizer" / "simplestories-1.json")

    # Load or initialize Agent
    agent_path = Path(agent_path)
    agent_size = 0
    if agent_path and agent_path.exists():
        print(f"\nLoading agent from: {agent_path}")
        agent = KAgent.load(agent_path, format=args.format, adapter=EventBus())
        agent_size = agent.frame_size()
        print(f"Loaded agent size: {agent_size:,} KLines")
    else:
        print("\nInitializing new agent...")
        agent = KAgent(adapter=EventBus())
        print(f"Initial agent size: {agent.frame_size():,} KLines")

    for text in stream_text(input_file):
        if _interrupted:
            break

        # Split into sentences
        sentences = split_into_sentences(text)

        # Encode all sentences in batch
        for sentence in tqdm(sentences, desc=f"Encoding {len(sentences):,} sentences..."):
            if _interrupted:
                break
            nodes = agent.tokenizer.encode(sentence)
            kline = KLine(signature=agent.signifier.make_signature(nodes), nodes=nodes)
            agent.rationalise(kline)

    # Report agent size
    final_size = agent.frame_size()
    print(f"\nFinal agent size: {final_size:,} KLines")
    if final_size > agent_size:
        print(f"Extended by: {final_size - agent_size:,} KLines")

    # Save agent
    print(f"\nSaving agent to: {agent_path}")
    agent.save(agent_path, format=args.format)
    actual_size = Path(agent_path).stat().st_size
    print(f"Saved: {actual_size / 1024:.1f} KB ({actual_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
