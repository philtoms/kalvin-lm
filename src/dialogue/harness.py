"""The lean training harness.

A minimal, synchronous, non-judging loop. Compile a KScript source, feed each
compiled entry to the engine one at a time, and present the engine's
``(batch, observations)`` response. The harness is feeder, driver, and
presenter — it never judges. The trainer (a pi agent, outside the loop) reads
the trace, makes decisions, edits the engine and/or the source, and re-runs.

Usage::

    PYTHONPATH=src python -m dialogue.harness path/to/curriculum.ks
    PYTHONPATH=src python -m dialogue.harness path/to/curriculum.ks -v
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from dialogue.engine import Engine, EngineState
from kalvin.kvalue import KValue
from kalvin.nlp_tokenizer import NLPTokenizer
from kalvin.significance import SIG_S1, SIG_S2, SIG_S3, SIG_S4
from kalvin.signifier import NLPSignifier
from ks.compiler import compile_source

_SIG_TO_BAND = {SIG_S1: "S1", SIG_S2: "S2", SIG_S3: "S3", SIG_S4: "S4"}
_BAND_ORDER = ("S1", "S2", "S3", "S4")


@dataclass
class StepResult:
    """One loop iteration: the input entry and the engine's response to it."""

    index: int
    entry: KValue
    batch: list[KValue] = field(default_factory=list)
    observations: list[KValue] = field(default_factory=list)


class Harness:
    """Compile a source, then feed each compiled entry to the engine in turn.

    The engine and its state are the only participants. The harness holds no
    verdict logic — it records what the engine returned and hands it to the
    presenter.
    """

    def __init__(
        self,
        signifier: NLPSignifier,
        tokenizer: NLPTokenizer,
        *,
        state: EngineState | None = None,
    ) -> None:
        self._signifier = signifier
        self._tokenizer = tokenizer
        self._engine = Engine(signifier)
        self._state = state if state is not None else EngineState()

    @property
    def state(self) -> EngineState:
        return self._state

    def run(self, source: str) -> list[StepResult]:
        """Compile ``source`` and drive the engine one entry per step."""
        entries = compile_source(
            source, tokenizer=self._tokenizer, signifier=self._signifier, dev=True
        )
        results: list[StepResult] = []
        for i, entry in enumerate(entries):
            batch, observations = self._engine.rationalise(self._state, [entry])
            results.append(StepResult(i, entry, batch, observations))
        return results


# ── Presentation ──────────────────────────────────────────────────────────


def _label(signature: int, labels: dict[int, str], verbose: bool) -> str:
    name = labels.get(signature)
    if verbose:
        return f"{name}|0x{signature:x}" if name else f"0x{signature:x}"
    return name or f"0x{signature:x}"


def _render_kline(value: KValue, labels: dict[int, str], verbose: bool) -> str:
    sig = _label(value.kline.signature, labels, verbose)
    nodes = ", ".join(_label(n, labels, verbose) for n in value.kline.nodes)
    return f"{sig}:[{nodes}]"


def _band(value: KValue) -> str:
    return _SIG_TO_BAND.get(value.significance, f"0x{value.significance:x}")


def _sig_to_label(source: str, tokenizer: NLPTokenizer, signifier: NLPSignifier) -> dict[int, str]:
    """Recompile once to recover ``{signature: scripted label}`` for display."""
    entries = compile_source(source, tokenizer=tokenizer, signifier=signifier, dev=True)
    out: dict[int, str] = {}
    for e in entries:
        d = e.kline.dbg
        if d is None:
            continue
        label = d.label or d.decoded
        if label:
            out.setdefault(e.kline.signature, label)
    return out


def _render_step(step: StepResult, labels: dict[int, str], verbose: bool) -> str:
    lines = [f"── Step {step.index + 1}  in  {_band(step.entry)}  "
             f"{_render_kline(step.entry, labels, verbose)} ──"]
    if step.batch:
        for v in step.batch:
            lines.append(f"  out   {_band(v)}  {_render_kline(v, labels, verbose)}")
    else:
        lines.append("  out   (none)")
    if step.observations:
        for v in step.observations:
            lines.append(f"  ground  {_render_kline(v, labels, verbose)}")
    return "\n".join(lines)


def _render_grounded(state: EngineState, labels: dict[int, str], verbose: bool) -> str:
    if not state.grounded:
        return "  (grounded nothing)"
    lines = []
    for signature in sorted(state.grounded, key=lambda s: (s.bit_length(), s)):
        owner = _label(signature, labels, verbose)
        bucket = state.grounded[signature]
        for kl in bucket:
            nodes = ", ".join(_label(n, labels, verbose) for n in kl.nodes)
            lines.append(f"  {owner}:[{nodes}]")
    return "\n".join(lines)


def _render_summary(results: list[StepResult], state: EngineState,
                    labels: dict[int, str], verbose: bool) -> str:
    bands: Counter = Counter()
    for step in results:
        for v in step.batch:
            bands[_band(v)] += 1
    band_str = "  ".join(f"{b}={bands.get(b, 0)}" for b in _BAND_ORDER) or "-"
    return (
        f"── summary ──\n"
        f"  steps: {len(results)}\n"
        f"  batch by band: {band_str}\n"
        f"  grounded:\n{_render_grounded(state, labels, verbose)}"
    )


def present(results: list[StepResult], state: EngineState, source: str,
            tokenizer: NLPTokenizer, signifier: NLPSignifier, *, verbose: bool) -> None:
    labels = _sig_to_label(source, tokenizer, signifier)
    for step in results:
        print(_render_step(step, labels, verbose))
    print()
    print(_render_summary(results, state, labels, verbose))


# ── CLI ───────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a KScript source through the lean engine and present the trace.",
    )
    parser.add_argument("source", help="Path to a .ks KScript file")
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show hex signatures alongside scripted labels.",
    )
    args = parser.parse_args(argv)

    source_path = Path(args.source)
    try:
        source = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"harness: could not read {args.source!r}: {exc}", file=sys.stderr)
        return 2

    tok = NLPTokenizer()
    sigf = NLPSignifier()
    harness = Harness(sigf, tok)
    results = harness.run(source)
    present(results, harness.state, source, tok, sigf, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
