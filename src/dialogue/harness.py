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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from dialogue.engine import Engine, MisfitStrategy
from dialogue.engine_state import EngineState
from dialogue.expand_fit import ExpandFit
from dialogue.similar_fit import SimilarFit
from kalvin.kline import KLine
from kalvin.kvalue import KValue
from kalvin.nlp_tokenizer import NLPTokenizer
from kalvin.significance import SIG_S1, SIG_S2, SIG_S3, SIG_S4
from kalvin.signifier import NLPSignifier
from ks.compiler import compile_source

# Named cogitation strategies for the misfit (S2) arm of ``cogitate``.
# "similar_fit" — the work-list graft heuristic (the original scheme).
# "expand"      — grade grounded candidates via ``ExpandFit._expand`` and
#                 propose under the entry's signature at the computed band.
_STRATEGIES: dict[str, Callable[[EngineState], MisfitStrategy]] = {
    "similar_fit": SimilarFit,
    "expand": ExpandFit,
}

_SIG_TO_BAND = {SIG_S1: "S1", SIG_S2: "S2", SIG_S3: "S3", SIG_S4: "S4"}
_BAND_ORDER = ("S1", "S2", "S3", "S4")


def _identity_offers(entries: list[KValue]) -> dict[int, KValue]:
    """``{signature: X:[X] at S1}`` for every signature the curriculum defines
    as a self-identity. The first such entry per signature wins."""
    offers: dict[int, KValue] = {}
    for entry in entries:
        if entry.significance != SIG_S1:
            continue
        sig = entry.kline.signature
        if sig in offers:
            continue
        nodes = entry.kline.nodes
        if len(nodes) == 1 and nodes[0] == sig:
            offers[sig] = KValue(KLine(sig, [sig]), SIG_S1)
    return offers


@dataclass
class StepResult:
    """One loop iteration: the input entry and the engine's responses to it.

    A step may span several engine calls. ``offers`` are the S1 identities the
    harness fed to answer S4 asks the engine emitted — presented for
    observability, not as a verdict.
    """

    index: int
    entry: KValue
    batch: list[KValue] = field(default_factory=list)
    observations: list[KValue] = field(default_factory=list)
    offers: list[KValue] = field(default_factory=list)


class Harness:
    """Compile a source, then feed each compiled entry to the engine in turn.

    The engine and its state are the only participants. The harness holds no
    verdict logic — it records what the engine returned and hands it to the
    presenter. Both ``tokenizer`` and ``engine`` are supplied fully
    constructed; see :func:`make_engine` / :func:`load_engine` for the
    single construction path.
    """

    def __init__(
        self,
        tokenizer: NLPTokenizer,
        engine: Engine,
    ) -> None:
        self._tokenizer = tokenizer
        self._engine = engine

    @property
    def engine(self) -> Engine:
        return self._engine

    @property
    def state(self) -> EngineState:
        return self._engine.state

    @property
    def signifier(self) -> NLPSignifier:
        # make_engine/load_engine store an NLPSignifier; the state types it as
        # KSignifier.
        return cast(NLPSignifier, self._engine.state.signifier)

    def run(self, source: str) -> list[StepResult]:
        """Compile ``source`` and drive the engine one entry per step.

        After each step, scan the batch for S4 identity asks whose signature
        the curriculum defines as an S1 identity ``X:[X]`` and feed those
        identities directly. This answers K's asks without supervision and
        without judgement; it changes the order K assimilates the curriculum.
        """
        entries = compile_source(
            source, tokenizer=self._tokenizer, signifier=self.signifier, dev=True
        )
        identities = _identity_offers(entries)
        results: list[StepResult] = []
        for i, entry in enumerate(entries):
            batch, observations = self._engine.rationalise([entry])
            offered: list[KValue] = []
            seen: set[int] = set()
            while True:
                asks = {
                    v.kline.signature for v in batch
                    if v.significance == SIG_S4 and not v.kline.nodes
                }
                signature = next((s for s in asks if s in identities and s not in seen), None)
                if signature is None:
                    break
                seen.add(signature)
                offer = identities[signature]
                offered.append(offer)
                batch, obs = self._engine.rationalise([offer])
                observations.extend(obs)
            results.append(StepResult(i, entry, batch, observations, offered))
        return results


# ── Construction (single source of truth) ────────────────────────────────
#
# The factories wire signifier → state → strategy → engine → harness so the
# signifier lives in one place (the EngineState). ``misfit_cls`` is a strategy
# class (ExpandFit / SimilarFit) constructed against the fresh or loaded state.

def make_engine(
    tokenizer: NLPTokenizer,
    misfit_cls: Callable[[EngineState], MisfitStrategy],
) -> Harness:
    """Build a harness over a fresh state: new signifier → state → engine."""
    state = EngineState(NLPSignifier())
    misfit = misfit_cls(state)
    return Harness(tokenizer, Engine(state, misfit))


def load_engine(
    path: str | Path,
    tokenizer: NLPTokenizer,
    misfit_cls: Callable[[EngineState], MisfitStrategy],
) -> Harness:
    """Build a harness over a loaded prior state (reusing its signifier)."""
    signifier = NLPSignifier()
    state = EngineState.load(signifier, path)
    misfit = misfit_cls(state)
    return Harness(tokenizer, Engine(state, misfit))


# ── Presentation ──────────────────────────────────────────────────────────


def _label(signature: int, labels: dict[int, str], verbose: bool) -> str:
    name = labels.get(signature)
    if verbose:
        return f"{name}|0x{signature:x}" if name else f"0x{signature:x}"
    return name or f"0x{signature:x}"


def _render_kline_struct(kline: KLine, labels: dict[int, str], verbose: bool) -> str:
    sig = _label(kline.signature, labels, verbose)
    nodes = ", ".join(_label(n, labels, verbose) for n in kline.nodes)
    return f"{sig}:[{nodes}]"


def _render_kline(value: KValue, labels: dict[int, str], verbose: bool) -> str:
    return _render_kline_struct(value.kline, labels, verbose)


def _band(value: KValue) -> str:
    return _SIG_TO_BAND.get(value.significance, f"0x{value.significance:x}")


def _sig_to_label(source: str, tokenizer: NLPTokenizer, signifier: NLPSignifier) -> dict[int, str]:
    """Recompile once to recover ``{signature: scripted label}`` for display.

    Compiled-entry labels are authoritative; the encoder's ``node_labels``
    supplies words for single-token node values that never head an entry
    (e.g. MTS-expanded words like ``did``/``have``).
    """
    from ks.compiler import Compiler
    from ks.lexer import Lexer
    from ks.parser import Parser
    compiler = Compiler(tokenizer, signifier=signifier, dev=True)
    entries = compiler.compile(Parser(Lexer(source).tokenize()).parse())
    out: dict[int, str] = dict(compiler.node_labels)
    for e in entries:
        d = e.kline.dbg
        if d is None:
            continue
        label = d.label
        if label:
            out.setdefault(e.kline.signature, label)
    return out


def _render_step(step: StepResult, labels: dict[int, str], verbose: bool) -> str:
    lines = [f"── Step {step.index + 1}  in  {_band(step.entry)}  "
             f"{_render_kline(step.entry, labels, verbose)} ──"]
    for v in step.offers:
        lines.append(f"  offer   {_render_kline(v, labels, verbose)}")
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
    if not state.ltm:
        return "  (grounded nothing)"
    lines = []
    for signature in sorted(state.ltm, key=lambda s: (s.bit_length(), s)):
        owner = _label(signature, labels, verbose)
        bucket = state.ltm[signature]
        for kl in bucket:
            nodes = ", ".join(_label(n, labels, verbose) for n in kl.nodes)
            lines.append(f"      {owner}:[{nodes}]")
    return "\n".join(lines)


def _render_work_list(state: EngineState, labels: dict[int, str], verbose: bool) -> str:
    if not state.work_list:
        return "  (empty)"
    return "\n".join(
        f"      {_render_kline_struct(kl, labels, verbose)}" for kl in state.work_list
    )


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
        f"  grounded:\n{_render_grounded(state, labels, verbose)}\n"
        f"  work_list (pending at end of run):\n{_render_work_list(state, labels, verbose)}"
    )


def present(results: list[StepResult], state: EngineState, source: str,
            tokenizer: NLPTokenizer, signifier: NLPSignifier, *, verbose: bool) -> None:
    labels = _sig_to_label(source, tokenizer, signifier)
    last_annotation: str | None = None
    for step in results:
        annotation = step.entry.kline.dbg.annotation if step.entry.kline.dbg else ""
        if annotation and annotation != last_annotation:
            print(f"\n[{annotation}]")
            last_annotation = annotation
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
    parser.add_argument(
        "-s", "--strategy", choices=("similar_fit", "expand"), default="expand",
        help="Cogitation strategy for the misfit (S2) arm. "
             "'expand' (default) grades grounded candidates via "
             "kalvin.expand.expand; 'similar_fit' is the graft heuristic.",
    )
    args = parser.parse_args(argv)

    source_path = Path(args.source)
    try:
        source = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"harness: could not read {args.source!r}: {exc}", file=sys.stderr)
        return 2

    tok = NLPTokenizer()
    harness = make_engine(tok, _STRATEGIES[args.strategy])
    results = harness.run(source)
    present(results, harness.state, source, tok, harness.signifier, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
