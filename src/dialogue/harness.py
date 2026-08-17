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
# "similar_fit" — the STM graft heuristic (the original scheme).
# "expand"      — grade grounded candidates via ``ExpandFit._expand`` and
#                 propose under the entry's signature at the computed band.
_STRATEGIES: dict[str, Callable[[EngineState], MisfitStrategy]] = {
    "similar_fit": SimilarFit,
    "expand": ExpandFit,
}

_SIG_TO_BAND = {SIG_S1: "S1", SIG_S2: "S2", SIG_S3: "S3", SIG_S4: "S4"}
_BAND_ORDER = ("S1", "S2", "S3", "S4")


@dataclass
class StepResult:
    """One loop iteration: the input entry, the engine's asks, and the
    ratifying answers the harness fed back. """

    index: int
    entry: KValue
    batch: list[KValue] = field(default_factory=list)
    observations: list[KValue] = field(default_factory=list)
    answers: list[KValue] = field(default_factory=list)
    stopped_on: KValue | None = None


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
        """Compile ``source``, open each sub-script's dialogue, and let the
        engine drive.

        Each sub-script (annotation group) is opened with its first entry.
        From there the engine asks; the harness answers each ask from the
        script or the run stops.
        """
        entries = compile_source(
            source, tokenizer=self._tokenizer, signifier=self.signifier, dev=True
        )
        heads: dict[int, list[KValue]] = {}
        exact: dict[tuple[int, tuple[int, ...]], KValue] = {}
        words: set[int] = set()
        openers: list[KValue] = []
        opened: set[str] = set()
        for entry in entries:
            kline = entry.kline
            heads.setdefault(kline.signature, []).append(entry)
            exact.setdefault(
                (kline.signature, tuple(kline.nodes)), entry
            )
            if (
                kline.nodes != [kline.signature]
                and kline.signature == self.signifier.signature_of(kline.nodes)
            ):
                # A compound self-ref (e.g. DH unpacking to did, have): its
                # nodes are script-known words.
                words.update(kline.nodes)
            annotation = kline.dbg.annotation if kline.dbg else ""
            if annotation and annotation not in opened:
                opened.add(annotation)
                openers.append(entry)
        answered: set[tuple[int, tuple[int, ...]]] = set()
        results: list[StepResult] = []
        for i, opener in enumerate(openers):
            step = StepResult(i, opener)
            results.append(step)
            queue: list[KValue] = [opener]
            while queue:
                batch, observations = self._engine.rationalise([queue.pop(0)])
                step.observations.extend(observations)
                step.batch = batch
                for ask in batch:
                    reply = self._answer(ask, heads, exact, words, answered)
                    if reply is None:
                        step.stopped_on = ask
                        return results
                    step.answers.extend(reply)
                    queue.extend(reply)
        return results

    def _answer(
        self,
        ask: KValue,
        heads: dict[int, list[KValue]],
        exact: dict[tuple[int, tuple[int, ...]], KValue],
        words: set[int],
        answered: set[tuple[int, tuple[int, ...]]],
    ) -> list[KValue] | None:
        """The ratifying reply to ``ask``, or ``None`` when the script cannot
        answer it.

        An identity ask ``X:[]`` is answered by an identity ``X:[X]`` plus the
        script klines headed ``X``. A proposal ask ``A:[B]`` is answered by the
        matching script kline plus its countersignature ``B:[A]``.
        """
        kline = ask.kline
        key = (kline.signature, tuple(kline.nodes))
        if key in answered:
            return []
        answered.add(key)
        if not kline.nodes:
            script_klines = [
                e for e in heads.get(kline.signature, [])
                if e.kline.nodes != [kline.signature]
            ]
            is_word = kline.signature in words or any(
                e.kline.nodes == [kline.signature]
                for e in heads.get(kline.signature, [])
            )
            if not script_klines and not is_word:
                return None
            identity = KValue(
                KLine(kline.signature, [kline.signature]), SIG_S1
            )
            return [identity, *script_klines]
        hit = exact.get(key)
        if hit is None:
            return None
        countersigns = [
            KValue(KLine(node, [kline.signature]), SIG_S3)
            for node in kline.nodes
        ]
        return [hit, *countersigns]


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
    for v in step.answers:
        lines.append(f"  answer  {_render_kline(v, labels, verbose)}")
    if step.stopped_on is not None:
        lines.append(f"  stop    unanswerable ask  {_render_kline(step.stopped_on, labels, verbose)}")
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


def _render_stm(state: EngineState, labels: dict[int, str], verbose: bool) -> str:
    if not state.stm:
        return "  (empty)"
    return "\n".join(
        f"      {_render_kline_struct(kl, labels, verbose)}" for kl in state.stm
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
        f"  stm (attending to at end of run):\n{_render_stm(state, labels, verbose)}"
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
