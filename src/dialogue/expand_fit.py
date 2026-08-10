"""The expand S2 strategy — a faithful port of ``cogitator._run_work_item``.

For each grounded kline sharing a node value with the misfit ``entry`` (the
candidate set the production ``_route`` submits as S2/S3 work items), the
``(entry, candidate)`` pair is expanded via ``kalvin.expand.expand`` and
**every yield** is routed by band, exactly as the cogitator does:

- **S4** — skip.
- **S1** — the candidate is a structural exact match for the entry; ground it
  via the ``ground`` callback (the lean-engine analogue of the cogitator's
  ``on_s1`` promote) and stop expanding this pair.
- **S2/S3** — hand the yielded (possibly-misfit) kline to
  ``dialogue.proposals.propose_expansions``, which reshapes it into
  self-consistent proposal klines; each is emitted at its computed band.

The reshape step is what was missing from the earlier grade-only shortcut:
``propose_expansions`` turns a misfit candidate into underfit/overfit/dual
proposals rather than copying the candidate's nodes verbatim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dialogue.misfit import GroundedModel, similar_fit_candidates
from dialogue.proposals import propose_expansions
from kalvin.expand import expand
from kalvin.kline import KLine
from kalvin.kvalue import KValue
from kalvin.significance import SIG_S2, SIG_S3, BandLayout

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

    from dialogue.engine import EngineState
    from kalvin.abstract import KSignifier

__all__ = ["ExpandFit"]

_BAND_REP = {"S2": SIG_S2, "S3": SIG_S3}


class ExpandFit:
    """The alternative S2 strategy: route ``expand`` yields like the cogitator."""

    def __init__(self) -> None:
        self._layout = BandLayout()

    def propose(
        self,
        state: EngineState,
        signifier: KSignifier,
        entry: KLine,
        ground: Callable[[KLine], None],
    ) -> list[KValue]:
        model = GroundedModel(state)
        batch: list[KValue] = []
        for candidate in similar_fit_candidates(state, signifier, entry):
            batch.extend(self._expand_pair(model, signifier, entry, candidate, ground))
        return batch

    def _expand_pair(
        self,
        model: GroundedModel,
        signifier: KSignifier,
        entry: KLine,
        candidate: KLine,
        ground: Callable[[KLine], None],
    ) -> list[KValue]:
        """Route every yield of ``expand(entry, candidate)`` by band."""
        emissions: list[KValue] = []
        kv: KValue
        for kv in expand(model, entry, candidate, signifier):
            band = self._layout.classify(kv.significance)
            if band == "S4":
                continue
            if band == "S1":
                # Structural exact match — promote the candidate (the lean
                # analogue of on_s1) and stop expanding this pair.
                ground(candidate)
                return emissions
            emissions.extend(self._expansion_proposals(model, signifier, kv.kline, kv.significance))
        return emissions

    def _expansion_proposals(
        self,
        model: GroundedModel,
        signifier: KSignifier,
        candidate: KLine,
        significance: int,
    ) -> list[KValue]:
        """Reshape a graded S2/S3 candidate into self-consistent proposals."""
        rep = _BAND_REP.get(self._layout.classify(significance))
        if rep is None:
            return []
        return [
            KValue(proposal, rep)
            for proposal in propose_expansions(model, candidate, signifier)
        ]
