"""Misfit — classification and expansion proposals for non-canonical KLines.

A kline is a misfit when its signature and its nodes' signature disagree.
This module classifies the misfit type (underfitting, overfitting, or both)
and generates expansion proposals that satisfy:
  - No invention: every signature used exists in the model.
  - No orphan nodes: removed nodes form a companion kline.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from kalvin.kline import KLine
from kalvin.signature import make_signature

if TYPE_CHECKING:
    from kalvin.model import Model


def classify_misfit(kline: KLine) -> tuple[bool, bool]:
    """Classify a kline's misfit type.

    Returns (underfitting, overfitting):
    - underfitting: True if S & ~N != 0 (signature promises more than nodes deliver)
    - overfitting: True if N & ~S != 0 (nodes carry more than signature captures)
    """
    nodes_sig = make_signature(kline.nodes)
    underfit = (kline.signature & ~nodes_sig) != 0
    overfit = (nodes_sig & ~kline.signature) != 0
    return underfit, overfit


def generate_expansions(
    model: Model,
    kline: KLine,
    underfit_gap: int,
    overfit_mask: int,
) -> Iterator[tuple[KLine, list[KLine]]]:
    """Generate expansion proposals for a misfit kline.

    Each yield is (proposal_kline, companion_klines) where:
    - proposal_kline is the expanded version of the input
    - companion_klines are klines formed from removed nodes (may be empty)

    Expansion proposals satisfy:
    - No invention: every signature used exists in the model
    - No orphan nodes: removed nodes form a companion kline
    """
    if underfit_gap and overfit_mask:
        yield from _dual_expansions(model, kline, underfit_gap, overfit_mask)
    else:
        if underfit_gap:
            yield from _underfit_expansions(model, kline, underfit_gap)

        if overfit_mask:
            yield from _overfit_expansions(kline, overfit_mask)


def _underfit_expansions(
    model: Model, kline: KLine, gap: int
) -> Iterator[tuple[KLine, list[KLine]]]:
    """Add nodes whose signatures overlap and thus reduce the gap."""
    contributors = model.where(lambda k: (k.signature & gap) != 0)

    for contributor in contributors:
        expanded_nodes = list(kline.nodes) + list(contributor.nodes)
        expanded_sig = kline.signature
        proposal = KLine(expanded_sig, expanded_nodes, kline.dbg)

        new_nodes_sig = make_signature(expanded_nodes)
        if (new_nodes_sig & expanded_sig) != 0:
            yield (proposal, [])


def _split_excess(kline: KLine, excess: int) -> tuple[list[int], list[int]]:
    """Split kline nodes into (excess_nodes, remaining) by excess mask."""
    excess_nodes = [n for n in kline.nodes if (n & excess) != 0]
    remaining = [n for n in kline.nodes if n not in excess_nodes]
    return excess_nodes, remaining


def _overfit_expansions(kline: KLine, excess: int) -> Iterator[tuple[KLine, list[KLine]]]:
    """Remove nodes whose bits contribute to the excess."""
    excess_nodes, remaining = _split_excess(kline, excess)

    if not excess_nodes:
        return
    trimmed = KLine(kline.signature, remaining, kline.dbg)

    companion_sig = make_signature(excess_nodes)
    companion = KLine(companion_sig, excess_nodes)

    yield (trimmed, [companion])


def _dual_expansions(
    model: Model, kline: KLine, gap: int, excess: int
) -> Iterator[tuple[KLine, list[KLine]]]:
    """Atomic replacement: swap excess nodes for gap-filling nodes."""
    excess_nodes, remaining = _split_excess(kline, excess)

    contributors = model.where(lambda k: (k.signature & gap) != 0)

    for contributor in contributors:
        replacement_nodes = remaining + list(contributor.nodes)
        replacement = KLine(kline.signature, replacement_nodes, kline.dbg)

        companion_sig = make_signature(excess_nodes)
        companion = KLine(companion_sig, excess_nodes)

        yield (replacement, [companion])
