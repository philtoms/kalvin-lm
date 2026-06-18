"""Event enrichment for auto-tune.

Transforms raw harness WebSocket frames into the enriched auto-tune event
format defined in specs/auto-tune.md §Event Enrichment (rules 23–27).

Each event gets a monotonic ``seq`` counter supplied by the caller.
Rationalise and ratify events receive decompiled KScript source and a full
significance breakdown (raw, normalised, level).
"""

from __future__ import annotations

from functools import lru_cache

from kalvin.expand import boundaries, classify, normalise_significance
from kalvin.kline import KLine, kline_display
from kalvin.nlp_tokenizer import NLPTokenizer


@lru_cache(maxsize=1)
def _display_tokenizer() -> NLPTokenizer:
    """Lazily-built NLP tokenizer for kline display (cached; NLP data required)."""
    return NLPTokenizer.from_files()


# Public API


def enrich_event(raw_frame: dict, seq: int) -> dict:
    """Transform a raw harness WebSocket frame into an enriched auto-tune event.

    Args:
        raw_frame: Deserialized WebSocket JSON frame from the harness bus.
        seq: Monotonic sequence number assigned by the caller.

    Returns:
        Enriched event dict ready for writing to events.jsonl.

    Raises:
        ValueError: If the action type is unknown.
    """
    action = raw_frame["action"]
    message = raw_frame["message"]

    if action == "progress":
        return _enrich_progress(message, seq)
    elif action == "event":
        return _enrich_rationalise(message, seq)
    elif action == "ratify_request":
        return _enrich_ratify_request(message, seq)
    elif action == "notify":
        return _enrich_escalation(message, seq)
    else:
        raise ValueError(f"Unknown action: {action!r}")


# Action-specific enrichers


def _enrich_progress(message: dict, seq: int) -> dict:
    """Enrich a progress frame (rule 27)."""
    return {
        "seq": seq,
        "type": "progress",
        "status": message["status"],
        "lesson": message.get("lesson_label"),
        "lessons_total": message["lessons_total"],
        "lessons_completed": message["lessons_completed"],
    }


def _enrich_rationalise(message: object, seq: int) -> dict:
    """Enrich a rationalise event frame (rules 23–26)."""
    if isinstance(message, dict):
        kind = message["kind"]
        query = _to_kline(message["query"])
        proposal = _to_kline(message["proposal"])
        significance = message["significance"]
    else:
        kind = message.kind
        query = _to_kline(message.query)
        proposal = _to_kline(message.proposal)
        significance = message.significance

    return {
        "seq": seq,
        "type": "rationalise",
        "kind": kind,
        "significance": _build_significance(significance),
        "query": _build_kline_display(query),
        "proposal": _build_kline_display(proposal),
    }


def _enrich_ratify_request(message: dict, seq: int) -> dict:
    """Enrich a ratify request frame (rules 23–26)."""
    query = _to_kline(message["query"])
    proposal = _to_kline(message["proposal"])
    significance = message["significance"]

    return {
        "seq": seq,
        "type": "ratify_request",
        "query": _build_kline_display(query),
        "proposal": _build_kline_display(proposal),
        "significance": _build_significance(significance),
    }


def _enrich_escalation(message: dict, seq: int) -> dict:
    """Enrich a notify/escalation frame."""
    return {
        "seq": seq,
        "type": "escalation",
        "reason": message["reason"],
        "detail": message["detail"],
        "lesson_position": message["lesson_position"],
    }


# Helpers


def _build_significance(raw_sig: int) -> dict:
    """Build a Significance Object (§Significance Object).

    Returns dict with raw, normalised, and level fields.

    Normalisation is band-anchored via the shared
    ``normalise_significance`` helper: S1 → 1.0; S2 → linear in
    [0.50, 0.99]; S3 → asymptotic in (0.0, 0.50), never clamped;
    S4 → 0.0. The helper is the single source of truth.
    """
    s12, s23, s34 = boundaries()
    level = classify(raw_sig, s12, s23, s34)
    normalised = normalise_significance(raw_sig)
    return {
        "raw": raw_sig,
        "normalised": normalised,
        "level": level,
    }


def _build_kline_display(kline: KLine) -> dict:
    """Build a KLine Display Object (§KLine Display Object).

    Returns dict with raw (signature, nodes) and decompiled source.
    """
    source = _display_kline(kline)
    return {
        "raw": {"signature": kline.signature, "nodes": kline.nodes},
        "source": source,
    }


def _to_kline(obj: object) -> KLine:
    """Normalise a KLine instance or plain dict to a KLine instance."""
    if isinstance(obj, KLine):
        return obj
    if isinstance(obj, dict):
        return KLine(
            signature=obj["signature"],
            nodes=obj["nodes"],
        )
    raise TypeError(f"Cannot convert {type(obj).__name__} to KLine")


def _display_kline(kline: KLine) -> str:
    """Display a KLine as KScript source string.

    Returns "<unknown>" if display fails.
    """
    try:
        return kline_display(kline, _display_tokenizer())
    except Exception:
        pass
    return "<unknown>"
