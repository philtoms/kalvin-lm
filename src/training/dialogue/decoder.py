"""Single-stage dialogue-script decoder — see @specs/dialogue-driven-training.md §Decode.

``decode(script)`` turns a :class:`DialogueScript` into a flat ordered
``list[DecodedTurn]``, resolving every symbolic label against ``script.source``."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from kalvin.expand import SIG_S1, SIG_S2, SIG_S3, SIG_S4
from kalvin.kline import KLine
from kalvin.kvalue import KValue
from kalvin.nlp_tokenizer import COMPOUND_TOKEN
from ks.compiler import compile_source

if TYPE_CHECKING:
    from kalvin.abstract import KSignifier
    from kalvin.nlp_tokenizer import NLPTokenizer

# Significance by band lookup (independent of ``op``).
BAND_TO_SIG: dict[str, int] = {
    "S1": SIG_S1,
    "S2": SIG_S2,
    "S3": SIG_S3,
    "S4": SIG_S4,
}

Role = Literal["T", "K"]  # trainer (T) or trainee (K)
OnDivergence = Literal["fail", "accept"]
DIALOGUE_OPS = frozenset({"COUNTERSIGNS", "CANONIZES", "CONNOTES", "DENOTES", "IDENTITY", "UNKNOWN"})


class DecodeError(Exception):
    """A turn could not be decoded (unknown op, unresolved symbol, bad band, …)."""

# ── Typed script structures ────────────────────────────────────────────────


@dataclass(frozen=True)
class Turn:
    """One row of the dialogue script. Annotation-only turns (``notes``, no
    ``op``) are dropped at decode. ``close`` marks the run's terminal row."""

    role: Role
    op: str | None  # None on annotation-only turns
    signature: str | None
    nodes: tuple[str, ...]
    significance: str | None  # None on annotation-only turns
    notes: str = ""
    close: bool = False  # True when this turn closes a script (a boundary marker)
    # The raw JSON record this turn was loaded from (``None`` for turns built
    # synthetically, e.g. test fixtures). Carried for diagnostics so a trace
    # can show the script row verbatim alongside its decoded form.
    record: dict | None = None

    @property
    def is_annotation_only(self) -> bool:
        """An annotation-only turn carries ``notes`` but no ``op``."""
        return self.op is None


@dataclass(frozen=True)
class RunConfig:
    """Runner configuration, carried on a ``DialogueTable``'s optional ``run``
    section."""

    on_divergence: OnDivergence = "fail"


@dataclass(frozen=True)
class DialogueScript:
    """``source`` is the source of truth for kline structure; ``turns`` is the
    exchange; ``events`` (optional) holds expected K groundings; ``priors``
    (optional) are script files run before this one as a run sequence."""

    source: str
    turns: tuple[Turn, ...]
    events: tuple[Turn, ...] = ()
    # ``priors`` (optional): other script files run **before** this script's
    # own run, as a sequence of independent runs against the same actor
    # instances. Each is a path to a script file (resolved and decoded by the
    # sequencer, not merged in here).
    priors: tuple[str, ...] = ()
    run_config: RunConfig | None = None

    @property
    def has_run_config(self) -> bool:
        """True when this script carries a ``run`` section."""
        return self.run_config is not None


@dataclass(frozen=True)
class DecodedTurn:
    """A turn resolved to a :class:`KValue`, carrying ``role``/``op``/``close``."""

    role: Role
    op: str
    value: KValue
    close: bool = False  # True when this turn closes a script
    # The raw JSON record this decoded turn was loaded from (``None`` for
    # turns built synthetically). Carried for diagnostics.
    record: dict | None = None



# ── Symbol resolution against compiled source ─────────────────────────────


@dataclass(frozen=True)
class _ResolvedScript:
    """Compiled-source indices built at decode time."""

    canon_by_label: dict[str, KLine] = field(default_factory=dict)
    relation_by_label: dict[str, KLine] = field(default_factory=dict)
    compound_by_label: dict[str, KLine] = field(default_factory=dict)
    labels: dict[str, KLine] = field(default_factory=dict)


def _resolve_script(
    source: str,
    *,
    tokenizer: NLPTokenizer | None = None,
    signifier: KSignifier | None = None,
) -> tuple[list[KValue], _ResolvedScript]:
    """Compile ``source`` once and build the canon + label indices."""
    entries = compile_source(source, tokenizer=tokenizer, signifier=signifier, dev=True)

    resolved = _ResolvedScript()
    for e in entries:
        kl = e.kline
        d = kl.dbg
        if d is None:
            raise DecodeError(
                f"compiled entry 0x{kl.signature:x} has no debug info "
                "— dev compile invariant broken"
            )
        # Canon-by-label: populated for any compiled canon (COUNTERSIGNS or
        # CANONIZES with nodes) so node/signature resolution can prefer it.
        if d.op in ("CANONIZES", "COUNTERSIGNS") and kl.nodes and d.label:
            resolved.canon_by_label.setdefault(d.label, kl)
        # Compound-by-label: the compound-word identity (CANONIZES whose nodes
        # include COMPOUND_TOKEN), held separately from a same-label block-canon.
        if d.op == "CANONIZES" and d.label and COMPOUND_TOKEN in kl.nodes:
            resolved.compound_by_label.setdefault(d.label, kl)
        if d.op in ("COUNTERSIGNS", "CONNOTES", "DENOTES") and d.label:
            resolved.relation_by_label.setdefault(d.label, kl)
        # Label index: atom/compound dbg.label, and subword dbg.decoded.
        if d.label:
            resolved.labels.setdefault(d.label, kl)
        if d.decoded:
            resolved.labels.setdefault(d.decoded, kl)
    return entries, resolved


# ── The single-stage decode ──────────────────────────────────────────────


def _resolve_node_signatures(
    nodes: tuple[str, ...],
    resolved: _ResolvedScript,
    *,
    op: str,
) -> list[int]:
    """Resolve each node label to its canonical signature (canon-preferred,
    atom fallback). Shared by the CANONIZES and constructed-relation branches."""
    node_sigs: list[int] = []
    for n in nodes:
        ncanon = resolved.canon_by_label.get(n)
        if ncanon is not None:
            node_sigs.append(ncanon.signature)
            continue
        nkl = resolved.labels.get(n)
        if nkl is None:
            raise DecodeError(
                f"{op} node {n!r}: label not found in compiled source"
            )
        node_sigs.append(nkl.signature)
    return node_sigs


def _resolve_kline(
    op: str,
    signature: str,
    nodes: tuple[str, ...],
    resolved: _ResolvedScript,
) -> KLine:
    """Build the kline the turn declares — a resolver, not a gatekeeper (an
    author may declare a deliberate misfit)."""
    if op == "CANONIZES":
        node_sigs = _resolve_node_signatures(nodes, resolved, op="CANONIZES")
        sig_kl = resolved.canon_by_label.get(signature) or resolved.labels.get(signature)
        if sig_kl is None:
            raise DecodeError(
                f"CANONIZES signature {signature!r}: label not found in compiled source"
            )
        # Compound catch-up: if the signature names a compound-word and the
        # declared nodes are exactly its subwords, prepend COMPOUND_TOKEN so
        # the kline is the compound identity (not a misfit against the CT-encoded
        # signature). Gated on the subwords to avoid folding a same-label
        # block-canon (e.g. ``had => did have``) into the compound identity.
        node_sigs = _maybe_catch_up_compound(signature, node_sigs, resolved)
        return KLine(sig_kl.signature, node_sigs, dbg=sig_kl.dbg)

    if op == "UNKNOWN":
        kl = resolved.canon_by_label.get(signature) or resolved.labels.get(signature)
        if kl is None:
            raise DecodeError(
                f"UNKNOWN {signature!r}: label not found in compiled source"
            )
        if nodes:
            raise DecodeError(
                f"UNKNOWN {signature!r}: an Unknown carries no decomposition, "
                f"but nodes {nodes!r} were declared"
            )
        return KLine(kl.signature, [], dbg=kl.dbg)  # the S4 ask ``X:[]``

    if op == "IDENTITY":
        kl = resolved.canon_by_label.get(signature) or resolved.labels.get(signature)
        if kl is None:
            raise DecodeError(
                f"IDENTITY {signature!r}: label not found in compiled source"
            )
        if not nodes:
            # A scripted true identity with no declared nodes: the
            # self-referential form ``{X: [X]}`` (S1), the decodable identity
            # the UNKNOWN token cannot express.
            return KLine(kl.signature, [kl.signature], dbg=kl.dbg)
        node_sigs = _resolve_node_signatures(nodes, resolved, op="IDENTITY")
        node_sigs = _maybe_catch_up_compound(signature, node_sigs, resolved)
        return KLine(kl.signature, node_sigs, dbg=kl.dbg)

    # Constructed relation (CONNOTES/DENOTES/COUNTERSIGNS).
    node_sigs = _resolve_node_signatures(nodes, resolved, op="relation")
    sig_kl = (
        resolved.relation_by_label.get(signature)
        or resolved.canon_by_label.get(signature)
        or resolved.labels.get(signature)
    )
    if sig_kl is None:
        raise DecodeError(
            f"relation signature {signature!r}: label not found in compiled source"
        )
    return KLine(sig_kl.signature, node_sigs, dbg=sig_kl.dbg)


def _maybe_catch_up_compound(
    signature: str, node_sigs: list[int], resolved: _ResolvedScript
) -> list[int]:
    """Prepend COMPOUND_TOKEN when ``signature``'s compound identity's subwords
    equal ``node_sigs``."""
    compound = resolved.compound_by_label.get(signature)
    if compound is not None and COMPOUND_TOKEN not in node_sigs:
        subwords = [n for n in compound.nodes if n != COMPOUND_TOKEN]
        if list(node_sigs) == subwords:
            return [COMPOUND_TOKEN, *node_sigs]
    return node_sigs


def decode(
    script: DialogueScript,
    *,
    tokenizer: NLPTokenizer | None = None,
    signifier: KSignifier | None = None,
) -> list[DecodedTurn]:
    """Decode ``script.turns`` to a flat ordered ``list[DecodedTurn]``."""
    resolved = _resolve_script(
        script.source, tokenizer=tokenizer, signifier=signifier
    )[1]

    out = _decode_turns(script.turns, resolved, what="turn")
    # ``close`` markers and dialogue-mode invariants are validated on the
    # decoded list (they require symbol resolution to have run).
    _validate_close(out)
    if script.has_run_config:
        _validate_run(out)
    return out


def decode_events(
    script: DialogueScript,
    *,
    tokenizer: NLPTokenizer | None = None,
    signifier: KSignifier | None = None,
) -> list[DecodedTurn]:
    """Decode the script's ``events`` (expected K groundings) for white-box
    verification. Same resolution as :func:`decode`; no ``close`` semantics."""
    resolved = _resolve_script(
        script.source, tokenizer=tokenizer, signifier=signifier
    )[1]
    return _decode_turns(script.events, resolved, what="event")


def _decode_turns(
    turns: tuple[Turn, ...], resolved, *, what: str
) -> list[DecodedTurn]:
    """Resolve ``turns`` to :class:`DecodedTurn`\ s (shared by turns and events).
    Annotation-only rows are dropped; ``what`` labels the row kind in errors."""
    out: list[DecodedTurn] = []
    for idx, turn in enumerate(turns):
        if turn.is_annotation_only:
            continue
        assert turn.op is not None and turn.significance is not None  # annotation guard
        if turn.signature is None:
            raise DecodeError(f"{what} {idx}: structural row missing 'signature'")
        if turn.op not in DIALOGUE_OPS:
            raise DecodeError(f"{what} {idx}: unknown op {turn.op!r}")
        if turn.significance not in BAND_TO_SIG:
            raise DecodeError(
                f"{what} {idx}: unknown significance {turn.significance!r}"
            )

        kline = _resolve_kline(turn.op, turn.signature, turn.nodes, resolved)
        significance = BAND_TO_SIG[turn.significance]
        out.append(
            DecodedTurn(
                role=turn.role,
                op=turn.op,
                value=KValue(kline, significance),
                close=turn.close,
                record=turn.record,
            )
        )
    return out


# ── Loader ─────────────────────────────────────────────────────────────────


def _turn_from_dict(raw: dict) -> Turn:
    """Build a :class:`Turn` from a raw JSON dict. A structural turn (with
    ``op``) must also carry ``signature`` and ``significance``."""
    role = raw.get("role")
    if role not in ("T", "K"):
        raise DecodeError(f"turn role must be 'T' or 'K', got {role!r}")
    op = raw.get("op")
    if op is not None and op not in DIALOGUE_OPS:
        raise DecodeError(f"unknown op {op!r}")
    nodes = tuple(raw.get("nodes", ()) or ())
    if op is not None:
        if "signature" not in raw:
            raise DecodeError(f"structural turn missing 'signature': {raw!r}")
        if "significance" not in raw:
            raise DecodeError(f"structural turn missing 'significance': {raw!r}")
    close = raw.get("close")
    if close is not None and not (isinstance(close, bool) and close):
        raise DecodeError(
            f"'close' must be the boolean true (a source-boundary marker), got {close!r}"
        )
    return Turn(
        role=role,
        op=op,
        signature=raw.get("signature"),
        nodes=nodes,
        significance=raw.get("significance"),
        notes=raw.get("notes", ""),
        close=close,
        record=raw,
    )


# ── Dialogue-run decode-time validations ──────────────────────────────────


def turn_content_key(turn: DecodedTurn) -> tuple[str, int, tuple[int, ...], int]:
    """``(role, sig, nodes, significance)`` — :class:`KValue` equality ignores sig."""
    return (
        turn.role,
        turn.value.kline.signature,
        tuple(turn.value.kline.nodes),
        turn.value.significance,
    )


def _validate_close(decoded: list[DecodedTurn]) -> None:
    """No-op: a ``close:true`` is well-formed on any row; tables with none are valid."""
    return


def _validate_run(decoded: list[DecodedTurn]) -> None:
    """A run needs at least two turns (an opening and a close)."""
    if len(decoded) < 2:
        raise DecodeError(
            "dialogue-mode script needs at least two turns (an opening and a close)"
        )


def _run_config_from_dict(raw: dict) -> RunConfig:
    """Build a :class:`RunConfig`; unknown keys are a decode error."""
    on_divergence = raw.get("on_divergence", "fail")
    if on_divergence not in ("fail", "accept"):
        raise DecodeError(
            f"run.on_divergence must be 'fail' or 'accept', got {on_divergence!r}"
        )
    unknown = set(raw) - {"on_divergence"}
    if unknown:
        raise DecodeError(f"unknown run section keys: {sorted(unknown)!r}")
    return RunConfig(on_divergence=on_divergence)


def load_script_file(path: str | Path) -> DialogueScript:
    """Load a :class:`DialogueScript` from a JSON file (used by the sequencer)."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise DecodeError(f"dialogue script {str(path)!r} could not be read: {exc}") from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DecodeError(f"dialogue script {str(path)!r} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise DecodeError(f"dialogue script {str(path)!r} must be a JSON object")
    return load_script(raw)


def load_script(raw: dict) -> DialogueScript:
    """Parse a raw ``{source, turns[], run?, events?, priors?}`` dict into a
    :class:`DialogueScript` (one run). ``priors`` is a path list (a run
    sequence), not merged into ``turns``."""
    if "source" not in raw or not isinstance(raw["source"], str):
        raise DecodeError("dialogue script missing string 'source'")
    # ``source`` is a path (path-like) or inline KScript.
    source = raw["source"]
    looks_like_path = ("/" in source) or ("\\" in source) or source.endswith(".ks")
    if looks_like_path:
        try:
            source = Path(source).read_text(encoding="utf-8")
        except OSError as exc:
            raise DecodeError(
                f"dialogue script 'source' path {source!r} could not be read: {exc}"
            ) from exc
    if "turns" not in raw or not isinstance(raw["turns"], list):
        raise DecodeError("dialogue script missing list 'turns'")
    run_raw = raw.get("run")
    if run_raw is not None:
        if not isinstance(run_raw, dict):
            raise DecodeError("'run' section must be an object")
        run_config = _run_config_from_dict(run_raw)
    else:
        run_config = None
    turns = tuple(_turn_from_dict(t) for t in raw["turns"])
    events_raw = raw.get("events")
    if events_raw is not None:
        if not isinstance(events_raw, list):
            raise DecodeError("'events' must be a list of turn rows")
        events = tuple(_turn_from_dict(t) for t in events_raw)
    else:
        events = ()
    priors_raw = raw.get("priors")
    if priors_raw is not None:
        if not isinstance(priors_raw, list) or not all(
            isinstance(p, str) for p in priors_raw
        ):
            raise DecodeError("'priors' must be a list of script-file path strings")
        priors = tuple(priors_raw)
    else:
        priors = ()
    return DialogueScript(
        source=source, turns=turns, events=events, priors=priors, run_config=run_config,
    )
