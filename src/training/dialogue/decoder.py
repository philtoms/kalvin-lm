"""Single-stage dialogue-table decoder (spec ``@specs/dialogue-driven-training.md`` §Decoder).

This is Phase 1 of the plan. The decoder is a **pre-loop configuration stage**:
``decode(table)`` turns a :class:`DialogueTable` into a flat ordered list of
:class:`DecodedTurn`, resolving every symbolic label against ``script`` (the
single source of truth for kline structure). The training loop (Phases 2–3)
receives only the decoded list and never touches ``script`` or the symbol map.

Spec mapping
------------
- DDT-1 — :class:`DialogueTable`/``Turn`` typed structure (``script`` + ordered
  ``turns``; each turn: actor, op, signature, nodes, significance, notes).
- DDT-2 — :func:`decode` returns a flat ordered ``list[DecodedTurn]``.
- DDT-3 — single-stage: resolve the kline from ``script``, attach significance
  by band lookup, pass through ``actor``/``op``, ignore ``notes``.
- DDT-5 — canons retrieved by **node-list match** against compiled canons.
- DDT-28 — annotation-only turns (notes, no structural fields) are dropped at
  decode time (not submittable).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from kalvin.expand import SIG_S1, SIG_S2, SIG_S3, SIG_S4
from kalvin.kline import KLine
from kalvin.kvalue import KValue
from ks.compiler import compile_source

if TYPE_CHECKING:
    from kalvin.abstract import KSignifier
    from kalvin.nlp_tokenizer import NLPTokenizer

# ── Significance band lookup ──────────────────────────────────────────────
#
# Per spec §Decoder step 2: significance is attached by band lookup ("S1"→
# SIG_S1, …) from the turn. It is independent of ``op`` (DDT-3: op and
# significance are separate axes; they diverge in most turns and neither
# reconstructs the other). These are the same uint64 inverted-distance
# constants the compiler attaches to compiled entries (see ``kalvin.expand``).
BAND_TO_SIG: dict[str, int] = {
    "S1": SIG_S1,
    "S2": SIG_S2,
    "S3": SIG_S3,
    "S4": SIG_S4,
}

Role = Literal["T", "K"]  # a turn's role: trainer (T) or trainee (K)

# The runner's divergence policy (spec ``@specs/dialogue-runner.md``
# §Matching). Lives on :class:`RunConfig`, carried on the table's optional
# ``run`` section.
OnDivergence = Literal["fail", "accept"]

# The table's closed op vocabulary (spec §Dialogue Table). An unknown op is
# a decode error.
DIALOGUE_OPS = frozenset({"COUNTERSIGNED", "CANONIZED", "CONNOTED", "UNDERSIGNED", "IDENTITY"})


class DecodeError(Exception):
    """A turn could not be decoded: an unknown op, an unresolved symbol, an
    ambiguous/missing canon node-list match, or a band typo."""


# ── Typed table structures (DDT-1) ────────────────────────────────────────


@dataclass(frozen=True)
class Turn:
    """One row of the dialogue table (spec §Dialogue Table).

    Structural turns (those carrying ``op``) decode to a :class:`DecodedTurn`.
    Annotation-only turns (``notes`` but no ``op``) are dropped at decode time
    (DDT-28) — they are not submittable klines.

    ``close`` marks a turn as a script close (the table's explicit boundary
    marker). The runner reads it to know when a script ends (and, for
    multi-script, when to open the next). A turn without ``close`` is not a
    close. Closing is a runner concern and is role-agnostic — it may sit on
    either a T or a K row. See :class:`DecodedTurn.close`.
    """

    role: Role
    op: str | None  # None on annotation-only turns (DDT-28)
    signature: str | None
    nodes: tuple[str, ...]
    significance: str | None  # None on annotation-only turns
    notes: str = ""
    close: bool = False  # True when this turn closes a script (a boundary marker)
    # The raw JSON record this turn was loaded from (``None`` for turns built
    # synthetically, e.g. test fixtures). Carried for diagnostics so a trace
    # can show the table row verbatim alongside its decoded form.
    record: dict | None = None

    @property
    def is_annotation_only(self) -> bool:
        """An annotation-only turn carries ``notes`` but no structural fields.

        Such turns are dropped at decode (DDT-28): they are commentary, not
        submittable klines. A turn is annotation-only when it has no ``op``.
        """
        return self.op is None


@dataclass(frozen=True)
class RunConfig:
    """Runner configuration (spec ``@specs/dialogue-runner.md`` §The Table).

    Carried on a ``DialogueTable``'s optional ``run`` section. All run
    modifiers live in this one block so future run knobs extend it without
    touching the rest of the table.

    - ``on_divergence`` — the runner's divergence policy (default
      ``"fail"``).
    """

    on_divergence: OnDivergence = "fail"


@dataclass(frozen=True)
class DialogueTable:
    """The source artifact for a lesson (spec §Dialogue Table).

    ``script`` is the single source of truth for kline structure (canonical
    signatures, atom values, subword composition). ``turns`` is the exact T/K
    exchange — prescriptive, not predictive.

    ``run_config`` carries the runner modifiers (spec
    ``@specs/dialogue-runner.md`` §The Table): ``None`` (default, no ``run``
    section) means the defaults apply; a :class:`RunConfig` overrides them.
    The section's presence is **not** a regime selector — dialogue mode is the
    only run regime — it is purely the modifiers container. The runner consumes
    :class:`DecodedTurn`s, not the raw table.
    """

    script: str
    turns: tuple[Turn, ...]
    run_config: RunConfig | None = None

    @property
    def has_run_config(self) -> bool:
        """True when this table carries a ``run`` section (run modifiers present)."""
        return self.run_config is not None


@dataclass(frozen=True)
class DecodedTurn:
    """A turn resolved to a submittable structure (spec §Decoded Turn).

    ``role`` and ``op`` are carried alongside the KValue and **never folded
    into it** (DDT-3): ``op`` is the structural state; ``significance`` (on the
    KValue) is the dialogic stance. They are independent axes.

    ``close`` is carried through from :class:`Turn.close`: when ``True``, this
    turn is a script close (the table's explicit boundary marker). The runner
    — which owns the table cursor — reads it as the script-boundary signal;
    the trainer does not detect closes (it is told, via routing, that a script
    has ended).
    """

    role: Role
    op: str
    value: KValue
    close: bool = False  # True when this turn closes a script (a boundary marker)
    # The raw JSON record this decoded turn was loaded from (``None`` for
    # turns built synthetically, e.g. test fixtures or runner placeholders
    # reconstructed from a content key). Carried for diagnostics so a trace can
    # show the table row verbatim alongside its decoded form.
    record: dict | None = None


# backwards-compat alias for the documented `DECODEDTurn` spelling used in the
# spec text ("a flat ordered list of `DecodedTurn`"). Both names refer to the
# same class.
DECODEDTurn = DecodedTurn


# ── Symbol resolution against compiled script ─────────────────────────────


@dataclass(frozen=True)
class _ResolvedScript:
    """Compiled-script indices built once at decode time.

    - ``canon_by_label`` — canonical signature per label: a relation node whose
      label names a canon (e.g. ``Det``) resolves to the canon's signature, not
      the atom identity that shares the label. This keeps relation construction
      faithful to the compiled structure (the compiler binds ``A > Det`` to the
      Det canon signature).
    - ``relation_by_label`` — the compiled relation (COUNTERSIGNED/CONNOTED/
      UNDERSIGNED) per label, so a constructed-relation turn carries the
      relation's structural ``dbg.op`` (e.g. ``COUNTERSIGNED``), not the canon's
      (``CANONIZED``) when the signature label names both.
    - ``labels`` — every resolvable label (a compound/atom ``dbg.label`` or a
      subword atom ``dbg.decoded``), mapped to its compiled KLine. Used to
      resolve atom signatures (IDENTITY) and as the label fallback.
    """

    canon_by_label: dict[str, KLine] = field(default_factory=dict)
    relation_by_label: dict[str, KLine] = field(default_factory=dict)
    labels: dict[str, KLine] = field(default_factory=dict)


def _resolve_script(
    script: str,
    *,
    tokenizer: NLPTokenizer | None = None,
    signifier: KSignifier | None = None,
) -> tuple[list[KValue], _ResolvedScript]:
    """Compile ``script`` once and build the canon + label indices.

    Compilation reuses :func:`ks.compiler.compile_source` (plan task 1.2). The
    indices are:
      - ``canon_by_label`` — canon label → its canonical KLine signature.
      - ``relation_by_label`` — relation label → its compiled KLine.
      - ``labels`` — display label → KLine, for atom/compound resolution.
    """
    entries = compile_source(script, tokenizer=tokenizer, signifier=signifier, dev=True)

    resolved = _ResolvedScript()
    for e in entries:
        kl = e.kline
        d = kl.dbg
        # Compiled entries from ``compile_source(..., dev=True)`` always carry
        # ``dbg`` (op/label/decoded provenance). An entry without it is a
        # compiler invariant violation, not a decode error.
        if d is None:
            raise DecodeError(
                f"compiled entry 0x{kl.signature:x} has no debug info "
                "— dev compile invariant broken"
            )
        # Canon-by-label: the canonical signature for this label. Populated for
        # any compiled canon (COUNTERSIGNED canons like MHALL, and CANONIZED
        # canons like Det) so both node and signature resolution can prefer the
        # canon when a label names one.
        if d.op in ("CANONIZED", "COUNTERSIGNED") and kl.nodes and d.label:
            resolved.canon_by_label.setdefault(d.label, kl)
        if d.op in ("COUNTERSIGNED", "CONNOTED", "UNDERSIGNED") and d.label:
            # Relation-by-label: carries the relation's structural dbg.op, so a
            # constructed-relation turn reports e.g. COUNTERSIGNED, not the
            # CANONIZED of a canon that shares the label.
            resolved.relation_by_label.setdefault(d.label, kl)
        # Label index: compound/atom dbg.label, and subword atom dbg.decoded.
        if d.label:
            resolved.labels.setdefault(d.label, kl)
        if d.decoded:
            resolved.labels.setdefault(d.decoded, kl)
    return entries, resolved


def primaries_from_source(
    script: str,
    *,
    tokenizer: NLPTokenizer | None = None,
    signifier: KSignifier | None = None,
) -> list[KLine]:
    """The ordered script primaries (one per top-level KScript scope).

    A multi-script file (e.g. ``MHALL`` then ``WDMH``) has one primary per
    top-level ``OperatorScope``, in source order. Each primary is the first
    compiled entry whose ``dbg.label`` matches the scope's signature id — the
    kline a trainer opens (R1) for that script. Used by a multi-script trainer
    to open successive scripts after each close.

    The AST is the principled source of script boundaries (a compiled list
    alone cannot distinguish a primary from any other labelled entry); this
    helper keeps that AST knowledge in the decoder, next to the other
    compile-time indexing.
    """
    from ks.lexer import Lexer
    from ks.parser import OperatorScope, Parser

    kfile = Parser(Lexer(script).tokenize()).parse()
    scope_labels = [
        c.sig.id for c in kfile.constructs if isinstance(c, OperatorScope)
    ]
    if not scope_labels:
        return []
    entries = compile_source(script, tokenizer=tokenizer, signifier=signifier, dev=True)
    first_by_label: dict[str, KLine] = {}
    for e in entries:
        lbl = e.kline.dbg.label if e.kline.dbg else None
        if lbl and lbl not in first_by_label:
            first_by_label[lbl] = e.kline
    primaries: list[KLine] = []
    for label in scope_labels:
        kl = first_by_label.get(label)
        if kl is None:
            raise DecodeError(
                f"script primary {label!r} has no compiled entry — cannot resolve"
            )
        primaries.append(kl)
    return primaries


# ── The single-stage decode (DDT-2, DDT-3, DDT-28) ────────────────────────


def _resolve_node_signatures(
    nodes: tuple[str, ...],
    resolved: _ResolvedScript,
    *,
    op: str,
) -> list[int]:
    """Resolve each node label to its canonical signature.

    Shared by the CANONIZED and constructed-relation branches. A label that
    names a canon resolves to the canon's signature (so e.g. ``Det`` resolves to
    the Det canon signature, matching the compiler's binding); a pure-atom label
    (no canon) falls back to the label index.
    """
    node_sigs: list[int] = []
    for n in nodes:
        ncanon = resolved.canon_by_label.get(n)
        if ncanon is not None:
            node_sigs.append(ncanon.signature)
            continue
        nkl = resolved.labels.get(n)
        if nkl is None:
            raise DecodeError(
                f"{op} node {n!r}: label not found in compiled script"
            )
        node_sigs.append(nkl.signature)
    return node_sigs


def _resolve_kline(
    op: str,
    signature: str,
    nodes: tuple[str, ...],
    resolved: _ResolvedScript,
) -> KLine:
    """Resolve a turn's symbolic ``(op, signature, nodes)`` to a KLine.

    Three branches per spec §Decoder step 1. **The decoder is a resolver, not
    a gatekeeper**: it builds the kline the turn declares — the declared
    ``signature`` verbatim, the ``nodes`` resolved to their canonical
    signatures — and never second-guesses whether the signature "matches" the
    nodes. The script author is the golden master; an author may declare a
    signature that differs from the canon its nodes retrieve (a deliberate
    misfit — see ``scripts/dialogue-rationalisation-behaviours.md``).

    - **CANONIZED** — resolve each node label to its canonical signature
      (canon-preferred, atom fallback) and build ``KLine(signature, nodes)``
      with the declared signature verbatim. No canon retrieval by node-list,
      no signature-consistency check.
    - **IDENTITY** — resolve the atom by label, preferring the canon when the
      label names one (a label may name both a canon and its atoms; the
      identity names the concept, i.e. the canon).
    - **Constructed relation** (``CONNOTED`` / ``UNDERSIGNED`` / ``COUNTERSIGNED``)
      — resolve each node label to its canonical signature and reconstruct the
      relation KLine with the compiler's signifier (``make_signature``). A node
      label may be a compound (e.g. ``subject``) or an atom; both resolve via
      the label index.
    """
    if op == "CANONIZED":
        # Resolve each node label to its canonical signature, then build the
        # kline with the declared signature verbatim. The decoder does not
        # retrieve a canon by node-list and does not check that the declared
        # signature matches the canon those nodes would form: an author may
        # declare a deliberate misfit (e.g. a K-generated leap whose signature
        # is the query and whose nodes are the answer's atoms).
        node_sigs = _resolve_node_signatures(nodes, resolved, op="CANONIZED")
        # Carry the signature label's compiled ``dbg`` (op/label provenance) when
        # the label resolves; the turn's ``op`` is the dialogue vocabulary and
        # stays on the DecodedTurn.
        sig_kl = resolved.canon_by_label.get(signature)
        if sig_kl is None:
            sig_kl = resolved.labels.get(signature)
        if sig_kl is None:
            raise DecodeError(
                f"CANONIZED signature {signature!r}: label not found in compiled script"
            )
        return KLine(sig_kl.signature, node_sigs, dbg=sig_kl.dbg)

    if op == "IDENTITY":
        # Prefer the canon when the label names one. A KScript label may name
        # both a canon and its atoms (e.g. ``Det`` names the Det canon AND the
        # atoms D, et, which all carry dbg.label=='Det'); the atoms are compiled
        # before the canon, so a bare ``labels[label]`` lookup would resolve
        # to the first atom rather than the concept. An IDENTITY turn names the
        # *concept* (e.g. K asking "what is Det?" means Det the canon), so the
        # canon signature is correct; a pure-atom label (no canon) falls back
        # to the label index.
        kl = resolved.canon_by_label.get(signature)
        if kl is None:
            kl = resolved.labels.get(signature)
        if kl is None:
            raise DecodeError(
                f"IDENTITY {signature!r}: label not found in compiled script"
            )
        # An atom label may resolve to a multi-node canon (e.g. a subword word);
        # the identity is the bare signature with empty nodes. The compiled
        # entry's ``dbg`` (op/label/decoded provenance) is carried through so the
        # loop's diagnostics and the supply rule can read structure off it.
        return KLine(kl.signature, [], dbg=kl.dbg)

    # Constructed relation: resolve each node label to its canonical signature
    # (preferring the canon when the label names one — e.g. ``Det`` resolves to
    # the Det canon signature, matching the compiler's binding), then rebuild the
    # relation KLine. Atom labels (no canon) fall back to the label index.
    node_sigs = _resolve_node_signatures(nodes, resolved, op="relation")
    sig_kl = resolved.relation_by_label.get(signature)
    if sig_kl is None:
        # No compiled relation for this label: fall back to the canon signature
        # (same value) or the label index. The dbg may then be CANONIZED, which
        # is acceptable — a relation whose script declares no standalone relation
        # entry is constructed fresh.
        sig_canon = resolved.canon_by_label.get(signature)
        sig_kl = sig_canon if sig_canon is not None else resolved.labels.get(signature)
    if sig_kl is None:
        raise DecodeError(
            f"relation signature {signature!r}: label not found in compiled script"
        )
    # Carry the signature label's compiled ``dbg`` (its op/label provenance); the
    # turn's ``op`` is the dialogue vocabulary and stays on the DecodedTurn.
    return KLine(sig_kl.signature, node_sigs, dbg=sig_kl.dbg)


def decode(
    table: DialogueTable,
    *,
    tokenizer: NLPTokenizer | None = None,
    signifier: KSignifier | None = None,
) -> list[DecodedTurn]:
    """Pre-decode every turn into a flat ordered ``list[DecodedTurn]`` (DDT-2).

    Single-stage per turn (DDT-3): resolve the kline from ``script``, attach
    significance by band lookup, pass through ``actor``/``op``, ignore
    ``notes``. Annotation-only turns are dropped (DDT-28).

    This is a configuration-time function: call once, hand the result to the
    training loop. The loop never touches ``script`` again.
    """
    _entries, resolved = _resolve_script(
        table.script, tokenizer=tokenizer, signifier=signifier
    )

    out: list[DecodedTurn] = []
    for idx, turn in enumerate(table.turns):
        # DDT-28: annotation-only turns (notes, no op) are dropped at decode.
        if turn.is_annotation_only:
            continue
        assert turn.op is not None and turn.significance is not None  # annotation guard
        # A structural turn (op set) must carry ``signature`` — enforced at
        # load time (``_turn_from_dict``), so this is a defensive decode check.
        if turn.signature is None:
            raise DecodeError(f"turn {idx}: structural turn missing 'signature'")
        if turn.op not in DIALOGUE_OPS:
            raise DecodeError(f"turn {idx}: unknown op {turn.op!r}")
        if turn.significance not in BAND_TO_SIG:
            raise DecodeError(
                f"turn {idx}: unknown significance {turn.significance!r}"
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
    # ``close`` markers are validated on the decoded list (they are a property
    # of the exchange, but the per-turn shape check happens here alongside the
    # other decoded-list invariants).
    _validate_close(out)
    # Dialogue-mode invariants are checked on the decoded list (spec
    # @specs/dialogue-runner.md §Invariants). They require symbol resolution to
    # have run, so they live here, not in the loader.
    if table.has_run_config:
        _validate_run(out)
    return out


# ── Loader (DDT-1) ────────────────────────────────────────────────────────


def _turn_from_dict(raw: dict) -> Turn:
    """Build a :class:`Turn` from a raw JSON turn dict (DDT-1).

    Annotation-only turns carry ``notes`` but no ``op``; their ``signature``,
    ``nodes`` and ``significance`` default to absent (DDT-28). A turn with an
    ``op`` must also carry ``signature`` and ``significance``; a structural
    turn missing them is a malformed-table decode error.
    """
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
            f"'close' must be the boolean true (a script-boundary marker), got {close!r}"
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


# ── Dialogue-run decode-time validations (spec @specs/dialogue-runner.md §Invariants) ─


def turn_content_key(turn: DecodedTurn) -> tuple[str, int, tuple[int, ...], int]:
    """The content identity of a decoded turn (spec @specs/dialogue-runner.md
    §Matching): ``(role, kline_signature, kline_nodes_tuple, significance)``.

    Content matching keys on role + KLine + significance. The KLine is carried by
    its signature and nodes tuple (the two fields that define KLine equality,
    ``@specs/kline.md``). This helper exists because :class:`KValue.__eq__` is
    structural-only and ignores significance (``src/kalvin/kvalue.py``), so
    matching cannot use ``KValue`` equality directly — significance must be
    compared explicitly alongside the kline.
    """
    return (
        turn.role,
        turn.value.kline.signature,
        tuple(turn.value.kline.nodes),
        turn.value.significance,
    )


def _validate_close(decoded: list[DecodedTurn]) -> None:
    """Validate the ``close`` markers (spec §Dialogue Table).

    A ``close`` turn closes a script (a boundary marker). The marker is
    role-agnostic: it may sit on either a trainer (T) or trainee (K) row.
    Closing is a runner concern (the runner owns the table cursor and reads
    the marker as the script boundary); it is not tied to a role.

    Tables with no ``close`` markers are valid (single-script, backward
    compatible): the runner ends at the last row and no script boundary fires.
    """
    # Presence-only: a `close: true` is well-formed on any row. No role check —
    # the runner, not the decoder, decides how a close routes.
    return


def _validate_run(decoded: list[DecodedTurn]) -> None:
    """Validate the dialogue-mode invariants (spec @specs/dialogue-runner.md §Invariants).

    - The opening (``decoded[0]``) is a trainer (``T``) row.
    - The closing (``decoded[-1]``) is content-distinct from the opening **and**
      from every middle row: its ``(role, kline, significance)`` content key
      appears exactly once in the table. A closing that duplicates the opening
      would make coverage degenerate (the opening satisfies it); a closing that
      duplicates a middle row would make the positional "consumed last"
      semantics ambiguous (which occurrence is the closing?).
    """
    if len(decoded) < 2:
        raise DecodeError(
            "dialogue-mode table needs at least an opening and a closing turn"
        )
    if decoded[0].role != "T":
        raise DecodeError(
            f"dialogue-mode opening must be a trainer (T) row, got role {decoded[0].role!r}"
        )
    closing_key = turn_content_key(decoded[-1])
    middle_keys = {turn_content_key(t) for t in decoded[1:-1]}
    opening_key = turn_content_key(decoded[0])
    if closing_key == opening_key:
        raise DecodeError(
            "dialogue-mode opening and closing are content-equal "
            "(same role, kline, significance) — malformed table"
        )
    if closing_key in middle_keys:
        raise DecodeError(
            "dialogue-mode closing content also appears as a middle row "
            "— the closing must be a unique terminal content"
        )


def _run_config_from_dict(raw: dict) -> RunConfig:
    """Build a :class:`RunConfig` from a raw ``run`` section dict.

    Only the known modifier ``on_divergence`` (default ``"fail"``) is read; an
    unknown key is a decode error so future run knobs are added deliberately,
    not silently ignored.
    """
    on_divergence = raw.get("on_divergence", "fail")
    if on_divergence not in ("fail", "accept"):
        raise DecodeError(
            f"run.on_divergence must be 'fail' or 'accept', got {on_divergence!r}"
        )
    unknown = set(raw) - {"on_divergence"}
    if unknown:
        raise DecodeError(f"unknown run section keys: {sorted(unknown)!r}")
    return RunConfig(on_divergence=on_divergence)


def _load_table_file(path: Path) -> DialogueTable:
    """Load a :class:`DialogueTable` from a JSON file (used for ``priors``).

    The path is resolved against the file system (the cwd, like the ``script``
    field). A missing or unreadable file is a hard error. The loaded table's
    own ``priors`` (if any) are resolved recursively by :func:`load_table`, so
    a chain of priors composes; cycles would surface as an ``OSError`` (the
    second load of an already-open path) rather than a silent loop.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DecodeError(
            f"dialogue prior table {str(path)!r} could not be read: {exc}"
        ) from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DecodeError(
            f"dialogue prior table {str(path)!r} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise DecodeError(
            f"dialogue prior table {str(path)!r} must be a JSON object"
        )
    return load_table(raw)


def load_table(raw: dict) -> DialogueTable:
    """Parse a raw ``{script, turns[], run?}`` dict into a :class:`DialogueTable`.

    ``notes`` are carried on each :class:`Turn` (the decoder ignores them) but
    the structural fields are validated for shape here; symbol resolution
    happens later in :func:`decode`. A ``run`` section (optional) carries the
    runner modifiers (spec ``@specs/dialogue-runner.md`` §The Table); there
    is no top-level ``mode`` field — dialogue mode is the only run regime. Unknown
    keys inside ``run`` are rejected.
    """
    if "script" not in raw or not isinstance(raw["script"], str):
        raise DecodeError("dialogue table missing string 'script'")
    # ``script`` may be a path to a KScript file or inline source. The two are
    # disambiguated by **path-likeness**, not by existence: a value is treated
    # as a path when it contains a path separator (``/`` or ``\``) or ends in
    # ``.ks``. A path-like value is resolved against the file system — a missing
    # or unreadable file is a hard error (no silent inline fallback, which would
    # mask a typo or a wrong working directory). A non-path-like value is used
    # verbatim as inline KScript (the original behaviour).
    script = raw["script"]
    looks_like_path = ("/" in script) or ("\\" in script) or script.endswith(".ks")
    if looks_like_path:
        script_path = Path(script)
        try:
            script = script_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise DecodeError(
                f"dialogue table 'script' path {script!r} could not be read: {exc}"
            ) from exc
    if "turns" not in raw or not isinstance(raw["turns"], list):
        raise DecodeError("dialogue table missing list 'turns'")
    run_raw = raw.get("run")
    if run_raw is not None:
        if not isinstance(run_raw, dict):
            raise DecodeError("'run' section must be an object")
        run_config = _run_config_from_dict(run_raw)
    else:
        run_config = None
    turns = tuple(_turn_from_dict(t) for t in raw["turns"])
    # ``priors`` (optional) names other dialogue-table files whose turns run
    # before this table's own, in list order (spec §Dialogue Table). Each is
    # loaded recursively; its turns are prepended so a multi-file lesson reads
    # as one continuous exchange. A prior resolves its own ``script`` (its
    # turns' symbols must be resolvable there); only the turns are carried in.
    priors_raw = raw.get("priors")
    if priors_raw is not None:
        if not isinstance(priors_raw, list) or not all(
            isinstance(p, str) for p in priors_raw
        ):
            raise DecodeError("'priors' must be a list of table-file path strings")
        prior_turns: list[Turn] = []
        for prior_path in priors_raw:
            prior_table = _load_table_file(Path(prior_path))
            prior_turns.extend(prior_table.turns)
        turns = tuple(prior_turns) + turns
    return DialogueTable(script=script, turns=turns, run_config=run_config)
