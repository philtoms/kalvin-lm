"""Expand — graph expansion, significance computation, classification, and
expansion proposal pipeline.

This module owns the full significance → classification → expansion proposal
pipeline:

  1. **Significance computation** — expand() composes an 8-bit grade per
     query|candidate pair and yields QueryCandidate objects.
  2. **Band classification** — BandLayout maps bytes to S1/S2/S3/S4 bands.
     Only S2_S3_BOUNDARY is configurable.
  3. **Expansion proposals** — propose_expansions() classifies misfits and
     generates (proposal, significance) tuples for the caller to dispatch.
  4. **Structural grounding** — is_s1(), is_countersigned() verify S1 status;
     structural_significance() derives a stored KLine's band from its
     structure alone (the model-state S2→S1 countersigned fork is applied at
     the call site, e.g. KAgent).

The module reads from the Model (storage) but is a separate responsibility:
Model indexes and retrieves; Expand computes how far apart two KLines are.

Module-level constants and types:
  MAX_HOP, QueryCandidate,
  SIG_MASK, SIG8_MAX, SIG8_MIN, DEFAULT_S2_S3_BOUNDARY,
  SIG_S1, SIG_S2, SIG_S3, SIG_S4 (band-representative sentinels),
  BandLayout, distance_to_byte, Aggregator, DEFAULT_AGGREGATOR

Producer significance:
  band_significance — op → band-representative integer (KP-1)
  structural_significance — derive a KLine's structural band
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from kalvin.kline import KLine, is_canon, is_terminal, is_unknown
from kalvin.misfit import generate_expansions

if TYPE_CHECKING:
    from kalvin.abstract import KSignifier
    from kalvin.model import Model

_log = logging.getLogger(__name__)

# Upper bound on edge hop chain depth (edge_hops's traversal bound).
MAX_HOP = 100

# ─────────────────────────────────────────────────────────────────────
# 8-bit compositional significance.
#
# Significance occupies the LOW 8 BITS of an int; access is always via
# masking: ``sig & SIG_MASK``. It is a global linear inverted distance in
# ``(0x00, 0xFF)``: higher byte = closer match, no reshape at the S2|S3
# boundary. Saturation guards: ``0xFF`` is reachable ONLY by exact match
# (distance 0); ``0x00`` is reachable ONLY by a structural unresolvable;
# the interior ``(0x01..0xFE)`` is the open band of graded distance.
#
# Only ``S2_S3_BOUNDARY`` is configurable; S1|S2 and S3|S4 are fixed
# sentinels.
# ─────────────────────────────────────────────────────────────────────

#: Low-byte mask isolating the 8-bit significance.
SIG_MASK: int = 0xFF

#: The S1 sentinel / exact-match byte. Reachable only at distance 0.
SIG8_MAX: int = 0xFF

#: The S4 sentinel / structural-unresolvable byte. A computed (resolvable)
#: distance never yields this — only a structural unresolvable path does,
#: which does not go through ``distance_to_byte``.
SIG8_MIN: int = 0x00

#: Default for the one configurable boundary. S2 = [0x80, 0xFE];
#: S3 = [0x01, 0x7F]. Must be in [0x02, 0xFE] so both interior bands are
#: non-empty (see ``BandLayout.validate``).
DEFAULT_S2_S3_BOUNDARY: int = 0x80

#: Distance at which ``distance_to_byte`` floors at 0x01. Distances >= this
#: saturate to 0x01, never 0x00.
_MAX_INTERIOR_DISTANCE: int = 0xFE

# Band-representative significance values — the canonical bytes a producer
# stamps when asserting a band rather than computing a grade (the compiler,
# the countersign reciprocal, structural_significance). Single source of
# truth per @model spec §Band-representative Values. Computed values from
# expand() may be any byte within a band, not only the representative.
#
# Fixed (not derived from a BandLayout): structural significance marks *which
# band a structure claims*, independent of where the configurable
# S2_S3_BOUNDARY is drawn for computed grades. BandLayout exposes matching
# layout-derived representatives for classification.
SIG_S1 = 0xFF  # exact match  (== SIG8_MAX; the S1|S2 boundary)
SIG_S2 = 0xFE  # top of S2
SIG_S3 = 0x7F  # top of S3 at the default boundary (DEFAULT_S2_S3_BOUNDARY - 1)
SIG_S4 = 0x00  # the S4 sentinel  (== SIG8_MIN; structural unresolvable)

# Compile-time structural relationship (@CONTEXT.md §Structural Relationship) → band-
# representative significance. Producers that assert a band rather than compute
# a distance (the compiler, per @kvalue spec KP-1) look up here. CONNOTES and
# DENOTES both map to SIG_S3; unknown ops default to SIG_S4.
_OP_TO_SIG: dict[str, int] = {
    "COUNTERSIGNS": SIG_S1,
    "CANONIZES": SIG_S2,
    "CONNOTES": SIG_S3,
    "DENOTES": SIG_S3,
    "UNKNOWN": SIG_S4,
}


def band_significance(op: str) -> int:
    """Compile-time structural relationship → band-representative significance.

    Maps the closed set of structural relationships (@CONTEXT.md §Structural Relationship)
    to the maximal significance of each band. Used by producers that assert a
    band rather than compute a distance (the compiler, per @kvalue spec KP-1).
    Unknown ops default to ``SIG_S4``.
    """
    return _OP_TO_SIG.get(op, SIG_S4)


# ── Byte conversion (linear inverted distance) ──────────────────────


def distance_to_byte(distance: int) -> int:
    """Linear inverted ``distance`` → byte in ``[0x01, 0xFF]``.

    - ``distance == 0``        → ``0xFF`` (exact match — the only path to 0xFF)
    - ``1 <= distance <= 0xFE`` → ``0xFE..0x01`` (linear, monotone decreasing)
    - ``distance >= 0xFE``      → ``0x01`` (floors; never reaches 0x00)

    This function never emits ``0x00``: a *computed* distance is by definition
    resolvable, and only a structural unresolvable yields the ``SIG8_MIN``
    sentinel — that path does not go through here.
    """
    if distance < 0:
        raise ValueError(f"distance must be non-negative; got {distance}")
    if distance == 0:
        return SIG8_MAX
    if distance >= _MAX_INTERIOR_DISTANCE:
        return 0x01
    # Linear: distance 1 → 0xFE, ..., distance 0xFD → 0x02, distance 0xFE → 0x01.
    return SIG8_MAX - distance  # in [0x01, 0xFE] for distance in [1, 0xFE]


class BandLayout:
    """The four bands over the linear byte, derived from one boundary.

    Only ``s2_s3_boundary`` is configurable; S1|S2 and S3|S4 are fixed
    sentinels exposed as named constants.

        S1 = [0xFF]                       (only exact match — distance 0)
        S2 = [s2_s3_boundary, 0xFE]       (close, direct)
        S3 = [0x01, s2_s3_boundary - 1]    (indirect, decayed)
        S4 = [0x00]                       (only structural unresolvable)

    Band-representative values (the canonical bytes a producer stamps when
    it asserts a band rather than computes a distance)::

        sig_s1 = 0xFF
        sig_s2 = 0xFE                  (the top of S2)
        sig_s3 = s2_s3_boundary - 1     (the top of S3)
        sig_s4 = 0x00
    """

    __slots__ = ("s2_s3_boundary",)

    def __init__(self, s2_s3_boundary: int = DEFAULT_S2_S3_BOUNDARY) -> None:
        self.validate_boundary(s2_s3_boundary)
        self.s2_s3_boundary = s2_s3_boundary

    @staticmethod
    def validate_boundary(s2_s3_boundary: int) -> None:
        """Interior guard: the boundary must split the open band cleanly.

        ``[0x02, 0xFE]`` leaves room for non-empty S2 ``[b, 0xFE]`` and
        non-empty S3 ``[0x01, b-1]``.
        """
        if not isinstance(s2_s3_boundary, int) or not (
            0x02 <= s2_s3_boundary <= 0xFE
        ):
            raise ValueError(
                f"S2_S3_BOUNDARY must be an int in [0x02, 0xFE] to leave room "
                f"for non-empty S2 and S3 bands; got {s2_s3_boundary!r}"
            )

    # Fixed sentinels, as lowercase properties (ruff N802). The module-level
    # SIG_S1/SIG_S4 constants are uppercase (constants, not functions); these
    # are the layout-internal accessors.
    @property
    def sig_s1(self) -> int:
        return 0xFF

    @property
    def sig_s4(self) -> int:
        return 0x00

    # Boundary-derived representatives.
    @property
    def sig_s2(self) -> int:
        return 0xFE  # top of S2

    @property
    def sig_s3(self) -> int:
        return self.s2_s3_boundary - 1  # top of S3

    def classify(self, sig: int) -> str:
        """Classify a raw byte into S1/S2/S3/S4. Operates on the low byte only."""
        b = sig & SIG_MASK
        if b == self.sig_s1:
            return "S1"
        if b >= self.s2_s3_boundary:
            return "S2"
        if b >= 0x01:
            return "S3"
        return "S4"


# ── Pluggable seams for compositional significance ───────────────────
#
# A compositional significance has two functions:
#   1. DecayFunction  — leaf decay: reentrant hop count -> accountedness.
#   2. ComposeFunction — how per-node accountedness values combine.
# Both are Protocols with default implementations; expand() is parameterised
# over them via Aggregator.


@runtime_checkable
class DecayFunction(Protocol):
    """Leaf decay: reentrant hop count -> accountedness contribution in [0, 1].

    The caller (expand) supplies the boundary cases directly:
      - matched AND grounded     -> 1.0  (decay is never called)
      - matched but ungrounded   -> decay(1)  (one hop of doubt)
      - resolvable in h hops      -> decay(h)
      - unresolvable              -> 0.0  (decay is never called)
    """

    def __call__(self, hops: int) -> float: ...


@runtime_checkable
class ComposeFunction(Protocol):
    """Composition: per-slot accountedness values -> aggregate in [0, 1].

    The result is the accounted fraction consumed by the byte encoder.
    Implementations should be count-invariant (scaling the same accountedness
    distribution leaves the byte unchanged) unless they deliberately trade
    that property away.
    """

    def __call__(self, slot_values: Sequence[float]) -> float: ...


# ── Default decay functions ──────────────────────────────────────────


def asymptotic_decay(hops: int, k: int = 50) -> float:
    """The default decay curve: ``k / (k + hops)``.

    hops 0 -> 1.0, monotone decreasing, asymptotes to 0 as hops -> inf.
    Larger ``k`` decays more slowly (more tolerance for deep resolution).
    """
    if hops < 0:
        raise ValueError(f"hops must be non-negative; got {hops}")
    return k / (k + hops)


def harmonic_decay(hops: int) -> float:
    """``1 / (1 + hops)``. A standard harmonic decay; slower than small-k asymptote."""
    if hops < 0:
        raise ValueError(f"hops must be non-negative; got {hops}")
    return 1.0 / (1.0 + hops)


def linear_decay(hops: int, reach: int = 100) -> float:
    """Linear decay to 0 at ``hops == reach``; clamps at 0 beyond.

    ``reach`` is the hop count at which a node is considered fully unaccounted.
    """
    if hops < 0:
        raise ValueError(f"hops must be non-negative; got {hops}")
    if reach <= 0:
        raise ValueError(f"reach must be positive; got {reach}")
    return max(0.0, 1.0 - hops / reach)


def make_asymptotic_decay(k: int = 50) -> DecayFunction:
    """Curry ``k`` into an asymptotic_decay callable."""

    def _decay(hops: int) -> float:
        return asymptotic_decay(hops, k=k)

    return _decay


# ── Default compose functions ────────────────────────────────────────


def mean_compose(slot_values: Sequence[float]) -> float:
    """Default composition: arithmetic mean of per-slot accountedness.

    Count-invariant by construction: scaling the same accountedness
    distribution (repeating it) leaves the mean — and thus the byte —
    unchanged.
    """
    n = len(slot_values)
    if n == 0:
        # No slots — vacuously unaccounted. Callers only invoke compose when
        # there is at least one slot; this guard is defensive.
        return 0.0
    return sum(slot_values) / n


@dataclass(frozen=True)
class Aggregator:
    """The compose-on-return policy, bundling layout + the two seams.

    ``expand()`` is parameterised over a single ``Aggregator`` (keyword-only,
    defaulting to :data:`DEFAULT_AGGREGATOR`) so the call site stays readable
    while the recursion threads one object. Freezing makes it safe to share
    across recursive calls and threads.
    """

    layout: BandLayout = field(default_factory=BandLayout)
    decay: DecayFunction = field(default_factory=make_asymptotic_decay)
    compose: ComposeFunction = field(default=mean_compose)

    def compose_terminal(self, slot_values: list[float]) -> int:
        """Compose per-slot accountedness into the terminal significance byte.

        Maps the compose() of per-slot accountedness through the linear
        inverted-distance byte, with the two saturation guards:

          accounted_fraction == 1.0 -> 0xFF  (exact match only)
          accounted_fraction == 0.0 -> 0x00  (only total non-account)
          otherwise                 -> byte in (0x00, 0xFF)
        """
        frac = max(0.0, min(1.0, self.compose(slot_values)))  # clamp float noise
        if frac >= 1.0:
            return SIG8_MAX
        if frac <= 0.0:
            return SIG8_MIN
        # Map the unaccounted fraction (1 - frac) to an interior distance in
        # [1, 0xFE] so every resolvable pair lands strictly inside (0x00, 0xFF).
        interior_distance = 1 + round((1.0 - frac) * (_MAX_INTERIOR_DISTANCE - 1))
        return distance_to_byte(interior_distance)


#: Module-level default aggregator: default layout, asymptotic decay (k=50),
#: mean compose. cogitator uses this unless constructed otherwise.
DEFAULT_AGGREGATOR = Aggregator()



class QueryCandidate:
    """A single query|candidate pair yielded by graph expansion.

    Replaces the NamedTuple from model.py with a class for forward
    compatibility. Still usable as a tuple: (query, candidate, significance).
    """

    __slots__ = ("query", "candidate", "significance")

    def __init__(self, query: KLine, candidate: KLine, significance: int):
        self.query = query
        self.candidate = candidate
        self.significance = significance

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QueryCandidate):
            return NotImplemented
        return (
            self.query is other.query
            and self.candidate is other.candidate
            and self.significance == other.significance
        )

    def __repr__(self) -> str:
        return f"QueryCandidate(q={self.query!r}, c={self.candidate!r}, sig={self.significance:#x})"

    def __iter__(self):
        return iter((self.query, self.candidate, self.significance))


# Helper Functions


def edge_hops(model: Model, sig: int, signifier: KSignifier) -> Iterator[tuple[int, int]]:
    """Yield (hop_count, next_sig) for each non-canonical resolution step.

    Follows: resolve sig → kline → signifier.signature_of(kline.nodes) → repeat.
    Stops at a dead end, an identity kline, a canonical kline, or a cycle.
    """
    hop_count = 0
    visited: set[int] = set()
    while hop_count < MAX_HOP:
        if sig in visited:
            break  # cycle detected
        visited.add(sig)
        kline = model.find(sig)
        if kline is None or is_terminal(kline) or is_canon(kline, signifier):
            break
        hop_count += 1
        sig = signifier.signature_of(kline.nodes)
        yield hop_count, sig

# Structural Grounding


def is_s1(model: Model, kline: KLine, signifier: KSignifier) -> bool:
    """Determine if a kline is structurally grounded (S1).

    A kline is S1 if:
    1. Its signature fully describes its nodes (canonical), OR
    2. It is countersigned by another kline in the model.
    """
    if is_canon(kline, signifier):
        return True
    return is_countersigned(model, kline, signifier)


def is_countersigned(model: Model, kline: KLine, signifier: KSignifier) -> bool:
    """Check if kline is countersigned by any kline in the model.

    A kline is countersigned if its nodes_signature exists as a
    countersigning kline with one node — the countersigned kline's signature.

    Query = {Q: [A, B]}
    Countersigner = {AB: [Q]}

    A self-referential kline ``{S: [S]}`` is excluded: its nodes_signature
    is ``S`` and it is itself a one-node kline whose node is ``S``, so it
    would otherwise count as its own countersigner.
    """
    if is_terminal(kline):
        return False
    nodes_signature = signifier.signature_of(kline.nodes)
    for countersigner in model.find_all(nodes_signature):
        if len(countersigner.nodes) == 1 and countersigner.nodes[0] == kline.signature:
            return True
    return False


def structural_significance(kline: KLine, signifier: KSignifier) -> int:
    """Derive a kline's significance band from its structure alone.

    No model state — structure is an emergent property of the kline — this
    function composes the predicates rather than re-deriving structure inline.

    Mapping (@CONTEXT.md §Structural Relationship):

    - **S1** — a grounded Identity terminal or a canon. An Identity
      (self-referential ``{A:[A]}`` or a compound-word) is self-grounded;
      the empty form ``{A:[]}`` is an Unknown (S4), not S1. A canon
      ``{AB:[A, B]}`` is a grounded aggregation.
    - **S3** — a single-node, non-terminal relationship ``{A:[B]}``
      (connotation / denotation).
    - **S2** — a multi-node misfit (underfit / overfit / misfit).
    - **S4** — the empty Unknown frame ``{A:[]}``.
    """
    if is_terminal(kline):
        return SIG_S4 if is_unknown(kline) else SIG_S1
    if len(kline.nodes) == 1:
        return SIG_S3
    if is_canon(kline, signifier):
        return SIG_S1
    return SIG_S2


# Graph Expansion


def expand(
    model: Model,
    query: KLine,
    candidate: KLine,
    signifier: KSignifier,
    *,
    aggregator: Aggregator | None = None,
    _visited: set[tuple[int, int]] | None = None,
) -> Iterator[QueryCandidate]:
    """Expand a query-candidate pair, yielding connotations and terminal byte.

    Compose-on-return aggregation: topology is captured on descent (per-node
    accountedness retained as a float), and composition is applied on the
    return phase.

    Per-node accountedness:
      matched & grounded       -> 1.0
      matched but ungrounded   -> decay(1)   (one hop of doubt)
      resolvable in h hops      -> decay(h)
      unresolvable              -> 0.0

    Yield asymmetry: exact opposing matches and S3 connotation bridges
    recurse; signifies matches emit a side-candidate and do not recurse.
    The final yield is always the terminal QueryCandidate for the original
    pair.

    ``aggregator`` bundles the layout (S2_S3_BOUNDARY) and the two pluggable
    seams (DecayFunction, ComposeFunction); defaults to
    :data:`DEFAULT_AGGREGATOR`.
    """
    if aggregator is None:
        aggregator = DEFAULT_AGGREGATOR
    if _visited is None:
        _visited = set()

    key = (query.signature, candidate.signature)
    if key in _visited:
        return  # cycle detected
    _visited.add(key)

    q_set = set(query.nodes)
    c_set = set(candidate.nodes)
    mismatched_q = q_set - c_set
    mismatched_c = c_set - q_set
    matched = q_set & c_set

    s3_connotations: dict[int, int] = {}  # sig -> min hops from any query node

    # Per-node accountedness, in slot order. One float per slot.
    slot_values: list[float] = []

    decay = aggregator.decay

    for n in mismatched_q:
        accounted = 0.0  # unresolvable default (case F)
        q_kline = model.find(n)
        if q_kline is not None:
            for hops, match_sig in edge_hops(model, n, signifier):
                if match_sig in mismatched_c:
                    # case C: exact opposing match (S2 direct) -> recurse.
                    accounted = decay(hops)
                    c_kline = model.find(match_sig)
                    if c_kline is not None:
                        yield from expand(
                            model, q_kline, c_kline, signifier,
                            aggregator=aggregator, _visited=_visited,
                        )
                    break
                elif signifier.signifies(n, match_sig):
                    # case D: signifies (S2 loose) -> side-candidate, no recurse.
                    accounted = decay(hops)
                    c_kline = model.find(match_sig)
                    if c_kline is not None:
                        sig_byte = aggregator.compose_terminal([decay(hops)])
                        yield QueryCandidate(q_kline, c_kline, sig_byte)
                    break
                elif match_sig not in s3_connotations or hops < s3_connotations[match_sig]:
                    s3_connotations[match_sig] = hops
        slot_values.append(accounted)

    for n in mismatched_c:
        accounted = 0.0
        q_kline = model.find(n)
        if q_kline is not None:
            for hops, match_sig in edge_hops(model, n, signifier):
                if match_sig in mismatched_q:
                    # case C: exact opposing match (S2 direct) -> recurse.
                    accounted = decay(hops)
                    c_kline = model.find(match_sig)
                    if c_kline is not None:
                        yield from expand(
                            model, q_kline, c_kline, signifier,
                            aggregator=aggregator, _visited=_visited,
                        )
                    break
                elif signifier.signifies(n, match_sig):
                    # case D: signifies (S2 loose) -> side-candidate, no recurse.
                    accounted = decay(hops)
                    c_kline = model.find(match_sig)
                    if c_kline is not None:
                        sig_byte = aggregator.compose_terminal([decay(hops)])
                        yield QueryCandidate(q_kline, c_kline, sig_byte)
                    break
                elif match_sig in s3_connotations:
                    # case E: S3 connotation bridge -> recurse (no side-candidate).
                    s3_hop = s3_connotations[match_sig] + hops
                    accounted = decay(s3_hop)
                    c_kline = model.find(match_sig)
                    if c_kline is not None:
                        yield from expand(
                            model, q_kline, c_kline, signifier,
                            aggregator=aggregator, _visited=_visited,
                        )
                    break
        slot_values.append(accounted)

    # Matched nodes: grounded -> 1.0; matched-ungrounded -> decay(1).
    for n in matched:
        kl = model.find(n)
        if kl is not None and is_s1(model, kl, signifier):
            slot_values.append(1.0)
        else:
            # Ungrounded match OR not in model: one hop of doubt.
            slot_values.append(decay(1))

    if not slot_values:
        # Both klines node-less: vacuously fully accounted.
        slot_values = [1.0]

    significance = aggregator.compose_terminal(slot_values)
    yield QueryCandidate(query, candidate, significance)


# Promotion Helpers


def promote_participating(model: Model, query: KLine, candidate: KLine, signifier: KSignifier) -> None:
    """Promote klines that structurally participated in a ratification event.

    After S1 ratification between query and candidate, promote:
    1. The query and candidate themselves (always)
    2. Any STM kline whose signature is a node value in the query or
       candidate AND whose nodes are empty (Unknown frame), a single
       non-literal node (countersign/denote pair), or a canonical
       composition (canonization entry).

    Does NOT promote cogitator expansion proposals (multi-node non-
    canonical klines) that merely share signature bits.
    """
    # Signatures of node values participating in query/candidate.
    node_sigs: set[int] = set()
    for n in query.nodes:
        node_sigs.add(n)
    for n in candidate.nodes:
        node_sigs.add(n)
    node_sigs.add(query.signature)
    node_sigs.add(candidate.signature)

    to_promote: list[KLine] = []
    for kl in model.iter_stm():
        if kl.signature not in node_sigs:
            continue
        # Promote structural klines: Unknown frames, single-node entries,
        # or canonical compositions.
        if not kl.nodes:
            to_promote.append(kl)
        elif isinstance(kl.nodes, int):
            to_promote.append(kl)
        elif isinstance(kl.nodes, list) and len(kl.nodes) == 1:
            to_promote.append(kl)
        elif is_canon(kl, signifier):
            to_promote.append(kl)

    _log.info(
        "promote_participating: query=%#x candidate=%#x promoting %d structural + 2",
        query.signature,
        candidate.signature,
        len(to_promote),
    )

    to_promote.extend([query, candidate])

    for kl in to_promote:
        model.add_to_ltm(kl)


# Expansion Proposal Pipeline


def propose_expansions(
    model: Model,
    candidate: KLine,
    significance: int,
    signifier: KSignifier,
) -> Iterator[tuple[KLine, int]]:
    """Generate expansion proposals for a misfit candidate.

    Classifies the candidate's misfit type (underfitting, overfitting, or both)
    and generates expansion proposals. Each yield is ``(proposal_kline, significance)``.
    Yields nothing if the candidate is canonical or not a misfit.

    The caller is responsible for pairing proposals with the correct query kline
    for handler dispatch — for connotation yields from ``expand()``, this is
    ``qc.query``, not the original WorkItem's query.

    Expansion proposals must carry decomposition information, so terminal
    klines (the empty Unknown, self-referential ``{S: [S]}``, or a
    compound-word) are never emitted
    — neither as the proposal nor as a companion. A single removed node
    produces the companion ``{n: [n]}``, which is a terminal and is dropped.
    """
    if is_terminal(candidate) or is_canon(candidate, signifier):
        return  # terminal or canonical — nothing to expand

    underfit, overfit = signifier.classify_misfit(candidate.signature, candidate.nodes)

    if not underfit and not overfit:
        return

    candidate_sig = candidate.signature
    nodes_sig = signifier.signature_of(candidate.nodes)
    underfit_gap = signifier.residual(candidate_sig, nodes_sig)
    overfit_mask = signifier.residual(nodes_sig, candidate_sig)

    for proposal, companions in generate_expansions(model, candidate, underfit_gap, overfit_mask, signifier):
        if is_terminal(proposal):
            continue
        yield (proposal, significance)
        for companion in companions:
            if is_terminal(companion):
                continue
            yield (companion, significance)
