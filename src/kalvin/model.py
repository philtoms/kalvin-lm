"""Model — three-tier layered KLine collection (STM → Frame → Base).

The model provides storage, deduplication, lookup by signature, graph
traversal, and the significance API functions consumed by the pipeline.

See specs/model.md for the full specification.
"""

from __future__ import annotations

from typing import Callable, Iterator, NamedTuple

from kalvin.kline import KLine, KSig
from kalvin.stm import STM
from kalvin.signature import make_signature, signifies, is_literal_node

# _S3_BIAS — S3 connotation tier bias. Connotation hop counts are biased by this
# amount before quadratic packing, ensuring S3 packed distances exceed S2 distances
# for the same hop count while keeping both tiers close enough for temperature bridging.
# With _S3_BIAS=9, minimum S3 packed distance = _pack(2+9) = 121 > MAX_HOP(100).
_S3_BIAS = 9


def _pack(distance: int) -> int:
    """Non-linear distance packing via squaring.

    Quadratic growth keeps small distances close for fine-grained temperature
    discrimination, while large distances spread apart so the accumulation
    penalty grows super-linearly.
    """
    if distance <= 0:
        return 0
    return distance * distance

# Public significance constants
D_MAX = 0xFFFF_FFFF_FFFF_FFFF   # maximum distance, also the significance of zero distance
MASK64 = 0xFFFF_FFFF_FFFF_FFFF  # 64-bit mask for bitwise inversion

# MAX_HOP hyperparameter — upper bound on edge hop chain depth
MAX_HOP = 100


class QueryCandidate(NamedTuple):
    """A single query|candidate pair yielded by graph expansion."""
    query: KLine
    candidate: KLine
    significance: int


class Model:
    """Three-tier Model: STM → Frame → Base.

    - STM: bounded rolling window (default 256). add() writes here.
    - Frame: populated by promote() from STM.
    - Base: optional long-term knowledge store (read-only, set at construction).

    The caller sees a single unified API.
    """

    def __init__(self, base: Model | None = None, stm_bound: int = 256):
        self._base = base
        self._stm = STM(bound=stm_bound)

        # Frame storage — ordered dict for reverse-insertion-order iteration
        self._frame_list: list[KLine] = []
        self._frame_by_sig: dict[KSig, list[int]] = {}
        self._frame_dedup: set[tuple[KSig, tuple[int, ...]]] = set()

    # ── Storage Operations ────────────────────────────────────────────

    def add(self, kline: KLine, dedup: bool = False) -> bool:
        """Add a KLine to STM only.

        If kline.is_literal() and dedup=True, reject if an equal KLine
        exists in any tier. Non-literal Klines are always accepted.

        Returns True if added, False if rejected.
        """
        if dedup and kline.is_literal():
            if self._exists_any(kline):
                return False

        # Add to STM — always track in STM's dedup set so promote() and
        # _exists_any() can rely on it.  Model-level dedup is handled above.
        self._stm.add(kline, True)
        return True

    def exists(self, kline: KLine) -> bool:
        """Check if an equal KLine exists in any tier."""
        return self._exists_any(kline)

    def _exists_any(self, kline: KLine) -> bool:
        """Check STM, frame, then base."""
        key = (kline.signature, tuple(kline.nodes))
        if key in self._stm._dedup:
            return True
        if key in self._frame_dedup:
            return True
        if self._base:
            return self._base.exists(kline)
        return False

    def find(self, signature: KSig) -> KLine | None:
        """Find the most recently added KLine by signature.

        Searches STM, then frame, then base.
        """
        # STM first — filter for actual signature match (STM is dual-keyed)
        for kl in reversed(self._stm.find_by_signature(signature)):
            if kl.signature == signature:
                return kl
        # Frame
        indices = self._frame_by_sig.get(signature)
        if indices:
            for idx in reversed(indices):
                kl = self._frame_list[idx]
                if kl is not None:
                    return kl
        # Base
        if self._base:
            return self._base.find(signature)
        return None

    def find_all(self, signature: KSig) -> list[KLine]:
        """Return all KLines with the given signature across all tiers."""
        results: list[KLine] = []
        seen: set[tuple[KSig, tuple[int, ...]]] = set()

        # STM results — filter for actual signature match (STM is dual-keyed)
        for kl in self._stm.find_by_signature(signature):
            if kl.signature != signature:
                continue
            key = (kl.signature, tuple(kl.nodes))
            if key not in seen:
                seen.add(key)
                results.append(kl)

        # Frame results
        indices = self._frame_by_sig.get(signature, [])
        for idx in indices:
            kl = self._frame_list[idx]
            if kl is None:
                continue
            key = (kl.signature, tuple(kl.nodes))
            if key not in seen:
                seen.add(key)
                results.append(kl)

        # Base results
        if self._base:
            for kl in self._base.find_all(signature):
                key = (kl.signature, tuple(kl.nodes))
                if key not in seen:
                    seen.add(key)
                    results.append(kl)

        return results

    def find_by_nodes(self, nodes_signature: KSig) -> KLine | None:
        """Find the most recently added KLine whose nodes signature matches."""
        # STM first
        klines = self._stm.find_by_nodes(nodes_signature)
        if klines:
            return klines[-1]
        # Scan frame
        for kl in reversed(self._frame_list):
            ns = make_signature(kl.nodes)
            if ns == nodes_signature:
                return kl
        # Base
        if self._base:
            return self._base.find_by_nodes(nodes_signature)
        return None

    # ── Count ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Number of KLines in the frame (excluding STM and base)."""
        return sum(1 for kl in self._frame_list if kl is not None)

    def __iter__(self) -> Iterator[KLine]:
        return iter(kl for kl in self._frame_list if kl is not None)

    def __getitem__(self, signature: KSig) -> KLine | None:
        return self.find(signature)

    # ── Iteration ─────────────────────────────────────────────────────

    def klines(self) -> list[KLine]:
        """All KLines in reverse insertion order, deduplicated across tiers."""
        seen: set[tuple[KSig, tuple[int, ...]]] = set()
        results: list[KLine] = []

        # STM entries (most recent)
        for kl in reversed(self._stm._order):
            key = (kl.signature, tuple(kl.nodes))
            if key not in seen:
                seen.add(key)
                results.append(kl)

        # Frame entries not in STM
        for kl in reversed(self._frame_list):
            if kl is None:
                continue
            key = (kl.signature, tuple(kl.nodes))
            if key not in seen:
                seen.add(key)
                results.append(kl)

        # Base entries not in frame
        if self._base:
            for kl in self._base.klines():
                key = (kl.signature, tuple(kl.nodes))
                if key not in seen:
                    seen.add(key)
                    results.append(kl)

        return results

    def where(self, predicate: Callable[[KLine], bool] | KSig) -> list[KLine]:
        """Return KLines matching a predicate or signature overlap.

        If predicate is an int, it's treated as a signature for AND matching:
            where(sig) returns klines where kline.signature & sig != 0.
        """
        if isinstance(predicate, int):
            sig = predicate
            return [kl for kl in self.klines() if signifies(kl.signature, sig)]
        return [kl for kl in self.klines() if predicate(kl)]

    # ── Promotion ─────────────────────────────────────────────────────

    def promote(self, kline: KLine) -> bool:
        """Promote a KLine from STM to the frame.

        The KLine must currently be in STM. Returns True if added to frame,
        False if not in STM or already in frame.
        """
        key = (kline.signature, tuple(kline.nodes))
        # Must be in STM
        if key not in self._stm._dedup:
            return False
        # Already in frame
        if key in self._frame_dedup:
            return False
        # Add to frame
        idx = len(self._frame_list)
        self._frame_list.append(kline)
        if kline.signature not in self._frame_by_sig:
            self._frame_by_sig[kline.signature] = []
        self._frame_by_sig[kline.signature].append(idx)
        self._frame_dedup.add(key)
        return True

    def promote_all(self) -> int:
        """Promote all STM KLines to the frame."""
        count = 0
        for kl in list(self._stm._order):
            if self.promote(kl):
                count += 1
        return count

    # ── Graph Traversal ───────────────────────────────────────────────

    def resolve(self, node: int) -> KLine | None:
        """Resolve a node value to a KLine."""
        return self.find(node)

    def query_expand(self, kline: KLine, depth: int = 2) -> list[KLine]:
        """Expand graph from kline up to *depth* levels.

        depth=0 → []
        depth=1 → []
        depth=2 → direct children
        depth=N → children up to N-1 levels deep.
        """
        if depth <= 1:
            return []
        visited: set[int] = set()
        results: list[KLine] = []
        self._query_expand_inner(kline, depth, 1, visited, results)
        return results

    def _query_expand_inner(
        self,
        kline: KLine,
        max_depth: int,
        current_depth: int,
        visited: set[int],
        results: list[KLine],
    ) -> None:
        if id(kline) in visited:
            return
        visited.add(id(kline))

        if current_depth >= max_depth:
            return

        for node in kline.nodes:
            child = self.find(node)
            if child is not None:
                results.append(child)
                self._query_expand_inner(child, max_depth, current_depth + 1, visited, results)

    def descendants(self, node: int) -> set[int]:
        """Recursively collect all descendant node values."""
        visited: set[int] = set()
        result: set[int] = set()
        self._descendants_inner(node, visited, result)
        return result

    def _descendants_inner(self, node: int, visited: set[int], result: set[int]) -> None:
        if node in visited:
            return
        visited.add(node)
        kline = self.find(node)
        if kline is None:
            return
        for child_node in kline.nodes:
            result.add(child_node)
            self._descendants_inner(child_node, visited, result)

    def query(self, signature: KSig, depth: int = 1) -> list[KLine]:
        """Find all KLines with signature, then expand each."""
        matches = self.find_all(signature)
        results: list[KLine] = list(matches)
        for kl in matches:
            results.extend(self.query_expand(kl, depth))
        return results

    # ── Significance API ──────────────────────────────────────────────


    def _is_canon(self, kline: KLine) -> bool:
        """Test whether a kline is canonical (signature = make_signature of nodes)."""
        return kline.signature == make_signature(kline.nodes)

    def _edge_hops(self, sig: int) -> Iterator[tuple[int, int]]:
        """Yield (hop_count, next_sig) for each non-canonical resolution step.

        Follows: resolve sig → kline → make_signature(kline.nodes) → repeat.
        Stops at a dead end (unresolvable) or a canonical kline.
        Yields (hop_count, next_sig) at each step.
        """
        hop_count = 0
        while hop_count < MAX_HOP:
            kline = self.find(sig)
            if kline is None or self._is_canon(kline):
                break
            hop_count += 1
            sig = make_signature(kline.nodes)
            yield hop_count, sig

    def _as_kline(self, node: int) -> KLine:
        """Resolve a node value to a KLine.

        Raises ValueError if the node does not resolve.
        """
        kline = self.find(node)
        if kline is None:
            raise ValueError(f"Node {node:#x} does not resolve to any KLine")
        return kline

    def expand(
        self,
        query: KLine,
        candidate: KLine,
        distance: int = 0,
        *,
        _visited: set[tuple[int, int]] | None = None,
    ) -> Iterator[QueryCandidate]:
        """Expand a query-candidate pair, yielding connotations and terminal distance.

        For each discovered connotation (S2/S3 indirect relationship), recursively
        yields QueryCandidate items for the connotation pair.  The final yield is
        always the terminal QueryCandidate with the computed distance for the
        original pair.

        Distance is a single integer. S3 connotation hops are biased by
        ``_pack(hop_count + _S3_BIAS)`` to ensure S3 distances moderately
        exceed S2 distances — close enough for temperature to bridge.

        Parameters
        ----------
        query: The query KLine.
        candidate: The candidate KLine.
        distance: Accumulated hop distance (internal, for recursive calls).
        _visited: Internal cycle-detection set.

        Yields
        ------
        QueryCandidate
            Intermediate connotations (from recursive calls) followed by the
            terminal distance.
        """
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

        s3_connotations: dict[int, int] = {}  # sig → min hops from any query node

        # Accumulate distance from mismatched nodes
        # Direct edge resolutions (S2) add hop count directly.
        # Connotation resolutions (S3) add _pack(hop_count + _S3_BIAS).
        # Unresolved nodes add MAX_HOP.
        total_distance = 0

        # Mismatched query nodes
        for n in mismatched_q:
            hop_distance = MAX_HOP
            q_kline = self.find(n)
            if q_kline is not None:
                for hops, match_sig in self._edge_hops(n):
                    if match_sig in mismatched_c:
                        hop_distance = hops
                        c_kline = self._as_kline(match_sig)
                        yield from self.expand(
                            q_kline, c_kline, hops, _visited=_visited,
                        )
                        break
                    elif signifies(n, match_sig):
                        c_kline = self.find(match_sig)
                        if c_kline is not None:
                            sig_distance = distance + hops
                            significance = (~min(sig_distance, D_MAX - 1)) & MASK64
                            yield QueryCandidate(q_kline, c_kline, significance)
                        break
                    elif (match_sig not in s3_connotations
                        or hops < s3_connotations[match_sig]):
                        s3_connotations[match_sig] = hops
            total_distance += hop_distance

        # Mismatched candidate nodes
        for n in mismatched_c:
            hop_distance = MAX_HOP
            q_kline = self.find(n)
            if q_kline is not None:
                for hops, match_sig in self._edge_hops(n):
                    if match_sig in mismatched_q:
                        hop_distance = hops
                        c_kline = self._as_kline(match_sig)
                        yield from self.expand(
                            q_kline, c_kline, hops, _visited=_visited,
                        )
                        break
                    elif signifies(n, match_sig):
                        c_kline = self.find(match_sig)
                        if c_kline is not None:
                            sig_distance = distance + hops
                            significance = (~min(sig_distance, D_MAX - 1)) & MASK64
                            yield QueryCandidate(q_kline, c_kline, significance)
                        break
                    elif match_sig in s3_connotations:
                        s3_hop = s3_connotations[match_sig] + hops
                        c_kline = self._as_kline(match_sig)
                        yield from self.expand(
                            q_kline, c_kline,
                            _pack(s3_hop + _S3_BIAS),
                            _visited=_visited,
                        )
                        hop_distance = 0
                        break
            total_distance += hop_distance

        # Matched but not grounded — small S2 penalty
        for n in matched:
            kl = self.find(n)
            if kl is None or not self.is_s1(kl):
                total_distance += 1

        # Carry forward the incoming distance
        total_distance += distance

        # Clamp to avoid overflow, then invert to significance
        significance = (~min(total_distance, D_MAX - 1)) & MASK64
        yield QueryCandidate(query, candidate, significance)

    # ── Structural Grounding ──────────────────────────────────────────

    def is_s1(self, kline: KLine) -> bool:
        """Determine if a kline is structurally grounded (S1).

        A kline is S1 if:
        1. Its signature fully describes its nodes (canonical), OR
        2. It is countersigned by another kline in the model.
        """
        if self._is_canon(kline):
            return True
        return self.is_countersigned(kline)

    def is_countersigned(self, kline: KLine) -> bool:
        """Check if kline is countersigned by any kline in the model.

        A kline is countersigned if its nodes_signature exists as a
        countersigning kline with one node — the countersigned kline's signature.

        Query = {Q: [A, B]}
        Countersigner = {AB: [Q]}
        """
        nodes_signature = make_signature(kline.nodes)
        for countersigner in self.find_all(nodes_signature):
            if len(countersigner.nodes) == 1 and countersigner.nodes[0] == kline.signature:
                return True
        return False

    def promote_participating(self, query: KLine, candidate: KLine) -> int:
        """Promote all STM klines involved in a ratification event.

        After countersignature is detected between query and candidate,
        promote both plus any STM klines whose signatures appear in the
        union of their nodes.

        Returns the number of klines promoted.
        """
        # Collect all signatures from the participating pair
        node_sigs: set[int] = set()
        for n in query.nodes:
            if not is_literal_node(n):
                node_sigs.add(n)
        for n in candidate.nodes:
            if not is_literal_node(n):
                node_sigs.add(n)
        node_sigs.add(query.signature)
        node_sigs.add(candidate.signature)

        # Find all STM klines with matching signatures
        to_promote: list[KLine] = []
        for kl in self._stm._order:
            if kl.signature in node_sigs:
                to_promote.append(kl)

        # Also promote the query and candidate
        to_promote.extend([query, candidate])

        count = 0
        for kl in to_promote:
            if self.promote(kl):
                count += 1
        return count

    # ── Misfit Classification & Expansion ──────────────────────────────

    def classify_misfit(self, kline: KLine) -> tuple[bool, bool]:
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
        self,
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
        if underfit_gap:
            yield from self._underfit_expansions(kline, underfit_gap)

        if overfit_mask:
            yield from self._overfit_expansions(kline, overfit_mask)

        if underfit_gap and overfit_mask:
            yield from self._dual_expansions(kline, underfit_gap, overfit_mask)

    def _underfit_expansions(
        self, kline: KLine, gap: int
    ) -> Iterator[tuple[KLine, list[KLine]]]:
        """Add nodes whose signatures contribute to the gap."""
        contributors = self.where(lambda k: (k.signature & gap) != 0)

        for contributor in contributors:
            expanded_nodes = list(kline.nodes) + list(contributor.nodes)
            expanded_sig = kline.signature
            proposal = KLine(expanded_sig, expanded_nodes, kline.dbg_text)

            new_nodes_sig = make_signature(expanded_nodes)
            if (new_nodes_sig & expanded_sig) != 0:
                yield (proposal, [])

    def _split_excess(
        self, kline: KLine, excess: int
    ) -> tuple[list[int], list[int]]:
        """Split kline nodes into (excess_nodes, remaining) by excess mask."""
        excess_nodes = [n for n in kline.nodes
                        if not is_literal_node(n) and (n & excess) != 0]
        remaining = [n for n in kline.nodes if n not in excess_nodes]
        return excess_nodes, remaining

    def _overfit_expansions(
        self, kline: KLine, excess: int
    ) -> Iterator[tuple[KLine, list[KLine]]]:
        """Remove nodes whose bits contribute to the excess."""
        excess_nodes, remaining = self._split_excess(kline, excess)

        if not excess_nodes:
            return
        trimmed = KLine(kline.signature, remaining, kline.dbg_text)

        companion_sig = make_signature(excess_nodes)
        companion = KLine(companion_sig, excess_nodes)

        yield (trimmed, [companion])

    def _dual_expansions(
        self, kline: KLine, gap: int, excess: int
    ) -> Iterator[tuple[KLine, list[KLine]]]:
        """Atomic replacement: swap excess nodes for gap-filling nodes."""
        excess_nodes, remaining = self._split_excess(kline, excess)

        contributors = self.where(lambda k: (k.signature & gap) != 0)

        for contributor in contributors:
            replacement_nodes = remaining + list(contributor.nodes)
            replacement = KLine(kline.signature, replacement_nodes, kline.dbg_text)

            companion_sig = make_signature(excess_nodes)
            companion = KLine(companion_sig, excess_nodes)

            yield (replacement, [companion])

    # ── Properties ────────────────────────────────────────────────────

    @property
    def base(self) -> Model | None:
        return self._base

    @property
    def stm(self) -> STM:
        return self._stm

    # ── Compatibility ─────────────────────────────────────────────────

    def find_kline(self, signature: KSig) -> KLine | None:
        """Alias for find() — backwards compat."""
        return self.find(signature)

    def find_signed_klines(self, signature: KSig) -> list[KLine]:
        """Alias for find_all() — backwards compat."""
        return self.find_all(signature)

    def query_graph(self, query: KSig, depth: int = 1):
        """Alias for query() returning list — backwards compat."""
        return self.query(query, depth)

    def duplicate(self) -> Model:
        """Create a duplicate of this model's frame."""
        klines = [KLine(kl.signature, list(kl.nodes), kl.literal, kl.dbg_text)
                   for kl in self._frame_list if kl is not None]
        m = Model()
        for kl in klines:
            m.add(kl)
            m.promote(kl)
        return m

    def get_all_descendants(self, node: int, visited: set[int] | None = None) -> set[int]:
        """Backwards-compat alias for descendants()."""
        return self.descendants(node)

    @property
    def klines_prop(self) -> list[KLine]:
        """Backwards compat: return frame klines."""
        return [kl for kl in self._frame_list if kl is not None]

    def as_kline_list(self, limit: int = 0):
        """Backwards compat: iterate KLines."""
        items = [kl for kl in self._frame_list if kl is not None]
        if limit > 0:
            items = items[:limit]
        return reversed(items)

    def upgrade(self, kline: KLine, significance: KSig) -> None:
        """Upgrade significance — backwards compat."""
        kline.signature |= significance

    @property
    def kline(self) -> _KLineAccessor:
        return _KLineAccessor(self)


class _KLineAccessor:
    __slots__ = ("_model",)

    def __init__(self, model: Model):
        self._model = model

    def __getitem__(self, signature: KSig) -> KLine | None:
        return self._model.find(signature)
