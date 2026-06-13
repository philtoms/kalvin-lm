"""AST Emitter for KScript v3 — walks scope-model AST and emits SymbolicEntry tuples.

Central compilation stage that transforms the KScript v3 scope-model AST
(spec §3) into a list of symbolic entries (spec §6).  No token encoding
happens here — all values are strings.  The TokenEncoder (separate module)
converts SymbolicEntry tuples to encoded uint64 values.

**Scope processing rules (spec §7):**
  Each OperatorScope is processed by resolving its signature, collecting
  node identifiers from items and child_block, and emitting operator-specific
  entries:

  - IDENTITY (op=None):   {sig: []}   — bare identity
  - COUNTERSIGNED (==):   {sig: [node]}, {node: [sig]} per item  — bidirectional
  - UNDERSIGNED (=):      {node: [sig]} per item  — reversed direction
  - CONNOTED (>):         {sig: [node]} per item  — forward direction
  - CANONIZED (=>):       {sig: [all_nodes]}  — aggregated single entry

  Self-identity (A = A) collapses to IDENTITY with empty nodes (spec §7.3).

**MCS expansion (spec §8):**
  Multi-character identifiers trigger automatic emission of:
  1. One IDENTITY entry per resolved constituent character.
  2. One CANONIZE entry mapping the compound to its resolved components.

  MCS applies to identifiers wherever they appear — signature side or node
  side, any operator.  Single-character identifiers do NOT trigger MCS.

**MCS deduplication (§8.3):**
  CANONIZE entries are deduplicated on (sig, nodes).  Component IDENTITY
  entries are deduplicated across MCS calls via _mcs_identity_seen (a
  character emitted once is never emitted again).  Intra-expansion dedup
  prevents duplicate chars within a single compound (e.g., MHALL's second L).

**NLP binding integration (spec §10):**
  When a BindingScope is provided, single-character identifiers are resolved
  inline during the AST walk:

  1. Inline annotation first (Rule B4): S(ubject) → "Subject", immediate
     binding that bypasses the occurrence counter and retroactively patches
     the parent scope's MCS CANONIZE entry.
  2. BindingScope fallback (Rule B3): scope.resolve(char) walks the scope
     stack innermost-first with first-letter matching and occurrence counter.

  CANONIZE scope boundaries trigger push_scope/pop_scope on the BindingScope.
  Parent kline tracking is saved/restored for Rule B4 override patching.

  When scope is None, all binding logic is skipped.

**Key design constraints:**
  - nodes field is ALWAYS list[str] — never None, never a bare string,
    never singleton-unwrapped.  Singleton unwrapping happens in TokenEncoder.
  - No IDENTITY op — self-identity emits IDENTITY with empty nodes.
  - No general deduplication beyond MCS — CANONIZE dedup per §8.3,
    plus component IDENTITY dedup.

Spec references: §3 (Scope Model), §6 (Entry Model), §7 (Operator Rules),
§8 (MCS Expansion), §10 (NLP Binding Resolution).
"""

from __future__ import annotations

from typing import NamedTuple

from .ast import (
    Annotation,
    Block,
    ConstructItem,
    KScriptFile,
    OperatorScope,
    Signature,
)
from .binding_scope import BindingScope
from .token import TokenType

# ---------------------------------------------------------------------------
# SymbolicEntry — the output unit of the ASTEmitter
# ---------------------------------------------------------------------------


class SymbolicEntry(NamedTuple):
    """A symbolic (not yet tokenized) compilation entry.

    Attributes:
        sig:  The signature identifier string (possibly a resolved NLP word).
        nodes: Always a list — empty for IDENTITY, single-item for per-item
               operators, multi-item for CANONIZE aggregation.  Never None,
               never a bare string, never singleton-unwrapped.
        op:   One of "COUNTERSIGNED", "CANONIZED", "CONNOTED", "UNDERSIGNED",
               "IDENTITY".
        component_labels: Resolved words per signature character (for NLP
               mode).  None when not applicable.
    """

    sig: str
    nodes: list[str]
    op: str  # COUNTERSIGNED | CANONIZED | CONNOTED | UNDERSIGNED | IDENTITY
    component_labels: list[str] | None = None


# ---------------------------------------------------------------------------
# ASTEmitter
# ---------------------------------------------------------------------------


class ASTEmitter:
    """Walks a KScript v3 scope-model AST and emits SymbolicEntry tuples.

    Args:
        scope: Optional BindingScope for NLP binding resolution.
            When provided, single-character identifiers are resolved inline
            via scope.resolve().  When None, all identifiers
            pass through as raw characters.
        dev: Enable development/diagnostic mode.
    """

    def __init__(
        self,
        scope: BindingScope | None = None,
        dev: bool = False,
    ) -> None:
        self.entries: list[SymbolicEntry] = []
        self._scope = scope
        self._dev = dev

        # MCS dedup tracking for CANONIZE entries (§8.3).
        # Maps (sig, tuple(nodes)) → index in self.entries.
        self._mcs_canonize_seen: dict[tuple[str, tuple[str, ...]], int] = {}

        # MCS component IDENTITY dedup (§8.3 extended).
        # Tracks resolved characters already emitted as IDENTITY by MCS.
        self._mcs_identity_seen: set[str] = set()

        # Rule B4 parent kline tracking — saved/restored on scope entry/exit.
        self._parent_kline_chars: str | None = None
        self._parent_kline_canonize_idx: int | None = None

        # CANONIZE subscript identity context flag.
        # Set when processing children of a CANONIZE scope that has recursive
        # content (OperatorScope items or child_block). Only activated when
        # the CANONIZE scope's sig did NOT trigger MCS expansion (single-char
        # sig). Multi-char sig CANONIZE scopes have component identities
        # provided by MCS, so subscript identity is unnecessary.
        # Saved/restored alongside parent kline tracking.
        self._in_canonize_subscript: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def emit(self, file: KScriptFile) -> list[SymbolicEntry]:
        """Walk a KScriptFile AST and return the list of SymbolicEntry tuples."""
        for construct in file.constructs:
            self._process_construct(construct)
        return self.entries

    # ------------------------------------------------------------------
    # Construct dispatch
    # ------------------------------------------------------------------

    def _process_construct(self, construct: ConstructItem) -> None:
        """Dispatch a top-level construct to the appropriate handler."""
        if isinstance(construct, OperatorScope):
            self._process_scope(construct)
        elif isinstance(construct, Annotation):
            self._feed_annotation(construct)
        elif isinstance(construct, Block):
            for c in construct.constructs:
                self._process_construct(c)

    # ------------------------------------------------------------------
    # Core scope processing (Steps 2–3)
    # ------------------------------------------------------------------

    def _process_scope(self, scope: OperatorScope) -> None:
        """Process a single OperatorScope node.

        Algorithm:
        1. Resolve signature (inline annotation or BindingScope).
        2. Emit MCS for multi-char signature.
        3. If bare (op=None): emit IDENTITY and return.
        4. Collect node IDs from items and child_block.
        5. Emit MCS for multi-char node IDs.
        6. Resolve nodes.
        7. Emit operator-specific entries.
        8. Compile children (recursive).
        """
        # 1. Resolve signature
        sig_resolved = self._resolve_inline_or_scope(
            scope.sig.id,
            scope.inline_annotation,
        )

        # 2. MCS for signature
        mcs_idx = self._emit_mcs(scope.sig.id)

        # 3. Determine op
        op = self._op_to_str(scope.op)

        if op == "IDENTITY":
            # For multi-char sigs, _emit_mcs already emitted a CANONIZE
            # entry (mcs_idx is not None), introducing the compound.  Skip
            # IDENTITY — a compound cannot form one (spec §8).  For single-
            # char sigs, _emit_mcs returns None (no MCS) so emit IDENTITY here.
            if mcs_idx is None:
                self._emit_entry(sig_resolved, [], "IDENTITY")
            return

        # 4. Collect node IDs
        node_ids = self._collect_node_ids(scope)

        # 5. MCS for multi-char nodes
        for nid in node_ids:
            if len(nid) > 1:
                self._emit_mcs(nid)

        # 6. Resolve nodes
        resolved_nodes = self._resolve_nodes(node_ids, scope)

        # 7. Emit operator entries
        self._emit_operator_entries(sig_resolved, resolved_nodes, op)

        # 8. Compile children
        self._compile_children(scope, op, mcs_idx)

    # ------------------------------------------------------------------
    # Operator emission (Step 2)
    # ------------------------------------------------------------------

    def _emit_operator_entries(
        self,
        sig: str,
        nodes: list[str],
        op: str,
    ) -> None:
        """Emit operator-specific entries based on the operator type."""
        if op == "COUNTERSIGNED":
            # Per-item bidirectional
            for node in nodes:
                self._emit_entry(sig, [node], "COUNTERSIGNED")
                self._emit_entry(node, [sig], "COUNTERSIGNED")

        elif op == "UNDERSIGNED":
            # Per-item reversed
            for node in nodes:
                if node == sig:
                    # Self-identity → IDENTITY with empty nodes (§7.3)
                    self._emit_entry(sig, [], "IDENTITY")
                else:
                    self._emit_entry(node, [sig], "UNDERSIGNED")

        elif op == "CONNOTED":
            # Per-item forward
            for node in nodes:
                self._emit_entry(sig, [node], "CONNOTED")

        elif op == "CANONIZED":
            # Aggregated single entry
            self._emit_entry(sig, list(nodes), "CANONIZED")

    # ------------------------------------------------------------------
    # Node collection (Step 2)
    # ------------------------------------------------------------------

    def _collect_node_ids(self, scope: OperatorScope) -> list[str]:
        """Walk items and child_block to collect node identifier strings.

        - Signature items → item.id
        - OperatorScope items → item.sig.id
        - Annotation items → skipped
        - child_block constructs → extract sig IDs recursively
        """
        node_ids: list[str] = []

        for item in scope.items:
            if isinstance(item, Signature):
                node_ids.append(item.id)
            elif isinstance(item, OperatorScope):
                node_ids.append(item.sig.id)
            # Annotation → skip (not a node)

        if scope.child_block is not None:
            for construct in scope.child_block.constructs:
                self._collect_block_node_ids(construct, node_ids)

        return node_ids

    def _collect_block_node_ids(
        self,
        construct: ConstructItem,
        node_ids: list[str],
    ) -> None:
        """Recursively collect node IDs from a Block's constructs."""
        if isinstance(construct, OperatorScope):
            node_ids.append(construct.sig.id)
        elif isinstance(construct, Block):
            for c in construct.constructs:
                self._collect_block_node_ids(c, node_ids)
        # Annotation → skip

    # ------------------------------------------------------------------
    # MCS expansion (§8)
    # ------------------------------------------------------------------

    def _emit_mcs(self, sig: str) -> int | None:
        """Emit MCS entries for a multi-character identifier.

        1. One IDENTITY entry per resolved constituent character (deduped).
        2. One CANONIZE entry mapping the compound to its resolved components.

        Component IDENTITY deduplication (§8.3 extended):
          - Intra-expansion: duplicate chars within a compound (e.g., MHALL's
            second L) emit only one IDENTITY L.
          - Inter-expansion: if a char was already emitted by a previous MCS
            call, it is silently skipped.

        CANONIZE deduplication (§8.3):
          - Same (sig, nodes) pair is silently skipped.

        No IDENTITY entry is emitted for the compound itself.  A compound
        signature is the OR-reduction of multiple token IDs and cannot form
        an identity (spec §8; CONTEXT.md "Identity" glossary).

        Returns the index of the CANONIZE entry (for Rule B4), or None
        if no MCS was emitted (single-char identifier).
        """
        if len(sig) <= 1:
            return None

        chars = [self._resolve_char(c) for c in sig]

        # Component identities — with intra- and inter-expansion dedup
        seen_in_this_call: set[str] = set()
        for resolved_char in chars:
            if resolved_char in seen_in_this_call:
                continue  # intra-expansion dedup (e.g., second L in MHALL)
            seen_in_this_call.add(resolved_char)
            if resolved_char in self._mcs_identity_seen:
                continue  # inter-expansion dedup
            self._mcs_identity_seen.add(resolved_char)
            self._emit_entry(resolved_char, [], "IDENTITY")

        # MCS canonization
        key = (sig, tuple(chars))
        if key in self._mcs_canonize_seen:
            # Already emitted — return existing index for Rule B4
            return self._mcs_canonize_seen[key]

        self._emit_entry(sig, list(chars), "CANONIZED")
        canonize_idx = len(self.entries) - 1

        return canonize_idx

    # ------------------------------------------------------------------
    # Entry emission with CANONIZE dedup (§8.3, Step 4)
    # ------------------------------------------------------------------

    def _emit_entry(self, sig: str, nodes: list[str], op: str) -> None:
        """Emit a SymbolicEntry with CANONIZE deduplication.

        - CANONIZE entries: dedup on (sig, tuple(nodes)).  Duplicates
          are silently skipped.
        - All other ops: always emit (no dedup at this level).

        Note: IDENTITY dedup for MCS components is handled in _emit_mcs
        via _mcs_identity_seen, not here.
        """
        if op == "CANONIZED":
            key = (sig, tuple(nodes))
            if key in self._mcs_canonize_seen:
                return  # dedup — silently skip
            self._mcs_canonize_seen[key] = len(self.entries)

        self.entries.append(SymbolicEntry(sig=sig, nodes=nodes, op=op))

    # ------------------------------------------------------------------
    # Identity emission for CANONIZE subscript blocks
    # ------------------------------------------------------------------

    def _emit_identity_if_needed(self, sig: str) -> None:
        """Emit identity IDENTITY only if no IDENTITY entry for this sig exists.

        Used in CANONIZE subscript blocks to ensure every identifier appears
        as the signature of at least one emitted entry.

        Dedup checks (in order):
          1. _mcs_identity_seen — sig was already emitted as MCS component.
          2. Existing CANONIZE entry — sig is a compound already introduced
             by its CANONIZE entry from MCS.
          3. Existing IDENTITY entries — sig already has an IDENTITY entry.

        This prevents duplicate IDENTITY when MCS expansion already provided
        one for the same identifier, or when the identifier already appears
        as the signature of an IDENTITY entry.  The CANONIZE check blocks
        compounds (which cannot form an identity) without affecting single-char
        sigs that have only UNDERSIGN entries (e.g., D in §14.8).
        """
        if sig in self._mcs_identity_seen:
            return
        if any(e.sig == sig and e.op == "CANONIZED" for e in self.entries):
            return  # compound already introduced by its CANONIZED entry
        if any(e.sig == sig and e.op == "IDENTITY" for e in self.entries):
            return
        self._emit_entry(sig, [], "IDENTITY")

    # ------------------------------------------------------------------
    # Scope walk and child compilation (Step 3)
    # ------------------------------------------------------------------

    def _compile_children(
        self,
        scope: OperatorScope,
        op: str,
        mcs_idx: int | None,
    ) -> None:
        """After emitting operator entries, recursively process children.

        - For CANONIZE scopes: push/pop BindingScope, save/restore
          parent kline tracking for Rule B4.
        - Process nested OperatorScopes and Annotations from items.
        - Process child_block constructs.  Bare OperatorScope nodes
          (op=None) in a non-CANONIZE child_block are skipped — they
          are already collected as node identifiers by _collect_node_ids
          and used in the parent operator's emission.  Under CANONIZE,
          bare scopes in child_block still emit their own IDENTITY
          entries (they are subscript items with independent identity).

        **CANONIZE subscript identity (spec §7.6, §14.8, §14.9):**

        When a CANONIZE scope has recursive content (OperatorScope items
        or child_block), it forms a "subscript block". Every identifier
        within such a block must appear as the signature of at least one
        emitted entry. Where no operator entry provides this, an identity
        IDENTITY fills the gap.

        Subscript identity is only activated when the CANONIZE scope's
        sig did NOT trigger MCS expansion (mcs_idx is None). For single-
        char sigs (e.g., `A =>`), MCS does not fire, so subscript identity
        ensures all identifiers appear as signatures. For multi-char sigs
        (e.g., `SVO =>`, `ALL =>`), MCS provides component identities for
        all constituent characters, making subscript identity unnecessary
        and preventing extra entries like IDENTITY D in §14.11.

        Three categories receive identity via _emit_identity_if_needed:

        1. Leaf Signature items — bare identifiers in items lists produce
           no operator entry, so they get identity IDENTITY.
        2. UNDERSIGN OperatorScope sigs — UNDERSIGN emits entries with
           nodes as signatures (reversed direction), so the scope's own
           sig lacks identity and needs IDENTITY.
        3. (Not needed for CANONIZE/COUNTERSIGN/CONNOTATE scope sigs —
           they already produce entries with the scope's sig as signature.
           Not needed for bare OperatorScope nodes (op=None) — they emit
           IDENTITY via _process_scope step 3.)

        The _in_canonize_subscript flag does NOT propagate between
        CANONIZE scopes — each CANONIZE independently decides whether
        to activate subscript identity based on its own MCS status.

        _emit_identity_if_needed includes a dedup check to prevent
        duplicate IDENTITY when MCS expansion already provided one
        for the same identifier.
        """
        is_canonize = op == "CANONIZED"

        # Save parent kline tracking and CANONIZE subscript context
        saved_chars = self._parent_kline_chars
        saved_idx = self._parent_kline_canonize_idx
        saved_in_canonize = self._in_canonize_subscript

        # Set parent kline tracking for Rule B4 (multi-char CANONIZE sigs)
        if is_canonize and mcs_idx is not None:
            self._parent_kline_chars = scope.sig.id
            self._parent_kline_canonize_idx = mcs_idx

        # CANONIZE scope push
        if is_canonize and self._scope is not None:
            self._scope.push_scope()

        # Set CANONIZE subscript context when this CANONIZE scope has
        # recursive content (OperatorScope items or child_block), meaning
        # it's a "subscript block" rather than a flat aggregation like
        # `A => B C D`.
        # Subscript identity is only activated when this CANONIZE scope's
        # sig did NOT trigger MCS expansion (mcs_idx is None). Multi-char
        # sig CANONIZE scopes (e.g., SVO =>, ALL =>) have their component
        # identities provided by MCS, so subscript identity is unnecessary
        # and would produce extra entries (e.g., IDENTITY D in §14.11).
        if is_canonize:
            has_recursive_content = scope.child_block is not None or any(
                isinstance(item, OperatorScope) for item in scope.items
            )
            if has_recursive_content and mcs_idx is None:
                self._in_canonize_subscript = True

        # Process items
        for item in scope.items:
            if isinstance(item, OperatorScope):
                # In CANONIZE subscript blocks, UNDERSIGN scope sigs need
                # identity IDENTITY because UNDERSIGN emits entries with nodes
                # as sigs (e.g., D | [C] | UNDERSIGN), not the scope's own sig.
                # CANONIZE/COUNTERSIGN/CONNOTATE scope sigs already have identity
                # via their operator entries.
                if (
                    self._in_canonize_subscript
                    and item.op is not None
                    and self._op_to_str(item.op) == "UNDERSIGNED"
                ):
                    resolved = self._resolve_char(item.sig.id)
                    self._emit_identity_if_needed(resolved)
                self._process_scope(item)
            elif isinstance(item, Annotation):
                self._feed_annotation(item)
            # Note: bare Signature items in CANONIZE subscript blocks need
            # identity IDENTITY because they produce no operator entry.
            elif isinstance(item, Signature):
                if self._in_canonize_subscript:
                    resolved = self._resolve_char(item.id)
                    self._emit_identity_if_needed(resolved)

        # Process child_block constructs
        if scope.child_block is not None:
            for construct in scope.child_block.constructs:
                if (
                    isinstance(construct, OperatorScope)
                    and construct.op is None
                    and not is_canonize
                ):
                    # Bare scope in non-CANONIZE child_block — already a node
                    # of the parent operator (collected by _collect_node_ids).
                    # Skip to avoid spurious IDENTITY emission.
                    continue
                # In CANONIZE subscript blocks, UNDERSIGN scope sigs in
                # child_block need identity IDENTITY. Bare scopes (op=None)
                # already emit IDENTITY via _process_scope step 3.
                if (
                    self._in_canonize_subscript
                    and isinstance(construct, OperatorScope)
                    and construct.op is not None
                    and self._op_to_str(construct.op) == "UNDERSIGNED"
                ):
                    resolved = self._resolve_char(construct.sig.id)
                    self._emit_identity_if_needed(resolved)
                self._process_construct(construct)

        # CANONIZE scope pop
        if is_canonize and self._scope is not None:
            self._scope.pop_scope()

        # Restore parent kline tracking and CANONIZE subscript context
        self._parent_kline_chars = saved_chars
        self._parent_kline_canonize_idx = saved_idx
        self._in_canonize_subscript = saved_in_canonize

    # ------------------------------------------------------------------
    # Binding integration (§10, Step 5)
    # ------------------------------------------------------------------

    def _resolve_char(self, char: str) -> str:
        """Resolve a single character via BindingScope.

        Returns the resolved word if bound, otherwise the raw character.
        When scope is None, always returns the raw character.
        """
        if self._scope is not None:
            word = self._scope.resolve(char)
            if word is not None:
                return word
        return char

    def _resolve_inline_or_scope(
        self,
        sig: str,
        inline_annotation: Annotation | None,
    ) -> str:
        """Resolve a signature: inline annotation first, then BindingScope.

        - If inline annotation present: extract word, trigger Rule B4
          override patching, return the word.
        - If single-char sig with no inline: resolve via BindingScope.
        - If multi-char sig: return as-is (MCS handles individual chars).
        """
        if inline_annotation is not None:
            word = self._extract_inline_word(sig, inline_annotation)
            self._patch_parent_canonize(sig, word)
            return word
        if len(sig) == 1:
            return self._resolve_char(sig)
        return sig  # multi-char: MCS decomposition handles individual chars

    def _resolve_nodes(
        self,
        node_ids: list[str],
        scope: OperatorScope,
    ) -> list[str]:
        """Resolve node identifiers to their bound or raw forms.

        Handles node_inline_annotation for the first node (if present).
        All other nodes are resolved via _resolve_char.
        """
        resolved: list[str] = []
        first_inline = scope.node_inline_annotation
        for i, nid in enumerate(node_ids):
            if i == 0 and first_inline is not None:
                word = self._extract_inline_word(nid, first_inline)
                self._patch_parent_canonize(nid, word)
                resolved.append(word)
            else:
                resolved.append(self._resolve_char(nid))
        return resolved

    # ------------------------------------------------------------------
    # Word extraction helpers
    # ------------------------------------------------------------------

    def _extract_inline_word(self, sig_char: str, annotation: Annotation) -> str:
        """Extract word from an inline annotation.

        Strips outer parentheses and prepends sig_char.
        E.g. "S" + "(ubject)" → "Subject".  Case preserved.
        """
        text = annotation.text
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]
        return sig_char + text

    def _extract_words(self, text: str) -> list[str]:
        """Extract word list from a block annotation.

        Strips outer parentheses and splits on whitespace.
        E.g. "(Mary had a little lamb)" → ["Mary", "had", "a", "little", "lamb"].
        Returns empty list for empty text.
        """
        t = text.strip()
        if t.startswith("("):
            t = t[1:]
        if t.endswith(")"):
            t = t[:-1]
        t = t.strip()
        if not t:
            return []
        return t.split()

    def _feed_annotation(self, annotation: Annotation) -> None:
        """Feed a block annotation's words into the BindingScope."""
        if self._scope is not None:
            words = self._extract_words(annotation.text)
            if words:
                self._scope.add_words(words)

    # ------------------------------------------------------------------
    # Rule B4 override patching
    # ------------------------------------------------------------------

    def _patch_parent_canonize(self, char: str, word: str) -> None:
        """Rule B4 — inline override: patch parent MCS CANONIZE entry.

        When an inline binding fires for char C with resolved word W inside
        a subscript, retroactively patch the matching character in the already-
        emitted MCS CANONIZE entry for the parent kline.

        Example:
            Source: SVO => Block([S(ubject) = M])
            Before: CANONIZE("SVO", ["S","V","O"]) at parent_kline_canonize_idx
            After:  CANONIZE("SVO", ["Subject","V","O"]) — S patched at index 0

        Only patches the immediate parent — no propagation beyond one level.
        If char is not found in parent kline chars, this is a safe no-op.
        """
        if self._parent_kline_chars is None or self._parent_kline_canonize_idx is None:
            return
        idx = self._parent_kline_chars.find(char)
        if idx < 0:
            return  # safe no-op — char not in parent kline
        entry = self.entries[self._parent_kline_canonize_idx]
        if entry.op != "CANONIZED":
            return
        if isinstance(entry.nodes, list) and idx < len(entry.nodes):
            new_nodes = list(entry.nodes)
            new_nodes[idx] = word
            self.entries[self._parent_kline_canonize_idx] = entry._replace(
                nodes=new_nodes,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _op_to_str(op: TokenType | None) -> str:
        """Convert a TokenType operator to its string name, or 'IDENTITY'."""
        if op is None:
            return "IDENTITY"
        _map = {
            TokenType.COUNTERSIGN: "COUNTERSIGNED",
            TokenType.CANONIZE: "CANONIZED",
            TokenType.CONNOTATE: "CONNOTED",
            TokenType.UNDERSIGN: "UNDERSIGNED",
        }
        return _map.get(op, "IDENTITY")
