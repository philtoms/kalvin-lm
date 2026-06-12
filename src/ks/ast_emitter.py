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
  - COUNTERSIGN (==):     {sig: [node]}, {node: [sig]} per item  — bidirectional
  - UNDERSIGN (=):        {node: [sig]} per item  — reversed direction
  - CONNOTATE (>):        {sig: [node]} per item  — forward direction
  - CANONIZE (=>):        {sig: [all_nodes]}  — aggregated single entry

  Self-identity (A = A) collapses to IDENTITY with empty nodes (spec §7.3).

**MCS expansion (spec §8):**
  Multi-character identifiers trigger automatic emission of:
  1. One IDENTITY entry per resolved constituent character.
  2. One CANONIZE entry mapping the compound to its resolved components.

  MCS applies to identifiers wherever they appear — signature side or node
  side, any operator.  Single-character identifiers do NOT trigger MCS.

**MCS deduplication (§8.3):**
  Only CANONIZE entries are deduplicated on (sig, nodes).  All other
  operator entries emit freely.  This prevents duplicate MCS CANONIZE
  entries when the same compound identifier appears in multiple contexts.

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
  - No general deduplication — only CANONIZE dedup per §8.3.

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
    ScopeItem,
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
        op:   One of "COUNTERSIGN", "CANONIZE", "CONNOTATE", "UNDERSIGN",
               "IDENTITY".
        component_labels: Resolved words per signature character (for NLP
               mode).  None when not applicable.
    """

    sig: str
    nodes: list[str]
    op: str  # COUNTERSIGN | CANONIZE | CONNOTATE | UNDERSIGN | IDENTITY
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

        # MCS dedup tracking for CANONIZE entries only (§8.3).
        # Maps (sig, tuple(nodes)) → index in self.entries.
        self._mcs_canonize_seen: dict[tuple[str, tuple[str, ...]], int] = {}

        # Rule B4 parent kline tracking — saved/restored on scope entry/exit.
        self._parent_kline_chars: str | None = None
        self._parent_kline_canonize_idx: int | None = None

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
            scope.sig.id, scope.inline_annotation,
        )

        # 2. MCS for signature
        mcs_idx = self._emit_mcs(scope.sig.id)

        # 3. Determine op
        op = self._op_to_str(scope.op)

        if op == "IDENTITY":
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
        if op == "COUNTERSIGN":
            # Per-item bidirectional
            for node in nodes:
                self._emit_entry(sig, [node], "COUNTERSIGN")
                self._emit_entry(node, [sig], "COUNTERSIGN")

        elif op == "UNDERSIGN":
            # Per-item reversed
            for node in nodes:
                if node == sig:
                    # Self-identity → IDENTITY with empty nodes (§7.3)
                    self._emit_entry(sig, [], "IDENTITY")
                else:
                    self._emit_entry(node, [sig], "UNDERSIGN")

        elif op == "CONNOTATE":
            # Per-item forward
            for node in nodes:
                self._emit_entry(sig, [node], "CONNOTATE")

        elif op == "CANONIZE":
            # Aggregated single entry
            self._emit_entry(sig, list(nodes), "CANONIZE")

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

        1. One IDENTITY entry per resolved constituent character.
        2. One CANONIZE entry mapping the compound to its resolved components.

        Returns the index of the CANONIZE entry (for Rule B4), or None
        if no MCS was emitted (single-char identifier).
        """
        if len(sig) <= 1:
            return None

        chars = [self._resolve_char(c) for c in sig]

        # Component identities
        for resolved_char in chars:
            self._emit_entry(resolved_char, [], "IDENTITY")

        # MCS canonization
        key = (sig, tuple(chars))
        if key in self._mcs_canonize_seen:
            # Already emitted — return existing index for Rule B4
            return self._mcs_canonize_seen[key]

        self._emit_entry(sig, list(chars), "CANONIZE")
        return len(self.entries) - 1

    # ------------------------------------------------------------------
    # Entry emission with CANONIZE dedup (§8.3, Step 4)
    # ------------------------------------------------------------------

    def _emit_entry(self, sig: str, nodes: list[str], op: str) -> None:
        """Emit a SymbolicEntry with CANONIZE deduplication.

        - CANONIZE entries: dedup on (sig, tuple(nodes)).  Duplicates
          are silently skipped.
        - All other ops: always emit (no dedup).
        """
        if op == "CANONIZE":
            key = (sig, tuple(nodes))
            if key in self._mcs_canonize_seen:
                return  # dedup — silently skip
            self._mcs_canonize_seen[key] = len(self.entries)

        self.entries.append(SymbolicEntry(sig=sig, nodes=nodes, op=op))

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
          Bare Signature items are nodes of the parent operator, already
          collected by _collect_node_ids and processed in _process_scope
          steps 4–7 — no action needed here.
        - Process child_block constructs.  Bare OperatorScope nodes
          (op=None) in a non-CANONIZE child_block are skipped — they
          are already collected as node identifiers by _collect_node_ids
          and used in the parent operator's emission.  Under CANONIZE,
          bare scopes in child_block still emit their own IDENTITY
          entries (they are subscript items with independent identity).
        """
        is_canonize = op == "CANONIZE"

        # Save parent kline tracking
        saved_chars = self._parent_kline_chars
        saved_idx = self._parent_kline_canonize_idx

        # Set parent kline tracking for Rule B4 (multi-char CANONIZE sigs)
        if is_canonize and mcs_idx is not None:
            self._parent_kline_chars = scope.sig.id
            self._parent_kline_canonize_idx = mcs_idx

        # CANONIZE scope push
        if is_canonize and self._scope is not None:
            self._scope.push_scope()

        # Process items
        for item in scope.items:
            if isinstance(item, OperatorScope):
                self._process_scope(item)
            elif isinstance(item, Annotation):
                self._feed_annotation(item)
            # Note: bare Signature items are nodes of the parent operator,
            # already collected by _collect_node_ids and processed in
            # _process_scope steps 4–7. No action needed here.

        # Process child_block constructs
        if scope.child_block is not None:
            for construct in scope.child_block.constructs:
                if isinstance(construct, OperatorScope) and construct.op is None and not is_canonize:
                    # Bare scope in non-CANONIZE child_block — already a node
                    # of the parent operator (collected by _collect_node_ids).
                    # Skip to avoid spurious IDENTITY emission.
                    continue
                self._process_construct(construct)

        # CANONIZE scope pop
        if is_canonize and self._scope is not None:
            self._scope.pop_scope()

        # Restore parent kline tracking
        self._parent_kline_chars = saved_chars
        self._parent_kline_canonize_idx = saved_idx

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
        if entry.op != "CANONIZE":
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
        _MAP = {
            TokenType.COUNTERSIGN: "COUNTERSIGN",
            TokenType.CANONIZE: "CANONIZE",
            TokenType.CONNOTATE: "CONNOTATE",
            TokenType.UNDERSIGN: "UNDERSIGN",
        }
        return _MAP.get(op, "IDENTITY")
