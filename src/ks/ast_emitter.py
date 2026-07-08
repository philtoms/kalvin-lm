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

**MTS expansion (spec §8):**
  Multi-character all-uppercase identifiers (compounds: MHALL, SVO, ALL)
  trigger automatic emission of:
  1. One IDENTITY entry per resolved constituent character.
  2. One CANONIZE entry mapping the compound to its resolved components.

  MTS applies to compounds wherever they appear — signature side or node
  side, any operator.  Single-character identifiers and lowercase/mixed-case
  words (had, did, all) do NOT trigger MTS — they are single-word tokens,
  not multi-token compounds. The case distinction is what separates a
  compound from a word, both admitted by the case-insensitive SIGNATURE
  rule (§2).

**MTS deduplication (§8.3):**
  CANONIZE entries are deduplicated on (sig, nodes).  Component IDENTITY
  entries are deduplicated across MTS calls via _mts_identity_seen (a
  character emitted once is never emitted again).  Intra-expansion dedup
  prevents duplicate chars within a single compound (e.g., MHALL's second L).

**Word binding integration (spec §10):**
  When a BindingScope is provided, single-character identifiers are resolved
  inline during the AST walk:

  1. Inline annotation first (Rule B4): S(ubject) → "Subject", immediate
     binding that bypasses the occurrence counter and retroactively patches
     the parent scope's MTS CANONIZE entry.
  2. BindingScope fallback (Rule B3): scope.resolve(char) walks the scope
     stack innermost-first with first-letter matching and occurrence counter.

  CANONIZE scope boundaries trigger push_scope/pop_scope on the BindingScope.
  Parent kline tracking is saved/restored for Rule B4 override patching.

  When scope is None, all binding logic is skipped.

**Key design constraints:**
  - nodes field is ALWAYS list[str] — never None, never a bare string,
    never singleton-unwrapped.  Singleton unwrapping happens in TokenEncoder.
  - No IDENTITY op — self-identity emits IDENTITY with empty nodes.
  - No general deduplication beyond MTS — CANONIZE dedup per §8.3,
    plus component IDENTITY dedup.

Spec references: §3 (Scope Model), §6 (Entry Model), §7 (Operator Rules),
§8 (MTS Expansion), §10 (Word Binding Resolution).
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


class SymbolicEntry(NamedTuple):
    """A symbolic (not yet tokenized) compilation entry.

    Attributes:
        sig:  The signature identifier string (possibly a resolved word).
        nodes: Always a list — empty for IDENTITY, single-item for per-item
               operators, multi-item for CANONIZE aggregation.  Never None,
               never a bare string, never singleton-unwrapped.
        op:   One of "COUNTERSIGNED", "CANONIZED", "CONNOTED", "UNDERSIGNED",
               "IDENTITY".
        component_labels: Resolved words per signature character (for word
               mode).  None when not applicable.
    """

    sig: str
    nodes: list[str]
    op: str  # COUNTERSIGNED | CANONIZED | CONNOTED | UNDERSIGNED | IDENTITY
    component_labels: list[str] | None = None
    is_mts: bool = False  # True for §8 MTS-produced entries (component
                          # identity + MTS canonization). The TokenEncoder
                          # combines this with its own §11.3 BPE-MTS tag to
                          # push every MTS kline after compiled source.


class ASTEmitter:
    """Walks a KScript v3 scope-model AST and emits SymbolicEntry tuples.

    Args:
        scope: Optional BindingScope for word binding resolution.
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

        # MTS dedup tracking (§8.3).
        self._mts_canonize_seen: dict[tuple[str, tuple[str, ...]], int] = {}
        self._mts_identity_seen: set[str] = set()
        # Cached resolved components per identifier (§8.3) so the
        # BindingScope occurrence counter never re-advances for one.
        self._resolution_cache: dict[str, list[str]] = {}

        # Rule B4 parent kline tracking (saved/restored on scope entry/exit).
        self._parent_kline_chars: str | None = None
        self._parent_kline_canonize_idx: int | None = None

        # Set inside a single-char CANONIZE scope with recursive content
        # (subscript block); multi-char CANONIZE sigs get component identities
        # from MTS, so subscript identity is unnecessary.
        self._in_canonize_subscript: bool = False

    # Public API

    def emit(self, file: KScriptFile) -> list[SymbolicEntry]:
        """Walk a KScriptFile AST and return the list of SymbolicEntry tuples."""
        for construct in file.constructs:
            self._process_construct(construct)
        return self.entries

    # Construct dispatch

    def _process_construct(self, construct: ConstructItem) -> None:
        """Dispatch a top-level construct to the appropriate handler."""
        if isinstance(construct, OperatorScope):
            self._process_scope(construct)
        elif isinstance(construct, Annotation):
            self._feed_annotation(construct)
        elif isinstance(construct, Block):
            for c in construct.constructs:
                self._process_construct(c)

    # Core scope processing (Steps 2–3)

    def _process_scope(self, scope: OperatorScope) -> None:
        """Process a single OperatorScope: resolve sig, emit MTS, emit
        operator entries, then recurse into children."""
        sig_resolved = self._resolve_inline_or_scope(
            scope.sig.id,
            scope.inline_annotation,
        )
        # Pre-register this scope's inline (item) bindings before MTS, so the
        # signature's MTS char-expansion resolves each char to its inline word
        # (inline binds tighter than any looser word-list binding). Without
        # this, MTS runs before the children are processed and resolves chars
        # against the word list, producing a competing token for a char that
        # an inline annotation has already bound (Word Binding regression).
        self._register_inline_overrides(scope)
        mts_idx = self._emit_mts(scope.sig.id)
        op = self._op_to_str(scope.op)

        if op == "IDENTITY":
            # For multi-char sigs _emit_mts already introduced the compound
            # via CANONIZE (mts_idx is not None) — a compound can't form an
            # identity (§8). Single-char sigs get a bare IDENTITY here.
            if mts_idx is None:
                self._emit_entry(sig_resolved, [], "IDENTITY")
            return

        node_ids = self._collect_node_ids(scope)

        # A CANONIZE scope introduces a subscript scope (§3): push it BEFORE
        # expanding node MTS and resolving operands, so a node-compound's
        # chars (e.g. SVO's S,V,O) resolve against the subscript's bindings
        # (S->Subject, V->Verb, O->Object) rather than the outer scope where
        # they are unbound. BindingScope.push_scope resets parent counters too,
        # so duplicate chars (the two Ls in ALL => A=L L=L against 'Mary had
        # a Little Lamb') resolve to their distinct words. _compile_children
        # pops this scope after walking the block.
        is_canonize = op == "CANONIZED"
        pushed_scope = False
        if is_canonize and self._scope is not None:
            self._scope.push_scope()
            pushed_scope = True

        for nid in node_ids:
            if len(nid) > 1:
                self._emit_mts(nid)

        # A CANONIZE scope's nodes are its declared block operands — always.
        # The signature's MTS character-expansion is a separate decoding-aid
        # kline (emitted by _emit_mts above); it coexists with the block canon
        # and is never the canon's node-list. The two share a signature but are
        # distinct relationships: signature:block (the canon) and signature:MTS
        # (the decoding aid). Do not let the MTS cache override the block.
        #
        # Rule B4 parent-kline tracking must be active DURING _resolve_nodes,
        # because inline annotations on the block operands (e.g. S(ubject) in
        # `SVO => S(ubject) = M`) fire _patch_parent_canonize here, and the
        # MTS CANONIZE entry they must patch is the compound's own (kept
        # intact under the two-entry scheme). Setting it here (before
        # _resolve_nodes and _compile_children) and restoring after ensures
        # both operand resolution and the child walk see the correct parent.
        saved_chars = self._parent_kline_chars
        saved_idx = self._parent_kline_canonize_idx
        if is_canonize and mts_idx is not None:
            self._parent_kline_chars = scope.sig.id
            self._parent_kline_canonize_idx = mts_idx

        resolved_nodes = self._resolve_nodes(node_ids, scope)

        self._emit_operator_entries(sig_resolved, resolved_nodes, op)
        self._compile_children(scope, op, mts_idx, pushed_scope=pushed_scope)

        self._parent_kline_chars = saved_chars
        self._parent_kline_canonize_idx = saved_idx

    # Operator emission (Step 2)

    def _emit_operator_entries(
        self,
        sig: str,
        nodes: list[str],
        op: str,
    ) -> None:
        """Emit operator-specific entries based on the operator type."""
        if op == "COUNTERSIGNED":
            for node in nodes:
                self._emit_entry(sig, [node], "COUNTERSIGNED")
                self._emit_entry(node, [sig], "COUNTERSIGNED")

        elif op == "UNDERSIGNED":
            for node in nodes:
                if node == sig:
                    # Self-identity → IDENTITY with empty nodes (§7.3)
                    self._emit_entry(sig, [], "IDENTITY")
                else:
                    self._emit_entry(node, [sig], "UNDERSIGNED")

        elif op == "CONNOTED":
            for node in nodes:
                self._emit_entry(sig, [node], "CONNOTED")

        elif op == "CANONIZED":
            # A compound-headed CANONIZE scope produces TWO distinct
            # relationships that share one signature (spec §11.4):
            #   1. MTS CANONIZE  — compound → its declared character
            #      components (the decoding aid; already emitted by
            #      _emit_mts when the sig was expanded). This entry DEFINES
            #      the compound's signature: the encoder computes it as
            #      make_signature over these character components.
            #   2. Block canon    — compound → the block's resolved operands
            #      (the script's declared signature↔nodes relationship).
            # The block canon is emitted here as a SEPARATE CANONIZED entry
            # that reuses the compound's signature (a reference, not a
            # re-definition).
            #
            # These must not be conflated: when the block operands differ
            # from the compound's characters (a deliberate misfit, e.g.
            # `WDMH => M H W` omits D), the compound's signature is still the
            # OR of ALL its characters (W,D,M,H) so the block canon composes
            # into a misfit — the whole point of the script. Patching the MTS
            # entry's nodes with the block operands would drop the missing
            # character from the signature and collapse the misfit into a
            # full canon with the wrong signature.
            #
            # When the block operands equal the character components (the
            # common case, e.g. `SVO => S V O`), both entries share the same
            # (sig, nodes) and §8.3 CANONIZE dedup collapses them to one —
            # preserving the prior single-entry behaviour.
            #
            # Rule B4 inline-override patching is unaffected: it patches the
            # MTS entry directly via _parent_kline_canonize_idx, which the
            # MTS entry retains (it is not replaced here).
            self._emit_entry(sig, list(nodes), "CANONIZED")

    # Node collection (Step 2)

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
        # Annotation items are skipped (not nodes).

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

    # MTS expansion (§8)

    def _emit_mts(self, sig: str) -> int | None:
        """Emit MTS entries for a multi-character identifier.

        1. One IDENTITY entry per resolved constituent character (deduped).
        2. One CANONIZE entry mapping the compound to its resolved components.

        Component IDENTITY deduplication (§8.3 extended):
          - Intra-expansion: duplicate chars within a compound (e.g., MHALL's
            second L) emit only one IDENTITY L.
          - Inter-expansion: if a char was already emitted by a previous MTS
            call, it is silently skipped.

        CANONIZE deduplication (§8.3):
          - Same (sig, nodes) pair is silently skipped.

        No IDENTITY entry is emitted for the compound itself.  A compound
        signature is the OR-reduction of multiple token IDs and cannot form
        an identity (spec §8; CONTEXT.md "Identity" glossary).

        Returns the index of the CANONIZE entry (for Rule B4), or None
        if no MTS was emitted (single-char or non-uppercase identifier).
        """
        # MTS character-decomposition applies only to all-uppercase
        # multi-character identifiers (compounds: MHALL, SVO, ALL). A
        # lowercase/mixed-case multi-char identifier is a single word
        # (had, did, all) admitted by the case-insensitive SIGNATURE rule;
        # decomposing it by character would be wrong. (§8)
        if len(sig) <= 1 or not sig.isupper():
            return None

        # Resolve once on first expansion (§8.3); reuse the cached list so
        # the identifier resolves identically as node or signature.
        if sig in self._resolution_cache:
            chars = list(self._resolution_cache[sig])
        else:
            chars = [self._resolve_char(c) for c in sig]
            self._resolution_cache[sig] = list(chars)

        seen_in_this_call: set[str] = set()
        for resolved_char in chars:
            if resolved_char in seen_in_this_call:
                continue  # intra-expansion dedup (e.g., second L in MHALL)
            seen_in_this_call.add(resolved_char)
            if resolved_char in self._mts_identity_seen:
                continue  # inter-expansion dedup
            self._mts_identity_seen.add(resolved_char)
            self._emit_entry(resolved_char, [], "IDENTITY", is_mts=True)

        key = (sig, tuple(chars))
        if key in self._mts_canonize_seen:
            return self._mts_canonize_seen[key]  # already emitted

        self._emit_entry(sig, list(chars), "CANONIZED", is_mts=True)
        return len(self.entries) - 1

    # Entry emission with CANONIZE dedup (§8.3, Step 4)

    def _emit_entry(self, sig: str, nodes: list[str], op: str, *, is_mts: bool = False) -> None:
        """Emit a SymbolicEntry with CANONIZE deduplication.

        - CANONIZE entries: dedup on (sig, tuple(nodes)).  Duplicates
          are silently skipped.
        - All other ops: always emit (no dedup at this level).

        ``is_mts`` marks entries produced by §8 MTS expansion (component
        identity + MTS canonization) so the TokenEncoder can push them
        after compiled source in the final output.  Operator-produced
        entries, subscript identities, and single-char CANONIZE scopes
        carry the default (source).

        Note: IDENTITY dedup for MTS components is handled in _emit_mts
        via _mts_identity_seen, not here.
        """
        if op == "CANONIZED":
            key = (sig, tuple(nodes))
            if key in self._mts_canonize_seen:
                return
            self._mts_canonize_seen[key] = len(self.entries)

        self.entries.append(SymbolicEntry(sig=sig, nodes=nodes, op=op, is_mts=is_mts))

    # Identity emission for CANONIZE subscript blocks

    def _emit_identity_if_needed(self, sig: str) -> None:
        """Emit identity IDENTITY only if no IDENTITY entry for this sig exists.

        Used in CANONIZE subscript blocks to ensure every identifier appears
        as the signature of at least one emitted entry.

        Dedup checks (in order):
          1. _mts_identity_seen — sig was already emitted as MTS component.
          2. Existing CANONIZE entry — sig is a compound already introduced
             by its CANONIZE entry from MTS.
          3. Existing IDENTITY entries — sig already has an IDENTITY entry.

        This prevents duplicate IDENTITY when MTS expansion already provided
        one for the same identifier, or when the identifier already appears
        as the signature of an IDENTITY entry.  The CANONIZE check blocks
        compounds (which cannot form an identity) without affecting single-char
        sigs that have only UNDERSIGN entries (e.g., D in §14.8).
        """
        if sig in self._mts_identity_seen:
            return
        if any(e.sig == sig and e.op == "CANONIZED" for e in self.entries):
            return  # compound already introduced by its CANONIZE entry
        if any(e.sig == sig and e.op == "IDENTITY" for e in self.entries):
            return
        self._emit_entry(sig, [], "IDENTITY")

    # Scope walk and child compilation (Step 3)

    def _compile_children(
        self,
        scope: OperatorScope,
        op: str,
        mts_idx: int | None,
        *,
        pushed_scope: bool = False,
    ) -> None:
        """After emitting operator entries, recursively process children.

        CANONIZE scopes push/pop the BindingScope and save/restore parent
        kline tracking (Rule B4). Bare OperatorScope nodes (op=None) in a
        non-CANONIZE child_block are skipped — already collected as node
        identifiers by _collect_node_ids; under CANONIZE they still emit
        their own IDENTITY (independent subscript identity).

        **CANONIZE subscript identity (§7.6, §14.8, §14.9):**

        A CANONIZE scope with recursive content forms a "subscript block"
        where every identifier must appear as the signature of at least
        one emitted entry; identity IDENTITY fills any gap. Activated only
        when the CANONIZE sig did NOT trigger MTS (mts_idx is None) —
        multi-char sigs get component identities from MTS, so subscript
        identity would produce spurious entries (e.g. IDENTITY D in §14.11).

        _emit_identity_if_needed is applied to leaf Signature items (no
        operator entry) and to UNDERSIGN scope sigs (their entries use nodes
        as sigs, so the scope's own sig lacks identity). Not needed for
        CANONIZE/COUNTERSIGN/CONNOTATE scope sigs (already produce entries
        with the scope's sig) nor bare op=None scopes (emit IDENTITY in
        _process_scope). The flag does not propagate between CANONIZE scopes.
        """
        is_canonize = op == "CANONIZED"

        # Parent-kline tracking (Rule B4) is now set in _process_scope so it
        # is active during operand resolution; this method only walks children
        # and pops the subscript scope. _in_canonize_subscript still needs
        # save/restore (it is subscript-local and does not propagate).
        saved_in_canonize = self._in_canonize_subscript

        # The subscript scope was pushed in _process_scope (before operand
        # resolution); this method only walks children and pops it.

        # Activate subscript identity for a single-char CANONIZE sig
        # (mts_idx is None) with recursive content.
        if is_canonize:
            has_recursive_content = scope.child_block is not None or any(
                isinstance(item, OperatorScope) for item in scope.items
            )
            if has_recursive_content and mts_idx is None:
                self._in_canonize_subscript = True

        for item in scope.items:
            if isinstance(item, OperatorScope):
                # UNDERSIGN scope sigs in subscript blocks need identity
                # (their entries use nodes as sigs, not the scope's own sig).
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
            # Bare Signature items in subscript blocks need identity
            # (they produce no operator entry).
            elif isinstance(item, Signature):
                if self._in_canonize_subscript:
                    resolved = self._resolve_char(item.id)
                    self._emit_identity_if_needed(resolved)

        if scope.child_block is not None:
            for construct in scope.child_block.constructs:
                if (
                    isinstance(construct, OperatorScope)
                    and construct.op is None
                    and not is_canonize
                ):
                    # Bare node in non-CANONIZE child_block — already
                    # collected by _collect_node_ids; skip to avoid a
                    # spurious IDENTITY.
                    continue
                # UNDERSIGN scope sigs in subscript child_blocks need
                # identity; bare scopes (op=None) emit IDENTITY in
                # _process_scope.
                if (
                    self._in_canonize_subscript
                    and isinstance(construct, OperatorScope)
                    and construct.op is not None
                    and self._op_to_str(construct.op) == "UNDERSIGNED"
                ):
                    resolved = self._resolve_char(construct.sig.id)
                    self._emit_identity_if_needed(resolved)
                self._process_construct(construct)

        if pushed_scope and self._scope is not None:
            self._scope.pop_scope()

        self._in_canonize_subscript = saved_in_canonize

    # Binding integration (§10, Step 5)

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
        """Resolve a scope signature.

        Word Binding (top-level): a signature-prefix annotation binds
        fill-if-empty — it takes effect only when the character is currently
        unbound in the scope. If ``sig`` is already bound (e.g. H bound to
        'had' by an outer scope), the annotation is inert and the existing
        binding stands. This guarantees each identity is bound once, with no
        competing token.

        - If ``inline_annotation`` present AND ``sig`` is a single unbound char:
          extract word, trigger Rule B4 patching, return the word.
        - Else if single-char sig: resolve via BindingScope (may be unbound → raw char).
        - If multi-char sig: return as-is (MTS handles individual chars).
        """
        if inline_annotation is not None and len(sig) == 1:
            existing = self._scope.resolve(sig) if self._scope is not None else None
            if existing is None:
                word = self._extract_inline_word(sig, inline_annotation)
                self._patch_parent_canonize(sig, word)
                # Register the inline binding so it overrides any looser
                # word-list binding for this char everywhere it is resolved
                # (MTS char expansion, identity emission — not only here).
                if self._scope is not None:
                    self._scope.bind_override(sig, word)
                return word
            return existing  # already bound — top-level annotation is inert
        if len(sig) == 1:
            return self._resolve_char(sig)
        return sig

    def _resolve_nodes(
        self,
        node_ids: list[str],
        scope: OperatorScope,
    ) -> list[str]:
        """Resolve node identifiers to their bound or raw forms.

        Word Binding (inline): an inline annotation on a Signature item binds
        unconditionally to that item, overriding any outer binding. The
        per-item annotations come from the Signature items in ``scope.items``
        (and the subscript block), matched to ``node_ids`` in order; any node
        without an inline annotation resolves via ``_resolve_char``.
        """
        # Build a per-position inline-annotation map from the scope's Signature
        # items, in collection order, aligned with node_ids.
        inline_by_pos = self._collect_item_inline_annotations(scope)
        resolved: list[str] = []
        for i, nid in enumerate(node_ids):
            ann = inline_by_pos[i] if i < len(inline_by_pos) else None
            if ann is not None:
                word = self._extract_inline_word(nid, ann)
                self._patch_parent_canonize(nid, word)
                # Register the inline binding (see _resolve_inline_or_scope):
                # the inline word overrides any looser binding for this char
                # everywhere, so MTS/identity emission sees the same token.
                if self._scope is not None:
                    self._scope.bind_override(nid, word)
                resolved.append(word)
            else:
                resolved.append(self._resolve_char(nid))
        return resolved

    def _register_inline_overrides(self, scope: OperatorScope) -> None:
        """Pre-register this scope subtree's inline (item) bindings.

        Walks ``scope``'s items and child_block (one level into nested
        OperatorScopes' items) for ``Signature`` items carrying an
        ``inline_annotation``, and binds each via :meth:`bind_override` in the
        current scope. Called before the scope's MTS so the signature's
        char-expansion — which runs before children are processed — resolves
        each char to its inline word (inline binds tighter than word-list).
        """
        if self._scope is None:
            return

        def _bind_from_items(items) -> None:
            for item in items:
                if isinstance(item, Signature) and item.inline_annotation is not None:
                    if len(item.id) == 1:
                        word = self._extract_inline_word(item.id, item.inline_annotation)
                        self._scope.bind_override(item.id, word)

        _bind_from_items(scope.items)
        if scope.child_block is not None:
            for construct in scope.child_block.constructs:
                if isinstance(construct, OperatorScope):
                    _bind_from_items(construct.items)

    def _collect_item_inline_annotations(
        self, scope: OperatorScope
    ) -> list[Annotation | None]:
        """Inline annotations on this scope's Signature items, in node order.

        Mirrors ``_collect_node_ids``: walks items (Signature items) and the
        subscript child_block, returning each item's ``inline_annotation``
        (or None) aligned to the collected node positions.
        """
        anns: list[Annotation | None] = []
        for item in scope.items:
            if isinstance(item, Signature):
                anns.append(item.inline_annotation)
            elif isinstance(item, OperatorScope):
                anns.append(None)  # nested scope — no per-item inline here
        if scope.child_block is not None:
            for construct in scope.child_block.constructs:
                self._collect_block_item_inline_annotations(construct, anns)
        return anns

    def _collect_block_item_inline_annotations(
        self,
        construct: ConstructItem,
        anns: list[Annotation | None],
    ) -> None:
        if isinstance(construct, OperatorScope):
            anns.append(construct.inline_annotation)
        elif isinstance(construct, Block):
            for c in construct.constructs:
                self._collect_block_item_inline_annotations(c, anns)

    # Word extraction helpers

    def _extract_inline_word(self, sig_char: str, annotation: Annotation) -> str:
        """Extract word from an inline annotation.

        Strips outer parentheses and prepends sig_char, preserving case.
        E.g. "S" + "(ubject)" → "Subject".
        """
        text = annotation.text
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]
        return sig_char + text

    def _extract_words(self, text: str) -> list[str]:
        """Extract word list from a block annotation.

        Strips outer parentheses and splits on whitespace; empty for empty
        text. E.g. "(Mary had a little lamb)" → ["Mary","had","a","little","lamb"].
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

    # Rule B4 override patching

    def _patch_parent_canonize(self, char: str, word: str) -> None:
        """Rule B4 — inline override: patch parent MTS CANONIZE entry.

        When an inline binding fires for char C with resolved word W inside
        a subscript, retroactively patch the matching character in the already-
        emitted MTS CANONIZE entry for the parent kline.

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
            return  # no-op — char not in parent kline
        entry = self.entries[self._parent_kline_canonize_idx]
        if entry.op != "CANONIZED":
            return
        if isinstance(entry.nodes, list) and idx < len(entry.nodes):
            new_nodes = list(entry.nodes)
            new_nodes[idx] = word
            self.entries[self._parent_kline_canonize_idx] = entry._replace(
                nodes=new_nodes,
            )
            # Re-key the CANONIZE dedup registry (§8.3): the patched entry's
            # (sig, nodes) changed, so the stale key under which it was
            # registered no longer matches. Without this, a block-canon entry
            # whose operands equal the PATCHED component list (the common case,
            # e.g. `SVO => S(ubject) V O`) would fail to dedup against it and
            # emit a spurious duplicate. Remove the old key and register the
            # new one, pointing at the same entry index.
            if entry.is_mts:
                old_key = (entry.sig, tuple(entry.nodes))
                self._mts_canonize_seen.pop(old_key, None)
                self._mts_canonize_seen[(entry.sig, tuple(new_nodes))] = (
                    self._parent_kline_canonize_idx
                )

    # Helpers

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
