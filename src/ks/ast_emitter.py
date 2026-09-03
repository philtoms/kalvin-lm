"""AST Emitter for KScript v3 — walks scope-model AST and emits SymbolicEntry tuples.

Central compilation stage that transforms the KScript v3 scope-model AST
into a list of symbolic entries.  No token encoding happens here — all
values are strings.  The TokenEncoder (separate module) converts
SymbolicEntry tuples to encoded uint64 values.

**Scope processing rules:**
  Each OperatorScope is processed by resolving its signature, collecting
  node identifiers from items and child_block, and emitting operator-specific
  entries:

  - UNKNOWN (op=None):   {sig: []}   — bare unknown ask
  - COUNTERSIGNS (==):   {sig: [node]}, {node: [sig]} per item  — bidirectional
  - DENOTES (=):         {sig: [nodes]}  — forward direction
  - CONNOTES (>):        {sig+nodes: [nodes]}  — compound signature
  - RCONNOTES (<):       {sig+nodes: [sig]}   — compound signature, reversed
  - CANONIZES (=>):       {sig: [all_nodes]}  — aggregated single entry

  Self-identity (A = A) collapses to UNKNOWN with empty nodes.

**MTS expansion:**
  Multi-character all-uppercase identifiers (compounds: MHALL, SVO, ALL)
  trigger emission of exactly one CANONIZES entry mapping the compound to
  its resolved constituent characters. MTS emits only the canon — the
  characters are values inside the canon, not headed klines of their own.
  An author who wants a headed kline for a character writes it as a bare
  singleton.

  MTS applies to compounds wherever they appear — signature side or node
  side, any operator.  Single-character identifiers and lowercase/mixed-case
  words (had, did, all) do NOT trigger MTS — they are single-word tokens,
  not multi-token compounds. The case distinction is what separates a
  compound from a word, both admitted by the case-insensitive SIGNATURE
  rule.

**MTS deduplication:**
  CANONIZES entries are deduplicated on (sig, nodes).

**Word binding integration:**
  When a BindingScope is provided, single-character identifiers are resolved
  inline during the AST walk:

  1. Inline annotation first (Rule B4): S(ubject) → "Subject", immediate
     binding that bypasses the occurrence counter and retroactively patches
     the parent scope's MTS CANONIZES entry.
  2. BindingScope fallback (Rule B3): scope.resolve(char) walks the scope
     stack innermost-first with first-letter matching and occurrence counter.

  CANONIZES scope boundaries trigger push_scope/pop_scope on the BindingScope.
  Parent kline tracking is saved/restored for Rule B4 override patching.

  When scope is None, all binding logic is skipped.

**Key design constraints:**
  - nodes field is ALWAYS list[str] — never None, never a bare string,
    never singleton-unwrapped.  Singleton unwrapping happens in TokenEncoder.
  - No UNKNOWN op written — self-denote (A = A) emits UNKNOWN with empty nodes.
  - No general deduplication beyond CANONIZES dedup.
"""

from __future__ import annotations

from typing import NamedTuple
from collections import deque

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
        nodes: Always a list — empty for UNKNOWN, single-item for per-item
               operators, multi-item for CANONIZES aggregation.  Never None,
               never a bare string, never singleton-unwrapped.
        op:   One of "COUNTERSIGNS", "CANONIZES", "CONNOTES", "RCONNOTES",
               "DENOTES",
               "UNKNOWN".
        component_labels: Resolved words per signature character (for word
               mode).  None when not applicable.
    """

    sig: str
    nodes: list[str]
    op: str  # COUNTERSIGNS | CANONIZES | CONNOTES | DENOTES | IDENTITY | UNKNOWN
    component_labels: list[str] | None = None
    is_mts: bool = False  # True for MTS-produced entries (component
                          # identity + MTS canonization). The TokenEncoder
                          # combines this with its own BPE-MTS tag to
                          # push every MTS kline after compiled source.
    annotation: str = ""   # the owning scope's annotation text
    scope: int = 0         # nesting level; 0 at top level, +1 for MTS output
    is_ask: bool = False   # ASK_BPE_TOKEN bit: the sig is the original
                           # canonical signature; the bit marks the kline
                           # as an ask (TokenEncoder ORs it into the sig)


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

        # MTS dedup tracking.
        self._mts_canonize_seen: dict[tuple[str, tuple[str, ...]], tuple[int, bool]] = {}
        # Cached resolved components per identifier so the
        # BindingScope occurrence counter never re-advances for one.
        self._resolution_cache: dict[str, list[str]] = {}
        # Per-char queues of already-resolved node occurrences (innermost
        # scope's resolution is consumed first by child scope sigs), so each
        # textual occurrence resolves exactly once — the canon's operand
        # resolution is reused by the child kline, not re-resolved against
        # the occurrence counter.
        self._node_res_q: dict[str, deque] | None = None

        # Rule B4 parent kline tracking (saved/restored on scope entry/exit).
        self._parent_kline_chars: str | None = None
        self._parent_kline_canonize_idx: int | None = None

        # Set inside a single-char CANONIZES scope with recursive content
        # (subscript block); multi-char CANONIZES sigs trigger MTS (the
        # canon kline), so subscript identity filling is suppressed for them.
        self._in_canonize_subscript: bool = False

        # Scope/annotation tracking for KDbg.annotation and KDbg.scope.
        self._scope_annotation: str = ""
        self._pending_annotation: str = ""

    # Public API

    def emit(self, file: KScriptFile) -> list[SymbolicEntry]:
        """Walk a KScriptFile AST and return the list of SymbolicEntry tuples."""
        self._process_constructs(file.constructs)
        return self.entries

    # Construct dispatch

    def _process_constructs(self, constructs: list) -> None:
        """Dispatch constructs with one-step lookahead.

        An annotation binds to a following scope (its pending annotation);
        one not followed by a scope is a sigless ask.
        """
        for i, construct in enumerate(constructs):
            nxt = constructs[i + 1] if i + 1 < len(constructs) else None
            if isinstance(construct, OperatorScope):
                self._process_scope(construct)
            elif isinstance(construct, Annotation):
                self._pending_annotation = self._annotation_text(construct)
                self._feed_annotation(construct)
                if not isinstance(nxt, OperatorScope):
                    self._emit_ask(self._pending_annotation)
            elif isinstance(construct, Block):
                self._process_constructs(construct.constructs)

    def _emit_ask(self, text: str) -> None:
        """Emit a sigless annotation as an ask kline:
        ``ABC|ASK_BPE_TOKEN:[a big cat]``.

        The canonical signature is the annotation's word initials (one
        uppercased letter per word); the nodes are the words. The ASK bit
        marks it as an ask — any signature can be one.
        """
        words = self._extract_words(f"({text})")
        if not words:
            return
        sig = "".join(w[:1].upper() for w in words)
        saved = self._scope_annotation
        self._scope_annotation = text
        self._emit_entry(sig, words, "ASK", is_ask=True)
        self._scope_annotation = saved

    @staticmethod
    def _annotation_text(annotation: Annotation) -> str:
        """The annotation's text with surrounding parens stripped."""
        text = getattr(annotation, "text", "") or ""
        if len(text) >= 2 and text[0] == "(" and text[-1] == ")":
            return text[1:-1]
        return text

    # Core scope processing (Steps 2–3)

    def _process_scope(self, scope: OperatorScope) -> None:
        """Process a single OperatorScope: resolve sig, emit MTS, emit
        operator entries, then recurse into children.

        The scope's annotation is its own — the pending scope annotation, or
        its signature's inline annotation (resolved to the full bound word,
        e.g. "S" + "(ubject)" → "Subject") — and does **not** propagate to
        child scopes (each kline owns its own annotation). MTS spawned by the
        scope's signature inherits this scope's annotation.
        """
        if scope.inline_annotation is not None and len(scope.sig.id) == 1:
            annotation = self._extract_inline_word(scope.sig.id, scope.inline_annotation)
        else:
            annotation = self._pending_annotation
        self._pending_annotation = ""
        saved_annotation = self._scope_annotation
        self._scope_annotation = annotation
        try:
            self._process_scope_body(scope)
        finally:
            self._scope_annotation = saved_annotation

    def _process_scope_body(self, scope: OperatorScope) -> None:
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
        prev_len = len(self.entries)
        mts_idx = self._emit_mts(scope.sig.id)
        mts_created = len(self.entries) > prev_len
        op = self._op_to_str(scope.op)

        if op == "UNKNOWN":
            # For multi-char sigs _emit_mts already introduced the compound
            # via CANONIZES (mts_idx is not None) — a compound can't form an
            # identity. Single-char sigs refine by Word Binding:
            # word-bound → self-referential IDENTITY {S:[S]} (S1); unbound →
            # empty UNKNOWN {S:[]} (S4). Binding is the sole discriminator.
            if mts_idx is None:
                if sig_resolved != scope.sig.id:
                    self._emit_entry(sig_resolved, [sig_resolved], "IDENTITY")
                else:
                    self._emit_entry(sig_resolved, [], "UNKNOWN")
            else:
                # A bare compound is an ask. When this scope created the MTS
                # canon, it becomes the ask in place (leaving the dedup
                # registry — a later authored canon for the same compound is a
                # distinct relationship). When the canon is shared with an
                # earlier authored scope (dedup hit), it stands untouched and
                # the ask is a fresh entry with the canon's nodes. Either way
                # the ask keeps the compound's original canonical signature;
                # the ASK_BPE_TOKEN bit marks it as an ask.
                canon = self.entries[mts_idx]
                # The ask is an authored statement of THIS scope — it takes
                # the current scope's annotation and authored provenance,
                # not the cached MTS canon's.
                ask = canon._replace(
                    op="ASK", is_ask=True, is_mts=False, scope=0,
                    annotation=self._scope_annotation,
                )
                if mts_created:
                    self._mts_canonize_seen.pop((canon.sig, tuple(canon.nodes)), None)
                    self.entries[mts_idx] = ask
                else:
                    self.entries.append(ask)
            return

        node_ids = self._collect_node_ids(scope)

        # A CANONIZES scope introduces a subscript scope: push it BEFORE
        # expanding node MTS and resolving operands, so a node-compound's
        # chars (e.g. SVO's S,V,O) resolve against the subscript's bindings
        # (S->Subject, V->Verb, O->Object) rather than the outer scope where
        # they are unbound. BindingScope.push_scope resets parent counters too,
        # so duplicate chars (the two Ls in ALL => A=L L=L against 'Mary had
        # a Little Lamb') resolve to their distinct words. _compile_children
        # pops this scope after walking the block.
        is_canonize = op == "CANONIZES"
        pushed_scope = False
        if is_canonize and self._scope is not None:
            self._scope.push_scope()
            pushed_scope = True

        for nid in node_ids:
            if len(nid) > 1:
                self._emit_mts(nid)

        # A CANONIZES scope's nodes are its declared block operands — always.
        # The signature's MTS character-expansion is a separate decoding-aid
        # kline (emitted by _emit_mts above); it coexists with the block canon
        # and is never the canon's node-list. The two share a signature but are
        # distinct relationships: signature:block (the canon) and signature:MTS
        # (the decoding aid). Do not let the MTS cache override the block.
        #
        # Rule B4 parent-kline tracking must be active DURING _resolve_nodes,
        # because inline annotations on the block operands (e.g. S(ubject) in
        # `SVO => S(ubject) = M`) fire _patch_parent_canonize here, and the
        # MTS CANONIZES entry they must patch is the compound's own (kept
        # intact under the two-entry scheme). Setting it here (before
        # _resolve_nodes and _compile_children) and restoring after ensures
        # both operand resolution and the child walk see the correct parent.
        saved_chars = self._parent_kline_chars
        saved_idx = self._parent_kline_canonize_idx
        if is_canonize and mts_idx is not None:
            self._parent_kline_chars = scope.sig.id
            self._parent_kline_canonize_idx = mts_idx

        resolved_nodes = self._resolve_nodes(node_ids, scope)

        saved_q = self._node_res_q
        q: dict[str, deque] = {}
        for nid, word in zip(node_ids, resolved_nodes):
            if len(nid) == 1 and word != nid:
                q.setdefault(nid, deque()).append(word)
        self._node_res_q = q

        self._emit_operator_entries(sig_resolved, resolved_nodes, op)
        self._compile_children(scope, op, mts_idx, pushed_scope=pushed_scope)

        self._parent_kline_chars = saved_chars
        self._parent_kline_canonize_idx = saved_idx
        self._node_res_q = saved_q

    # Operator emission (Step 2)

    def _emit_operator_entries(
        self,
        sig: str,
        nodes: list[str],
        op: str,
    ) -> None:
        """Emit operator-specific entries based on the operator type."""
        if op == "COUNTERSIGNS":
            for node in nodes:
                self._emit_entry(sig, [node], "COUNTERSIGNS")
                self._emit_entry(node, [sig], "COUNTERSIGNS")

        elif op == "DENOTES":
            if nodes == [sig]:
                # Self-denote → self-referential IDENTITY {S:[S]}.
                # Binding-independent: once the author writes the
                # self-reference, the structure is fixed at S1.
                self._emit_entry(sig, [sig], "IDENTITY")
            elif nodes:
                self._emit_entry(sig, list(nodes), "DENOTES")

        elif op == "CONNOTES":
            if nodes == [sig]:
                self._emit_entry(sig, [sig], "IDENTITY")
            elif nodes:
                self._emit_entry(sig + "".join(nodes), list(nodes), "CONNOTES")

        elif op == "RCONNOTES":
            if nodes == [sig]:
                self._emit_entry(sig, [sig], "IDENTITY")
            elif nodes:
                # Reversed reading: node first, sig second (A < B reads
                # "B is a kind of A", so the identifier is BA).
                self._emit_entry("".join(nodes) + sig, [sig], "CONNOTES")

        elif op == "CANONIZES":
            # A compound-headed CANONIZES scope produces TWO distinct
            # relationships that share one signature:
            #   1. MTS CANONIZES  — compound → its declared character
            #      components (the decoding aid; already emitted by
            #      _emit_mts when the sig was expanded). This entry DEFINES
            #      the compound's signature: the encoder computes it as
            #      signature_of over these character components.
            #   2. Block canon    — compound → the block's resolved operands
            #      (the script's declared signature↔nodes relationship).
            # The block canon is emitted here as a SEPARATE CANONIZES entry
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
            # (sig, nodes) and CANONIZES dedup collapses them to one —
            # preserving the prior single-entry behaviour.
            #
            # Rule B4 inline-override patching is unaffected: it patches the
            # MTS entry directly via _parent_kline_canonize_idx, which the
            # MTS entry retains (it is not replaced here).
            if nodes:
                self._emit_entry(sig, list(nodes), "CANONIZES")

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

    # MTS expansion

    def _emit_mts(self, sig: str) -> int | None:
        """Emit the MTS canon entry for a multi-character identifier.

        MTS emits exactly one CANONIZES entry mapping the compound to its
        resolved constituent characters (one node per character, preserving
        repeats). MTS emits no per-character component entries: the characters
        are values inside the canon, not headed klines. An author who wants a
        headed kline for a character writes it as a bare singleton.

        CANONIZES deduplication: the same (sig, nodes) pair is silently
        skipped.

        A compound signature is the OR-reduction of multiple token IDs.

        Returns the index of the CANONIZES entry (for Rule B4), or None
        if no MTS was emitted (single-char or non-uppercase identifier).
        """
        # MTS character-decomposition applies only to all-uppercase
        # multi-character identifiers (compounds: MHALL, SVO, ALL). A
        # lowercase/mixed-case multi-char identifier is a single word
        # (had, did, all) admitted by the case-insensitive SIGNATURE rule;
        # decomposing it by character would be wrong.
        if len(sig) <= 1 or not sig.isupper():
            return None

        # Resolve once on first expansion; reuse the cached list so
        # the identifier resolves identically as node or signature. MTS emits
        # only the canon kline — its nodes are the resolved characters (one
        # per character, preserving repeats). MTS no longer emits per-character
        # component entries: components are values inside the canon, not headed
        # klines. An author who wants a
        # headed kline for a character writes it as a bare singleton.
        if sig in self._resolution_cache:
            chars = list(self._resolution_cache[sig])
        else:
            # MTS is a decoding aid — its char resolution must not consume
            # the occurrence counters that belong to the identity occurrences
            # emitted later as item klines (e.g. the two Ls in ALL => L > M(od)
            # / L > O must resolve to little and lamb respectively).
            snap = self._scope.counters_snapshot() if self._scope is not None else None
            chars = [self._resolve_char(c) for c in sig]
            if snap is not None:
                self._scope.counters_restore(snap)
            self._resolution_cache[sig] = list(chars)

        key = (sig, tuple(chars))
        hit = self._mts_canonize_seen.get(key)
        if hit is not None:
            return hit[0]  # already emitted
        self._emit_entry(sig, list(chars), "CANONIZES", is_mts=True)
        idx = len(self.entries) - 1
        # A word-bound token is the script's own word: emit its identity
        # (X:[X]) alongside the canon. Unbound raw chars stay unheaded —
        # an unannotated character is an ask, not a known word.
        for c, word in zip(sig, chars):
            if word == c:
                continue
            if any(
                e.sig == word and e.op in ("IDENTITY", "UNKNOWN")
                for e in self.entries
            ):
                continue
            self._emit_entry(word, [word], "IDENTITY", is_mts=True)
        return idx

    # Entry emission with CANONIZES dedup

    def _emit_entry(self, sig: str, nodes: list[str], op: str, *, is_mts: bool = False, is_ask: bool = False) -> None:
        """Emit a SymbolicEntry.

        CANONIZES dedup applies only to MTS expansion (one decoding aid per
        compound): authored CANONIZES entries always emit — a repeated
        authored canon is a temporally distinct encounter (a second ask),
        identical in content. Bucket order is the temporal axis; K does not
        utilise it yet.

        ``is_mts`` marks entries produced by MTS expansion (component
        identity + MTS canonization) so the TokenEncoder can push them
        after compiled source in the final output.  Operator-produced
        entries, subscript identities, and single-char CANONIZES scopes
        carry the default (source).
        """
        if op == "CANONIZES":
            # Temporal distinctness: authored authored repeats both emit (a
            # second ask). Skip only when either side is MTS — an authored
            # subscript canon and its MTS twin are one compile artifact.
            key = (sig, tuple(nodes))
            hit = self._mts_canonize_seen.get(key)
            if hit is not None and (is_mts or hit[1]):
                if is_mts:
                    return
                return
            self._mts_canonize_seen[key] = (len(self.entries), is_mts)

        self.entries.append(SymbolicEntry(
            sig=sig, nodes=nodes, op=op, is_mts=is_mts, is_ask=is_ask,
            annotation=self._scope_annotation,
            scope=1 if is_mts else 0,
        ))

    # Identity emission for CANONIZES subscript blocks

    def _emit_identity_if_needed(self, raw_id: str) -> None:
        """Emit a component entry for ``raw_id`` if none exists.

        Used in CANONIZES subscript blocks to ensure every identifier appears
        as the signature of at least one emitted entry. Applies the binding-
        aware rule: word-bound → self-referential IDENTITY {w:[w]};
        unbound → empty UNKNOWN {w:[]}.

        Dedup checks (in order):
          1. Existing CANONIZES entry — sig is a compound already introduced
             by its CANONIZES entry from MTS.
          2. Existing IDENTITY or UNKNOWN entries — sig already has one.

        This prevents duplicate entries when the identifier already appears
        as the signature of an IDENTITY/UNKNOWN entry.  The CANONIZES check
        blocks compounds (which cannot form an identity) without affecting
        single-char sigs that have only DENOTES entries.
        """
        resolved = self._resolve_char(raw_id)
        if any(e.sig == resolved and e.op == "CANONIZES" for e in self.entries):
            return  # compound already introduced by its CANONIZES entry
        if any(e.sig == resolved and e.op in ("IDENTITY", "UNKNOWN") for e in self.entries):
            return
        if resolved != raw_id:
            self._emit_entry(resolved, [resolved], "IDENTITY")
        else:
            self._emit_entry(resolved, [], "UNKNOWN")

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

        CANONIZES scopes push/pop the BindingScope and save/restore parent
        kline tracking (Rule B4). Bare OperatorScope nodes (op=None) in a
        non-CANONIZES child_block are skipped — already collected as node
        identifiers by _collect_node_ids; under CANONIZES they still emit
        their own UNKNOWN (independent subscript identity).

        **CANONIZES subscript identity:**

        A CANONIZES scope with recursive content forms a "subscript block"
        where every identifier must appear as the signature of at least
        one emitted entry; identity UNKNOWN fills any gap. Activated only
        when the CANONIZES sig did NOT trigger MTS (mts_idx is None) —
        multi-char sigs trigger MTS (the canon kline), so subscript identity
        is suppressed for them.

        _emit_identity_if_needed is applied to leaf Signature items (no
        operator entry). Not needed for CANONIZES/COUNTERSIGNS/CONNOTES/
        DENOTES scope sigs (all produce entries with the scope's sig)
        nor bare op=None scopes (emit UNKNOWN in _process_scope). The flag does not propagate between CANONIZES scopes.
        """
        is_canonize = op == "CANONIZES"

        # Parent-kline tracking (Rule B4) is now set in _process_scope so it
        # is active during operand resolution; this method only walks children
        # and pops the subscript scope. _in_canonize_subscript still needs
        # save/restore (it is subscript-local and does not propagate).
        saved_in_canonize = self._in_canonize_subscript

        # The subscript scope was pushed in _process_scope (before operand
        # resolution); this method only walks children and pops it.

        # Activate subscript identity for a single-char CANONIZES sig
        # (mts_idx is None) with recursive content.
        if is_canonize:
            has_recursive_content = scope.child_block is not None or any(
                isinstance(item, OperatorScope) for item in scope.items
            )
            if has_recursive_content and mts_idx is None:
                self._in_canonize_subscript = True

        for item in scope.items:
            if isinstance(item, OperatorScope):
                self._process_scope(item)
            elif isinstance(item, Annotation):
                self._feed_annotation(item)
            # Bare Signature items in subscript blocks need identity
            # (they produce no operator entry).
            elif isinstance(item, Signature):
                if self._in_canonize_subscript:
                    self._emit_identity_if_needed(item.id)

        if scope.child_block is not None:
            for construct in scope.child_block.constructs:
                if (
                    isinstance(construct, OperatorScope)
                    and construct.op is None
                    and not is_canonize
                ):
                    # Bare node in non-CANONIZES child_block — already
                    # collected by _collect_node_ids; skip to avoid a
                    # spurious UNKNOWN.
                    continue
                # Bare scopes (op=None) emit UNKNOWN in _process_scope.
                self._process_constructs([construct])

        if pushed_scope and self._scope is not None:
            self._scope.pop_scope()

        self._in_canonize_subscript = saved_in_canonize

    # Binding integration

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
            q = self._node_res_q
            if q is not None and q.get(sig):
                return q[sig].popleft()
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
        """Pre-register this scope's direct inline (item) bindings.

        Walks ``scope``'s own items for ``Signature`` items carrying an
        ``inline_annotation`` and binds each via :meth:`bind_override` in the
        current scope. Called before the scope's MTS so the signature's
        char-expansion — which runs before children are processed — resolves
        each char to its inline word (inline binds tighter than word-list).

        Only direct items are pre-registered. Inline annotations on items of
        *nested* scopes (in ``child_block``) are registered by those scopes'
        own processing (after their subscript scope is pushed, via
        :meth:`_resolve_nodes`), so they bind at the correct scope level.
        Descending into ``child_block`` here would register a nested item's
        override in the *parent* scope, leaking it across scope boundaries —
        e.g. a nested ``DH = h(ad)`` would force ``H → had`` everywhere instead
        of only within ``DH``'s own MTS, stealing the binding from a header
        word (``H → have``) that the parent compound's MTS should resolve.
        """
        if self._scope is None:
            return

        for item in scope.items:
            if isinstance(item, Signature) and item.inline_annotation is not None:
                if len(item.id) == 1:
                    word = self._extract_inline_word(item.id, item.inline_annotation)
                    self._scope.bind_override(item.id, word)

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
                # A nested scope's node is its head; the head's sig-side
                # inline annotation (W > Q(uery)) binds that node to the word.
                anns.append(item.inline_annotation)
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
        """Rule B4 — inline override: patch parent MTS CANONIZES entry.

        When an inline binding fires for char C with resolved word W inside
        a subscript, retroactively patch the matching character in the already-
        emitted MTS CANONIZES entry for the parent kline.

        Example:
            Source: SVO => Block([S(ubject) = M])
            Before: CANONIZES("SVO", ["S","V","O"]) at parent_kline_canonize_idx
            After:  CANONIZES("SVO", ["Subject","V","O"]) — S patched at index 0

        Only patches the immediate parent — no propagation beyond one level.
        If char is not found in parent kline chars, this is a safe no-op.
        """
        if self._parent_kline_chars is None or self._parent_kline_canonize_idx is None:
            return
        idx = self._parent_kline_chars.find(char)
        if idx < 0:
            return  # no-op — char not in parent kline
        entry = self.entries[self._parent_kline_canonize_idx]
        if entry.op != "CANONIZES":
            return
        if isinstance(entry.nodes, list) and idx < len(entry.nodes):
            new_nodes = list(entry.nodes)
            new_nodes[idx] = word
            self.entries[self._parent_kline_canonize_idx] = entry._replace(
                nodes=new_nodes,
            )
            # Re-key the CANONIZES dedup registry: the patched entry's
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
                    self._parent_kline_canonize_idx, True
                )

    # Helpers

    @staticmethod
    def _op_to_str(op: TokenType | None) -> str:
        """Convert a TokenType operator to its string name, or 'UNKNOWN'."""
        if op is None:
            return "UNKNOWN"
        _map = {
            TokenType.COUNTERSIGNS: "COUNTERSIGNS",
            TokenType.CANONIZES: "CANONIZES",
            TokenType.CONNOTES: "CONNOTES",
            TokenType.RCONNOTES: "RCONNOTES",
            TokenType.DENOTES: "DENOTES",
        }
        return _map.get(op, "UNKNOWN")
