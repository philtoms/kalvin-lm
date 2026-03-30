"""Compiler for KScript with eager emit and two-step compilation.

Key design:
- Eager emit: emit CompiledEntry immediately when construct is completed
- Two-step compilation per construct:
  Step 1: Collect CLNs, emit MCS (if applicable), emit main entry, handle BWD
  Step 2: Process subscripts recursively
- BWD operators use CLNs from current construct (S2=ALL, S3=CLNs[-1])
- Duplicates allowed (by design for graph completeness)
"""

from kalvin.abstract import KLine, KNodes, KSig
from kalvin.mod_tokenizer import Mod32Tokenizer, ModTokenizer, PACKED_BIT
from kalvin.significance import Int32Significance

from .parser import (
    Construct,
    KScriptFile,
    Literal,
    Node,
    Script,
    Signature,
    ConstructType,
)

# Significance instance for encoding
_sig = Int32Significance()


class CompiledEntry(KLine):
    """A single compiled KLine entry.

    Nodes semantics:
    - None: identity entry (sig exists with no children)
    - int: single token ID link (countersign, undersign)
    - list[int]: nodes list (connotate, canonize)
    """

    def __init__(self, signature: KSig, nodes: KNodes, dbg_text: str = ""):
        super().__init__(signature=signature, nodes=nodes, dbg_text=dbg_text)

    @classmethod
    def add_significance(cls, token_id: int, sig_level: str) -> int:
        """Add significance bits to token ID based on sig level.

        S1 (countersign): bit 63
        S2 (canonize): bit 55
        S3 (connotate): bit 32
        S4 (undersign/identity): no bits
        """
        if sig_level == "S1":
            return token_id | _sig.S1
        elif sig_level == "S2":
            return token_id | _sig.S2
        elif sig_level == "S3":
            return token_id | _sig.S3
        else:  # S4 or unknown
            return token_id

    @classmethod
    def encode(
        cls,
        sig: str,
        nodes: str | None | list[str],
        tokenizer: ModTokenizer,
        *,
        sig_level: str = "S4",
        dbg_text: str = ""
    ) -> "CompiledEntry":
        """Encode string signature/nodes to token IDs.

        Signatures (uppercase strings) are packed (PACKED_BIT set).
        Literals are unpacked: each char encoded as (ord(char) << 1).
        """
        sig_id = tokenizer.encode(sig, pack=True)[0]
        sig_id = cls.add_significance(sig_id, sig_level)

        if nodes is None:
            return cls(signature=sig_id, nodes=None, dbg_text=dbg_text)
        elif isinstance(nodes, str):
            if nodes.isupper() and nodes.isalpha():
                node_id = tokenizer.encode(nodes, pack=True)[0]
                return cls(signature=sig_id, nodes=node_id, dbg_text=dbg_text)
            else:
                node_ids = [ord(c) << 1 for c in nodes]
                return cls(signature=sig_id, nodes=node_ids, dbg_text=dbg_text)
        else:
            all_node_ids: list[int] = []
            for n in nodes:
                if n.isupper() and n.isalpha():
                    all_node_ids.append(tokenizer.encode(n, pack=True)[0])
                else:
                    all_node_ids.extend(ord(c) << 1 for c in n)
            return cls(signature=sig_id, nodes=all_node_ids, dbg_text=dbg_text)

    def decode(self, tokenizer: ModTokenizer) -> tuple[str, str | None | list[str]]:
        """Decode token IDs back to strings."""
        sig_token = self.signature & _sig.TOKEN_MASK
        sig = tokenizer.decode([sig_token], pack=None)

        if self._nodes is None:
            return sig, None
        elif isinstance(self._nodes, int):
            decoded = tokenizer.decode([self._nodes], pack=None)
            return sig, decoded
        else:
            all_unpacked = all((n & PACKED_BIT) == 0 for n in self._nodes)

            if all_unpacked:
                decoded = "".join(chr(n >> 1) for n in self._nodes)
                return sig, decoded
            else:
                decoded_nodes: list[str] = []
                literal_chars: list[int] = []

                for n in self._nodes:
                    if (n & PACKED_BIT) == 0:
                        literal_chars.append(n)
                    else:
                        if literal_chars:
                            decoded_nodes.append("".join(chr(c >> 1) for c in literal_chars))
                            literal_chars = []
                        decoded_nodes.append(tokenizer.decode([n], pack=None))

                if literal_chars:
                    decoded_nodes.append("".join(chr(c >> 1) for c in literal_chars))

                return sig, decoded_nodes


class Compiler:
    """Compiles KScript AST to CompiledEntry list.

    Two-step compilation per construct:
    Step 1: Collect sig and CLNs, emit MCS, emit main entry, handle BWD
    Step 2: Process subscripts recursively

    Eager emit - no buffering, emit immediately.
    """

    def __init__(self, tokenizer: ModTokenizer | None = None, dev: bool = False):
        self.entries: list[CompiledEntry] = []
        self.tokenizer = tokenizer or Mod32Tokenizer()
        self.dev = dev

    def _format_dbg_text(self, sig: str, nodes: str | None | list[str]) -> str:
        """Format textual representation of an entry for debugging.

        Args:
            sig: Signature string
            nodes: Nodes as string, None, or list of strings

        Returns:
            Human-readable string like "A: B" or "ABC: [A, B, C]"
        """
        if nodes is None:
            return f"{sig}: None"
        elif isinstance(nodes, str):
            return f"{sig}: {nodes}"
        else:
            return f"{sig}: {nodes}"

    def compile(self, file: KScriptFile) -> list[CompiledEntry]:
        """Compile a KScriptFile to entries."""
        for script in file.scripts:
            self._compile_script(script)
        return self.entries

    def _compile_script(self, script: Script) -> None:
        """Compile a script and all its constructs."""
        for construct in script.constructs:
            self._compile_construct(construct)

    def _compile_construct(self, construct: Construct) -> None:
        """Compile a single construct in two steps.

        Step 1: Emit MCS, emit main entry, handle BWD
        Step 2: Process subscripts recursively
        """
        # === STEP 1: Emit ===

        # MCS expansion for signature (if multi-char)
        self._emit_mcs(construct.sig.id)

        # Emit main construct entry
        self._emit_construct_entry(construct)

        # Handle BWD if present
        if construct.bwd:
            bwd_sig, bwd_op, bwd_clns = construct.bwd
            bwd_level = "S2" if bwd_op == "<=" else "S3"
            cln_strings = [n.id for n in bwd_clns]
            dbg = self._format_dbg_text(bwd_sig.id, cln_strings) if self.dev else ""
            self.entries.append(
                CompiledEntry.encode(bwd_sig.id, cln_strings, self.tokenizer, sig_level=bwd_level, dbg_text=dbg)
            )
            # MCS for BWD signature
            self._emit_mcs(bwd_sig.id)

        # === STEP 2: Process subscripts recursively ===
        for subscript in construct.subscripts:
            self._compile_construct(subscript)

    def _emit_construct_entry(self, construct: Construct) -> None:
        """Emit the main entry for a construct based on its operator type."""
        sig = construct.sig
        op = construct.op
        cln_strings = [n.id for n in construct.clns]

        if op == "identity" or op == "":
            # Identity: {sig: None}
            dbg = self._format_dbg_text(sig.id, None) if self.dev else ""
            self.entries.append(
                CompiledEntry.encode(sig.id, None, self.tokenizer, sig_level="S4", dbg_text=dbg)
            )

        elif op == "==":
            # Countersign: {sig: node} AND {node: sig}
            for cln in cln_strings:
                dbg = self._format_dbg_text(sig.id, cln) if self.dev else ""
                self.entries.append(
                    CompiledEntry.encode(sig.id, cln, self.tokenizer, sig_level="S1", dbg_text=dbg)
                )
                # Reverse entry if node is signature
                if cln.isupper() and cln.isalpha():
                    dbg_rev = self._format_dbg_text(cln, sig.id) if self.dev else ""
                    self.entries.append(
                        CompiledEntry.encode(cln, sig.id, self.tokenizer, sig_level="S1", dbg_text=dbg_rev)
                    )
                    self._emit_mcs(cln)

        elif op == "=>":
            # Canonize fwd: {sig: [nodes...]}
            dbg = self._format_dbg_text(sig.id, cln_strings) if self.dev else ""
            self.entries.append(
                CompiledEntry.encode(sig.id, cln_strings, self.tokenizer, sig_level="S2", dbg_text=dbg)
            )
            # Entity emissions for CLN signatures
            for cln in cln_strings:
                if cln.isupper() and cln.isalpha():
                    self._emit_entity(cln)

        elif op == ">":
            # Connotate fwd: {sig: [node]} AND {node: None}
            for cln in cln_strings:
                dbg = self._format_dbg_text(sig.id, [cln]) if self.dev else ""
                self.entries.append(
                    CompiledEntry.encode(sig.id, [cln], self.tokenizer, sig_level="S3", dbg_text=dbg)
                )
                self._emit_entity(cln)

        elif op == "=":
            # Undersign: {sig: node} AND {node: None}
            for cln in cln_strings:
                dbg = self._format_dbg_text(sig.id, cln) if self.dev else ""
                self.entries.append(
                    CompiledEntry.encode(sig.id, cln, self.tokenizer, sig_level="S4", dbg_text=dbg)
                )
                self._emit_entity(cln)

        elif op == "<=":
            # Canonize bwd: emitted separately via construct.bwd
            pass

        elif op == "<":
            # Connotate bwd: emitted separately via construct.bwd
            pass

    def _emit_mcs(self, sig: str) -> bool:
        """Emit MCS canonization and identity entries for multi-char signatures.

        Emits:
        - {sig: [component chars]} with S2 significance
        - {component: None} for each char with S4 significance

        Returns:
            True if MCS was emitted, False otherwise
        """
        if len(sig) <= 1:
            return False

        # MCS canonization: {sig: [A, B, C, ...]}
        component_chars = list(sig)
        dbg = self._format_dbg_text(sig, component_chars) if self.dev else ""
        self.entries.append(
            CompiledEntry.encode(sig, component_chars, self.tokenizer, sig_level="S2", dbg_text=dbg)
        )

        # Component identities
        for char in component_chars:
            self._emit_entity(char)

        return True

    def _emit_entity(self, sig: str) -> None:
        """Emit entity entry for a signature (identity with S4)."""
        dbg = self._format_dbg_text(sig, None) if self.dev else ""
        self.entries.append(
            CompiledEntry.encode(sig, None, self.tokenizer, sig_level="S4", dbg_text=dbg)
        )

    def _node_to_string(self, node: Node) -> str:
        """Convert a Node to its string representation."""
        if isinstance(node, Signature):
            return node.id
        elif isinstance(node, Literal):
            return node.id
        else:
            return str(node)


# Convenience function
def compile_source(source: str, tokenizer: ModTokenizer | None = None, dev: bool = False) -> list[CompiledEntry]:
    """Compile KScript source string to entries."""
    from .lexer import Lexer
    from .parser import Parser

    tokens = Lexer(source).tokenize()
    kscript_file = Parser(tokens).parse()
    return Compiler(tokenizer, dev=dev).compile(kscript_file)
