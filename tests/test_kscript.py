"""Tests for KScript compiler - rebuilt from spec."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from kalvin.abstract import KLine
from kalvin.mod_tokenizer import Mod64Tokenizer
from kalvin.significance import Int32Significance
from kscript import KScript
from kscript.compiler import Compiler, CompiledEntry
from kscript.decompiler import Decompiler
from kscript.lexer import Lexer
from kscript.parser import Parser

# Shared significance instance for tests
_sig = Int32Significance()

# Shared tokenizer for tests
_tokenizer = Mod64Tokenizer()


def compile_source(source: str) -> list[CompiledEntry]:
    """Helper to compile source string to entries."""
    tokens = Lexer(source).tokenize()
    kscript_file = Parser(tokens).parse()
    return Compiler(_tokenizer).compile(kscript_file)


def entries_to_dict(entries: list[CompiledEntry]) -> dict[str, list[str] | str | None]:
    """Convert entries to dict for easier comparison (decoded)."""
    result = {}
    for e in entries:
        sig, nodes = e.decode(_tokenizer)
        result[sig] = nodes
    return result


# =============================================================================
# 1. Token Module Tests
# =============================================================================

class TestTokenTypes:
    """Tests for token types."""

    def test_token_type_count(self) -> None:
        """Test all 15 token types defined."""
        from kscript.token import TokenType
        assert len(TokenType) == 15

    def test_token_creation(self) -> None:
        """Test Token dataclass creation."""
        from kscript.token import Token, TokenType
        token = Token(TokenType.SIGNATURE, "A", 1, 1)
        assert token.type == TokenType.SIGNATURE
        assert token.value == "A"
        assert token.line == 1
        assert token.column == 1

    def test_token_frozen(self) -> None:
        """Test Token is frozen (immutable)."""
        from kscript.token import Token, TokenType
        token = Token(TokenType.SIGNATURE, "A", 1, 1)
        try:
            token.value = "B"  # type: ignore
            assert False, "Token should be frozen"
        except AttributeError:
            pass


# =============================================================================
# 2. Lexer Module Tests
# =============================================================================

class TestLexer:
    """Tests for the lexer."""

    def test_tokenize_signature(self) -> None:
        """Test tokenizing a simple signature."""
        from kscript.token import TokenType
        tokens = Lexer("A").tokenize()
        assert len(tokens) == 2  # SIGNATURE + EOF
        assert tokens[0].type == TokenType.SIGNATURE
        assert tokens[0].value == "A"

    def test_tokenize_multi_char_signature(self) -> None:
        """Test tokenizing multi-character signature."""
        from kscript.token import TokenType
        tokens = Lexer("MHALL").tokenize()
        assert tokens[0].type == TokenType.SIGNATURE
        assert tokens[0].value == "MHALL"

    def test_tokenize_countersign(self) -> None:
        """Test tokenizing countersign operator."""
        from kscript.token import TokenType
        tokens = Lexer("==").tokenize()
        assert tokens[0].type == TokenType.COUNTERSIGN
        assert tokens[0].value == "=="

    def test_tokenize_canonize_fwd(self) -> None:
        """Test tokenizing forward canonize."""
        from kscript.token import TokenType
        tokens = Lexer("=>").tokenize()
        assert tokens[0].type == TokenType.CANONIZE_FWD

    def test_tokenize_canonize_bwd(self) -> None:
        """Test tokenizing backward canonize."""
        from kscript.token import TokenType
        tokens = Lexer("<=").tokenize()
        assert tokens[0].type == TokenType.CANONIZE_BWD

    def test_tokenize_connotate_fwd(self) -> None:
        """Test tokenizing forward connotate."""
        from kscript.token import TokenType
        tokens = Lexer(">").tokenize()
        assert tokens[0].type == TokenType.CONNOTATE_FWD

    def test_tokenize_connotate_bwd(self) -> None:
        """Test tokenizing backward connotate."""
        from kscript.token import TokenType
        tokens = Lexer("<").tokenize()
        assert tokens[0].type == TokenType.CONNOTATE_BWD

    def test_tokenize_undersign(self) -> None:
        """Test tokenizing undersign."""
        from kscript.token import TokenType
        tokens = Lexer("=").tokenize()
        assert tokens[0].type == TokenType.UNDERSIGN

    def test_tokenize_string(self) -> None:
        """Test tokenizing string literals."""
        from kscript.token import TokenType
        tokens = Lexer('"hello"').tokenize()
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == '"hello"'

    def test_tokenize_string_with_escape(self) -> None:
        """Test tokenizing string with escape."""
        from kscript.token import TokenType
        tokens = Lexer(r'"\"hello\""').tokenize()
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == r'"\"hello\""'

    def test_tokenize_number(self) -> None:
        """Test tokenizing number literals."""
        from kscript.token import TokenType
        tokens = Lexer("42").tokenize()
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "42"

    def test_tokenize_comment(self) -> None:
        """Test tokenizing comments."""
        from kscript.token import TokenType
        tokens = Lexer("(this is a comment)").tokenize()
        assert tokens[0].type == TokenType.COMMENT

    def test_tokenize_nested_comment(self) -> None:
        """Test nested comments (greedy match)."""
        from kscript.token import TokenType
        tokens = Lexer("(comment (nested))").tokenize()
        assert tokens[0].type == TokenType.COMMENT
        assert tokens[0].value == "(comment (nested))"

    def test_tokenize_unterminated_comment(self) -> None:
        """Test unterminated comment (greedy to EOL)."""
        from kscript.token import TokenType
        tokens = Lexer("(no closing").tokenize()
        assert tokens[0].type == TokenType.COMMENT

    def test_tokenize_empty_comment(self) -> None:
        """Test empty comment."""
        from kscript.token import TokenType
        tokens = Lexer("()").tokenize()
        assert tokens[0].type == TokenType.COMMENT
        assert tokens[0].value == "()"

    def test_tokenize_indentation(self) -> None:
        """Test tokenizing indentation."""
        from kscript.token import TokenType
        source = "A\n  B"
        tokens = Lexer(source).tokenize()
        assert any(t.type == TokenType.INDENT for t in tokens)
        assert any(t.type == TokenType.DEDENT for t in tokens)

    def test_tokenize_string_literal_lowercase(self) -> None:
        """Test tokenizing lowercase identifier as STRING_LITERAL."""
        from kscript.token import TokenType
        tokens = Lexer("zed").tokenize()
        assert tokens[0].type == TokenType.STRING_LITERAL
        assert tokens[0].value == "zed"

    def test_tokenize_string_literal_mixed_case(self) -> None:
        """Test tokenizing mixed case identifier as STRING_LITERAL."""
        from kscript.token import TokenType
        tokens = Lexer("Hello").tokenize()
        assert tokens[0].type == TokenType.STRING_LITERAL
        assert tokens[0].value == "Hello"

    def test_tokenize_string_literal_alphanumeric(self) -> None:
        """Test tokenizing alphanumeric identifier as STRING_LITERAL."""
        from kscript.token import TokenType
        tokens = Lexer("item123").tokenize()
        assert tokens[0].type == TokenType.STRING_LITERAL
        assert tokens[0].value == "item123"

    def test_tokenize_uppercase_remains_signature(self) -> None:
        """Test that uppercase-only identifier remains SIGNATURE."""
        from kscript.token import TokenType
        tokens = Lexer("FOO").tokenize()
        assert tokens[0].type == TokenType.SIGNATURE
        assert tokens[0].value == "FOO"

    def test_tokenize_numeric_start_is_number(self) -> None:
        """Test that numeric-start identifier is NUMBER."""
        from kscript.token import TokenType
        tokens = Lexer("123abc").tokenize()
        # Numbers consume digits, so this is NUMBER "123" followed by STRING_LITERAL "abc"
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "123"
        assert tokens[1].type == TokenType.STRING_LITERAL
        assert tokens[1].value == "abc"


class TestLexerIndentation:
    """Tests for lexer indentation handling."""

    def test_indent_dedent(self) -> None:
        """Test INDENT and DEDENT tokens."""
        from kscript.token import TokenType
        source = "A\n  B\nC"
        tokens = Lexer(source).tokenize()
        types = [t.type for t in tokens]
        assert TokenType.INDENT in types
        assert TokenType.DEDENT in types

    def test_multiple_indents(self) -> None:
        """Test multiple indentation levels."""
        from kscript.token import TokenType
        source = "A\n  B\n    C"
        tokens = Lexer(source).tokenize()
        types = [t.type for t in tokens]
        assert types.count(TokenType.INDENT) == 2

    def test_dedent_multiple_levels(self) -> None:
        """Test dedenting multiple levels at once."""
        from kscript.token import TokenType
        source = "A\n  B\n    C\nD"
        tokens = Lexer(source).tokenize()
        types = [t.type for t in tokens]
        assert types.count(TokenType.DEDENT) == 2


# =============================================================================
# 3. AST Module Tests
# =============================================================================

class TestAST:
    """Tests for AST nodes."""

    def test_signature_dataclass(self) -> None:
        """Test Signature dataclass."""
        from kscript.ast import Signature
        sig = Signature("A", None, 1, 1)
        assert sig.id == "A"
        assert sig.comment is None

    def test_string_literal_dataclass(self) -> None:
        """Test StringLiteral dataclass."""
        from kscript.ast import StringLiteral
        lit = StringLiteral('"hello"', 1, 1)
        assert lit.id == '"hello"'

    def test_number_literal_dataclass(self) -> None:
        """Test NumberLiteral dataclass."""
        from kscript.ast import NumberLiteral
        lit = NumberLiteral("42", 1, 1)
        assert lit.id == "42"

    def test_construct_type_enum(self) -> None:
        """Test ConstructType enum."""
        from kscript.ast import ConstructType
        assert len(ConstructType) == 6
        assert ConstructType.COUNTERSIGN.value == "=="

    def test_construct_dataclass(self) -> None:
        """Test Construct dataclass."""
        from kscript.ast import Construct, ConstructType, Signature
        sig = Signature("B", None, 1, 3)
        construct = Construct(ConstructType.COUNTERSIGN, [sig], 1)
        assert construct.type == ConstructType.COUNTERSIGN
        assert len(construct.nodes) == 1
        assert construct.has_leading_nodes is False


# =============================================================================
# 4. Parser Module Tests
# =============================================================================

class TestParser:
    """Tests for the parser."""

    def test_parse_identity_script(self) -> None:
        """Test parsing an identity script."""
        tokens = Lexer("A").tokenize()
        kscript = Parser(tokens).parse()
        assert len(kscript.scripts) == 1
        assert kscript.scripts[0].signature.id == "A"
        assert len(kscript.scripts[0].constructs) == 0

    def test_parse_countersign(self) -> None:
        """Test parsing a countersign construct."""
        tokens = Lexer("A == B").tokenize()
        kscript = Parser(tokens).parse()
        assert len(kscript.scripts) == 1
        script = kscript.scripts[0]
        assert len(script.constructs) == 1
        construct = script.constructs[0]
        assert construct.type.name == "COUNTERSIGN"
        assert len(construct.nodes) == 1

    def test_parse_multiple_scripts(self) -> None:
        """Test parsing multiple scripts."""
        tokens = Lexer("A\nB").tokenize()
        kscript = Parser(tokens).parse()
        assert len(kscript.scripts) == 2

    def test_parse_canonize_multi(self) -> None:
        """Test parsing multi-node canonize."""
        tokens = Lexer("A => B C D").tokenize()
        kscript = Parser(tokens).parse()
        script = kscript.scripts[0]
        assert script.constructs[0].type.name == "CANONIZE_FWD"
        assert len(script.constructs[0].nodes) == 3

    def test_parse_subscript(self) -> None:
        """Test parsing subscripts."""
        source = "A =>\n  B\n  C"
        tokens = Lexer(source).tokenize()
        kscript = Parser(tokens).parse()
        script = kscript.scripts[0]
        assert len(script.subscripts) == 2

    def test_parse_chained_constructs(self) -> None:
        """Test parsing chained constructs."""
        tokens = Lexer("A => B => C").tokenize()
        kscript = Parser(tokens).parse()
        script = kscript.scripts[0]
        assert len(script.constructs) == 2


class TestParserSubscripts:
    """Tests for parser subscript handling."""

    def test_nested_subscripts(self) -> None:
        """Test parsing nested subscripts."""
        source = "A =>\n  B =>\n    C"
        tokens = Lexer(source).tokenize()
        kscript = Parser(tokens).parse()
        script = kscript.scripts[0]
        assert len(script.subscripts) == 1
        assert len(script.subscripts[0].subscripts) == 1



    def test_parse_nested_subscripts(self) -> None:
        """Test parsing nested subscripts."""
        source = "A =>\n  B =>\n    C"
        tokens = Lexer(source).tokenize()
        kscript = Parser(tokens).parse()
        script = kscript.scripts[0]
        assert len(script.subscripts) == 1
        assert len(script.subscripts[0].subscripts) == 1


class TestParserStringLiterals:
    """Tests for parsing string literal nodes."""

    def test_parse_string_literal_node(self) -> None:
        """Test parsing a string literal as a node."""
        from kscript.ast import StringLiteral, Signature
        tokens = Lexer("A => Y zed").tokenize()
        kscript = Parser(tokens).parse()
        script = kscript.scripts[0]
        construct = script.constructs[0]
        assert len(construct.nodes) == 2
        # Y is SIGNATURE, zed is STRING_LITERAL
        assert isinstance(construct.nodes[0], Signature)
        assert isinstance(construct.nodes[1], StringLiteral)
        assert construct.nodes[0].id == "Y"
        assert construct.nodes[1].id == "zed"

    def test_parse_mixed_string_literal_nodes(self) -> None:
        """Test parsing mixed string literal and signature nodes."""
        from kscript.ast import StringLiteral, Signature
        tokens = Lexer("A => foo bar BAZ").tokenize()
        kscript = Parser(tokens).parse()
        script = kscript.scripts[0]
        construct = script.constructs[0]
        assert len(construct.nodes) == 3
        # foo is STRING_LITERAL, bar is STRING_LITERAL, BAZ is SIGNATURE
        assert isinstance(construct.nodes[0], StringLiteral)
        assert isinstance(construct.nodes[1], StringLiteral)
        assert isinstance(construct.nodes[2], Signature)
        assert construct.nodes[0].id == "foo"
        assert construct.nodes[1].id == "bar"
        assert construct.nodes[2].id == "BAZ"

    def test_parse_string_literal_in_countersign(self) -> None:
        """Test parsing string literal in countersign."""
        from kscript.ast import StringLiteral
        tokens = Lexer("A == hello").tokenize()
        kscript = Parser(tokens).parse()
        script = kscript.scripts[0]
        construct = script.constructs[0]
        assert len(construct.nodes) == 1
        assert isinstance(construct.nodes[0], StringLiteral)
        assert construct.nodes[0].id == "hello"

    def test_parse_backward_canonize_with_string_literal(self) -> None:
        """Test backward canonize with string literal as child node.

        Note: Scripts must start with SIGNATURE, so we test A <= zed
        where A is the signature and zed is the string literal child.
        """
        from kscript.ast import StringLiteral
        tokens = Lexer("A <= zed").tokenize()
        kscript = Parser(tokens).parse()
        script = kscript.scripts[0]
        assert len(script.constructs) == 1
        construct = script.constructs[0]
        assert construct.type.name == "CANONIZE_BWD"
        # zed is STRING_LITERAL (the parent in backward canonize)
        assert len(construct.nodes) == 1
        assert isinstance(construct.nodes[0], StringLiteral)
        assert construct.nodes[0].id == "zed"


# =============================================================================
# 5. Compiler Module Tests
# =============================================================================

class TestCompiler:
    """Tests for the compiler."""

    def test_compile_identity(self) -> None:
        """Test compiling identity script."""
        entries = compile_source("A")
        assert len(entries) == 1
        sig, nodes = entries[0].decode(_tokenizer)
        assert sig == "A"
        assert nodes is None

    def test_compile_countersign(self) -> None:
        """Test compiling countersign (bidirectional)."""
        entries = compile_source("A == B")
        d = entries_to_dict(entries)
        assert d["A"] == "B"
        assert d["B"] == "A"

    def test_compile_undersign(self) -> None:
        """Test compiling undersign."""
        entries = compile_source("A = B")
        assert len(entries) == 2
        sig1, nodes1 = entries[0].decode(_tokenizer)
        sig2, nodes2 = entries[1].decode(_tokenizer)
        assert sig1 == "A" and nodes1 == "B"
        assert sig2 == "B" and nodes2 is None

    def test_compile_connotate_fwd(self) -> None:
        """Test compiling forward connotate."""
        entries = compile_source("A > B")
        assert len(entries) == 2
        sig1, nodes1 = entries[0].decode(_tokenizer)
        sig2, nodes2 = entries[1].decode(_tokenizer)
        assert sig1 == "A" and nodes1 == ["B"]
        assert sig2 == "B" and nodes2 is None

    def test_compile_connotate_bwd(self) -> None:
        """Test compiling backward connotate."""
        entries = compile_source("A < B")
        assert len(entries) == 2
        sig1, nodes1 = entries[0].decode(_tokenizer)
        sig2, nodes2 = entries[1].decode(_tokenizer)
        assert sig1 == "B" and nodes1 == ["A"]
        assert sig2 == "A" and nodes2 is None

    def test_compile_canonize_fwd(self) -> None:
        """Test compiling forward canonize."""
        entries = compile_source("AB => C D")
        d = entries_to_dict(entries)
        assert d["AB"] == ["C", "D"]

    def test_compile_string_literal(self) -> None:
        """Test compiling string literal."""
        entries = compile_source(r'A = "\"hello\""')
        d = entries_to_dict(entries)
        assert d["A"] == r'"\"hello\""'

    def test_compile_number_literal(self) -> None:
        """Test compiling number literal."""
        entries = compile_source("A = 42")
        d = entries_to_dict(entries)
        assert d["A"] == "42"

    def test_compile_incomplete_construct(self) -> None:
        """Test incomplete construct falls back to identity."""
        entries = compile_source("A ==")
        d = entries_to_dict(entries)
        assert d["A"] is None

    def test_compile_literal_countersign(self) -> None:
        """Test literal countersign recovers undersign."""
        entries = compile_source("A == 1")
        assert len(entries) == 1
        sig, nodes = entries[0].decode(_tokenizer)
        assert sig == "A"
        assert nodes == "1"

    def test_compile_multiple_constructs(self) -> None:
        """Test multiple constructs with immediate binding."""
        entries = compile_source("A => B => C")
        d = entries_to_dict(entries)
        assert d["A"] == ["B"]
        assert d["B"] == ["C"]

    def test_compile_subscript_as_nodes(self) -> None:
        """Test subscript signatures as nodes."""
        source = "A =>\n  B\n  C"
        entries = compile_source(source)
        d = entries_to_dict(entries)
        assert d["A"] == ["B", "C"]
        assert d["B"] is None
        assert d["C"] is None

    def test_compile_unquoted_string_literal_unpacked(self) -> None:
        """Test unquoted string literal compiles to unpacked chars.

        Spec: String literal compiles to unpacked chars
        - WHEN compiler processes construct `A => zed` where `zed` is a StringLiteral
        - THEN the nodes are encoded as [ord('z')<<1, ord('e')<<1, ord('d')<<1] = [244, 202, 204]
        """
        from kalvin.mod_tokenizer import PACKED_BIT
        entries = compile_source("A => zed")
        assert len(entries) == 1
        entry = entries[0]

        # Verify unpacked encoding (no PACKED_BIT set)
        assert isinstance(entry._nodes, list)
        for node_id in entry._nodes:
            assert (node_id & PACKED_BIT) == 0, f"Node {node_id} has PACKED_BIT set"

        # Verify specific encoding: ord(c) << 1
        expected = [ord('z') << 1, ord('e') << 1, ord('d') << 1]
        assert entry._nodes == expected, f"Expected {expected}, got {entry._nodes}"

        # Decode to verify roundtrip works
        sig, nodes = entry.decode(_tokenizer)
        assert sig == "A"
        # Note: all-literal nodes decode to a single string
        assert nodes == "zed"

    def test_compile_string_literal_no_reverse(self) -> None:
        """Test string literal in countersign gets no reverse entry.

        Spec: String literal gets no reverse entry
        - WHEN compiler processes countersign `A == zed` where `zed` is a StringLiteral
        - THEN only one entry is created {A: zed} without reverse {zed: A}
        """
        entries = compile_source("A == zed")
        # Should only have ONE entry, not two (no reverse for literals)
        assert len(entries) == 1, f"Expected 1 entry, got {len(entries)}"

        sig, nodes = entries[0].decode(_tokenizer)
        assert sig == "A"
        # Literal nodes decode to string (all unpacked tokens form one value)
        assert nodes == "zed"

    def test_compile_mixed_signature_and_string_literal(self) -> None:
        """Test canonize with both signature and string literal nodes.

        The decode returns a list where:
        - Signatures are decoded individually
        - Consecutive literal chars are grouped into one string
        """
        entries = compile_source("A => B zed C")
        assert len(entries) == 1
        sig, nodes = entries[0].decode(_tokenizer)
        assert sig == "A"
        # Mixed nodes: B (signature), zed (literal), C (signature)
        assert nodes == ["B", "zed", "C"]


class TestCompilerIntegration:
    """Integration tests for compiler."""

    def test_example_compilation(self) -> None:
        """Test the example from the spec."""
        from textwrap import dedent
        source = dedent("""\
            (Mary had a little lamb)
            MHALL == SVO =>
               S(ubject) = M
               V(erb) = H
               O(bject) = ALL =>
                 A = D(et)
                 L = M(od)
                 L > O
            X==
            """)
        entries = compile_source(source)

        # Verify we get entries
        assert len(entries) > 0

        # Decode entries for checking
        decoded = [e.decode(_tokenizer) for e in entries]
        sigs = [sig for sig, _ in decoded]

        # Verify specific signatures exist
        assert "X" in sigs


# =============================================================================
# 6. Output Module Tests
# =============================================================================

class TestOutput:
    """Tests for output functions."""

    def test_write_json(self, tmp_path: Path) -> None:
        """Test writing to JSON file."""
        model = KScript("A == B")
        json_path = tmp_path / "model.json"
        model.output(json_path)

        content = json_path.read_text()
        data = json.loads(content)
        assert len(data) == 2
        assert {"A": "B"} in data
        assert {"B": "A"} in data

    def test_write_jsonl(self, tmp_path: Path) -> None:
        """Test writing to JSONL file."""
        model = KScript("A == B")
        jsonl_path = tmp_path / "model.jsonl"
        model.output(jsonl_path)

        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"A": "B"}
        assert json.loads(lines[1]) == {"B": "A"}

    def test_write_bin(self, tmp_path: Path) -> None:
        """Test writing to binary file."""
        model = KScript("A == B")
        bin_path = tmp_path / "model.bin"
        model.output(bin_path)

        content = bin_path.read_bytes()
        assert content[:4] == b"KSC1"

    def test_read_bin(self, tmp_path: Path) -> None:
        """Test reading from binary file."""
        from kscript.output import read_bin
        model = KScript("A == B")
        bin_path = tmp_path / "model.bin"
        model.output(bin_path)

        entries = read_bin(bin_path)
        assert len(entries) == 2

    def test_load_json(self, tmp_path: Path) -> None:
        """Test loading from JSON file."""
        json_path = tmp_path / "model.json"
        json_path.write_text('[{"A": "B"}, {"B": "A"}]')

        model = KScript(json_path)
        assert len(model.entries) == 2
        d = entries_to_dict(model.entries)
        assert d["A"] == "B"
        assert d["B"] == "A"

    def test_load_jsonl(self, tmp_path: Path) -> None:
        """Test loading from JSONL file."""
        jsonl_path = tmp_path / "model.jsonl"
        jsonl_path.write_text('{"A": "B"}\n{"B": "A"}\n')

        model = KScript(jsonl_path)
        assert len(model.entries) == 2
        d = entries_to_dict(model.entries)
        assert d["A"] == "B"
        assert d["B"] == "A"


class TestOutputRoundtrip:
    """Tests for round-trip through formats."""

    def test_roundtrip_json(self, tmp_path: Path) -> None:
        """Test roundtrip through JSON."""
        original = KScript("A == B")
        json_path = tmp_path / "model.json"
        original.output(json_path)

        loaded = KScript(json_path)
        assert loaded.to_jsonl() == original.to_jsonl()

    def test_roundtrip_jsonl(self, tmp_path: Path) -> None:
        """Test roundtrip through JSONL."""
        original = KScript("A == B")
        jsonl_path = tmp_path / "model.jsonl"
        original.output(jsonl_path)

        loaded = KScript(jsonl_path)
        assert loaded.to_jsonl() == original.to_jsonl()

    def test_roundtrip_bin(self, tmp_path: Path) -> None:
        """Test roundtrip through binary."""
        original = KScript("A == B")
        bin_path = tmp_path / "model.bin"
        original.output(bin_path)

        loaded = KScript(bin_path)
        assert loaded.to_jsonl() == original.to_jsonl()


# =============================================================================
# 7. API Module Tests
# =============================================================================

class TestKScriptAPI:
    """Tests for the KScript API class."""

    def test_inline_source(self) -> None:
        """Test inline source compilation."""
        model = KScript("A == B")
        assert len(model.entries) == 2
        d = entries_to_dict(model.entries)
        assert d["A"] == "B"
        assert d["B"] == "A"

    def test_extend_base(self) -> None:
        """Test extending a base model."""
        base = KScript("A == B")
        extended = KScript("C => A D", base)
        assert len(extended.entries) == 3
        d = entries_to_dict(extended.entries)
        assert "A" in d
        assert "B" in d
        assert "C" in d

    def test_to_jsonl_with_duplicates(self) -> None:
        """Test to_jsonl preserves multiple entries with same signature."""
        model = KScript("A > B\nA > C")
        m = model.to_jsonl()
        assert '{"A": ["B"]}' in m
        assert '{"A": ["C"]}' in m


class TestKScriptAPIIntegration:
    """Integration tests for KScript API."""

    def test_extend_from_json(self, tmp_path: Path) -> None:
        """Test extending model from JSON file."""
        base_path = tmp_path / "base.json"
        base_path.write_text('[{"A": "B"}]')

        extended = KScript("C == D", KScript(base_path))
        assert len(extended.entries) == 3  # A:B, C:D, D:C
        d = entries_to_dict(extended.entries)
        assert d["A"] == "B"
        assert d["C"] == "D"


# =============================================================================
# 8. CLI Module Tests
# =============================================================================

class TestCLI:
    """Smoke tests for CLI."""

    def test_cli_compile_default_output(self, tmp_path: Path, monkeypatch) -> None:
        """Test CLI compiles .ks to default .jsonl output."""
        import subprocess

        # Create input file
        ks_file = tmp_path / "test.ks"
        ks_file.write_text("A == B")

        # Run CLI
        result = subprocess.run(
            ["python", "-m", "kscript", str(ks_file)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )

        # Check output file exists
        output_file = tmp_path / "test.jsonl"
        assert output_file.exists(), f"Output file not created: {result.stderr}"

        # Check content
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"A": "B"}

    def test_cli_compile_custom_output(self, tmp_path: Path) -> None:
        """Test CLI with custom output path."""
        import subprocess

        ks_file = tmp_path / "test.ks"
        ks_file.write_text("A == B")
        json_file = tmp_path / "custom.json"

        result = subprocess.run(
            ["python", "-m", "kscript", str(ks_file), "-out", str(json_file)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )

        assert json_file.exists(), f"Output file not created: {result.stderr}"
        data = json.loads(json_file.read_text())
        assert {"A": "B"} in data

    def test_cli_missing_input(self, tmp_path: Path) -> None:
        """Test CLI exits with error on missing input."""
        import subprocess

        result = subprocess.run(
            ["python", "-m", "kscript", "nonexistent.ks"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "not found" in result.stderr.lower()

    def test_cli_binary_output(self, tmp_path: Path) -> None:
        """Test CLI produces binary output with .bin suffix."""
        import subprocess

        ks_file = tmp_path / "test.ks"
        ks_file.write_text("A == B")
        bin_file = tmp_path / "output.bin"

        result = subprocess.run(
            ["python", "-m", "kscript", str(ks_file), "-out", str(bin_file)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )

        assert bin_file.exists()
        content = bin_file.read_bytes()
        assert content[:4] == b"KSC1"


# =============================================================================
# 9. Decompiler Module Tests
# =============================================================================


def make_kline(sig: int, nodes, tokenizer: Mod64Tokenizer) -> KLine:
    """Helper to create a KLine with encoded signature and nodes."""
    return KLine(signature=sig, nodes=nodes)


def encode_sig(name: str, level: str, tokenizer: Mod64Tokenizer) -> int:
    """Encode a signature name with significance bits.

    Args:
        name: Signature name to encode
        level: Significance level ("S1", "S2", "S3", "S4")
        tokenizer: Tokenizer for encoding
    """
    base = tokenizer.encode(name, pack=True)[0]
    if level == "S1":
        return base | _sig.S1
    elif level == "S2":
        return base | _sig.S2_RANGE  # Sets bit 55
    elif level == "S3":
        return base | _sig.S3_RANGE  # Sets bit 32
    else:  # S4
        return base  # No significance bits


class TestDecompilerSignificance:
    """Tests for significance level detection."""

    def test_detect_s1_countersign(self) -> None:
        """Test S1 (countersign) detection."""
        decompiler = Decompiler(_tokenizer)
        sig = encode_sig("A", "S1", _tokenizer)
        level = decompiler._get_significance_level(sig)
        assert level == "S1"

    def test_detect_s2_canonize(self) -> None:
        """Test S2 (canonize) detection."""
        decompiler = Decompiler(_tokenizer)
        sig = encode_sig("A", "S2", _tokenizer)
        level = decompiler._get_significance_level(sig)
        assert level == "S2"

    def test_detect_s3_connotate(self) -> None:
        """Test S3 (connotate) detection."""
        decompiler = Decompiler(_tokenizer)
        sig = encode_sig("A", "S3", _tokenizer)
        level = decompiler._get_significance_level(sig)
        assert level == "S3"

    def test_detect_s4_undersign(self) -> None:
        """Test S4 (undersign) detection - no bits set."""
        decompiler = Decompiler(_tokenizer)
        sig = _tokenizer.encode("A", pack=True)[0]  # No significance bits
        level = decompiler._get_significance_level(sig)
        assert level == "S4"


class TestDecompilerCountersign:
    """Tests for countersign pair handling."""

    def test_countersign_pair_dedup(self) -> None:
        """Test countersign pair emits once."""
        sig_a = encode_sig("A", "S1", _tokenizer)
        sig_b = encode_sig("B", "S1", _tokenizer)
        node_a = _tokenizer.encode("A", pack=True)[0]
        node_b = _tokenizer.encode("B", pack=True)[0]

        klines = [
            KLine(signature=sig_a, nodes=node_b),
            KLine(signature=sig_b, nodes=node_a),
        ]

        decompiler = Decompiler(_tokenizer)
        result = decompiler.decompile(klines)

        # Should emit "A == B" once, not both directions
        assert "A == B" in result
        assert result.count("==") == 1

    def test_countersign_single_emits(self) -> None:
        """Test single countersign emits correctly."""
        sig_a = encode_sig("A", "S1", _tokenizer)
        node_b = _tokenizer.encode("B", pack=True)[0]

        klines = [
            KLine(signature=sig_a, nodes=node_b),
        ]

        decompiler = Decompiler(_tokenizer)
        result = decompiler.decompile(klines)

        assert "A == B" in result


class TestDecompilerSubscripts:
    """Tests for subscript reconstruction."""

    def test_canonize_multi_node_subscripts(self) -> None:
        """Test multi-node canonize uses subscripts."""
        sig_cd = encode_sig("CD", "S2", _tokenizer)
        sig_c = encode_sig("C", "S3", _tokenizer)
        sig_d = encode_sig("D", "S3", _tokenizer)
        node_c = _tokenizer.encode("C", pack=True)[0]
        node_d = _tokenizer.encode("D", pack=True)[0]
        node_1 = _tokenizer.encode("1", pack=True)[0]
        node_2 = _tokenizer.encode("2", pack=True)[0]

        klines = [
            KLine(signature=sig_cd, nodes=[node_c, node_d]),
            KLine(signature=sig_c, nodes=node_1),
            KLine(signature=sig_d, nodes=node_2),
        ]

        decompiler = Decompiler(_tokenizer)
        result = decompiler.decompile(klines)

        # Should emit subscript structure
        assert "CD =>" in result
        assert "  C >" in result
        assert "  D >" in result

    def test_canonize_single_node_inline(self) -> None:
        """Test single-node canonize emits inline."""
        sig_a = encode_sig("A", "S2", _tokenizer)
        node_b = _tokenizer.encode("B", pack=True)[0]

        klines = [
            KLine(signature=sig_a, nodes=[node_b]),
        ]

        decompiler = Decompiler(_tokenizer)
        result = decompiler.decompile(klines)

        # Should emit inline
        assert "A => B" in result


class TestDecompilerIdentity:
    """Tests for identity recovery."""

    def test_missing_entry_is_identity(self) -> None:
        """Test missing entry emits as identity."""
        sig_cd = encode_sig("CD", "S2", _tokenizer)
        node_c = _tokenizer.encode("C", pack=True)[0]
        node_d = _tokenizer.encode("D", pack=True)[0]

        # CD => [C, D] but C and D have no entries
        klines = [
            KLine(signature=sig_cd, nodes=[node_c, node_d]),
        ]

        decompiler = Decompiler(_tokenizer)
        result = decompiler.decompile(klines)

        # Should emit C and D as identity subscripts
        assert "CD =>" in result
        assert "  C" in result
        assert "  D" in result

    def test_s4_undersign_emits_sig_only(self) -> None:
        """Test S4 level emits just signature."""
        sig_a = _tokenizer.encode("A", pack=True)[0]  # No significance bits

        klines = [
            KLine(signature=sig_a, nodes=None),
        ]

        decompiler = Decompiler(_tokenizer)
        result = decompiler.decompile(klines)

        # Should emit just "A"
        lines = result.strip().split("\n")
        assert "A" in lines[0]
        assert "=>" not in lines[0]
        assert "==" not in lines[0]
        assert ">" not in lines[0]


class TestDecompilerErrors:
    """Tests for error surfacing."""

    def test_orphan_detection(self) -> None:
        """Test orphaned KLine detection."""
        # Create: A == B, and orphaned X => Y
        sig_a = encode_sig("A", "S1", _tokenizer)
        sig_x = encode_sig("X", "S2", _tokenizer)  # Orphan - not reachable from A
        node_b = _tokenizer.encode("B", pack=True)[0]
        node_y = _tokenizer.encode("Y", pack=True)[0]

        klines = [
            KLine(signature=sig_a, nodes=node_b),  # A == B
            KLine(signature=sig_x, nodes=[node_y]),  # X => Y (orphan - not reachable from A)
        ]

        decompiler = Decompiler(_tokenizer)
        result = decompiler.decompile(klines)

        # Should surface orphan
        assert "!!! ORPHAN:" in result
        assert "X" in result  # Orphan should mention X

    def test_broken_chain_reference(self) -> None:
        """Test broken chain reference - token that can't be decoded."""
        from unittest.mock import Mock

        # Create a mock tokenizer that raises an exception for certain tokens
        mock_tokenizer = Mock()
        mock_tokenizer.decode = Mock(side_effect=ValueError("Invalid token"))

        sig_a = encode_sig("A", "S1", _tokenizer)
        node_b = _tokenizer.encode("B", pack=True)[0]

        klines = [
            KLine(signature=sig_a, nodes=node_b),
        ]

        decompiler = Decompiler(mock_tokenizer)
        result = decompiler.decompile(klines)

        # Should surface broken reference since tokenizer raises exception
        assert "!!! BROKEN:" in result


class TestDecompilerIntegration:
    """Integration tests for full decompilation."""

    def test_roundtrip_countersign_canonize(self) -> None:
        """Test decompilation of countersign + canonize chain."""
        # Simulating: AB == CD => C D
        # CD needs BOTH S1 (countersign pair) and S2 (canonize) - different sigs!
        sig_ab_s1 = encode_sig("AB", "S1", _tokenizer)  # AB with countersign bit
        sig_cd_s1 = encode_sig("CD", "S1", _tokenizer)  # CD with countersign bit (pair)
        sig_cd_s2 = encode_sig("CD", "S2", _tokenizer)  # CD with canonize bit
        sig_c_s3 = encode_sig("C", "S3", _tokenizer)
        sig_d_s3 = encode_sig("D", "S3", _tokenizer)

        node_ab = _tokenizer.encode("AB", pack=True)[0]
        node_cd = _tokenizer.encode("CD", pack=True)[0]
        node_c = _tokenizer.encode("C", pack=True)[0]
        node_d = _tokenizer.encode("D", pack=True)[0]
        node_1 = _tokenizer.encode("1", pack=True)[0]
        node_2 = _tokenizer.encode("2", pack=True)[0]

        klines = [
            KLine(signature=sig_ab_s1, nodes=node_cd),  # AB == CD
            KLine(signature=sig_cd_s1, nodes=node_ab),  # CD == AB (pair)
            KLine(signature=sig_cd_s2, nodes=[node_c, node_d]),  # CD => [C, D]
            KLine(signature=sig_c_s3, nodes=node_1),  # C > 1
            KLine(signature=sig_d_s3, nodes=node_2),  # D > 2
        ]

        decompiler = Decompiler(_tokenizer)
        result = decompiler.decompile(klines)

        # Should emit: AB == CD => \n  C > 1\n  D > 2
        assert "AB == CD" in result
        assert "CD =>" in result
        assert "C >" in result
        assert "D >" in result


# =============================================================================
# 10. Multi-Character Signature (MCS) Tests
# =============================================================================


class TestMCSCanonization:
    """Tests for multi-character signature expansion."""

    def test_mcs_simple_expansion(self) -> None:
        """Test simple MCS expansion emits canonization + identities.

        Spec: Simple MCS expansion
        - WHEN compiler processes signature `ABC`
        - THEN entry `{ABC: [A, B, C]}` is emitted with S2 significance
        - AND entries `{A: null}`, `{B: null}`, `{C: null}` are emitted with S4
        """
        entries = compile_source("ABC")

        # Should have 4 entries: MCS canonization + 3 identities
        assert len(entries) == 4

        d = entries_to_dict(entries)

        # MCS canonization: {ABC: [A, B, C]}
        assert d["ABC"] == ["A", "B", "C"]

        # Component identities
        assert d["A"] is None
        assert d["B"] is None
        assert d["C"] is None

    def test_mcs_in_construct_position(self) -> None:
        """Test MCS in construct position.

        Spec: MCS in construct position
        - WHEN compiler processes `ABC => X`
        - THEN entry `{ABC: [A, B, C]}` is emitted (implicit MCS expansion)
        - AND entry `{ABC: [X]}` is emitted (explicit construct)
        """
        entries = compile_source("ABC => X")

        # Should have: MCS canonization + 3 identities + construct
        assert len(entries) == 5

        # First entry is MCS canonization
        sig, nodes = entries[0].decode(_tokenizer)
        assert sig == "ABC"
        assert nodes == ["A", "B", "C"]

        # Next 3 are component identities
        for i in range(1, 4):
            sig, nodes = entries[i].decode(_tokenizer)
            assert nodes is None
            assert sig in "ABC"

        # Last entry is the construct
        sig, nodes = entries[4].decode(_tokenizer)
        assert sig == "ABC"
        assert nodes == ["X"]

    def test_single_char_no_expansion(self) -> None:
        """Test single-character signature bypass.

        Spec: Single-character bypass
        - WHEN compiler processes signature `A`
        - THEN no MCS expansion entry is emitted (single-char is atomic)
        """
        entries = compile_source("A")

        # Should have only 1 entry (identity), no MCS expansion
        assert len(entries) == 1

        d = entries_to_dict(entries)
        assert d["A"] is None

    def test_mcs_two_characters(self) -> None:
        """Test two-character MCS expansion."""
        entries = compile_source("AB")

        # Should have 3 entries: canonization + 2 identities
        assert len(entries) == 3

        d = entries_to_dict(entries)
        assert d["AB"] == ["A", "B"]
        assert d["A"] is None
        assert d["B"] is None

    def test_mcs_with_countersign(self) -> None:
        """Test MCS expansion with countersign construct."""
        entries = compile_source("ABC == X")

        # Should have: MCS canonization + 3 identities + 2 countersign entries
        assert len(entries) == 6

        # First entry is MCS canonization
        sig, nodes = entries[0].decode(_tokenizer)
        assert sig == "ABC"
        assert nodes == ["A", "B", "C"]

        # Next 3 are component identities
        for i in range(1, 4):
            sig, nodes = entries[i].decode(_tokenizer)
            assert nodes is None
            assert sig in "ABC"

        # Last 2 are countersign entries
        sig, nodes = entries[4].decode(_tokenizer)
        assert sig == "ABC"
        assert nodes == "X"

        sig, nodes = entries[5].decode(_tokenizer)
        assert sig == "X"
        assert nodes == "ABC"


class TestMCSOrdering:
    """Tests for MCS entry ordering."""

    def test_mcs_emitted_before_construct(self) -> None:
        """Test MCS entries emitted before constructs that reference them.

        Spec: Ordering with explicit constructs
        - WHEN compiler processes `ABC == X`
        - THEN entries are emitted in order:
          1. {ABC: [A, B, C]} (MCS canonization)
          2. {A: null}, {B: null}, {C: null} (identities)
          3. {ABC: X}, {X: ABC} (countersign construct)
        """
        entries = compile_source("ABC == X")

        # First 4 entries should be MCS-related
        sig, nodes = entries[0].decode(_tokenizer)
        assert sig == "ABC"
        assert nodes == ["A", "B", "C"]

        # Next 3 should be identities
        for i in range(1, 4):
            sig, nodes = entries[i].decode(_tokenizer)
            assert nodes is None
            assert sig in "ABC"

        # Last 2 should be countersign entries
        sig, nodes = entries[4].decode(_tokenizer)
        assert sig == "ABC"
        assert nodes == "X"

        sig, nodes = entries[5].decode(_tokenizer)
        assert sig == "X"
        assert nodes == "ABC"


class TestMCSSignificance:
    """Tests for MCS significance levels."""

    def test_mcs_canonization_s2_significance(self) -> None:
        """Test MCS canonization uses S2 significance.

        Spec: S2 significance encoding
        - WHEN MCS canonization entry `{ABC: [A, B, C]}` is emitted
        - THEN the signature has bit 55 set (S2 indicator)
        """
        entries = compile_source("ABC")

        # First entry is MCS canonization
        mcs_entry = entries[0]

        # Check signature has S2 bit set
        sig_token = mcs_entry.signature & _sig.TOKEN_MASK
        sig_s2 = sig_token | _sig.S2

        # The signature should have S2 bits (CANONIZE_FWD)
        assert mcs_entry.signature == sig_s2 or (mcs_entry.signature & _sig.S2) != 0

    def test_mcs_identity_s4_significance(self) -> None:
        """Test component identities use S4 significance (no bits).

        Spec: Component identities use S4
        - WHEN component identity entries are emitted
        - THEN they have no significance bits (S4)
        """
        entries = compile_source("ABC")

        # Entries 1-3 are identities (A, B, C)
        for i in range(1, 4):
            entry = entries[i]
            # S4 = no significance bits, just the token
            sig_token = entry.signature & _sig.TOKEN_MASK
            # Identity entries use UNDERSIGN which has no bits
            # So signature should equal just the token (no extra bits)
            assert entry.signature == sig_token or (entry.signature & ~_sig.TOKEN_MASK) == 0
