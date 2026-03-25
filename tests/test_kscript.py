"""Tests for KScript compiler - rebuilt from spec."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from kalvin.mod_tokenizer import Mod64Tokenizer
from kscript import KScript
from kscript.compiler import Compiler, CompiledEntry
from kscript.lexer import Lexer
from kscript.parser import Parser

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
        """Test all 14 token types defined."""
        from kscript.token import TokenType
        assert len(TokenType) == 14

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
