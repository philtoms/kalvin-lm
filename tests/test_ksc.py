"""Tests for KScript compiler."""

import json
from pathlib import Path
from textwrap import dedent

from ksc import KScript
from ksc.compiler import Compiler, CompiledEntry
from ksc.lexer import Lexer
from ksc.parser import Parser


def compile_source(source: str) -> list[CompiledEntry]:
    """Helper to compile source string to entries."""
    tokens = Lexer(source).tokenize()
    kscript_file = Parser(tokens).parse()
    return Compiler().compile(kscript_file)


def entries_to_dict(entries: list[CompiledEntry]) -> dict[str, list[str]]:
    """Convert entries to dict for easier comparison."""
    return {e.signature: e.nodes for e in entries}


class TestLexer:
    """Tests for the lexer."""

    def test_tokenize_signature(self) -> None:
        """Test tokenizing a simple signature."""
        tokens = Lexer("A").tokenize()
        assert len(tokens) == 2  # SIGNATURE + EOF
        assert tokens[0].type.name == "SIGNATURE"
        assert tokens[0].value == "A"

    def test_tokenize_operators(self) -> None:
        """Test tokenizing operators."""
        tokens = Lexer("A == B").tokenize()
        assert tokens[1].type.name == "COUNTERSIGN"
        assert tokens[1].value == "=="

    def test_tokenize_string(self) -> None:
        """Test tokenizing string literals."""
        tokens = Lexer('A = "hello"').tokenize()
        assert any(t.type.name == "STRING" and t.value == '"hello"' for t in tokens)

    def test_tokenize_string_with_escape(self) -> None:
        """Test tokenizing string with escape."""
        tokens = Lexer(r'A = "\"hello\""').tokenize()
        assert any(t.type.name == "STRING" for t in tokens)

    def test_tokenize_number(self) -> None:
        """Test tokenizing number literals."""
        tokens = Lexer("A = 42").tokenize()
        assert any(t.type.name == "NUMBER" and t.value == "42" for t in tokens)

    def test_tokenize_comment(self) -> None:
        """Test tokenizing comments."""
        tokens = Lexer("(this is a comment)").tokenize()
        assert tokens[0].type.name == "COMMENT"

    def test_tokenize_nested_comment(self) -> None:
        """Test nested comments (greedy match)."""
        tokens = Lexer("(comment (nested))").tokenize()
        assert tokens[0].type.name == "COMMENT"
        assert tokens[0].value == "(comment (nested))"

    def test_tokenize_unterminated_comment(self) -> None:
        """Test unterminated comment (greedy to EOL)."""
        tokens = Lexer("(no closing").tokenize()
        assert tokens[0].type.name == "COMMENT"

    def test_tokenize_indentation(self) -> None:
        """Test tokenizing indentation."""
        source = "A\n  B"
        tokens = Lexer(source).tokenize()
        assert any(t.type.name == "INDENT" for t in tokens)
        assert any(t.type.name == "DEDENT" for t in tokens)


class TestParser:
    """Tests for the parser."""

    def test_parse_identity_script(self) -> None:
        """Test parsing an identity script."""
        tokens = Lexer("A").tokenize()
        kscript = Parser(tokens).parse()
        assert len(kscript.scripts) == 1
        assert kscript.scripts[0].signature.name == "A"
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


class TestCompiler:
    """Tests for the compiler."""

    def test_compile_identity(self) -> None:
        """Test compiling identity script."""
        entries = compile_source("A")
        assert len(entries) == 1
        assert entries[0].signature == "A"
        assert entries[0].nodes is None

    def test_compile_countersign(self) -> None:
        """Test compiling countersign (bidirectional)."""
        entries = compile_source("A == B")
        d = entries_to_dict(entries)
        assert d["A"] == "B"
        assert d["B"] == "A"

    def test_compile_undersign(self) -> None:
        """Test compiling undersign."""
        entries = compile_source("A = B")
        d = entries_to_dict(entries)
        assert d["A"] == "B"

    def test_compile_connotate_fwd(self) -> None:
        """Test compiling forward connotate."""
        entries = compile_source("A > B")
        d = entries_to_dict(entries)
        assert d["A"] == ["B"]

    def test_compile_connotate_bwd(self) -> None:
        """Test compiling backward connotate."""
        entries = compile_source("A < B")
        d = entries_to_dict(entries)
        assert d["B"] == ["A"]

    def test_compile_canonize_fwd(self) -> None:
        """Test compiling forward canonize."""
        entries = compile_source("AB => C D")
        d = entries_to_dict(entries)
        assert d["AB"] == ["C", "D"]

    def test_compile_canonize_bwd(self) -> None:
        """Test compiling backward canonize."""
        # Backward canonize: A <= B C means {A: [B, C]}
        # The last node is the parent, the rest are children
        entries = compile_source("X <= A B")
        d = entries_to_dict(entries)
        assert d["B"] == ["A"]

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


class TestKScriptAPI:
    """Tests for the KScript API class."""

    def test_inline_source(self) -> None:
        """Test inline source compilation."""
        model = KScript("A == B")
        assert len(model.entries) == 2
        assert model.to_dict()["A"] == "B"
        assert model.to_dict()["B"] == "A"

    def test_extend_base(self) -> None:
        """Test extending a base model."""
        base = KScript("A == B")
        extended = KScript("C => A D", base)
        d = extended.to_dict()
        assert "A" in d
        assert "B" in d
        assert "C" in d
        # Canonize doesn't create identity for nodes, so D is in C's nodes
        assert d["C"] == ["A", "D"]

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        model = KScript("A => B C")
        d = model.to_dict()
        assert d["A"] == ["B", "C"]

    def test_to_model(self) -> None:
        """Test to_model preserves duplicate signatures."""
        model = KScript("A == B")
        m = model.to_model()
        assert len(m) == 2
        assert {"A": "B"} in m
        assert {"B": "A"} in m

    def test_to_model_with_duplicates(self) -> None:
        """Test to_model preserves multiple entries with same signature."""
        model = KScript("A > B\nA > C")
        m = model.to_model()
        assert len(m) == 2
        # Both entries should be preserved
        assert {"A": ["B"]} in m
        assert {"A": ["C"]} in m


class TestJSONIO:
    """Tests for JSON/JSONL input/output."""

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

    def test_load_json(self, tmp_path: Path) -> None:
        """Test loading from JSON file."""
        # Write JSON file manually
        json_path = tmp_path / "model.json"
        json_path.write_text('[{"A": "B"}, {"B": "A"}]')

        model = KScript(json_path)
        assert len(model.entries) == 2
        d = model.to_dict()
        assert d["A"] == "B"
        assert d["B"] == "A"

    def test_load_jsonl(self, tmp_path: Path) -> None:
        """Test loading from JSONL file."""
        jsonl_path = tmp_path / "model.jsonl"
        jsonl_path.write_text('{"A": "B"}\n{"B": "A"}\n')

        model = KScript(jsonl_path)
        assert len(model.entries) == 2
        d = model.to_dict()
        assert d["A"] == "B"
        assert d["B"] == "A"

    def test_roundtrip_json(self, tmp_path: Path) -> None:
        """Test roundtrip through JSON."""
        original = KScript("A == B")
        json_path = tmp_path / "model.json"
        original.output(json_path)

        loaded = KScript(json_path)
        assert loaded.to_model() == original.to_model()

    def test_roundtrip_jsonl(self, tmp_path: Path) -> None:
        """Test roundtrip through JSONL."""
        original = KScript("A == B")
        jsonl_path = tmp_path / "model.jsonl"
        original.output(jsonl_path)

        loaded = KScript(jsonl_path)
        assert loaded.to_model() == original.to_model()

    def test_extend_from_json(self, tmp_path: Path) -> None:
        """Test extending model from JSON file."""
        base_path = tmp_path / "base.json"
        base_path.write_text('[{"A": "B"}]')

        extended = KScript("C == D", KScript(base_path))
        assert len(extended.entries) == 3  # A:B, C:D, D:C
        d = extended.to_dict()
        assert d["A"] == "B"
        assert d["C"] == "D"


class TestIntegration:
    """Integration tests."""

    def test_example_compilation(self) -> None:
        """Test the example from the plan."""
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

        # Verify specific signatures exist in entries
        sigs = [e.signature for e in entries]
        assert "MHALL" in sigs
        assert "SVO" in sigs
        assert "X" in sigs

        # Check countersign entries (MHALL and SVO are bidirectional)
        mhall_entries = [e for e in entries if e.signature == "MHALL"]
        assert any(e.nodes == "SVO" for e in mhall_entries)

        # SVO has two entries: one from countersign, one from canonize
        svo_entries = [e for e in entries if e.signature == "SVO"]
        assert any(e.nodes == "MHALL" for e in svo_entries)  # countersign
        assert any(e.nodes == ["S", "V", "O"] for e in svo_entries)  # canonize

        # Verify X is identity (incomplete construct)
        x_entries = [e for e in entries if e.signature == "X"]
        assert any(e.nodes is None for e in x_entries)
