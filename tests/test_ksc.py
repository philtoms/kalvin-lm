"""Tests for KScript compiler."""

import json
from pathlib import Path
from textwrap import dedent

from kalvin.mod_tokenizer import Mod32Tokenizer
from ksc import KScript
from ksc.compiler import Compiler, CompiledEntry
from ksc.lexer import Lexer
from ksc.parser import Parser

# Shared tokenizer for tests
_tokenizer = Mod32Tokenizer()


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
        # Should produce 2 entries: {A: B} and {B: None}
        assert len(entries) == 2
        sig1, nodes1 = entries[0].decode(_tokenizer)
        sig2, nodes2 = entries[1].decode(_tokenizer)
        assert sig1 == "A" and nodes1 == "B"
        assert sig2 == "B" and nodes2 is None

    def test_compile_connotate_fwd(self) -> None:
        """Test compiling forward connotate."""
        entries = compile_source("A > B")
        # Should produce 2 entries: {A: [B]} and {B: None}
        assert len(entries) == 2
        sig1, nodes1 = entries[0].decode(_tokenizer)
        sig2, nodes2 = entries[1].decode(_tokenizer)
        assert sig1 == "A" and nodes1 == ["B"]
        assert sig2 == "B" and nodes2 is None

    def test_compile_connotate_bwd(self) -> None:
        """Test compiling backward connotate."""
        entries = compile_source("A < B")
        # Should produce 2 entries: {B: [A]} and {A: None}
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

    def test_compile_canonize_bwd(self) -> None:
        """Test compiling backward canonize."""
        # Backward canonize: A <= B C means {A: [B, C]}
        # The last node is the parent, the rest are children
        entries = compile_source("X <= A B")
        d = entries_to_dict(entries)
        assert d["B"] == ["A"]

    def test_compile_string_literal(self) -> None:
        """Test compiling string literal.

        Note: String characters are encoded by OR-ing bits together.
        When decoded, characters are sorted by bit position.
        """
        entries = compile_source(r'A = "\"hello\""')
        d = entries_to_dict(entries)
        # Original: "\"hello\"" -> decoded order: FILPS2 (sorted by bit pos)
        assert d["A"] == "FILPS2"

    def test_compile_number_literal(self) -> None:
        """Test compiling number literal.

        Note: Number characters are encoded by OR-ing bits together.
        When decoded, characters are sorted by bit position.
        """
        entries = compile_source("A = 42")
        d = entries_to_dict(entries)
        # Original: "42" -> decoded order: "24" (sorted by bit pos)
        assert d["A"] == "24"

    def test_compile_incomplete_construct(self) -> None:
        """Test incomplete construct falls back to identity."""
        entries = compile_source("A ==")
        d = entries_to_dict(entries)
        assert d["A"] is None

    def test_compile_literal_countersign(self) -> None:
        """Test literal countersign recovers undersign.

        Per recovery semantics: A == 1 => {A: 1} only (no reverse)
        Literals should never be used as signatures.
        """
        entries = compile_source("A == 1")
        # Should produce only 1 entry: {A: 1}, not {1: A}
        assert len(entries) == 1
        sig, nodes = entries[0].decode(_tokenizer)
        assert sig == "A"
        assert nodes == "1"  # "1" decodes as "1" (single char)

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
        assert len(m) == 4
        # Both connotate entries should be preserved
        assert {"A": ["B"]} in m
        assert {"A": ["C"]} in m
        # Plus identity entries for B and C
        assert {"B": None} in m
        assert {"C": None} in m


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
        """Test the example from the plan.

        Note: Multi-character signatures are encoded by OR-ing character bits together.
        When decoded, characters are sorted by bit position, not original order.
        E.g., "MHALL" decodes as "AHLM" (A, H, L, M sorted by bit position).
        """
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

        # Decode entries for easier checking
        # Note: multi-char signatures decode with chars sorted by bit position
        decoded = [e.decode(_tokenizer) for e in entries]
        sigs = [sig for sig, _ in decoded]

        # Verify specific signatures exist (decoded order may differ from source)
        # MHALL -> AHLM (sorted by bit position)
        # SVO -> OSV (sorted by bit position)
        assert "AHLM" in sigs
        assert "OSV" in sigs
        assert "X" in sigs

        # Check countersign entries (bidirectional)
        mhall_entries = [nodes for sig, nodes in decoded if sig == "AHLM"]
        assert "OSV" in mhall_entries

        # SVO has two entries: one from countersign, one from canonize
        svo_entries = [nodes for sig, nodes in decoded if sig == "OSV"]
        assert "AHLM" in svo_entries  # countersign
        # canonize entries preserve source order: S,V,O
        assert ["S", "V", "O"] in svo_entries

        # Verify X is identity (incomplete construct)
        x_entries = [nodes for sig, nodes in decoded if sig == "X"]
        assert None in x_entries
