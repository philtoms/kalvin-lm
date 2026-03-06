"""Tests for KScript lexer, parser, and interpreter."""

import pytest

from kscript import (
    InterpretResult,
    KLineExpr,
    KScript,
    Lexer,
    LexerError,
    ParseError,
    Parser,
    TokenType,
    interpret_script,
    parse,
    Token,
    CHAR_BIT,
    BIT_CHAR,
    encode_mod,
    decode_mod,
)


class TestLexer:
    """Tests for the KScript lexer."""

    def test_tokenize_keywords(self):
        """Test tokenizing load and save keywords."""
        lexer = Lexer("load save")
        tokens = lexer.tokenize()
        assert len(tokens) == 3  # load, save, EOF
        assert tokens[0].type == TokenType.LOAD
        assert tokens[1].type == TokenType.SAVE
        assert tokens[2].type == TokenType.EOF

    def test_tokenize_significance_operators(self):
        """Test tokenizing significance operators."""
        lexer = Lexer("= => > < !=")
        tokens = lexer.tokenize()
        assert tokens[0].type == TokenType.S1
        assert tokens[1].type == TokenType.S2
        assert tokens[2].type == TokenType.S3_FORWARD
        assert tokens[3].type == TokenType.S3_BACKWARD
        assert tokens[4].type == TokenType.S4

    def test_tokenize_identifiers(self):
        """Test tokenizing identifiers."""
        lexer = Lexer("hello world")
        tokens = lexer.tokenize()
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "hello"
        assert tokens[1].type == TokenType.IDENTIFIER
        assert tokens[1].value == "world"

    def test_tokenize_paths(self):
        """Test tokenizing file paths."""
        lexer = Lexer("/path/to/file.bin")
        tokens = lexer.tokenize()
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "/path/to/file.bin"

    def test_tokenize_inline_comment(self):
        """Test that comments in parentheses are stripped."""
        lexer = Lexer("V(erb)")
        tokens = lexer.tokenize()
        assert len(tokens) == 2  # V, EOF
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "V"

    def test_tokenize_comment_splice(self):
        """Test that inline comments splice identifiers."""
        lexer = Lexer("hel(comment)lo")
        tokens = lexer.tokenize()
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "hello"

    def test_tokenize_comment_between_tokens(self):
        """Test comments between tokens."""
        lexer = Lexer("hello (this is a comment) world")
        tokens = lexer.tokenize()
        assert len(tokens) == 3  # hello, world, EOF
        assert tokens[0].value == "hello"
        assert tokens[1].value == "world"

    def test_tokenize_nested_comments(self):
        """Test nested parentheses in comments."""
        lexer = Lexer("hello (nested (comment) here) world")
        tokens = lexer.tokenize()
        assert len(tokens) == 3
        assert tokens[0].value == "hello"
        assert tokens[1].value == "world"

    def test_unterminated_comment_raises_error(self):
        """Test that unterminated comments raise LexerError."""
        lexer = Lexer("hello (unterminated")
        with pytest.raises(LexerError):
            lexer.tokenize()


class TestParser:
    """Tests for the KScript parser."""

    def test_parse_load(self):
        """Test parsing load statement."""
        script = parse("load /path/to/model.bin")
        assert len(script.statements) == 1
        stmt = script.root
        assert isinstance(stmt, type(parse("load x").root))
        assert stmt.path.name == "/path/to/model.bin"

    def test_parse_save(self):
        """Test parsing save statement."""
        script = parse("save /path/to/output.bin")
        assert len(script.statements) == 1
        stmt = script.root
        assert stmt.path.name == "/path/to/output.bin"

    def test_parse_save_without_path(self):
        """Test parsing save without path."""
        script = parse("save")
        assert len(script.statements) == 1
        stmt = script.root
        assert stmt.path is None

    def test_parse_simple_kline(self):
        """Test parsing simple KLine."""
        script = parse("hello")
        assert len(script.statements) == 1
        kline = script.root
        assert isinstance(kline, KLineExpr)
        assert kline.sig.name == "hello"
        assert kline.significance is None
        assert kline.nodes == []

    def test_parse_kline_with_s1(self):
        """Test parsing KLine with S1 significance."""
        script = parse("greeting = hello world")
        kline = script.root
        assert kline.significance.value == "="
        assert len(kline.nodes) == 2
        assert kline.nodes[0].identifier.name == "hello"
        assert kline.nodes[1].identifier.name == "world"

    def test_parse_kline_with_s2(self):
        """Test parsing KLine with S2 significance."""
        script = parse("greeting => hello")
        kline = script.root
        assert kline.significance.value == "=>"

    def test_parse_kline_with_s3_forward(self):
        """Test parsing KLine with S3 forward significance."""
        script = parse("greeting > hello")
        kline = script.root
        assert kline.significance.value == ">"

    def test_parse_kline_with_s3_backward(self):
        """Test parsing KLine with S3 backward significance."""
        script = parse("greeting < hello")
        kline = script.root
        assert kline.significance.value == "<"

    def test_parse_kline_with_s4(self):
        """Test parsing KLine with S4 significance."""
        script = parse("greeting != hello")
        kline = script.root
        assert kline.significance.value == "!="

    def test_parse_multiple_statements(self):
        """Test parsing multiple statements."""
        script = parse("""
            load /path/to/model.bin
            greeting = hello world
            question > greeting
            save /path/to/output.bin
        """)
        assert len(script.statements) == 4

    def test_parse_with_inline_comments(self):
        """Test parsing with inline comments."""
        script = parse("sit > V(erb)")
        kline = script.root
        assert kline.sig.name == "sit"
        assert kline.nodes[0].identifier.name == "V"


class TestSingleCharEncoding:
    """Tests for single character encoding/decoding."""

    def test_char_bit_a_returns_1(self):
        """CHAR_BIT['A'] returns 1 (bit 0)."""
        assert CHAR_BIT["A"] == 1

    def test_char_bit_b_returns_2(self):
        """CHAR_BIT['B'] returns 2 (bit 1)."""
        assert CHAR_BIT["B"] == 2

    def test_char_bit_c_returns_4(self):
        """CHAR_BIT['C'] returns 4 (bit 2)."""
        assert CHAR_BIT["C"] == 4

    def test_char_bit_z_returns_correct_bit(self):
        """CHAR_BIT['Z'] returns bit 25."""
        assert CHAR_BIT["Z"] == 1 << 25

    def test_char_bit_6_same_as_a(self):
        """Digit 6 maps to same bit as A (position 32 % 32 = 0)."""
        assert CHAR_BIT["6"] == CHAR_BIT["A"] == 1

    def test_char_bit_0_returns_bit_26(self):
        """CHAR_BIT['0'] returns bit 26."""
        assert CHAR_BIT["0"] == 1 << 26

    def test_char_bit_5_returns_bit_31(self):
        """CHAR_BIT['5'] returns bit 31."""
        assert CHAR_BIT["5"] == 1 << 31

    def test_bit_char_1_returns_a(self):
        """BIT_CHAR[1] returns 'A'."""
        assert BIT_CHAR[1] == "A"

    def test_bit_char_2_returns_b(self):
        """BIT_CHAR[2] returns 'B'."""
        assert BIT_CHAR[2] == "B"

    def test_encode_mod_without_training(self):
        """Encoding single char works without trained """
        result = encode_mod("A")
        assert result == 1

    def test_encode_mod_uppercase(self):
        """Encoding uppercase A returns 1."""
        result = encode_mod("A")
        assert result == 1

    def test_encode_mod_lowercase(self):
        """Encoding lowercase 'a' returns same as 'A'."""
        assert encode_mod("a") == encode_mod("A")

    def test_encode_mod_b(self):
        """Encoding 'B' returns 2."""
        assert encode_mod("B") == 2

    def test_decode_mod_without_training(self):
        """Decoding single bit works without trained """
        result = decode_mod(1)
        assert result == "A"

    def test_decode_mod_2_returns_b(self):
        """Decoding 2 returns 'B'."""
        assert decode_mod(2) == "B"

    def test_decode_mod_4_returns_c(self):
        """Decoding 4 returns 'C'."""
        assert decode_mod(4) == "C"

    def test_single_char_roundtrip_a(self):
        """Roundtrip single 'A' works without training."""
        encoded = encode_mod("A")
        decoded = decode_mod(encoded)
        assert decoded == "A"

    def test_single_char_roundtrip_z(self):
        """Roundtrip single 'Z' works without training."""
        encoded = encode_mod("Z")
        decoded = decode_mod(encoded)
        assert decoded == "Z"


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_script(self):
        """Test a complete KScript example."""
        source = """
            load /path/to/model.bin

            greeting = hello world
            question > greeting

            save /path/to/output.bin
        """
        result = interpret_script(source)

        assert result.load_paths == ["/path/to/model.bin"]
        assert result.save_path == "/path/to/output.bin"
        assert len(result.model) >= 1

    def test_nlp_style_script(self):
        """Test NLP-style script with inline comments."""
        source = """
            sit > V(erb)
            N(oun) = cat dog
            S(entence) => sit N(oun)
        """
        result = interpret_script(source)

        # Comments should be stripped
        assert len(result.model) >= 3

    def test_multiline_kline(self):
        """Test multi-line KLine with indented nodes."""
        source = """
            MHALL = SVO =>
                S < M
                V < H
                O < ALL
        """
        result = interpret_script(source)

        # Should have multiple KLines
        assert len(result.model) >= 4  # MHALL, S, V, O (at minimum)

    def test_parse_with_nested_statements(self):
        """Test parsing with inline comments."""
        script = parse(
        """(Mary had a little lamb - anything in brackets is a comment btw)
        MHALL = SVO =>
            M = S(ubject)
            H = V(erb)
            ALL = O(bject) =>
                A = D(et)
                L = M(od)
                L = O
        """)
        kline = script.root
        assert kline.sig.name == "MHALL"
        # First relationship (=) has SVO as identifier
        assert kline.nodes[0].identifier.name == "SVO"
        # Second relationship (=>) has nested KLines
        assert len(kline.relationships) == 2
        nested = kline.relationships[1].nodes
        # M, H, ALL are children of MHALL's =>
        assert len(nested) == 3  # M, H, ALL
        # Check M = S
        assert nested[0].nested_kline.sig.name == "M"
        assert nested[0].nested_kline.nodes[0].identifier.name == "S"
        # Check H = V
        assert nested[1].nested_kline.sig.name == "H"
        assert nested[1].nested_kline.nodes[0].identifier.name == "V"
        # Check ALL = O => (nested S2 with A, L, L)
        all_kline = nested[2].nested_kline
        assert all_kline.sig.name == "ALL"
        assert all_kline.nodes[0].identifier.name == "O"
        # ALL has 2 relationships: = O, => [A, L, L]
        assert len(all_kline.relationships) == 2
        # The => relationship has A, L, L as siblings
        all_s2 = all_kline.relationships[1]
        assert all_s2.significance.value == "=>"
        assert len(all_s2.nodes) == 3  # A, L, L
        assert all_s2.nodes[0].nested_kline.sig.name == "A"
        assert all_s2.nodes[1].nested_kline.sig.name == "L"
        assert all_s2.nodes[2].nested_kline.sig.name == "L"
