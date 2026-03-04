"""Tests for KScript lexer, parser, and compiler."""

import pytest

from kscript import (
    CompileResult,
    KLineExpr,
    KScript,
    Lexer,
    LexerError,
    ParseError,
    Parser,
    TokenType,
    compile_script,
    parse,
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

    def test_tokenize_attention(self):
        """Test tokenizing attention operator."""
        lexer = Lexer("?")
        tokens = lexer.tokenize()
        assert tokens[0].type == TokenType.ATTENTION

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

    def test_parse_kline_with_attention(self):
        """Test parsing KLine with attention marker."""
        script = parse("hello ?")
        kline = script.root
        assert kline.attention is True

    def test_parse_kline_with_significance_and_attention(self):
        """Test parsing KLine with significance and attention."""
        script = parse("question > greeting ?")
        kline = script.root
        assert kline.significance.value == ">"
        assert kline.attention is True

    def test_parse_multiple_statements(self):
        """Test parsing multiple statements."""
        script = parse("""
            load /path/to/model.bin
            greeting = hello world
            question > greeting ?
            save /path/to/output.bin
        """)
        assert len(script.statements) == 4

    def test_parse_with_inline_comments(self):
        """Test parsing with inline comments."""
        script = parse("sit > V(erb)")
        kline = script.root
        assert kline.sig.name == "sit"
        assert kline.nodes[0].identifier.name == "V"


class TestCompiler:
    """Tests for the KScript compiler."""

    def test_compile_simple_kline(self):
        """Test compiling simple KLine."""
        result = compile_script("hello")
        assert len(result.model) == 1

    def test_compile_kline_with_nodes(self):
        """Test compiling KLine with nodes."""
        result = compile_script("greeting = hello world")
        assert len(result.model) >= 1
        # The main kline should have nodes
        kline = result.model[0]
        assert len(kline.nodes) == 2

    def test_compile_load_save_paths(self):
        """Test that load/save paths are captured."""
        result = compile_script("""
            load /input.bin
            save /output.bin
        """)
        assert result.load_paths == ["/input.bin"]
        assert result.save_path == "/output.bin"

    def test_compile_attention_tracking(self):
        """Test that attention markers are tracked."""
        result = compile_script("hello ?")
        assert len(result.attention_klines) == 1

    def test_compile_symbol_table(self):
        """Test that symbol table is populated with KLine sig names."""
        result = compile_script("greeting = hello")
        assert "greeting" in result.symbol_table
        # Note: "hello" is a node, not a KLine sig, so it's not in symbol_table
        # but it is tracked in the token_map internally


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_script(self):
        """Test a complete KScript example."""
        source = """
            load /path/to/model.bin

            greeting = hello world
            question > greeting ?

            save /path/to/output.bin
        """
        result = compile_script(source)

        assert result.load_paths == ["/path/to/model.bin"]
        assert result.save_path == "/path/to/output.bin"
        assert len(result.model) >= 1
        assert len(result.attention_klines) == 1

    def test_nlp_style_script(self):
        """Test NLP-style script with inline comments."""
        source = """
            sit > V(erb)
            N(oun) = cat dog
            S(entence) => sit N(oun) ?
        """
        result = compile_script(source)

        # Comments should be stripped
        assert len(result.model) >= 3
        assert len(result.attention_klines) == 1

    def test_multiline_kline(self):
        """Test multi-line KLine with indented nodes."""
        source = """
            MHALL = SVO =>
                S < M
                V < H
                O < ALL
        """
        result = compile_script(source)

        # Should have multiple KLines
        assert len(result.model) >= 4  # MHALL, S, V, O (at minimum)

