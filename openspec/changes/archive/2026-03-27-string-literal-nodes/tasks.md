## 1. Lexer Changes

- [x] 1.1 Add `STRING_LITERAL` to `TokenType` enum in `token.py`
- [x] 1.2 Add `_read_string_literal()` method to `Lexer` class in `lexer.py`
- [x] 1.3 Update `_next_token()` to recognize lowercase/mixed identifiers as STRING_LITERAL
- [x] 1.4 Add tests for STRING_LITERAL tokenization

## 2. Parser Changes

- [x] 2.1 Handle `TokenType.STRING_LITERAL` in `_try_parse_node()` in `parser.py`
- [x] 2.2 Update `_try_collect_leading_nodes()` to include STRING_LITERAL tokens (SKIPPED - leading nodes are SIGNATURE-only per design)
- [x] 2.3 Add tests for parsing constructs with string literal nodes

## 3. Compiler Integration

- [x] 3.1 Verify compiler handles StringLiteral from STRING_LITERAL tokens correctly
- [x] 3.2 Add tests for compiling string literal nodes to unpacked sequences
