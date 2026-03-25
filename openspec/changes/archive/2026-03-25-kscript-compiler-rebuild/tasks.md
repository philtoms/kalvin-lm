## 1. Token Module

- [x] 1.1 Define TokenType enum with all 14 token types (COUNTERSIGN, CANONIZE_FWD, CANONIZE_BWD, CONNOTATE_FWD, CONNOTATE_BWD, UNDERSIGN, SIGNATURE, STRING, NUMBER, COMMENT, NEWLINE, INDENT, DEDENT, EOF)
- [x] 1.2 Create Token dataclass with type, value, line, column fields
- [x] 1.3 Write unit tests for token types

## 2. Lexer Module

- [x] 2.1 Implement Lexer class with source, pos, line, column state
- [x] 2.2 Implement multi-char operator tokenization (==, =>, <=) before single-char
- [x] 2.3 Implement SIGNATURE tokenization for [A-Z]+ with optional inline comment
- [x] 2.4 Implement STRING tokenization with escape sequence support
- [x] 2.5 Implement NUMBER tokenization for [0-9]+
- [x] 2.6 Implement COMMENT tokenization with nested paren handling
- [x] 2.7 Implement Python-style INDENT/DEDENT tracking with stack
- [x] 2.8 Implement NEWLINE and EOF tokens
- [x] 2.9 Write lexer unit tests covering all token types
- [x] 2.10 Write lexer integration tests for indentation scenarios

## 3. AST Module

- [x] 3.1 Define Signature dataclass (id, comment, line, column)
- [x] 3.2 Define StringLiteral dataclass (id, line, column)
- [x] 3.3 Define NumberLiteral dataclass (id, line, column)
- [x] 3.4 Define Node type alias (Signature | StringLiteral | NumberLiteral)
- [x] 3.5 Define ConstructType enum with all 6 operators
- [x] 3.6 Define Construct dataclass with has_leading_nodes flag
- [x] 3.7 Define Script dataclass with recursive subscripts
- [x] 3.8 Define KScriptFile dataclass

## 4. Parser Module

- [x] 4.1 Implement Parser class with token list and position
- [x] 4.2 Implement top-level script parsing (column 1 signatures)
- [x] 4.3 Implement construct parsing with operator detection
- [x] 4.4 Implement node parsing (signature, string, number)
- [x] 4.5 Implement immediate binding for chained constructs
- [x] 4.6 Implement backward canonize with leading nodes detection
- [x] 4.7 Implement recursive subscript parsing with INDENT/DEDENT
- [x] 4.8 Write parser unit tests for each construct type
- [x] 4.9 Write parser integration tests for nested subscripts

## 5. Compiler Module

- [x] 5.1 Implement CompiledEntry class extending KLine
- [x] 5.2 Implement encode() with PACKED_BIT for signatures
- [x] 5.3 Implement encode() with literal encoding (ord << 1) for strings/numbers
- [x] 5.4 Implement decode() with auto-detection of packed vs literal
- [x] 5.5 Implement Compiler class with tokenizer dependency
- [x] 5.6 Implement identity script compilation (no constructs → {sig: None})
- [x] 5.7 Implement countersign compilation (bidirectional, literal recovery)
- [x] 5.8 Implement canonize forward compilation (multi-node)
- [x] 5.9 Implement canonize backward compilation (with leading nodes)
- [x] 5.10 Implement connotate forward/backward compilation
- [x] 5.11 Implement undersign compilation
- [x] 5.12 Implement subscript signatures as nodes
- [x] 5.13 Implement duplicate signature preservation
- [x] 5.14 Write compiler unit tests for each construct type
- [x] 5.15 Write compiler integration test with full example from spec

## 6. Output Module

- [x] 6.1 Implement write_json() for JSON array format
- [x] 6.2 Implement write_jsonl() for line-delimited format
- [x] 6.3 Implement write_bin() with KSC1 magic header
- [x] 6.4 Implement binary entry encoding (node_type byte, variable nodes)
- [x] 6.5 Implement read_bin() for binary format
- [x] 6.6 Implement read_json() for JSON/JSONL input
- [x] 6.7 Write output unit tests for each format
- [x] 6.8 Write round-trip tests for all formats

## 7. API Module

- [x] 7.1 Implement KScript class constructor with source detection
- [x] 7.2 Implement file type detection (.ks, .json, .jsonl, .bin)
- [x] 7.3 Implement base model extension
- [x] 7.4 Implement output() with format detection by suffix
- [x] 7.5 Implement to_jsonl() method
- [x] 7.6 Implement entries property
- [x] 7.7 Write API unit tests for all construction modes
- [x] 7.8 Write API integration tests with base model extension

## 8. CLI Module

- [x] 8.1 Implement argument parsing (input, -out)
- [x] 8.2 Implement default output path generation
- [x] 8.3 Implement error handling with exit codes
- [x] 8.4 Write CLI smoke tests

## 9. Integration

- [x] 9.1 Verify KScript API compatibility with existing TUI
- [x] 9.2 Verify CompiledEntry works with Kalvin.rationalise()
- [x] 9.3 Run full test suite against spec scenarios
- [x] 9.4 Update documentation in CLAUDE.md if needed
