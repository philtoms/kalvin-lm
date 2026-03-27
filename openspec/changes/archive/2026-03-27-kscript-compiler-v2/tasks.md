## 1. Parser Rebuild

- [x] 1.1 Define new AST nodes for recursive construct structure
- [x] 1.2 Implement `parse_construct()` with recursive grammar
- [x] 1.3 Implement FWD operator handling (`=>`, `>`, `==`, `=`)
- [x] 1.4 Implement BWD operator handling (`<=`, `<`) with construct binding
- [x] 1.5 Implement sequence parsing (construct construct*)
- [x] 1.6 Implement subscript layout normalization (INDENT/DEDENT → inline)
- [x] 1.7 Implement script boundary detection (column 1)
- [x] 1.8 Implement error recovery (literal at start, empty constructs)

## 2. Compiler Rebuild

- [x] 2.1 Define new compilation context (track current construct, accumulated nodes)
- [x] 2.2 Implement identity construct compilation
- [x] 2.3 Implement countersign (`==`) compilation with entity emission
- [x] 2.4 Implement connotate fwd (`>`) compilation with entity emission
- [x] 2.5 Implement undersign (`=`) compilation with entity emission
- [x] 2.6 Implement canonize fwd (`=>`) compilation with entity emission
- [x] 2.7 Implement canonize bwd (`<=`) compilation - binds ALL left nodes
- [x] 2.8 Implement connotate bwd (`<`) compilation - binds CLOSEST left node
- [x] 2.9 Implement MCS expansion for signatures in sig position
- [x] 2.10 Implement entity emission rules (non-FWD continuation, nodes)

## 3. Test Updates

- [x] 3.1 Add parser tests for new grammar structure
- [x] 3.2 Add parser tests for BWD construct binding
- [x] 3.3 Add parser tests for subscript normalization
- [x] 3.4 Add parser tests for script boundaries
- [x] 3.5 Add compiler tests for identity constructs
- [x] 3.6 Add compiler tests for FWD operators with entities
- [x] 3.7 Add compiler tests for BWD operators (`<=` ALL vs `<` CLOSEST)
- [x] 3.8 Add compiler tests for MCS expansion in sig position
- [x] 3.9 Add compiler tests for MCS on BWD right side (expand) vs left side (no expand)
- [x] 3.10 Add compiler tests for entity emission rules
- [x] 3.11 Add integration tests for complex chains (`A => B <= C => D`)
- [x] 3.12 Add integration tests for MCS with BWD (`AB => A B <= CD`)
- [x] 3.13 Update/remove obsolete tests from previous implementation

## 4. Edge Cases & Error Handling

- [x] 4.1 Handle literal at script start (ignore until first signature)
- [x] 4.2 Handle empty construct recovery (`A =>` → identity)
- [x] 4.3 Handle literal in sig position mid-script (treat as node)
- [ ] 4.4 Handle deeply nested constructs (recursion limits)
- [x] 4.5 Handle mixed literals and signatures in constructs

## 5. Verification

- [x] 5.1 Run full test suite and ensure all tests pass
- [x] 5.2 Verify example from spec compiles correctly
- [x] 5.3 Verify no regression in output format (JSON/JSONL/binary)
- [x] 5.4 Verify KScript API continues to work
- [x] 5.5 Verify CLI continues to work
