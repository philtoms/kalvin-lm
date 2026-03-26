## 1. Core Structure

- [x] 1.1 Create `src/kscript/decompiler.py` module
- [x] 1.2 Define `DecompilerError` exception class for error surfacing
- [x] 1.3 Define `Decompiler` class with `decompile(klines, tokenizer) -> str` method

## 2. Significance Detection

- [x] 2.1 Implement `get_significance_level(signature) -> str` to detect S1/S2/S3/S4
- [x] 2.2 Implement `get_construct_op(level) -> str` mapping levels to operators
- [x] 2.3 Add significance bit constants (S1_BIT=63, S2_BIT=62, S3_BIT=61)

## 3. Primary Processing

- [x] 3.1 Build lookup dict: `signature -> KLine`
- [x] 3.2 Implement recursive `emit(sig, indent)` function
- [x] 3.3 Track processed signatures to avoid duplicates
- [x] 3.4 Handle missing entries as identity (just output decoded signature)

## 4. Construct Handling

- [x] 4.1 Implement countersign pair detection and deduplication
- [x] 4.2 Implement canonize subscript emission for multi-node constructs
- [x] 4.3 Implement connotate inline emission
- [x] 4.4 Implement identity emission for S4 level

## 5. Error Surfacing

- [x] 5.1 Detect and emit `!!! ORPHAN:` for unreachable KLines
- [x] 5.2 Detect and emit `!!! BROKEN:` for missing chain references

## 6. Testing

- [x] 6.1 Unit tests for significance level detection
- [x] 6.2 Unit tests for countersign pair handling
- [x] 6.3 Unit tests for subscript reconstruction
- [x] 6.4 Unit tests for identity recovery
- [x] 6.5 Unit tests for error surfacing
- [x] 6.6 Integration test: full decompilation from compiled KScript
