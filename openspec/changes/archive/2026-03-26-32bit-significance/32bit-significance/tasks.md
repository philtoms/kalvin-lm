## 1. Int32Significance Class

- [x] 1.1 Create `Int32Significance` class in `src/kalvin/significance.py`
- [x] 1.2 Define 32-bit constants (S1=bit31, S2=bit23+range, S3=bit0+range, S4=clear)
- [x] 1.3 Define shifted 64-bit instance constants for direct signature use
- [x] 1.4 Implement `get_level(sig) -> str` with hierarchical detection
- [x] 1.5 Implement `get_construct_op(level) -> str` mapping to operators
- [x] 1.6 Define range masks (S2_RANGE, S3_RANGE, SIG_MASK, TOKEN_MASK)

## 2. Compiler Updates

- [x] 2.1 Update compiler to use `Int32Significance` constants for encoding
- [x] 2.2 Ensure S1 constructs set bit 63
- [x] 2.3 Ensure S2 constructs set bit 55 with degree in 56-62
- [x] 2.4 Ensure S3 constructs set bit 32 with degree in 33-54
- [x] 2.5 Ensure S4 constructs have no significance bits set

## 3. Decompiler Updates

- [x] 3.1 Replace ad-hoc bit constants with `Int32Significance` instance
- [x] 3.2 Update `TOKEN_MASK` to `(1 << 32) - 1`
- [x] 3.3 Update `_get_significance_level()` to use hierarchical detection
- [x] 3.4 Update `_get_construct_op()` to map S4 to `=` (undersign)
- [x] 3.5 Verify countersign deduplication still works with new bit positions

## 4. Testing

- [x] 4.1 Unit tests for `Int32Significance.get_level()` (S1/S2/S3/S4)
- [x] 4.2 Unit tests for `Int32Significance.get_construct_op()`
- [x] 4.3 Unit tests for token mask stripping significance
- [x] 4.4 Integration test: compile → decompile roundtrip with all construct types
- [x] 4.5 Update existing decompiler tests for new bit positions
