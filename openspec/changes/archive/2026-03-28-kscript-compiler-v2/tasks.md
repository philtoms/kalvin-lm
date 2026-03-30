# Implementation Tasks

## 1. Parser Implementation

- [x] Implement new `CLn_based node collection` in `parser_v2.py`
- [x] Implement eager emit compilation model
- [x] Remove "current owner" state tracking
- [x] Support all construct types (identity, countersign, undersign, canonize_fwd, canonize_bwd, connotate_fwd, connotate_bwd)

## 2. Compiler implementation

- [x] Implement `compiler_v2.py` with two-step compilation
- [x] Support MCS expansion for all signatures
- [x] Handle BWD emission inline
- [x] Process subscripts recursively

- [x] Update `__init__.py` to entry point
- [x] Create `tests/test_kscript_v2.py` with new test cases
- [x] Deprecate old modules once passing

## 3. Test coverage

- [x] Port existing test cases from `test_kscript.py` to new file
- [x] Add integration test
