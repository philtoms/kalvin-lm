## 1. Update Default Tokenizer

- [x] 1.1 Update `KScript.__init__` in `src/kscript/__init__.py` to use `Mod32Tokenizer` as default
- [x] 1.2 Update `Compiler.__init__` in `src/kscript/compiler.py` to use `Mod32Tokenizer` as default
- [x] 1.3 Update CLI default in `src/kscript/__main__.py` to use `Mod32Tokenizer`

## 2. Update Documentation

- [x] 2.1 Add bit allocation comment to `src/kalvin/mod_tokenizer.py` documenting reserved upper bits
- [x] 2.2 Update CLAUDE.md to document 64-bit signature allocation (bits 0-32 for chars, 33-63 reserved)

## 3. Verify & Test

- [x] 3.1 Run existing test suite to verify no regressions
- [x] 3.2 Verify character encoding stays in lower 32 bits with manual test
