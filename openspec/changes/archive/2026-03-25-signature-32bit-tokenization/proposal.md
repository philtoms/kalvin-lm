## Why

The 64-bit signature space is currently fully consumed by character tokenization (Mod64Tokenizer), leaving no room for significance encoding. By switching to Mod32Tokenizer, character encoding uses only the lower 32 bits (bits 1-32), freeing the upper 32 bits (bits 33-63) for significance levels. This enables future ranking/matching semantics without expanding the signature width.

## What Changes

- **BREAKING**: Default tokenizer changes from `Mod64Tokenizer` to `Mod32Tokenizer`
- Signatures will be encoded in lower 32 bits only (bits 1-32)
- Upper 32 bits (bits 33-63) become reserved for significance encoding
- Character collisions increase slightly (mod 32 vs mod 64) but remain acceptable for the alphabet

## Capabilities

### New Capabilities

- `signature-significance`: Reserved upper 32-bit space for significance encoding in 64-bit signatures

### Modified Capabilities

- `kscript-compiler`: Default tokenizer requirement changes from Mod64Tokenizer to Mod32Tokenizer

## Impact

**Affected Code:**
- `src/kscript/__init__.py` - KScript class default tokenizer
- `src/kscript/compiler.py` - Compiler default tokenizer
- `src/kscript/__main__.py` - CLI default tokenizer

**Compatibility:**
- Existing `.bin` files encoded with Mod64Tokenizer will decode incorrectly with Mod32Tokenizer
- JSON/JSONL files are unaffected (stored as decoded strings)
- Consumers must recompile `.ks` sources or migrate binary files

**Dependencies:**
- No external dependency changes
- `kalvin.mod_tokenizer` already provides both tokenizer variants
