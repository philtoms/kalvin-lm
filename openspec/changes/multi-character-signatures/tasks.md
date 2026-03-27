## 1. MCS Expansion Logic

- [x] 1.1 Implement `_expand_mcs(sig: str)` method to check and emit MCS canonization
- [x] 1.2 Implement `_emit_mcs_identities(sig: str)` method to emit component identity entries
- [x] 1.3 Integrate MCS expansion in `_compile_script()` before processing constructs
- [x] 1.4 Integrate MCS expansion in `_compile_construct()` for signature nodes (SKIPPED - not needed)

## 2. Entry Emission

- [x] 2.1 Emit MCS canonization entry with S2 significance (CANONIZE_FWD construct type)
- [x] 2.2 Emit component identity entries with S4 significance (UNDERSIGN/identity)
- [x] 2.3 Ensure MCS entries emitted before constructs that reference them

## 3. Testing

- [x] 3.1 Add test for simple MCS expansion (`ABC` → canonization + identities)
- [x] 3.2 Add test for MCS in construct position (`ABC => X`)
- [x] 3.3 Add test for single-character bypass (`A` → no expansion)
