# Delta Spec: KScript Compiler

## Purpose

Modifies the KScript compiler to use Mod32Tokenizer as the default, confining character encoding to lower 32 bits.

## ADDED Requirements

### Requirement: Default Tokenizer
The KScript compiler SHALL use Mod32Tokenizer as the default tokenizer for signature encoding.

#### Scenario: KScript API default tokenizer
- **WHEN** `KScript("A == B")` is called without tokenizer argument
- **THEN** Mod32Tokenizer is used for encoding
- **AND** signatures are encoded in bits 1-32

#### Scenario: Compiler default tokenizer
- **WHEN** `Compiler()` is instantiated without tokenizer argument
- **THEN** Mod32Tokenizer is used for encoding

#### Scenario: CLI default tokenizer
- **WHEN** `python -m kscript script.ks` is called
- **THEN** Mod32Tokenizer is used for encoding
- **AND** output binary files use 32-bit character encoding

---

### Requirement: Tokenizer Override
The system SHALL allow explicit tokenizer override when needed.

#### Scenario: Custom tokenizer in API
- **WHEN** `KScript("A == B", tokenizer=custom_tokenizer)` is called
- **THEN** the provided tokenizer is used instead of the default

#### Scenario: Custom tokenizer in Compiler
- **WHEN** `Compiler(tokenizer=custom_tokenizer)` is instantiated
- **THEN** the provided tokenizer is used for compilation
