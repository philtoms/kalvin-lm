## ADDED Requirements

### Requirement: Decompile rationalise response to KScript
The system SHALL decompile the list of KLines returned by `Kalvin.rationalise()` into human-readable KScript source code.

#### Scenario: Basic response decompilation
- **WHEN** rationalise returns a list of KLines
- **THEN** system uses Decompiler to convert KLines to KScript source
- **AND** decompiled source preserves construct operators (==, =>, >, =)
- **AND** decompiled source preserves subscript structure

#### Scenario: Empty response handling
- **WHEN** rationalise returns an empty list
- **THEN** decompiled output is empty string

### Requirement: Response decompilation uses shared tokenizer
The system SHALL use a shared Mod32Tokenizer instance for all decompilation operations.

#### Scenario: Shared tokenizer initialization
- **WHEN** KScriptApp initializes
- **THEN** a Decompiler with Mod32Tokenizer is created
- **AND** same tokenizer is reused for all response decompilations

### Requirement: Decompilation errors surface with markers
The system SHALL display decompiler error markers (e.g., `!!! BROKEN`, `!!! ORPHAN`) in the response output.

#### Scenario: Broken reference display
- **WHEN** decompiler encounters a signature with no token mapping
- **THEN** response displays `!!! BROKEN: expected {signature}`

#### Scenario: Orphaned KLine display
- **WHEN** decompiler finds an unreachable KLine
- **THEN** response displays `!!! ORPHAN: {signature}`
