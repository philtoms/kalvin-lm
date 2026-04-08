# Spec: Response Browser

## Purpose

Display area for Kalvin responses in the KScript TUI. Shows responses as a scrollable list of decompiled KScript blocks with click-to-select functionality.

## Requirements

### Requirement: Display responses as scrollable list
The system SHALL display Kalvin responses as a vertically scrollable list of decompiled KScript blocks.

#### Scenario: Response list grows during execution
- **WHEN** KLines are being executed
- **THEN** each response is decompiled to KScript source
- **AND** decompiled source is appended to the response list
- **AND** the list is scrollable
- **AND** the most recent response is visible

### Requirement: Display KLine as decompiled KScript
The system SHALL display each response KLine as decompiled KScript source text.

#### Scenario: KScript format display
- **WHEN** a KLine response is received
- **THEN** it is decompiled using the shared Decompiler
- **AND** displayed as multi-line KScript source text
- **AND** preserves construct operators and subscript indentation

### Requirement: Click response to select
The system SHALL allow user to click a response block in the list.

#### Scenario: Click response block
- **WHEN** user clicks a response block in the list
- **THEN** the block is visually selected
- **AND** execution is halted if running
- **AND** the decompiled KScript source is appended to editor content

### Requirement: Click halts execution
The system SHALL halt automatic execution when a response is clicked.

#### Scenario: Halt on click during execution
- **WHEN** execution is RUNNING
- **AND** user clicks a response item
- **THEN** execution halts immediately
- **AND** clicked item's KScript source is appended to editor

### Requirement: Clear responses list
The system SHALL allow user to clear the responses list without affecting editor or Kalvin state.

#### Scenario: Clear responses
- **WHEN** user clicks Clear button
- **THEN** the responses list is emptied
- **AND** editor content is preserved
- **AND** Kalvin model state is preserved
