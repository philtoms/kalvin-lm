# Spec: KLine Execution

## Purpose

Execution engine for feeding compiled KScript entries to Kalvin. Supports automatic, stepped, and halted execution modes with a clear state machine.

## Requirements

### Requirement: Compile editor content to KLines
The system SHALL compile the editor's KScript content into a list of CompiledEntry objects.

#### Scenario: Compile valid script
- **WHEN** user clicks Run button with valid KScript in editor
- **THEN** the script is compiled using KScript API
- **AND** resulting CompiledEntry list is prepared for execution

### Requirement: Feed KLines to Kalvin automatically
The system SHALL automatically feed compiled KLines to Kalvin one at a time when Run is clicked.

#### Scenario: Automatic execution
- **WHEN** user clicks Run button
- **THEN** each compiled entry is converted to a KLine
- **AND** each KLine is fed to `kalvin.rationalise()`
- **AND** execution proceeds without user intervention
- **AND** UI remains responsive during execution

### Requirement: Halt automatic execution
The system SHALL allow user to halt automatic execution.

#### Scenario: User halts during execution
- **WHEN** execution is running
- **AND** user clicks Halt button
- **THEN** execution stops after current KLine completes
- **AND** remaining KLines are preserved for step or resume

### Requirement: Step through KLines manually
The system SHALL allow user to execute one KLine at a time when halted.

#### Scenario: Step execution
- **WHEN** execution is halted with pending KLines
- **AND** user clicks Step button
- **THEN** the next pending KLine is fed to Kalvin
- **AND** execution remains halted

### Requirement: Resume automatic execution
The system SHALL allow user to resume automatic execution from halted state.

#### Scenario: Resume execution
- **WHEN** execution is halted with pending KLines
- **AND** user clicks Resume button
- **THEN** automatic execution continues from the next pending KLine

### Requirement: Execution state machine
The system SHALL maintain execution state with clear transitions.

#### Scenario: State transitions
- **WHEN** app starts
- **THEN** execution state is IDLE
- **WHEN** user clicks Run from IDLE
- **THEN** state transitions to RUNNING
- **WHEN** user clicks Halt from RUNNING
- **THEN** state transitions to HALTED
- **WHEN** user clicks Resume from HALTED
- **THEN** state transitions to RUNNING
- **WHEN** user clicks Step from HALTED
- **THEN** state remains HALTED
- **WHEN** all KLines are processed
- **THEN** state transitions to IDLE
