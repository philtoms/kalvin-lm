# Spec: Kalvin Persistence

## Purpose

Save and load Kalvin model state in the KScript TUI. Allows users to persist their trained model across sessions.

## Requirements

### Requirement: Save Kalvin model state
The system SHALL allow user to save the current Kalvin model state to disk.

#### Scenario: Save to binary format
- **WHEN** user clicks Save.k button and provides a filename with `.bin` extension
- **THEN** the Kalvin model is serialized to binary format
- **AND** the file is saved to the specified location

#### Scenario: Save to JSON format
- **WHEN** user clicks Save.k button and provides a filename with `.json` extension
- **THEN** the Kalvin model is serialized to JSON format
- **AND** the file is saved to the specified location

### Requirement: Load Kalvin model state
The system SHALL allow user to load a previously saved Kalvin model from disk.

#### Scenario: Load from binary format
- **WHEN** user clicks Load.k button and selects a `.bin` file
- **THEN** the Kalvin model is deserialized from binary format
- **AND** the loaded model replaces the current model state

#### Scenario: Load from JSON format
- **WHEN** user clicks Load.k button and selects a `.json` file
- **THEN** the Kalvin model is deserialized from JSON format
- **AND** the loaded model replaces the current model state

### Requirement: Kalvin state accumulates across runs
The system SHALL preserve Kalvin model state across multiple Run operations.

#### Scenario: State accumulates
- **WHEN** user runs a script that adds KLines to the model
- **AND** user runs another script
- **THEN** the second run's KLines are added to the existing model
- **AND** previous model state is not lost

#### Scenario: State persists after clear
- **WHEN** user clicks Clear button
- **AND** user runs another script
- **THEN** Kalvin model retains all previously rationalized KLines
