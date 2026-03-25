## ADDED Requirements

### Requirement: Display responses as scrollable list
The system SHALL display Kalvin responses as a vertically scrollable list of JSON items.

#### Scenario: Response list grows during execution
- **WHEN** KLines are being executed
- **THEN** each response is appended to the response list
- **AND** the list is scrollable
- **AND** the most recent response is visible

### Requirement: Display KLine as JSON
The system SHALL display each response KLine in JSON format.

#### Scenario: JSON format display
- **WHEN** a KLine response is received
- **THEN** it is displayed as a JSON object
- **AND** format is `{"signature": nodes}` where nodes can be null, string, or array

### Requirement: Click response to select
The system SHALL allow user to click a response item in the list.

#### Scenario: Click response item
- **WHEN** user clicks a response item in the list
- **THEN** the item is visually selected
- **AND** execution is halted if running
- **AND** the item's JSON is appended to editor content

### Requirement: Click halts execution
The system SHALL halt automatic execution when a response is clicked.

#### Scenario: Halt on click during execution
- **WHEN** execution is RUNNING
- **AND** user clicks a response item
- **THEN** execution halts immediately
- **AND** clicked item's JSON is appended to editor

### Requirement: Clear responses list
The system SHALL allow user to clear the responses list without affecting editor or Kalvin state.

#### Scenario: Clear responses
- **WHEN** user clicks Clear button
- **THEN** the responses list is emptied
- **AND** editor content is preserved
- **AND** Kalvin model state is preserved
