# Spec: KScript Editor

## Purpose

Text editing area for KScript source code in the KScript TUI application. Supports loading, saving, and direct input of KScript content.

## Requirements

### Requirement: Editor displays KScript source as plain text
The system SHALL provide a text editing area that displays KScript source code without syntax highlighting.

#### Scenario: Editor displays loaded script
- **WHEN** user loads a `.ks` file
- **THEN** the editor displays the file contents as plain text

#### Scenario: Editor accepts direct input
- **WHEN** user types in the editor
- **THEN** the text is displayed in the editor area

### Requirement: Load KScript file from disk
The system SHALL allow users to load a `.ks` file into the editor via file dialog.

#### Scenario: User loads script file
- **WHEN** user clicks Load.ks button and selects a file
- **THEN** the file contents are displayed in the editor
- **AND** the default directory for subsequent loads is set to that file's directory

#### Scenario: Sticky default directory
- **WHEN** user loads a script from a directory
- **AND** user clicks Load.ks again
- **THEN** the file dialog opens in the previously used directory

### Requirement: Save KScript file to disk
The system SHALL allow users to save editor contents as a `.ks` file.

#### Scenario: User saves script file
- **WHEN** user clicks Save.ks button and provides a filename
- **THEN** the editor contents are saved to the specified file
- **AND** the file has `.ks` extension

### Requirement: Append content to editor
The system SHALL allow programmatic appending of content to the editor.

#### Scenario: Response click appends to editor
- **WHEN** user clicks a response item
- **THEN** the clicked item's JSON representation is appended to the editor content
- **AND** existing editor content is preserved
