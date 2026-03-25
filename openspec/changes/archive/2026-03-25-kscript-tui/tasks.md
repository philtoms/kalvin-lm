## 1. Project Setup

- [x] 1.1 Create `ui/kscript/` module directory structure
- [x] 1.2 Create `ui/kscript/__init__.py` with module exports
- [x] 1.3 Create `ui/kscript/__main__.py` entry point
- [x] 1.4 Create `ui/kscript/regions/__init__.py`

## 2. Core Application

- [x] 2.1 Create `ui/kscript/app.py` with KScriptApp class skeleton
- [x] 2.2 Implement horizontal layout with placeholder regions
- [x] 2.3 Add app-level state management (Kalvin instance, execution state)
- [x] 2.4 Implement keyboard bindings (ctrl+q quit, ctrl+r run, etc.)

## 3. Toolbar Region

- [x] 3.1 Create `ui/kscript/regions/toolbar.py` with ToolbarRegion class
- [x] 3.2 Implement action buttons: Load.ks, Save.k, Load.k, Run, Halt, Step, Clear
- [x] 3.3 Add status indicator (IDLE/RUNNING/HALTED)
- [x] 3.4 Wire button events to app-level handlers

## 4. Editor Region

- [x] 4.1 Create `ui/kscript/regions/editor.py` with EditorRegion class
- [x] 4.2 Add TextArea widget for script editing (plain text)
- [x] 4.3 Implement `get_script()` method to retrieve editor content
- [x] 4.4 Implement `set_script()` method to set editor content
- [x] 4.5 Implement `append_to_script()` method for response click integration

## 5. Responses Region

- [x] 5.1 Create `ui/kscript/regions/responses.py` with ResponsesRegion class
- [x] 5.2 Add ListView widget for scrollable response display
- [x] 5.3 Implement `add_response()` method to append KLine as JSON
- [x] 5.4 Implement `clear()` method to empty the response list
- [x] 5.5 Wire click event to halt execution and append to editor

## 6. File Dialogs

- [x] 6.1 Create `ui/kscript/dialogs.py` with LoadScriptDialog class
- [x] 6.2 Create SaveStateDialog for saving Kalvin model
- [x] 6.3 Create LoadStateDialog for loading Kalvin model
- [x] 6.4 Implement sticky directory default (persist last used directory)

## 7. Execution Engine

- [x] 7.1 Implement execution state machine (IDLE → RUNNING → HALTED)
- [x] 7.2 Create `compile_script()` method using KScript API
- [x] 7.3 Create `entry_to_kline()` helper to convert CompiledEntry to KLine
- [x] 7.4 Implement async `feed_klines()` with cancellation support
- [x] 7.5 Implement `run()` action to start automatic execution
- [x] 7.6 Implement `halt()` action to pause execution
- [x] 7.7 Implement `step()` action for single KLine execution
- [x] 7.8 Implement `resume()` action to continue from halted state

## 8. Kalvin Integration

- [x] 8.1 Initialize Kalvin instance at app startup
- [x] 8.2 Wire `rationalise()` calls into execution loop
- [x] 8.3 Implement save model functionality (binary and JSON)
- [x] 8.4 Implement load model functionality (binary and JSON)

## 9. Integration & Testing

- [x] 9.1 Wire all regions together in KScriptApp
- [x] 9.2 Test load/compile/run workflow with sample `.ks` file
- [x] 9.3 Test halt/step/resume controls
- [x] 9.4 Test response click → append to editor flow
- [x] 9.5 Test Kalvin state persistence (save/load)
