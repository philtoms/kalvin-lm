## 1. Decompiler Integration

- [x] 1.1 Add Decompiler import to KScriptApp
- [x] 1.2 Initialize shared Decompiler with Mod32Tokenizer in KScriptApp.__init__
- [x] 1.3 Add _decompile_response(klines: list[KLine]) -> str helper method

## 2. Response Display

- [x] 2.1 Modify ResponseItem to store decompiled KScript source
- [x] 2.2 Update ResponseItem.compose to display multi-line KScript instead of JSON
- [x] 2.3 Update ResponseItem CSS for variable-height multi-line content
- [x] 2.4 Modify ResponsesRegion.add_response to accept decompiled source

## 3. Click Handler

- [x] 3.1 Update on_responses_region_response_clicked to append KScript source to editor
- [x] 3.2 Ensure ResponseClicked message carries decompiled source

## 4. App Integration

- [x] 4.1 Wire decompiler into _feed_klines flow
- [x] 4.2 Wire decompiler into action_step_script flow
- [x] 4.3 Test end-to-end: compile → run → decompile → display → click → edit

## 5. Testing

- [x] 5.1 Add test for decompiler integration in app
- [x] 5.2 Test multi-construct response display
- [x] 5.3 Test click appends KScript (not JSON) to editor
