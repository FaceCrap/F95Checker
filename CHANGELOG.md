### Added:
- Insecure SSL option (by @Willy-JL)

### Updated:
- Rename "Last Played" to "Last Launched" and add timeline event when manually setting launched date (by @Willy-JL)
- Save URIs and relative exe path correctly in Launched timeline event (by @Willy-JL)

### Fixed:
- Fix Windows start with system setting and quotes usage (#156 by @oneshoekid & @Willy-JL)
- Redraw screen when DDL is extracting to show when complete (by @Willy-JL)
- Improved Developer name sanitization for some characters like `()[]{}\` (by @Willy-JL)

### Removed:
- Nothing

### Known Issues:
- Sorting can be sporadically break/change with some actions, seems to be memory corruption inside (py)imgui, re-launch to fix it or change sorting manually
- MacOS webview in frozen binaries remains blank, run from source instead