# TODO

## Turn detection improvements

- [ ] **Transcript accumulation**: Buffer for a short window (~500ms) after transcription. If another segment arrives in that window, concatenate before responding. Catches mid-sentence pauses without relying solely on linguistic heuristics.
- [x] **LLM-based turn detection**: Uses Haiku to classify whether a transcript is a complete conversational turn. Replaces the suffix heuristic. Runs during the debounce window. Falls back to flush after 4s timeout if held.

## Call transfer improvements

- [ ] **Whisper announcement on transfer**: Use Twilio's `<Dial url="...">` to play a short context message to the user before bridging (e.g. "Incoming transfer: dental appointment scheduling"). Gives the user context about who's on the line and why.
