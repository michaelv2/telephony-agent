# TODO

## Turn detection improvements

- [ ] **Transcript accumulation**: Buffer for a short window (~500ms) after transcription. If another segment arrives in that window, concatenate before responding. Catches mid-sentence pauses without relying solely on linguistic heuristics.
- [ ] **LLM-based turn detection**: Send transcript to a fast classifier (e.g. Haiku) with the question "is this a complete conversational turn?" before routing to the main agent. More accurate than suffix heuristics, adds ~200-300ms.

## Call transfer improvements

- [ ] **Whisper announcement on transfer**: Use Twilio's `<Dial url="...">` to play a short context message to the user before bridging (e.g. "Incoming transfer: dental appointment scheduling"). Gives the user context about who's on the line and why.
