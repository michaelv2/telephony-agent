# TODO

## Turn detection improvements

- [ ] **Transcript accumulation**: Buffer for a short window (~500ms) after transcription. If another segment arrives in that window, concatenate before responding. Catches mid-sentence pauses without relying solely on linguistic heuristics.
- [x] **LLM-based turn detection**: Uses Haiku to classify whether a transcript is a complete conversational turn. Replaces the suffix heuristic. Runs during the debounce window. Falls back to flush after 4s timeout if held.

## Latency optimizations

- [ ] **Speculative Claude call during Haiku classification**: For ambiguous turns that go to Haiku, start the Claude API call in parallel. If Haiku says COMPLETE, Claude is already streaming (~1s saved). If INCOMPLETE, cancel it. Low waste since the fast-path handles obvious cases.
- [ ] **TTS prefetching**: Start TTS for sentence N+1 while sending audio chunks for sentence N. Reduces gaps between sentences on multi-sentence responses (~0.5-1s savings per sentence).
- [ ] **First-chunk clause splitting**: For the first chunk only, split on clause boundaries (commas) instead of sentence boundaries to get something to TTS faster. Revert to sentence boundaries after the first chunk is playing (~0.3-0.5s off time-to-first-audio).

## Call transfer improvements

- [ ] **Whisper announcement on transfer**: Use Twilio's `<Dial url="...">` to play a short context message to the user before bridging (e.g. "Incoming transfer: dental appointment scheduling"). Gives the user context about who's on the line and why.
