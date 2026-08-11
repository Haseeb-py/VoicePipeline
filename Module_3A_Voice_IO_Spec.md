# Module 3A — Voice I/O Pipeline

**Part of:** EchoAI — AI-Driven Voice Sales & Support System
**Depends on:** Module 2 (Script & Persona Setup)

### Development Mode

Module 3A is designed to operate independently before Module 2 is completed.

During development, Module 3A shall include a built-in Persona Manager with predefined persona configurations for standalone testing.

When Module 2 becomes available, the built-in Persona Manager will be replaced by persona configurations supplied by Module 2 without requiring architectural changes to Module 3A.

**Feeds:** Module 3B (Conversation State & Script Flow) and Module 4 (Emotion Engine) — both consume this module's live transcript stream
**Primary constraint:** latency. Every design decision below is made to minimize round-trip time from "customer stops speaking" to "AI starts responding."

---

## 1. Objective

Build a real-time, bidirectional voice pipeline that:
1. Captures live customer audio over WebRTC
2. Transcribes it to text in real time (STT)
3. Synthesizes the AI's text response back into audio (TTS)
4. Supports mid-sentence interruption from the customer (barge-in)
5. Applies persona/tone settings from Module 2 to the AI's voice
6. Outputs a live transcript stream for downstream modules to consume


This module is responsible for Voice Input and Output.

In the final EchoAI architecture, AI response generation is performed exclusively by Module 5.

During standalone development, however, Module 3A temporarily uses the Groq API as a development response generator so that the complete Voice I/O pipeline can be developed and demonstrated independently.

This temporary implementation must be replaceable without modifying the rest of Module 3A.


---

## 2. Tech Stack

| Layer | Technology | Role |
|---|---|---|
| Pipeline framework | **Pipecat AI** (`pipecat-ai`, Python) | Orchestrates the STT → LLM → TTS loop; handles streaming, VAD, interruption out of the box |
| Real-time transport | **LiveKit** (Cloud) | WebRTC audio in/out between customer and server |
| STT (primary) | **Deepgram** (streaming API) | Real-time speech-to-text |
| STT (fallback) | **OpenAI Whisper** | Non-blocking secondary pass for low-confidence Deepgram results |
| TTS | **Deepgram Aura** (or equivalent low-latency TTS — ElevenLabs / PlayHT / Cartesia if already decided) | Real-time text-to-speech, streamed |
| LLM (response generation, used only for this module's own turn-taking logic, not sales logic) | **Groq API** | Fast inference — Groq's hardware is optimized for low-latency generation |LLM (Temporary Development Only)

Groq API is used as a temporary response generator until Module 5 is completed.

After Module 5 is implemented, Groq will no longer be called directly by Module 3A. Instead, Module 3A will obtain response text from Module 5 while preserving the exact same Voice I/O pipeline.
| Language/runtime | **Python 3.11**, `asyncio` | Pipecat's pipeline model is async; blocking calls anywhere break the latency budget |
| Containerization | **Docker** | Deployment consistency |
| Telephony (later milestone, not required for initial build) | **Twilio Programmable Voice** | Bridges to real PSTN phone numbers |


## Persona Management

Module 3A must not hardcode persona behavior.

Instead, personas shall be represented as configuration objects.

Each persona configuration contains all information required to control both the language model and the text-to-speech engine.

Example attributes include:

- Persona name
- Description
- System prompt for the LLM
- Preferred TTS voice model
- Speech rate
- Temperature
- Greeting style
- Optional future parameters

The active persona configuration should be loaded by a Persona Manager component.

During development, Persona Manager loads local configuration files.

After Module 2 is implemented, Persona Manager receives the same configuration from Module 2 without requiring any changes to the Voice I/O pipeline.

Example Persona Configuration

{
  "id": "professional_support",

  "display_name": "Professional Support",

  "description": "Formal BPO customer support representative.",

  "system_prompt": "...",

  "tts": {
      "provider": "Deepgram",
      "voice": "aura-2-thalia-en",
      "speech_rate": 1.0
  },

  "llm": {
      "temperature": 0.4
  }
}

## Persona Manager

Module 3A must access persona information only through PersonaManager.

Module 3A must never read JSON files directly.

Development

PersonaManager

↓

Local Persona JSON Files

Production

PersonaManager

↓

Module 2 API

This abstraction ensures Module 3A remains unchanged after Module 2 is integrated.


## Development Persona Selector

To allow independent testing before Module 2 exists, Module 3A shall provide a temporary Persona Selector within the testing UI.

The selector allows switching between predefined BPO personas.

Example personas:

- Professional Support
- Friendly Support
- Technical Support
- Sales Representative
- Complaint Resolution
- Customer Success
- Luxury Concierge

Changing the selected persona shall immediately update:

- LLM system prompt
- TTS voice model
- Speech settings
- Any other persona-dependent behavior

No application restart should be required.

This selector exists only during standalone development.

Once Module 2 is integrated, the selector will be removed.

### Key concepts the implementation must respect
- **Streaming, not batch**: use Deepgram's *interim* transcript results, not just final ones. Use TTS's streaming playback (start playing audio as it's generated), not "wait for full clip."
- **VAD (Voice Activity Detection)**: determines when the customer has stopped talking (to trigger a response) and when they've started talking again mid-AI-response (to trigger barge-in).
- **Barge-in**: if VAD detects customer speech while TTS is playing, immediately cancel/stop the TTS output and return to listening.
- **Persistent WebSocket connections**: open the connection to Deepgram/TTS once per call and reuse it — do not reconnect per turn.

---

## 3. Prerequisites / Environment

### Accounts & API keys needed
```
DEEPGRAM_API_KEY=
GROQ_API_KEY=
LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
TTS_API_KEY=            # Deepgram Aura or chosen alternative
```

### Local setup
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install pipecat-ai livekit-agents python-dotenv
```

### Reference starting point
Do not build the pipeline from a blank file. Start from Pipecat's official example pipeline that combines LiveKit transport + Deepgram STT + a TTS provider (available in Pipecat's public examples repository). Adapt that working example rather than re-deriving the architecture.

---

## 4. Functional Requirements

| ID | Requirement |
|---|---|
| FR-1 | Open a LiveKit WebRTC audio session per call |
| FR-2 | Stream customer audio to Deepgram STT in real time; consume interim + final transcripts |
| FR-3 | Run VAD continuously to detect start/end of customer speech |
| FR-4 | On detected end-of-speech, hand the transcript off downstream (to the response-generation step) |
| FR-5 | Stream the AI's response text to TTS and stream the resulting audio back over the WebRTC session as it's generated |
| FR-6 | On detecting customer speech while TTS is playing, immediately stop TTS output (barge-in) and resume listening |
| FR-7 | Run Whisper as a non-blocking fallback check when Deepgram's confidence score is below a defined threshold; never let it gate the primary response path |
| FR-8 | Apply voice/tone settings pulled from Module 2's persona configuration when initializing the TTS session |
| FR-9 | Emit a structured, timestamped transcript event stream (see §7 schema) for Module 3B and Module 4 to consume |
| FR-10 | Maintain the STT/TTS connections as persistent WebSockets for the duration of the call, not per-turn |
FR-11

Provide a temporary Persona Selector UI during standalone development.

FR-12

Load persona definitions from external configuration files.

FR-13

Changing persona shall immediately update:

- LLM prompt
- TTS voice
- Speech settings

without restarting the application.

---

## 5. Non-Functional Requirements

- **Latency target: sub-500ms** from end-of-customer-speech to first-audio-byte of AI response.
- **Server region**: deploy in the same region as the STT/TTS/LLM providers to minimize network round-trip.
- Pipeline must run on `asyncio` throughout — no blocking I/O calls in the audio processing path.
- Whisper fallback must run truly async and must not add latency to the primary Deepgram path under any circumstance.

---

## 6. Build Plan 

1. Clone/adapt Pipecat's LiveKit + Deepgram + TTS example pipeline or you can use livekit agents framework
2. STT

↓

Groq

↓

Deepgram Aura

↓

Speaker
3. Wire in persona/tone config from Module 2 into the TTS initialization.
4. Stabilize enough to survive 5–10 consecutive test calls without crashing.

1. Add timestamp logging at three checkpoints: `audio_received`, `stt_result`, `tts_first_byte_sent`.
2. Run 10 test calls; review the logged deltas to find the actual bottleneck before changing anything.
3. Confirm streaming (not batch) is active end-to-end — this is usually already correct in the Pipecat template; verify it wasn't broken during Day 1 edits.
4. Implement/verify barge-in (VAD-triggered TTS cancellation).
5. Confirm server region matches provider regions.
6. Confirm persistent WebSocket usage (no reconnect-per-turn).
7. Test from a real network (phone hotspot or remote host), not localhost — WebRTC behaves differently under real jitter.


## Development Testing UI

The standalone testing interface should include:

• Start Call

• End Call

• Persona Selector

• Current Connection Status

• Listening Status

• Live Interim Transcript

• Final Transcript

• Generated AI Response

• AI Speaking Indicator

• Transcript Event Log

• Latency Metrics

The interface should allow developers to verify every component of Module 3A independently.



### Explicitly out of scope for this initial build
- Twilio/PSTN bridging
- Concurrency/load testing beyond a light sanity check (2–3 simultaneous calls)
- Whisper fallback tuning beyond "it works and doesn't block"
## Temporary Response Generation

During development, every request sent to Groq shall contain:

1. Persona system prompt

2. Conversation history

3. Latest customer transcript

Groq must never receive only the customer transcript.

The active persona configuration supplies the system prompt.

Changing persona automatically changes the prompt sent to Groq.

---

## 7. Output Schema — Transcript Event Stream

This is the interface Module 3B and Module 4 consume. Emit one event per finalized utterance turn:

```json
{
  "call_id": "string",
  "turn_id": "integer",
  "speaker": "customer | ai",
  "text": "string",
  "is_final": "boolean",
  "confidence": "float (0-1)",
  "timestamp_start": "ISO 8601",
  "timestamp_end": "ISO 8601"
}
```

Interim (non-final) results may also be emitted with `is_final: false` for consumers that want live partial updates (e.g. Module 4's live sentiment tracking); Module 3B should generally only act on `is_final: true` events.

---
## Response Generator Abstraction

Module 3A must not call Groq directly.

Instead, all response generation shall go through a ResponseGenerator interface.

Development

ResponseGenerator

↓

GroqResponseGenerator

Production

ResponseGenerator

↓

Module5ResponseGenerator

This allows Module 5 to replace Groq without modifying any other component.

## 8. Definition of Done

- [ ] One full test call: customer speaks, AI responds, round trip is sub-second
- [ ] Barge-in works: interrupting the AI mid-sentence stops its speech promptly and the system starts listening again
- [ ] Persona/tone from Module 2 is audibly reflected in the TTS voice
- [ ] Whisper fallback triggers on low-confidence Deepgram results without adding perceptible delay to normal turns
- [ ] Transcript events are emitted in the schema above and can be consumed by a downstream listener
- [ ] Latency checkpoints are logged and reviewed at least once (not just implemented — actually run and read)
- [ ] Tested on a real network connection, not just localhost
- [ ] Runs for at least 5–10 consecutive calls without crashing
-- [ ] Persona switching immediately changes both AI wording and synthesized voice.

- [ ] Groq produces persona-consistent responses during standalone development.

- [ ] PersonaManager can swap between local JSON and Module 2 without code changes elsewhere.

---



## 9. Known Pitfalls to Avoid

- Using only *final* STT results instead of streaming interim ones — kills latency
- Batching TTS (waiting for full synthesis before playing) instead of streaming playback
- Reconnecting to Deepgram/TTS on every turn instead of keeping the WebSocket open for the call's duration
- Letting Whisper's fallback check block or delay the primary response path
- Testing only on localhost, where WebRTC latency looks artificially better than on a real network
- VAD tuned too aggressively (false interruptions from background noise) or too passively (AI talks over the customer) — this needs real listening/testing, not just default config
