# Voice AI Assistant (Whisper -> Ollama -> TTS)

Flow: caller audio -> Whisper (speech-to-text) -> Ollama local LLM -> edge-tts (text-to-speech) -> spoken reply.
Per-session memory keeps the last few turns. Every response reports latency for each stage.

Choose English or हिन्दी in the browser before speaking or typing. The selection
sets the transcription language, reply language, and speaking voice. API clients
can pass `language` as `en` or `hi` in the `/voice` form or `/chat` JSON.
Without this field, Whisper detects the input language and TTS uses the default voice.

## Setup steps

1. Install Python 3.10 or newer.
2. Install Ollama from https://ollama.com and pull a small free model:
   ollama pull llama3.2:3b
3. Create a virtual environment and install dependencies:
   python -m venv venv
   venv\Scripts\activate          (Windows)   |   source venv/bin/activate   (Mac/Linux)
   pip install -r requirements.txt
4. Make sure Ollama is running (it starts automatically on Windows/Mac; otherwise run `ollama serve`).
5. Start the API:
   uvicorn app:app --reload --port 8000
   (The first voice request downloads the Whisper model. Text chat does not need it.)
6. Open http://localhost:8000/docs and try:
   - GET /health        -> checks the API and Ollama
   - POST /chat         -> text-only test, e.g. {"text": "Hello", "speak": true}
   - POST /voice        -> upload a .wav / .mp3 / .m4a and get transcript, reply, audio_url, latency
7. Or test from the terminal:
   python test_client.py sample.wav
   Run it again with the printed session_id to see conversation memory working:
   python test_client.py sample2.wav <session_id>

## Settings (environment variables, all optional)

OLLAMA_MODEL   default llama3.2:3b (try qwen2.5:3b or phi3 if slow)
WHISPER_MODEL  tiny / base / small (smaller = faster, less accurate)
TTS_VOICE      default en-IN-NeerjaNeural (Indian English); try en-US-JennyNeural
HINDI_TTS_VOICE  default hi-IN-SwaraNeural (Hindi)
MAX_TURNS      conversation pairs kept in memory, default 6
SYSTEM_PROMPT  change the bot's persona or domain

## Notes

- Speech-to-text uses faster-whisper, an optimized implementation of OpenAI Whisper.
- edge-tts needs an internet connection. For fully offline speech, swap text_to_speech() for pyttsx3 or Piper.
- Session memory is in-process (a dict), so it resets when the server restarts. Use Redis for production.
