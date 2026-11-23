"""
Voice AI Assistant: Speech-to-Text (Whisper) -> LLM (Ollama) -> Text-to-Speech (edge-tts)
FastAPI backend with per-session memory and per-stage latency reporting.
"""
import asyncio
import os
import re
import shutil
import tempfile
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal, Optional

import edge_tts
import httpx
from faster_whisper import WhisperModel
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ---------------- Config (override with environment variables) ----------------
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
WHISPER_SIZE = os.getenv("WHISPER_MODEL", "base")  # tiny / base / small
TTS_VOICE = os.getenv("TTS_VOICE", "en-IN-NeerjaNeural")
LANGUAGE_VOICES = {"en": TTS_VOICE, "hi": os.getenv("HINDI_TTS_VOICE", "hi-IN-SwaraNeural")}
Language = Literal["en", "hi"]
LANGUAGE_PROMPTS = {
    "en": "Reply in English.",
    "hi": "Reply in natural Hindi using Devanagari script.",
}
MAX_TURNS = int(os.getenv("MAX_TURNS", "6"))  # user+assistant pairs kept in memory
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful voice assistant on a phone call. "
    "Reply in 1-3 short, natural spoken sentences. No markdown, no lists, no emojis.",
)

APP_DIR = Path(__file__).resolve().parent
AUDIO_DIR = APP_DIR / "audio_out"
AUDIO_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Voice AI Assistant", version="1.0")

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(APP_DIR / "index.html")

# Defer model loading so the page and text chat can start without a download.
@lru_cache(maxsize=1)
def get_whisper_model() -> WhisperModel:
    return WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")

# session_id -> list of {"role": ..., "content": ...}
sessions: Dict[str, List[dict]] = {}
# A fixed pool keeps lock storage bounded while serializing each session.
session_locks = tuple(asyncio.Lock() for _ in range(64))


def session_lock(session_id: str) -> asyncio.Lock:
    return session_locks[hash(session_id) % len(session_locks)]


# ---------------- Pipeline stages ----------------
def _transcribe_sync(path: str, language: Optional[Language] = None) -> str:
    segments, _info = get_whisper_model().transcribe(
        path,
        language=language,
        initial_prompt="यह हिन्दी में एक बातचीत है।" if language == "hi" else None,
        beam_size=1,
        vad_filter=True,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


async def speech_to_text(path: str, language: Optional[Language] = None) -> str:
    return await asyncio.to_thread(_transcribe_sync, path, language)


async def ask_llm(
    session_id: str, user_text: str, language: Optional[Language] = None
) -> str:
    async with session_lock(session_id):
        history = sessions.get(session_id, [])
        user_message = {"role": "user", "content": user_text}
        prompt = SYSTEM_PROMPT
        if language:
            prompt = f"{prompt} {LANGUAGE_PROMPTS[language]}"
        messages = [{"role": "system", "content": prompt}, *history, user_message]
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                r = await client.post(
                    f"{OLLAMA_BASE}/api/chat",
                    json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
                )
                r.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(502, f"Ollama error: {e}. Is `ollama serve` running?") from e

        try:
            reply = r.json()["message"]["content"].strip()
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            raise HTTPException(502, "Ollama returned an invalid reply.") from e
        if not reply:
            raise HTTPException(502, "Ollama returned an empty reply.")

        completed = [*history, user_message, {"role": "assistant", "content": reply}]
        if MAX_TURNS > 0:
            sessions[session_id] = completed[-MAX_TURNS * 2:]
        return reply


async def text_to_speech(text: str, language: Optional[Language] = None) -> str:
    name = f"{uuid.uuid4().hex}.mp3"
    voice = LANGUAGE_VOICES[language or "en"]
    await edge_tts.Communicate(text, voice).save(str(AUDIO_DIR / name))
    return name


def ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


# ---------------- Endpoints ----------------
@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            response = await client.get(f"{OLLAMA_BASE}/api/tags")
            response.raise_for_status()
        ollama = "up"
    except httpx.HTTPError:
        ollama = "down"
    return {"status": "ok", "ollama": ollama, "model": OLLAMA_MODEL, "whisper": WHISPER_SIZE}


@app.post("/voice")
async def voice(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    language: Optional[Language] = Form(None),
):
    """Audio in -> transcript + reply text + reply audio + latency per stage."""
    session_id = session_id or uuid.uuid4().hex
    suffix = Path(audio.filename or "in.wav").suffix or ".wav"
    total = time.perf_counter()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(audio.file, tmp)
        tmp_path = tmp.name
    try:
        t = time.perf_counter()
        transcript = await speech_to_text(tmp_path, language)
        stt_ms = ms(t)
    finally:
        os.remove(tmp_path)

    if not transcript:
        raise HTTPException(422, "No speech detected in the audio.")

    t = time.perf_counter()
    reply = await ask_llm(session_id, transcript, language)
    llm_ms = ms(t)

    t = time.perf_counter()
    audio_name = await text_to_speech(reply, language)
    tts_ms = ms(t)

    return {
        "session_id": session_id,
        "transcript": transcript,
        "reply": reply,
        "audio_url": f"/audio/{audio_name}",
        "latency_ms": {"stt": stt_ms, "llm": llm_ms, "tts": tts_ms, "total": ms(total)},
    }


class ChatIn(BaseModel):
    text: str
    session_id: Optional[str] = None
    speak: bool = False
    language: Optional[Language] = None


@app.post("/chat")
async def chat(body: ChatIn):
    """Text-only path for testing the LLM (and optionally TTS) without audio."""
    session_id = body.session_id or uuid.uuid4().hex
    total = time.perf_counter()
    t = time.perf_counter()
    reply = await ask_llm(session_id, body.text, body.language)
    out = {"session_id": session_id, "reply": reply, "latency_ms": {"llm": ms(t)}}
    if body.speak:
        t = time.perf_counter()
        out["audio_url"] = f"/audio/{await text_to_speech(reply, body.language)}"
        out["latency_ms"]["tts"] = ms(t)
    out["latency_ms"]["total"] = ms(total)
    return out


@app.get("/audio/{name}")
async def get_audio(name: str):
    if not re.fullmatch(r"[0-9a-f]{32}\.mp3", name):
        raise HTTPException(400, "Invalid file name")
    path = AUDIO_DIR / name
    if not path.exists():
        raise HTTPException(404, "Audio not found")
    return FileResponse(path, media_type="audio/mpeg")


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    async with session_lock(session_id):
        sessions.pop(session_id, None)
    return {"cleared": session_id}
