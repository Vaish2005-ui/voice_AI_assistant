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
from pathlib import Path
from typing import Dict, List, Optional

import edge_tts
import httpx
from faster_whisper import WhisperModel
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from fastapi.responses import RedirectResponse

# ---------------- Config (override with environment variables) ----------------
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
WHISPER_SIZE = os.getenv("WHISPER_MODEL", "base")  # tiny / base / small
TTS_VOICE = os.getenv("TTS_VOICE", "en-IN-NeerjaNeural")
MAX_TURNS = int(os.getenv("MAX_TURNS", "6"))  # user+assistant pairs kept in memory
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful voice assistant on a phone call. "
    "Reply in 1-3 short, natural spoken sentences. No markdown, no lists, no emojis.",
)

AUDIO_DIR = Path("audio_out")
AUDIO_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Voice AI Assistant", version="1.0")

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("index.html")

# Loaded once at startup. int8 on CPU keeps it light.
whisper_model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")

# session_id -> list of {"role": ..., "content": ...}
sessions: Dict[str, List[dict]] = {}


# ---------------- Pipeline stages ----------------
def _transcribe_sync(path: str) -> str:
    segments, _info = whisper_model.transcribe(path, beam_size=1, vad_filter=True)
    return " ".join(seg.text.strip() for seg in segments).strip()


async def speech_to_text(path: str) -> str:
    return await asyncio.to_thread(_transcribe_sync, path)


async def ask_llm(session_id: str, user_text: str) -> str:
    history = sessions.setdefault(session_id, [])
    history.append({"role": "user", "content": user_text})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history[-MAX_TURNS * 2:]
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{OLLAMA_BASE}/api/chat",
                json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
            )
            r.raise_for_status()
    except httpx.HTTPError as e:
        history.pop()  # don't keep a user turn that got no reply
        raise HTTPException(502, f"Ollama error: {e}. Is `ollama serve` running?")
    reply = r.json()["message"]["content"].strip()
    history.append({"role": "assistant", "content": reply})
    sessions[session_id] = history[-MAX_TURNS * 2:]
    return reply


async def text_to_speech(text: str) -> str:
    name = f"{uuid.uuid4().hex}.mp3"
    await edge_tts.Communicate(text, TTS_VOICE).save(str(AUDIO_DIR / name))
    return name


def ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


# ---------------- Endpoints ----------------
@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            await client.get(f"{OLLAMA_BASE}/api/tags")
        ollama = "up"
    except httpx.HTTPError:
        ollama = "down"
    return {"status": "ok", "ollama": ollama, "model": OLLAMA_MODEL, "whisper": WHISPER_SIZE}


@app.post("/voice")
async def voice(audio: UploadFile = File(...), session_id: Optional[str] = Form(None)):
    """Audio in -> transcript + reply text + reply audio + latency per stage."""
    session_id = session_id or uuid.uuid4().hex
    suffix = Path(audio.filename or "in.wav").suffix or ".wav"
    total = time.perf_counter()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(audio.file, tmp)
        tmp_path = tmp.name
    try:
        t = time.perf_counter()
        transcript = await speech_to_text(tmp_path)
        stt_ms = ms(t)
    finally:
        os.remove(tmp_path)

    if not transcript:
        raise HTTPException(422, "No speech detected in the audio.")

    t = time.perf_counter()
    reply = await ask_llm(session_id, transcript)
    llm_ms = ms(t)

    t = time.perf_counter()
    audio_name = await text_to_speech(reply)
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


@app.post("/chat")
async def chat(body: ChatIn):
    """Text-only path for testing the LLM (and optionally TTS) without audio."""
    session_id = body.session_id or uuid.uuid4().hex
    total = time.perf_counter()
    t = time.perf_counter()
    reply = await ask_llm(session_id, body.text)
    out = {"session_id": session_id, "reply": reply, "latency_ms": {"llm": ms(t)}}
    if body.speak:
        t = time.perf_counter()
        out["audio_url"] = f"/audio/{await text_to_speech(reply)}"
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
    sessions.pop(session_id, None)
    return {"cleared": session_id}
