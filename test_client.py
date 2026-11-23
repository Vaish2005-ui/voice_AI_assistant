"""Usage: python test_client.py path/to/audio.wav [session_id]"""
import sys
import requests

BASE = "http://localhost:8000"

path = sys.argv[1]
session = sys.argv[2] if len(sys.argv) > 2 else None

with open(path, "rb") as f:
    r = requests.post(
        f"{BASE}/voice",
        files={"audio": f},
        data={"session_id": session} if session else {},
        timeout=300,
    )
r.raise_for_status()
data = r.json()
print("Session   :", data["session_id"])
print("You said  :", data["transcript"])
print("Bot reply :", data["reply"])
print("Latency   :", data["latency_ms"])

audio = requests.get(BASE + data["audio_url"], timeout=60)
with open("reply.mp3", "wb") as out:
    out.write(audio.content)
print("Saved reply audio to reply.mp3")
