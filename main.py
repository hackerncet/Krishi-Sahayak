import os
import base64
import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Allow all origins for testing (lock down later)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables (set on DigitalOcean App Platform)
AGENT_ENDPOINT = os.getenv("AGENT_ENDPOINT")
AGENT_API_KEY = os.getenv("AGENT_API_KEY")
BHASHINI_ASR_KEY = os.getenv("BHASHINI_ASR_KEY")
BHASHINI_TTS_KEY = os.getenv("BHASHINI_TTS_KEY")
BHASHINI_USER_ID = os.getenv("BHASHINI_USER_ID")

LANGUAGE_MAP = {
    "hi": "Hindi", "bn": "Bengali", "te": "Telugu", "ta": "Tamil",
    "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
    "pa": "Punjabi", "or": "Odia", "as": "Assamese", "ur": "Urdu",
    "sa": "Sanskrit", "sd": "Sindhi", "ne": "Nepali", "ks": "Kashmiri",
    "doi": "Dogri", "mni": "Manipuri", "sat": "Santali", "brx": "Bodo",
    "mai": "Maithili", "gom": "Konkani"
}

class ChatRequest(BaseModel):
    prompt: str
    language: str = "hi"

# ── Call your Agent Platform agent ──
async def call_agent(user_prompt: str) -> str:
    if not AGENT_ENDPOINT or not AGENT_API_KEY:
        raise HTTPException(status_code=500, detail="Agent not configured")
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            AGENT_ENDPOINT,
            headers={
                "Authorization": f"Bearer {AGENT_API_KEY}",
                "Content-Type": "application/json"
            },
            json={"messages": [{"role": "user", "content": user_prompt}]},
            timeout=30.0
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Agent error: {resp.text}")
        data = resp.json()
        return data["choices"][0]["message"]["content"]

# ── Bhashini Speech-to-Text ──
async def bhashini_stt(audio_base64: str, language: str) -> str:
    url = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/compute"
    payload = {
        "pipelineTasks": [{
            "taskType": "asr",
            "config": {
                "language": {"sourceLanguage": language},
                "audioFormat": "wav",
                "samplingRate": 16000
            }
        }],
        "inputData": {"audio": [{"audioContent": audio_base64}]}
    }
    headers = {
        "Authorization": f"Bearer {BHASHINI_ASR_KEY}",
        "Content-Type": "application/json",
        "userID": BHASHINI_USER_ID or ""
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload, headers=headers, timeout=20.0)
        if resp.status_code != 200:
            raise Exception(f"Bhashini ASR failed: {resp.text}")
        result = resp.json()
        return result["pipelineResponse"][0]["output"][0]["source"]

# ── Bhashini Text-to-Speech ──
async def bhashini_tts(text: str, language: str) -> str:
    url = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/compute"
    payload = {
        "pipelineTasks": [{
            "taskType": "tts",
            "config": {
                "language": {"sourceLanguage": language},
                "gender": "female"
            }
        }],
        "inputData": {"input": [{"source": text}]}
    }
    headers = {
        "Authorization": f"Bearer {BHASHINI_TTS_KEY}",
        "Content-Type": "application/json",
        "userID": BHASHINI_USER_ID or ""
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload, headers=headers, timeout=30.0)
        if resp.status_code != 200:
            raise Exception(f"Bhashini TTS failed: {resp.text}")
        result = resp.json()
        audio_b64 = result["pipelineResponse"][0]["audio"][0]["audioContent"]
        return f"data:audio/wav;base64,{audio_b64}"

# ── Endpoints ──
@app.post("/chat/voice")
async def chat_voice(audio: UploadFile = File(...), language: str = Form("hi")):
    audio_bytes = await audio.read()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    transcript = await bhashini_stt(audio_b64, language)
    if not transcript.strip():
        raise HTTPException(status_code=400, detail="Could not transcribe audio.")
    ai_text = await call_agent(transcript)
    audio_url = await bhashini_tts(ai_text, language)
    return {
        "transcript": transcript,
        "response_text": ai_text,
        "audio_url": audio_url,
        "language": language
    }

@app.post("/chat/text")
async def chat_text(request: ChatRequest):
    ai_text = await call_agent(request.prompt)
    return {"response": ai_text, "language": request.language}

@app.get("/")
async def root():
    return {"status": "Krishi Sahayak is live", "agent": AGENT_ENDPOINT is not None}