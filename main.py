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
DO_MODEL_KEY = os.getenv("DO_MODEL_KEY")  # Your DigitalOcean Model Access Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Optional, for Option B

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
async def call_agent(user_prompt: str, language: str = "hi") -> str:
    if not AGENT_ENDPOINT or not AGENT_API_KEY:
        raise HTTPException(status_code=500, detail="Agent not configured")
    
    lang_name = LANGUAGE_MAP.get(language, "Hindi")
    
    # Build a clear instruction for the agent
    instruction = (
        f"You must respond in {lang_name} only. "
        "Be direct, brief, and actionable. "
        "If the user provides a photo, analyse it and give a specific diagnosis. "
        "Do not ask unnecessary questions. "
        "Keep the answer under 3 short paragraphs."
    )
    full_prompt = f"{instruction}\n\nFarmer's question: {user_prompt}"

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            AGENT_ENDPOINT,
            headers={
                "Authorization": f"Bearer {AGENT_API_KEY}",
                "Content-Type": "application/json"
            },
            json={"messages": [{"role": "user", "content": full_prompt}]},
            timeout=60.0
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

# --- New Image Analysis Function (Option A: DigitalOcean Native) ---
async def analyze_image_digitalocean(image_bytes: bytes, user_question: str, language: str) -> str:
    """Analyzes an image using DigitalOcean's native vision model (Ministral 3 14B)."""
    if not DO_MODEL_KEY:
        raise HTTPException(status_code=500, detail="DigitalOcean Model Access Key not configured")
    
    # Encode image to base64
    img_base64 = base64.b64encode(image_bytes).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{img_base64}"
    
    vision_prompt = f"You are Krishi Sahayak, an expert agricultural advisor. Analyze this crop photo carefully and answer the farmer's question. Always respond in {LANGUAGE_MAP.get(language, 'English')}. Focus on diagnosing diseases, pests, nutrient deficiencies, or growth issues visible in the image. Give organic solutions first. The farmer says: '{user_question}'."
    
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.digitalocean.com/v2/gen-ai/inference/chat/completions",
            headers={
                "Authorization": f"Bearer {DO_MODEL_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mistral-ministral-3-14b",  # Vision-capable model
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": vision_prompt},
                            {"type": "image_url", "image_url": {"url": data_uri}}
                        ]
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 500
            },
            timeout=90.0
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Vision API error: {resp.text}")
        data = resp.json()
        return data["choices"][0]["message"]["content"]

# --- Alternative: Option B (OpenAI GPT‑4o‑mini) ---
async def analyze_image_openai(image_bytes: bytes, user_question: str, language: str) -> str:
    """Analyzes an image using OpenAI's GPT‑4o‑mini (requires your own OpenAI API key)."""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    img_base64 = base64.b64encode(image_bytes).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{img_base64}"
    
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are Krishi Sahayak, an expert agricultural advisor. Analyze crop photos and answer questions in {LANGUAGE_MAP.get(language, 'English')}. Give organic solutions first."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_question},
                            {"type": "image_url", "image_url": {"url": data_uri}}
                        ]
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 500
            },
            timeout=90.0
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"OpenAI API error: {resp.text}")
        data = resp.json()
        return data["choices"][0]["message"]["content"]

# ── Endpoints ──
# --- New /chat/image Endpoint ---
@app.post("/chat/image")
async def chat_image(
    prompt: str = Form(...),
    language: str = Form("hi"),
    image: UploadFile = File(...)
):
    """Receives an image and a text prompt, then returns a vision-based diagnosis."""
    image_bytes = await image.read()
    img_base64 = base64.b64encode(image_bytes).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{img_base64}"
    lang_name = LANGUAGE_MAP.get(language, "Hindi")
    
    instruction = (
        f"Analyse the crop photo carefully. "
        f"Give a specific diagnosis and treatment in {lang_name}. "
        "Be brief, direct, and avoid unnecessary questions. "
        "If you cannot see clearly, state that, but still try to help."
    )
    full_text = f"{instruction}\n\nFarmer's additional message: {prompt}"
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": full_text},
                {"type": "image_url", "image_url": {"url": data_uri}}
            ]
        }
    ]
    
    if not AGENT_ENDPOINT or not AGENT_API_KEY:
        raise HTTPException(status_code=500, detail="Agent not configured")

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                AGENT_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {AGENT_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={"messages": messages},
                timeout=60.0
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Agent error: {resp.text}")
            data = resp.json()
            vision_response = data["choices"][0]["message"]["content"]
    except Exception as e:
        # Fallback: send text-only to your main agent if vision fails
        fallback_prompt = f"[Farmer uploaded a crop image but vision analysis failed. Their question: {prompt}]"
        vision_response = await call_agent(fallback_prompt, language)
    
    return {
        "response": vision_response,
        "language": language,
        "mode": "vision"
    }

@app.post("/chat/voice")
async def chat_voice(audio: UploadFile = File(...), language: str = Form("hi")):
    audio_bytes = await audio.read()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    transcript = await bhashini_stt(audio_b64, language)
    if not transcript.strip():
        raise HTTPException(status_code=400, detail="Could not transcribe audio.")
    ai_text = await call_agent(transcript, language)
    audio_url = await bhashini_tts(ai_text, language)
    return {
        "transcript": transcript,
        "response_text": ai_text,
        "audio_url": audio_url,
        "language": language
    }

@app.post("/chat/text")
async def chat_text(request: ChatRequest):
    ai_text = await call_agent(request.prompt, request.language)
    return {"response": ai_text, "language": request.language}

@app.get("/")
async def root():
    return {"status": "Krishi Sahayak is live", "agent": AGENT_ENDPOINT is not None}