import os, json, time, uuid, base64, io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from google.genai.errors import APIError
import chromadb
from gtts import gTTS

# --- 1. 8D VECTOR REGISTRY ---
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="apex_oracle_v6")
collection.add(
    embeddings=[[172, 85, 15, 400, 45, 98, 120, 95], [165, 90, 10, 520, 42, 92, 110, 88]],
    documents=["Stephen Curry", "Michael Jordan ('88)"],
    ids=["curry", "mj"]
)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

@app.get("/")
def root():
    return {"status": "Apex Oracle Engine Active", "docs": "/docs"}

# Plain JSON schema for Gemini (no additionalProperties — API does not support it)
ORACLE_SCHEMA = {
    "type": "object",
    "properties": {
        "athlete_action": {"type": "string"},
        "stats": {
            "type": "object",
            "properties": {
                "release_velocity": {"type": "string"},
                "shot_arc": {"type": "string"},
                "nil_valuation": {"type": "string"},
                "fluidity": {"type": "integer"},
            },
            "required": ["release_velocity", "shot_arc", "nil_valuation", "fluidity"],
        },
        "scout_report": {"type": "string"},
        "athlete_feedback": {"type": "string"},
    },
    "required": ["athlete_action", "stats", "scout_report", "athlete_feedback"],
}

def _normalize_analysis(data: dict) -> dict:
    """Ensure frontend always gets a consistent shape with safe defaults."""
    stats = data.get("stats") or {}
    return {
        **data,
        "athlete_action": data.get("athlete_action") or "—",
        "stats": {
            "release_velocity": stats.get("release_velocity") or "—",
            "shot_arc": stats.get("shot_arc") or "—",
            "nil_valuation": stats.get("nil_valuation") or "—",
            "fluidity": stats.get("fluidity") or 0,
        },
        "scout_report": data.get("scout_report") or "—",
        "athlete_feedback": data.get("athlete_feedback") or "—",
        "pro_match": data.get("pro_match") or "—",
    }

@app.post("/analyze-video")
async def analyze_video(video: UploadFile = File(...)):
    safe_name = f"temp_{uuid.uuid4()}.mp4"
    with open(safe_name, "wb") as b:
        b.write(await video.read())
    try:
        try:
            video_file = client.files.upload(file=safe_name)
            while getattr(getattr(video_file, "state", None), "name", None) == "PROCESSING":
                time.sleep(1)
                video_file = client.files.get(name=video_file.name)
            prompt = "Act as an elite NBA Biomechanics Director. Extract 8D joint metrics, Release Velocity (ms), Shot Arc, and NIL valuation. Include a brief scout_report and athlete_feedback. Output valid JSON with keys: athlete_action, stats (release_velocity, shot_arc, nil_valuation, fluidity), scout_report, athlete_feedback."
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[video_file, prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=ORACLE_SCHEMA),
            )
        except APIError as e:
            status = getattr(e, "code", 503)
            msg = "Analysis service temporarily unavailable. Please try again in a moment."
            if status == 429:
                msg = "Rate limit exceeded. Please try again later."
            raise HTTPException(status_code=min(status, 503), detail=msg)
        except Exception as e:
            raise HTTPException(status_code=503, detail="Analysis service error. Please try again.")
        data = json.loads(response.text)
        fluidity = (data.get("stats") or {}).get("fluidity", 0)
        results = collection.query(
            query_embeddings=[[172, 85, 15, 400, 45, fluidity, 110, 90]],
            n_results=1,
        )
        data["pro_match"] = (results.get("documents") or [[]])[0][0] if results.get("documents") else "—"
        return _normalize_analysis(data)
    finally:
        if os.path.exists(safe_name):
            os.remove(safe_name)

def _placeholder_card_svg(match: str, score) -> str:
    """Return base64-encoded SVG placeholder when Imagen is unavailable."""
    safe = (match or "Prospect")[:40].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="400" height="533" viewBox="0 0 400 533">
      <rect width="400" height="533" fill="#0a0a0a" stroke="#0df246" stroke-width="2"/>
      <text x="200" y="180" text-anchor="middle" fill="#0df246" font-family="sans-serif" font-size="14">FLUIDITY {score}</text>
      <text x="200" y="220" text-anchor="middle" fill="#94a3b8" font-family="sans-serif" font-size="12">{safe}</text>
      <text x="200" y="280" text-anchor="middle" fill="#64748b" font-family="sans-serif" font-size="11">Image generation unavailable</text>
      <text x="200" y="305" text-anchor="middle" fill="#475569" font-family="sans-serif" font-size="10">Imagen API quota or access limited</text>
    </svg>"""
    return base64.b64encode(svg.encode("utf-8")).decode("utf-8")


@app.post("/generate-metric-card")
async def generate_metric_card(req: dict):
    match = (req or {}).get("match", "Prospect")
    score = (req or {}).get("score", 0)
    prompt = f"Futuristic holographic sports card for {match}. Text: 'FLUIDITY {score}'. Neon accents."
    try:
        result = client.models.generate_images(
            model="imagen-4.0-generate-001",
            prompt=prompt,
            config=types.GenerateImagesConfig(number_of_images=1),
        )
        b64 = base64.b64encode(result.generated_images[0].image.image_bytes).decode("utf-8")
        return {"status": "success", "image_base64": b64}
    except Exception:
        return {"status": "fallback", "image_base64": _placeholder_card_svg(match, score)}

@app.post("/generate-audio-brief")
async def generate_audio_brief(body: dict = None):
    body = body or {}
    text = (body.get("text") or body.get("evaluation") or "").strip()
    if not text:
        return {"status": "error", "message": "Missing 'text' in request body."}
    try:
        tts = gTTS(text=text, lang="en", tld="co.uk")
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return {"status": "success", "audio_base64": base64.b64encode(fp.read()).decode("utf-8")}
    except Exception as e:
        return {"status": "error", "message": str(e)}