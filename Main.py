"""
Main.py
=======
Kids Speech Learning App - FastAPI Backend
-------------------------------------------
Local run:
    uvicorn Main:app --reload

Railway run (automatic):
    uvicorn Main:app --host 0.0.0.0 --port $PORT

APIs:
    POST /speak    -> TTS audio (English + Urdu)
    POST /check    -> Score + Progress save
    GET  /content  -> Flutter ko words do
    POST /content  -> Teacher words add kare
    GET  /progress -> Parent dashboard ke liye
"""

import os
import json
import shutil
import uvicorn
from datetime import date, datetime
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials, db
from ai_module import ai_speak, ai_check, ai_result, get_recording_path

# =============================================
# APP SETUP
# =============================================

app = FastAPI(title="Kids Speech Learning API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================
# FIREBASE SETUP
# =============================================
# Local: serviceAccountKey.json file se load hota hai
# Railway: FIREBASE_CREDENTIALS env variable se load hota hai (poora JSON string)

firebase_creds_env = os.environ.get("FIREBASE_CREDENTIALS")

if firebase_creds_env:
    # Railway / production: env variable se JSON parse karo
    cred_dict = json.loads(firebase_creds_env)
    cred = credentials.Certificate(cred_dict)
    print("✅ Firebase credentials loaded from environment variable")
else:
    # Local development: file se load karo
    cred = credentials.Certificate("serviceAccountKey.json")
    print("✅ Firebase credentials loaded from local file")

firebase_admin.initialize_app(cred, {
    "databaseURL": "https://hopelearn-app-default-rtdb.firebaseio.com"
})
print("✅ Firebase connected!")

# =============================================
# API 1: POST /speak
# TTS — English ya Urdu
# =============================================

@app.post("/speak")
async def speak(data: dict):
    """
    Flutter bheje:
        { "text": "Cat",  "language": "english" }
        { "text": "بلی", "language": "urdu"    }

    Response: mp3 audio file
    """
    language = data.get("language", "english")
    result   = await ai_speak(data, language)

    if not result["success"]:
        return result

    return FileResponse(
        result["audio_path"],
        media_type="audio/mpeg",
        filename=os.path.basename(result["audio_path"])
    )

# =============================================
# API 2: POST /check
# STT — Score + Progress Save
# =============================================

@app.post("/check")
async def check(
    audio      : UploadFile = File(...),
    expected   : str        = Form(...),
    language   : str        = Form("english"),   # english ya urdu
    module_name: str        = Form("General"),   # Animals, Sentences etc
    child_name : str        = Form("Child")      # Bacche ka naam
):
    """
    Flutter bheje:
        audio       -> recording file
        expected    -> "Cat" ya "بلی"
        language    -> "english" ya "urdu"
        module_name -> "Animals"
        child_name  -> "Ali"

    Response:
        {
            "success" : True,
            "expected": "cat",
            "heard"   : "cat",
            "score"   : 95,
            "correct" : True,
            "status"  : "correct",
            "message" : "Good Job! 🌟",
            "next"    : True
        }
    """
    # Step 1: Recording save karo
    rec_path = get_recording_path()
    with open(rec_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    # Step 2: AI check karo
    check_result = ai_check(rec_path, expected, language)
    final_result = ai_result(check_result["score"])

    # Step 3: Progress Firebase mein save karo
    try:
        db.reference(f"progress/{child_name}/{module_name}").push({
            "word"    : expected,
            "score"   : check_result["score"],
            "correct" : check_result["correct"],
            "heard"   : check_result["heard"],
            "language": language,
            "date"    : str(date.today()),
            "time"    : datetime.now().strftime("%H:%M")
        })
        print(f"✅ Progress saved: {child_name} → {module_name} → {expected} → {check_result['score']}%")
    except Exception as e:
        print(f"⚠️ Progress save error: {e}")

    # Step 4: Cleanup
    if os.path.exists(rec_path):
        os.remove(rec_path)

    # Step 5: Flutter ko bhejo
    return {
        **check_result,
        **final_result
    }

# =============================================
# API 3: POST /content
# Teacher words add kare
# =============================================

@app.post("/content")
async def add_content(data: dict):
    """
    Flutter bheje:
        {
            "screen_name": "Animals",
            "type"       : "word",
            "language"   : "english",
            "items"      : ["Cat", "Dog", "Bird"]
        }

    Urdu:
        {
            "screen_name": "جانور",
            "type"       : "word",
            "language"   : "urdu",
            "items"      : ["بلی", "کتا", "پرندہ"]
        }
    """
    try:
        screen_name = data.get("screen_name", "unnamed")
        language    = data.get("language", "english")

        # Firebase mein save
        db.reference(f"screens/{screen_name}").set(data)

        # TTS — sab words ki audio pehle se banao
        for item in data.get("items", []):
            await ai_speak(item, language)

        return {
            "success": True,
            "message": f"'{screen_name}' save ho gaya!"
        }

    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }

# =============================================
# API 4: GET /content
# Flutter ko words do
# =============================================

@app.get("/content")
async def get_content():
    """
    Response:
        {
            "success": True,
            "screens": [ ... ]
        }
    """
    try:
        ref  = db.reference("screens")
        data = ref.get()

        if not data:
            return {"success": True, "screens": []}

        return {
            "success": True,
            "screens": list(data.values())
        }

    except Exception as e:
        return {"success": False, "message": str(e)}

# =============================================
# API 5: GET /progress/{child_name}
# Parent Dashboard ke liye
# =============================================

@app.get("/progress/{child_name}")
async def get_progress(child_name: str):
    """
    Parent dashboard:
        GET /progress/Ali

    Response:
        {
            "success"     : True,
            "child_name"  : "Ali",
            "progress"    : {
                "Animals" : [
                    { "word": "Cat", "score": 95, "correct": true, "date": "2026-04-01" }
                ]
            },
            "total_words" : 10,
            "correct_words": 8,
            "average_score": 87
        }
    """
    try:
        ref  = db.reference(f"progress/{child_name}")
        data = ref.get()

        if not data:
            return {
                "success"      : True,
                "child_name"   : child_name,
                "progress"     : {},
                "total_words"  : 0,
                "correct_words": 0,
                "average_score": 0
            }

        # Stats calculate karo
        all_scores  = []
        total       = 0
        correct     = 0

        for module, attempts in data.items():
            if isinstance(attempts, dict):
                for attempt in attempts.values():
                    total += 1
                    all_scores.append(attempt.get("score", 0))
                    if attempt.get("correct"):
                        correct += 1

        avg = round(sum(all_scores) / len(all_scores)) if all_scores else 0

        return {
            "success"      : True,
            "child_name"   : child_name,
            "progress"     : data,
            "total_words"  : total,
            "correct_words": correct,
            "average_score": avg
        }

    except Exception as e:
        return {"success": False, "message": str(e)}

# =============================================
# RUN
# =============================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("Main:app", host="0.0.0.0", port=port, reload=False)
