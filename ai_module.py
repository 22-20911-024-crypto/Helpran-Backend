"""
ai_module.py
============
Kids Speech Learning App - Final AI Module
Fixed:
  - PASS_SCORE = 70 (Urdu + English relaxed)
  - Urdu TTS retry logic
  - Better Urdu comparison
  - Pakistani accent fix
"""

import os
import re
import sys
import uuid
import asyncio
import edge_tts
from faster_whisper import WhisperModel
from difflib import SequenceMatcher

# =============================================
# WINDOWS FIX
# =============================================
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# =============================================
# SETTINGS
# =============================================

PASS_SCORE        = 70
AUDIO_FOLDER      = "audio_files"
RECORDINGS_FOLDER = "recordings"
WHISPER_SIZE      = "small"

VOICES = {
    "english": "en-US-AriaNeural",
    "urdu"   : "ur-PK-UzmaNeural",
    "en"     : "en-US-AriaNeural",
    "ur"     : "ur-PK-UzmaNeural",
}
WHISPER_LANG = {
    "english": "en",
    "urdu"   : "ur",
    "en"     : "en",
    "ur"     : "ur",
}

# =============================================
# SETUP
# =============================================

os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(RECORDINGS_FOLDER, exist_ok=True)

print("Whisper model load ho raha hai...")
whisper_model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
print("Whisper ready! ✅")

# =============================================
# FUNCTION 1: TTS
# =============================================

async def ai_speak(input_data, language: str = "english") -> dict:
    try:
        text  = _extract_text(input_data)
        lang  = language.lower().strip()
        voice = VOICES.get(lang, VOICES["english"])

        if not text:
            return {
                "success"   : False,
                "error_code": "NO_TEXT",
                "message"   : "Text nahi mila",
                "text"      : ""
            }

        safe_name  = re.sub(r"[^a-zA-Z0-9\u0600-\u06FF]", "_", text.strip())[:50]
        audio_path = os.path.join(AUDIO_FOLDER, f"{lang}_{safe_name}.mp3")

        if os.path.exists(audio_path):
            print(f"✅ Audio already hai: {audio_path}")
        else:
            print(f"🔊 Audio ban rahi hai: '{text}' [{lang}]")
            success = False
            for attempt in range(3):
                try:
                    tts = edge_tts.Communicate(text, voice=voice)
                    await tts.save(audio_path)
                    success = True
                    break
                except Exception as e:
                    print(f"⚠️ Attempt {attempt+1} failed: {e}")
                    await asyncio.sleep(1)

            if not success:
                return {
                    "success"   : False,
                    "error_code": "TTS_FAILED",
                    "message"   : "Audio nahi ban saki",
                    "text"      : text
                }

            print(f"✅ Audio ban gayi: {audio_path}")

        return {
            "success"    : True,
            "audio_path" : audio_path,
            "text"       : text,
            "language"   : lang,
            "error_code" : None
        }

    except Exception as e:
        print(f"❌ TTS Error: {e}")
        return {
            "success"   : False,
            "error_code": "TTS_FAILED",
            "message"   : str(e),
            "text"      : str(input_data)
        }

# =============================================
# FUNCTION 2: STT — Audio Suno, Score Do
# =============================================

def ai_check(audio_path: str, expected_input, language: str = "english") -> dict:
    try:
        expected_text = _extract_text(expected_input)
        lang          = language.lower().strip()
        whisper_lang  = WHISPER_LANG.get(lang, "en")

        if not expected_text:
            return {
                "success"   : False,
                "error_code": "NO_EXPECTED",
                "message"   : "Expected text nahi mila",
                "score"     : 0,
                "correct"   : False
            }

        if not os.path.exists(audio_path):
            return {
                "success"   : False,
                "error_code": "FILE_NOT_FOUND",
                "message"   : f"File nahi mili: {audio_path}",
                "score"     : 0,
                "correct"   : False
            }

        print(f"🎤 Sun raha hai: {audio_path} [{lang}]")
        print(f"📝 Expected: '{expected_text}'")

        # ← initial_prompt add kiya — word hint deta hai Whisper ko
        segments, _ = whisper_model.transcribe(
            audio_path,
            beam_size=5,
            language=whisper_lang,
            initial_prompt=f"The child is saying the word: {expected_text}",
            temperature=0.0,
            vad_filter=True,
            vad_parameters={
                "threshold"              : 0.3,
                "min_speech_duration_ms" : 100,
                "min_silence_duration_ms": 500
            }
        )
        heard_text = " ".join([s.text.strip() for s in segments])
        print(f"👂 Suna: '{heard_text}'")

        expected_clean = _clean_text(expected_text)
        heard_clean    = _clean_text(heard_text)
        score          = _smart_score(expected_clean, heard_clean, lang)
        print(f"📊 Score: {round(score)}%")

        return {
            "success"   : True,
            "expected"  : expected_clean,
            "heard"     : heard_clean,
            "score"     : round(score),
            "correct"   : score >= PASS_SCORE,
            "pass_score": PASS_SCORE,
            "language"  : lang,
            "error_code": None
        }

    except Exception as e:
        print(f"❌ STT Error: {e}")
        return {
            "success"   : False,
            "error_code": "STT_FAILED",
            "message"   : str(e),
            "score"     : 0,
            "correct"   : False
        }

# =============================================
# FUNCTION 3: Result
# =============================================

def ai_result(score: int) -> dict:
    if score >= PASS_SCORE:
        print(f"🌟 Good Job! Score: {score}%")
        return {
            "status" : "correct",
            "message": "Good Job! 🌟",
            "score"  : score,
            "next"   : True
        }
    else:
        print(f"🔄 Try Again! Score: {score}%")
        return {
            "status" : "repeat",
            "message": "Try Again! 🔄",
            "score"  : score,
            "next"   : False
        }

# =============================================
# HELPER: Unique Recording Path
# =============================================

def get_recording_path() -> str:
    unique_name = str(uuid.uuid4())[:8]
    return os.path.join(RECORDINGS_FOLDER, f"{unique_name}.wav")

# =============================================
# HELPER FUNCTIONS
# =============================================

def _extract_text(input_data) -> str:
    if isinstance(input_data, str):
        return input_data.strip()
    elif isinstance(input_data, dict):
        for key in ["text", "word", "sentence", "content", "data"]:
            if key in input_data:
                return str(input_data[key]).strip()
    return str(input_data).strip()


def _clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z\u0600-\u06FF\s]", "", text)
    return " ".join(text.split())


def _smart_score(expected: str, heard: str, lang: str = "en") -> float:
    if not heard:
        return 0.0
    if expected == heard:
        return 100.0

    # Direct similarity
    direct = SequenceMatcher(None, expected, heard).ratio() * 100

    # Sliding window
    heard_words    = heard.split()
    expected_words = expected.split()
    exp_len        = len(expected_words)
    best           = direct

    for i in range(len(heard_words) - exp_len + 1):
        chunk = " ".join(heard_words[i:i + exp_len])
        s = SequenceMatcher(None, expected, chunk).ratio() * 100
        if s > best:
            best = s

    # Urdu bonus
    if lang in ["ur", "urdu"] and best > 20:
        best = min(best * 1.2, 100)

    # Pakistani English accent fix
    if lang in ["en", "english"]:
        exp_first   = expected[:2].lower() if len(expected) >= 2 else expected
        heard_first = heard[:2].lower() if len(heard) >= 2 else heard
        if exp_first == heard_first:
            best = max(best, 75.0)
        if len(expected.split()) == 1 and len(heard.split()) == 1:
            len_diff = abs(len(expected) - len(heard))
            if len_diff <= 2 and best > 30:
                best = min(best * 1.3, 100)

    # Pakistani accent map
    ACCENT_MAP = {
        "cat"  : ["cat", "kat", "get", "cut", "cot", "gat", "kit", "cap"],
        "dog"  : ["dog", "dok", "doc", "dag", "log"],
        "cow"  : ["cow", "go", "ko", "gao", "gow", "how"],
        "bird" : ["bird", "berd", "beard", "wird", "word"],
        "fish" : ["fish", "fis", "wish", "dish"],
        "hen"  : ["hen", "han", "pen", "when", "then"],
        "happy": ["happy", "hapi", "happi", "hepy"],
        "sad"  : ["sad", "sat", "said", "bad"],
        "angry": ["angry", "angri", "hungry"],
        "red"  : ["red", "read", "rid", "bed"],
        "blue" : ["blue", "bloo", "blow", "blew"],
        "green": ["green", "grin", "greet", "grain"],
        "eye"  : ["eye", "i", "ay", "aye"],
        "ear"  : ["ear", "here", "year", "are"],
        "nose" : ["nose", "noze", "nos", "knows"],
        "hand" : ["hand", "han", "and", "sand"],
        "head" : ["head", "had", "hed", "bed"],
        "water": ["water", "vader", "wader", "waiter"],
        "food" : ["food", "good", "foot", "flood"],
        "help" : ["help", "kelp", "held", "hell"],
        "yes"  : ["yes", "yes", "yeah", "yeas"],
        "no"   : ["no", "know", "now", "nope"],
        "stop" : ["stop", "shop", "top", "step"],
    }

    exp_lower   = expected.lower().strip()
    heard_lower = heard.lower().strip()

    if exp_lower in ACCENT_MAP:
        if any(h in heard_lower for h in ACCENT_MAP[exp_lower]):
            best = max(best, 85.0)
            print(f"✅ Accent match: '{heard_lower}' → '{exp_lower}'")

    return best