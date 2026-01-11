# scripts/utils.py
import os
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import joblib
import tempfile
import speech_recognition as sr
import logging
from google import genai
from google.genai.errors import ClientError
import json
import html
try:
    from groq import Groq
except Exception:
    Groq = None

# configure Gemini client lazily to avoid requiring API key on import
# The google-genai SDK reads GOOGLE_API_KEY environment variable.
client = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Audio helpers
# -------------------------
def convert_mp3_to_wav(mp3_path, wav_path):
    """Convert mp3 -> wav using pydub (ffmpeg required)."""
    audio = AudioSegment.from_file(mp3_path)
    audio.export(wav_path, format="wav")
    return wav_path

def ensure_wav(path):
    """If path points to mp3, convert to wav and return new wav path; if wav return same."""
    if path.lower().endswith(".mp3"):
        base = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(os.path.dirname(path), base + ".wav")
        if not os.path.exists(out):
            convert_mp3_to_wav(path, out)
        return out
    return path

# -------------------------
# Feature extraction
# -------------------------
def extract_audio_features_file(file_path, sr=16000, n_mfcc=13):
    """Return 32-dim feature vector: mean MFCC (13), mean chroma (12), mean spectral contrast (7)."""
    y, sr = librosa.load(file_path, sr=sr)
    # MFCC mean (13)
    mfcc_mean = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T, axis=0)
    # Chroma mean (12)
    chroma_mean = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    # Spectral contrast mean (7)
    spec_contrast_mean = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)

    feat = np.hstack([mfcc_mean, chroma_mean, spec_contrast_mean])
    return feat

def feature_stats(feature_matrix):
    """Given shape (n_features, T) return stats per-feature: mean,std,min,max -> flatten."""
    means = np.mean(feature_matrix, axis=1)
    stds = np.std(feature_matrix, axis=1)
    mins = np.min(feature_matrix, axis=1)
    maxs = np.max(feature_matrix, axis=1)
    return np.hstack([means, stds, mins, maxs])

# -------------------------
# ASR (Speech-to-Text)
# -------------------------
def audiofile_to_text(file_path):
    """
    Use SpeechRecognition with local file and Google Web Speech API fallback.
    Note: This uses internet because we call Google's speech recognition service via SpeechRecognition library.
    For better, replace with Whisper or VOSK.
    """
    recognizer = sr.Recognizer()
    # ensure wav for speech_recognition
    wav_path = ensure_wav(file_path)
    with sr.AudioFile(wav_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        text = ""
    except Exception as e:
        logger.exception("ASR error: %s", e)
        text = ""
    return text

# -------------------------
# Gemini grammar correction / explanation
# -------------------------
def call_gemini_grammar_correct(text, provider: str = "auto"):
    """
    Call Gemini to analyze grammar: return structured JSON:
    {
      "corrected_text": "...",
      "mistakes": [
        {"original": "...", "suggestion": "...", "explanation": "...", "start": i, "end": j},
        ...
      ]
    }
    Uses a prompt asking Gemini to return JSON only.
    """
    # Initialize Gemini client lazily so other modules (e.g., preprocessing/training)
    # can import utils without requiring an API key.
    global client
    if client is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            # Try load from a local .env without extra deps
            candidate_paths = [
                os.path.join(os.getcwd(), ".env"),
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
            ]
            for p in candidate_paths:
                try:
                    if os.path.exists(p):
                        with open(p, "r") as f:
                            for line in f:
                                line = line.strip()
                                if not line or line.startswith("#"):
                                    continue
                                if line.startswith("GOOGLE_API_KEY"):
                                    k, v = line.split("=", 1)
                                    v = v.strip().strip('"').strip("'")
                                    if v:
                                        os.environ["GOOGLE_API_KEY"] = v
                                        api_key = v
                                        break
                        if api_key:
                            break
                except Exception:
                    pass
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set; grammar analysis requires a valid Gemini API key.")
        client = genai.Client(api_key=api_key)

    # Keep prompt explicit and request strict JSON output
    prompt = f"""
You are a precise grammar-checking assistant. Given the input text, return ONLY a JSON object with keys:
- corrected_text: the full corrected sentence(s) as a string.
- mistakes: a list of objects, each with:
    - original: the original erroneous substring
    - suggestion: corrected substring (or corrected sentence)
    - explanation: short rule explaining why it's wrong
    - start: character start index in original text (or -1 if unknown)
    - end: character end index (or -1)

Input text:
\"\"\"{text}\"\"\"

Return only valid JSON, nothing else.
"""
    # Prefer Groq (if GROQ_API_KEY is set) or force by provider; otherwise use Gemini
    out_text = None
    last_err = None
    groq_key = os.environ.get("GROQ_API_KEY")
    if Groq is not None and groq_key and provider in ("auto", "groq"):
        groq_models = [
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768",
            "llama-3.2-90b-text-preview"
        ]
        try:
            groq_client = Groq(api_key=groq_key)
            for m in groq_models:
                try:
                    resp = groq_client.chat.completions.create(
                        model=m,
                        messages=[
                            {"role": "system", "content": "You are a precise grammar-checking assistant. Return ONLY JSON as instructed by the user."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.2
                    )
                    if resp and resp.choices and resp.choices[0].message and resp.choices[0].message.content:
                        out_text = resp.choices[0].message.content.strip()
                        break
                except Exception as e:
                    last_err = e
                    continue
        except Exception as e:
            last_err = e

    if out_text is None and provider in ("auto", "gemini"):
        # Try a list of model names to maximize compatibility with the installed SDK/account
        model_candidates = [
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-2.0-flash"
        ]
        resp = None
        for m in model_candidates:
            try:
                resp = client.models.generate_content(
                    model=m,
                    contents=[prompt]
                )
                break
            except ClientError as e:
                last_err = e
                continue
        if resp is None and last_err is not None:
            raise last_err
        # resp may have structure; text usually in resp.text
        out_text = resp.text.strip() if hasattr(resp, 'text') else str(resp)
    # try to parse JSON - model is instructed to return JSON only
    try:
        parsed = json.loads(out_text)
    except Exception:
        # As a fallback, try to extract JSON substring
        try:
            start = out_text.find('{')
            end = out_text.rfind('}') + 1
            parsed = json.loads(out_text[start:end])
        except Exception:
            parsed = {
                "corrected_text": "",
                "mistakes": [],
                "raw": out_text
            }
    return parsed

# -------------------------
# Utility: highlight mistakes in HTML
# -------------------------
def highlight_text_html(original_text, mistakes):
    """
    Return HTML string where each mistake.original is wrapped with <mark> and tooltip with suggestion/explanation.
    If start/end indexes are provided, use them; otherwise, do simple replace (first occurrence).
    """
    out = html.escape(original_text)
    # apply replacements from end to start (to not break indices)
    if isinstance(mistakes, list) and len(mistakes) > 0:
        # if they give start/end, we use them, otherwise naive replace
        try:
            # attempt index-based
            chars = list(out)
            offset = 0
            for m in sorted(mistakes, key=lambda x: x.get('start', -1), reverse=True):
                s = m.get('start', -1)
                e = m.get('end', -1)
                suggestion = html.escape(m.get('suggestion', ''))
                explanation = html.escape(m.get('explanation', ''))
                if s is not None and s >= 0 and e is not None and e >= s:
                    # replace slice with marked HTML
                    before = ''.join(chars[:s])
                    target = ''.join(chars[s:e])
                    after = ''.join(chars[e:])
                    mark = f'<mark title="{suggestion} — {explanation}">{target}</mark>'
                    out = before + mark + after
                    chars = list(out)
            # fallback: if nothing done, do simple replace
            if out == html.escape(original_text):
                for m in mistakes:
                    orig = html.escape(m.get('original', ''))
                    sugg = html.escape(m.get('suggestion', ''))
                    expl = html.escape(m.get('explanation', ''))
                    if orig:
                        out = out.replace(orig, f'<mark title="{sugg} — {expl}">{orig}</mark>', 1)
        except Exception:
            # naive replace
            out = html.escape(original_text)
            for m in mistakes:
                orig = html.escape(m.get('original', ''))
                sugg = html.escape(m.get('suggestion', ''))
                expl = html.escape(m.get('explanation', ''))
                if orig:
                    out = out.replace(orig, f'<mark title="{sugg} — {expl}">{orig}</mark>', 1)
    return out
