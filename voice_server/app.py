import os
import struct
import subprocess
import tempfile
from flask import Flask, request, jsonify, Response
import google.generativeai as genai

app = Flask(__name__)

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel(
    "gemini-3.6-flash",
    system_instruction="You are a helpful voice assistant on a small robot speaker. "
                        "Listen to the audio question and answer in under 2 short "
                        "sentences, plain text, no markdown."
)


def text_to_speech_wav(text):
    """Uses espeak-ng (installed via Dockerfile) to synthesize a simple
    robotic-voice WAV file, and reads back its real sample rate/channels
    from the file header so the ESP32 can configure I2S correctly
    regardless of espeak-ng's default output format."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tts_path = tmp.name
    subprocess.run(
        ["espeak-ng", "-v", "en", "-s", "150", "-w", tts_path, text],
        check=True, timeout=20,
    )
    with open(tts_path, "rb") as f:
        wav_bytes = f.read()
    os.remove(tts_path)

    # Canonical WAV header offsets: channels @22 (uint16), sample rate @24 (uint32)
    channels = struct.unpack("<H", wav_bytes[22:24])[0]
    sample_rate = struct.unpack("<I", wav_bytes[24:28])[0]
    return wav_bytes, sample_rate, channels


@app.route("/voice-query", methods=["POST"])
def voice_query():
    # The ESP32 sends the raw WAV bytes as the request body (see wifi_voice.cpp).
    audio_bytes = request.get_data()
    if not audio_bytes:
        return jsonify({"reply": "No audio received"}), 400

    try:
        # Gemini can take audio directly - no separate transcription step needed.
        response = model.generate_content([
            {"mime_type": "audio/wav", "data": audio_bytes},
            "Answer the question asked in this audio clip.",
        ])
        reply_text = response.text.strip()

        wav_bytes, sample_rate, channels = text_to_speech_wav(reply_text)

        resp = Response(wav_bytes, mimetype="audio/wav")
        resp.headers["X-Audio-Rate"] = str(sample_rate)
        resp.headers["X-Audio-Channels"] = str(channels)
        return resp

    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"}), 500


@app.route("/", methods=["GET"])
def health():
    return "Voice server is running."


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
