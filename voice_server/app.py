import os
from flask import Flask, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel(
    "gemini-2.0-flash",
    system_instruction="You are a helpful voice assistant on a small robot speaker. "
                        "Listen to the audio question and answer in under 2 short "
                        "sentences, plain text, no markdown."
)

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
        return jsonify({"reply": reply_text})

    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"}), 500


@app.route("/", methods=["GET"])
def health():
    return "Voice server is running."


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
