import os
import struct
from flask import Flask, request, Response
from openai import OpenAI

app = Flask(__name__)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

CHAT_MODEL = "gpt-4o-mini"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "alloy"


def read_wav_header_info(wav_bytes):
    """Reads real sample rate/channels back out of a WAV file's own
    header, so we tell the ESP32 the truth regardless of what the
    TTS model actually returns."""
    channels = struct.unpack("<H", wav_bytes[22:24])[0]
    sample_rate = struct.unpack("<I", wav_bytes[24:28])[0]
    return sample_rate, channels


@app.route("/voice-query", methods=["POST"])
def voice_query():
    # The ESP32 sends the raw WAV bytes as the request body (see wifi_voice.cpp).
    audio_bytes = request.get_data()
    if not audio_bytes:
        return Response("No audio received", status=400)

    try:
        # Step 1: speech-to-text
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=("query.wav", audio_bytes, "audio/wav"),
        )
        question_text = transcript.text.strip()
        if not question_text:
            question_text = "The user said something unclear."

        # Step 2: generate a short answer
        chat = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful voice assistant on a small robot speaker. Keep answers under 2 short sentences, plain text, no markdown, no emojis."},
                {"role": "user", "content": question_text},
            ],
        )
        reply_text = chat.choices[0].message.content.strip()

        # Step 3: natural-sounding speech
        speech = client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=reply_text,
            response_format="wav",
        )
        wav_bytes = speech.read()
        sample_rate, channels = read_wav_header_info(wav_bytes)

        resp = Response(wav_bytes, mimetype="audio/wav")
        resp.headers["X-Audio-Rate"] = str(sample_rate)
        resp.headers["X-Audio-Channels"] = str(channels)
        return resp

    except Exception as e:
        return Response(f"Error: {str(e)}", status=500)


@app.route("/", methods=["GET"])
def health():
    return "Voice server is running."


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
