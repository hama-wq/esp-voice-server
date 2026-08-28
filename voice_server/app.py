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
    """Read sample rate and channel count from the WAV header."""
    if len(wav_bytes) < 28:
        raise ValueError("Invalid WAV file returned by TTS")

    channels = struct.unpack("<H", wav_bytes[22:24])[0]
    sample_rate = struct.unpack("<I", wav_bytes[24:28])[0]

    return sample_rate, channels


@app.route("/voice-query", methods=["POST"])
def voice_query():
    # ESP32 sends raw WAV audio in the request body.
    audio_bytes = request.get_data()

    if not audio_bytes:
        return Response(
            "No audio received",
            status=400,
            content_type="text/plain; charset=utf-8"
        )

    try:
        # Step 1: Speech-to-text
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=("query.wav", audio_bytes, "audio/wav"),
        )

        question_text = transcript.text.strip()

        if not question_text:
            question_text = "The user said something unclear."

        print("Transcript:", question_text, flush=True)

        # Step 2: Generate answer
        chat = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Alexander, a helpful voice assistant "
                        "inside a small robot speaker. "
                        "Answer the user's question clearly and naturally. "
                        "Keep answers under 2 short sentences. "
                        "Plain text only. No markdown. No emojis."
                    ),
                },
                {
                    "role": "user",
                    "content": question_text,
                },
            ],
        )

        reply_text = chat.choices[0].message.content.strip()

        print("Reply:", reply_text, flush=True)

        # Step 3: Convert answer to speech
        speech = client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=reply_text,
            response_format="wav",
        )

        wav_bytes = speech.read()

        sample_rate, channels = read_wav_header_info(wav_bytes)

        print(
            f"TTS audio: {sample_rate} Hz, {channels} channel(s), "
            f"{len(wav_bytes)} bytes",
            flush=True
        )

        # Send WAV audio back to ESP32
        resp = Response(
            wav_bytes,
            status=200,
            content_type="audio/wav"
        )

        resp.headers["X-Audio-Rate"] = str(sample_rate)
        resp.headers["X-Audio-Channels"] = str(channels)

        return resp

    except Exception as e:
        # Always return UTF-8 text so Arabic/Kurdish/other Unicode
        # characters in OpenAI errors cannot cause another encoding error.
        error_message = f"Error: {str(e)}"

        print(error_message, flush=True)

        return Response(
            error_message,
            status=500,
            content_type="text/plain; charset=utf-8"
        )


@app.route("/", methods=["GET"])
def health():
    return "Voice server is running."


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
