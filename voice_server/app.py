import os
import struct
from flask import Flask, request, Response
import google.generativeai as genai
from google import genai as genai_new
from google.genai import types

app = Flask(__name__)

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
chat_model = genai.GenerativeModel(
    "gemini-3.6-flash",
    system_instruction="You are a helpful voice assistant on a small robot speaker. "
                        "Listen to the audio question and answer in under 2 short "
                        "sentences, plain text, no markdown, no emojis."
)

# Natural-sounding AI voice (replaces the earlier espeak-ng robotic voice).
tts_client = genai_new.Client(api_key=os.environ.get("GEMINI_API_KEY"))
TTS_MODEL = "gemini-2.5-flash-preview-tts"
TTS_VOICE = "Kore"          # see Gemini TTS docs for other prebuilt voice names
TTS_SAMPLE_RATE = 24000     # Gemini TTS's fixed output rate
TTS_CHANNELS = 1


def build_wav_header(data_size, sample_rate, channels, bits=16):
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 1, channels, sample_rate, byte_rate, block_align, bits,
        b"data", data_size,
    )


def text_to_speech(text):
    """Generates natural AI speech via Gemini TTS. Returns a WAV file
    (we add the header ourselves since Gemini returns raw PCM only) so
    the ESP32's existing header-skip logic keeps working unchanged."""
    response = tts_client.models.generate_content(
        model=TTS_MODEL,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=TTS_VOICE)
                )
            ),
        ),
    )
    pcm_data = response.candidates[0].content.parts[0].inline_data.data
    header = build_wav_header(len(pcm_data), TTS_SAMPLE_RATE, TTS_CHANNELS)
    return header + pcm_data, TTS_SAMPLE_RATE, TTS_CHANNELS


@app.route("/voice-query", methods=["POST"])
def voice_query():
    # The ESP32 sends the raw WAV bytes as the request body (see wifi_voice.cpp).
    audio_bytes = request.get_data()
    if not audio_bytes:
        return Response("No audio received", status=400)

    try:
        # Step 1: understand the spoken question and generate a text answer.
        response = chat_model.generate_content([
            {"mime_type": "audio/wav", "data": audio_bytes},
            "Answer the question asked in this audio clip.",
        ])
        reply_text = response.text.strip()
        if not reply_text:
            reply_text = "I did not catch that."

        # Step 2: convert that answer into natural-sounding speech.
        wav_bytes, sample_rate, channels = text_to_speech(reply_text)

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
