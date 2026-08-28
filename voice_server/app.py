import os
import re
import struct
from flask import Flask, request, Response
from openai import OpenAI

app = Flask(__name__)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

CHAT_MODEL = "gpt-4o-mini"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "alloy"
DEFAULT_WAKE_WORD = "Alexander"


def read_wav_header_info(wav_bytes):
    """Reads real sample rate/channels back out of a WAV file's own
    header, so we tell the ESP32 the truth regardless of what the
    TTS model actually returns."""
    channels = struct.unpack("<H", wav_bytes[22:24])[0]
    sample_rate = struct.unpack("<I", wav_bytes[24:28])[0]
    return sample_rate, channels


def build_wav_header(data_size, sample_rate, channels, bits=16):
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 1, channels, sample_rate, byte_rate, block_align, bits,
        b"data", data_size,
    )


def strip_wake_word(text, wake_word):
    """Checks whether the wake word appears at (or very near) the
    start of what was said, and if so, returns the rest of the
    sentence with it removed. Returns None if the wake word wasn't
    found near the start."""
    stripped = text.strip()
    # Allow a little leading noise ("uh, Alexander, ...") by checking
    # within roughly the first few words, not requiring it be the
    # very first character.
    lookahead = " ".join(stripped.split()[:4])
    match = re.search(re.escape(wake_word), lookahead, re.IGNORECASE)
    if not match:
        return None
    # Remove the wake word (and any immediately following comma) from
    # the full text, once, at the position it was found.
    pattern = re.compile(re.escape(wake_word) + r"[,]?\s*", re.IGNORECASE)
    remainder = pattern.sub("", stripped, count=1).strip()
    return remainder


@app.route("/voice-query", methods=["POST"])
def voice_query():
    # The ESP32 streams raw PCM (chunked transfer) with the actual
    # rate/channels and the expected wake word in custom headers -
    # see wifi_voice.cpp. Flask/gunicorn dechunk this automatically,
    # so by the time we're here we just have the full raw audio.
    pcm_bytes = request.get_data()
    if not pcm_bytes:
        return Response("No audio received", status=400)

    input_rate = int(request.headers.get("X-Input-Rate", 16000))
    input_channels = int(request.headers.get("X-Input-Channels", 1))
    wake_word = request.headers.get("X-Wake-Word", DEFAULT_WAKE_WORD)

    # We only know the final length now that the full body has
    # arrived, so we build the WAV header here rather than on-device.
    wav_bytes = build_wav_header(len(pcm_bytes), input_rate, input_channels) + pcm_bytes

    try:
        # Step 1: speech-to-text on the whole thing
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=("query.wav", wav_bytes, "audio/wav"),
        )
        heard_text = transcript.text.strip()

        # Step 2: only proceed if the wake word was actually said
        question_text = strip_wake_word(heard_text, wake_word)
        if question_text is None:
            return Response(status=204)  # wake word missing - stay silent
        if not question_text:
            question_text = "The user said something unclear after the wake word."

        # Step 3: generate a short answer
        chat = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful voice assistant on a small robot speaker. Keep answers under 2 short sentences, plain text, no markdown, no emojis."},
                {"role": "user", "content": question_text},
            ],
        )
        reply_text = chat.choices[0].message.content.strip()

        # Step 4: natural-sounding speech
        speech = client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=reply_text,
            response_format="wav",
        )
        reply_wav_bytes = speech.read()
        sample_rate, channels = read_wav_header_info(reply_wav_bytes)

        resp = Response(reply_wav_bytes, mimetype="audio/wav")
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
