import os
import re
import struct
import urllib.parse
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


def is_time_request(text):
    """Checks whether the question is asking for the current time.
    Deliberately NOT handled by GPT - the model has no access to a
    real clock and would either refuse or guess. Answered instead
    using the device's own RTC time, sent with every request."""
    t = text.lower()
    patterns = [
        r"\bwhat time is it\b",
        r"\bwhat(?:'s| is) the time\b",
        r"\bwhat time is this\b",
        r"\btell me the time\b",
        r"\bcurrent time\b",
        r"\bdo you know the time\b",
        r"\bwhat time do (i|we) have\b",
    ]
    return any(re.search(p, t) for p in patterns)


def format_spoken_time(time_str):
    """Converts the device's 'HH:MM:SS' (24-hour) into a natural
    spoken time like '2:32 PM'. Returns None if it can't be parsed."""
    try:
        hour_s, minute_s, _ = time_str.split(":")
        hour, minute = int(hour_s), int(minute_s)
    except (ValueError, AttributeError):
        return None
    period = "AM" if hour < 12 else "PM"
    hour12 = hour % 12
    if hour12 == 0:
        hour12 = 12
    return f"{hour12}:{minute:02d} {period}"


def parse_timer_request(text):
    """Checks whether the question is asking to set a timer/countdown
    (e.g. "set a timer for 10 minutes", "timer for 1 hour and 30
    seconds"). Returns the total duration in seconds if so, else None.
    Deliberately NOT handled by GPT - a duration needs to be exact,
    and this is far more reliable than hoping the model gets it right
    and phrases its reply in a way we can parse back out."""
    t = text.lower()
    if "timer" not in t:
        return None
    total_seconds = 0
    found = False
    for match in re.finditer(r"(\d+)\s*(hour|hr|minute|min|second|sec)s?", t):
        num = int(match.group(1))
        unit = match.group(2)
        if unit in ("hour", "hr"):
            total_seconds += num * 3600
        elif unit in ("minute", "min"):
            total_seconds += num * 60
        else:
            total_seconds += num
        found = True
    return total_seconds if found and total_seconds > 0 else None


def describe_duration(total_seconds):
    """Turns a second count back into a short spoken phrase, e.g.
    '10 minutes' or '1 hour and 30 minutes'."""
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    parts = []
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds:
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
    if not parts:
        return "0 seconds"
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def strip_wake_word(text, wake_word):
    """Checks whether the wake word (or a close transcription variant
    of it, e.g. Whisper hearing 'Aleksandra' for 'Alexander') appears
    near the start of what was said, and if so, returns the rest of
    the sentence with it removed. Returns None if no reasonable match
    is found near the start."""
    stripped = text.strip()
    words = stripped.split()
    lookahead_words = words[:4]
    lookahead = " ".join(lookahead_words)

    match = re.search(re.escape(wake_word), lookahead, re.IGNORECASE)
    if match:
        # Anchored to the very front, so this only strips wake-word
        # occurrences right at the start - and repeats, since Whisper
        # sometimes transcribes a standalone wake word twice in a row
        # (e.g. "Alexander, Alexander.").
        pattern = re.compile(r"^\s*" + re.escape(wake_word) + r"[,.!?]*\s*", re.IGNORECASE)
        remainder = stripped
        while True:
            new_remainder = pattern.sub("", remainder, count=1)
            if new_remainder == remainder:
                break
            remainder = new_remainder
        return remainder.strip()

    # Fuzzy fallback: a near-miss transcription of the name (shares
    # the same first few letters) still counts, since speech-to-text
    # on names isn't perfectly reliable.
    prefix = wake_word[:4].lower()
    for i, w in enumerate(lookahead_words):
        clean_w = re.sub(r"[^a-zA-Z]", "", w)
        if len(clean_w) >= 4 and clean_w.lower().startswith(prefix):
            remainder_words = words[:i] + words[i + 1:]
            return " ".join(remainder_words).strip()

    return None


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
    # Set by the device when it's already in a follow-up window (you
    # already said the wake word once and got a "Yes?") - in that case
    # whatever you say next IS the question, no wake word needed again.
    skip_wake_word = request.headers.get("X-Skip-Wake-Word", "0") == "1"
    device_time = request.headers.get("X-Device-Time")

    # We only know the final length now that the full body has
    # arrived, so we build the WAV header here rather than on-device.
    wav_bytes = build_wav_header(len(pcm_bytes), input_rate, input_channels) + pcm_bytes

    try:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=("query.wav", wav_bytes, "audio/wav"),
            language="en",
            prompt=f"The assistant's name is {wake_word}.",
        )
        heard_text = transcript.text.strip()
        print(f">>> Heard: '{heard_text}' | skip_wake_word={skip_wake_word} | Expected wake word: '{wake_word}'", flush=True)

        await_followup = False

        if skip_wake_word:
            # Already in a follow-up window - treat everything said as
            # the question directly, no wake word required.
            question_text = heard_text if heard_text else "The user said something unclear."
        else:
            question_text = strip_wake_word(heard_text, wake_word)
            if question_text is None:
                return Response(status=204)  # wake word missing entirely - stay silent
            print(f">>> After stripping wake word, remainder = '{question_text}'", flush=True)
            if not question_text:
                # Just the wake word alone, nothing else said yet -
                # acknowledge and open a follow-up window instead of
                # answering anything.
                reply_text = "Yes?"
                await_followup = True

        timer_seconds = None
        if not await_followup:
            print(f">>> Device time header = '{device_time}', looks like a time question = {is_time_request(question_text)}", flush=True)
            timer_seconds = parse_timer_request(question_text)
            if timer_seconds:
                reply_text = f"Timer set for {describe_duration(timer_seconds)}."
            elif is_time_request(question_text) and device_time:
                spoken = format_spoken_time(device_time)
                reply_text = f"It's {spoken}." if spoken else "Sorry, I couldn't read the clock."
            else:
                chat = client.chat.completions.create(
                    model=CHAT_MODEL,
                    max_tokens=60,
                    messages=[
                        {"role": "system", "content": "You are a helpful voice assistant on a small robot speaker. Keep answers under 2 short sentences, plain text, no markdown, no emojis."},
                        {"role": "user", "content": question_text},
                    ],
                )
                reply_text = chat.choices[0].message.content.strip()

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
        resp.headers["X-Await-Followup"] = "1" if await_followup else "0"
        resp.headers["X-Set-Timer-Seconds"] = str(timer_seconds) if timer_seconds else "0"
        return resp

    except Exception as e:
        return Response(f"Error: {str(e)}", status=500)


@app.route("/alarm-sound", methods=["GET"])
def alarm_sound():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alarm_sound.wav")
    if not os.path.exists(path):
        return Response("No alarm sound file uploaded", status=404)
    with open(path, "rb") as f:
        wav_bytes = f.read()
    sample_rate, channels = read_wav_header_info(wav_bytes)
    resp = Response(wav_bytes, mimetype="audio/wav")
    resp.headers["X-Audio-Rate"] = str(sample_rate)
    resp.headers["X-Audio-Channels"] = str(channels)
    return resp


@app.route("/", methods=["GET"])
def health():
    return "Voice server is running."


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
