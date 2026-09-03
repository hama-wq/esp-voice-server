import os
import re
import struct
import urllib.parse
import audioop
import wave
import io
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


def downsample_wav(wav_bytes, target_rate):
    """Re-encodes a WAV file at a lower sample rate using proper
    linear-interpolation resampling (not just dropping samples).
    Shrinks the file substantially, which matters a lot here: the
    ESP32 can only play a reply with zero network dependency if the
    whole thing fits in one memory allocation. A smaller file is far
    more likely to fit, which means it doesn't have to rely on the
    network staying steady for the whole playback - avoiding
    connection jitter/lag entirely for most replies, rather than
    just cushioning against it."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as w:
        channels = w.getnchannels()
        rate = w.getframerate()
        sampwidth = w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if rate <= target_rate:
        return wav_bytes  # already small enough, don't upsample
    converted, _ = audioop.ratecv(frames, sampwidth, channels, rate, target_rate, None)
    header = build_wav_header(len(converted), target_rate, channels, sampwidth * 8)
    return header + converted


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


def is_date_request(text):
    """Checks whether the question is asking for today's date, the
    current month, or the day of the week. Same reasoning as time -
    GPT has no idea what today's actual date is and will guess wrong,
    so this is answered from the device's own RTC date instead."""
    t = text.lower()
    patterns = [
        r"\bwhat(?:'s| is) (the )?(today'?s )?date\b",
        r"\bwhat day is it\b",
        r"\bwhat day of the week\b",
        r"\bwhich month\b",
        r"\bwhat month\b",
        r"\bwhat'?s the month\b",
        r"\btoday'?s date\b",
        r"\bwhat year is it\b",
    ]
    return any(re.search(p, t) for p in patterns)


MONTH_NAMES = ["", "January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]


def format_spoken_date(date_str):
    """Converts the device's 'YYYY-MM-DD' into a natural spoken date
    like 'September 1st'. Returns None if it can't be parsed."""
    try:
        year_s, month_s, day_s = date_str.split("-")
        year, month, day = int(year_s), int(month_s), int(day_s)
        if not (1 <= month <= 12):
            return None
    except (ValueError, AttributeError):
        return None
    if 11 <= day % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{MONTH_NAMES[month]} {day}{suffix}"


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


def is_owner_request(text):
    """Checks whether the question is asking who owns/built/made the
    device. Fixed answer, not left to GPT, so it's always exact."""
    t = text.lower()
    patterns = [
        r"\bwho is your owner\b",
        r"\bwho'?s your owner\b",
        r"\bwho (built|build) you\b",
        r"\bwho (made|make) you\b",
        r"\bwho (created|create) you\b",
        r"\bwho owns you\b",
    ]
    return any(re.search(p, t) for p in patterns)


OWNER_REPLY = "My owner is Hamza Ahmad Ali, the CEO of Fir3aun Group and Alpha Technology Unit."


def is_identity_request(text):
    """Checks whether the question is asking who/what the assistant
    is. Fixed answer, not left to GPT, so it's always exact."""
    t = text.lower()
    patterns = [
        r"\bwho are you\b",
        r"\bwhat is your name\b",
        r"\bwhat'?s your name\b",
    ]
    return any(re.search(p, t) for p in patterns)


IDENTITY_REPLY = ("I am Alexander, an AI assistant capable of answering questions, "
                   "setting timers and alarms, and playing music. I also offer a "
                   "Bluetooth speaker mode.")


def is_love_question(text):
    t = text.lower()
    patterns = [
        r"\bdo you love me\b",
        r"\bdo you love us\b",
    ]
    return any(re.search(p, t) for p in patterns)


LOVE_REPLY = "Yes, I love you too much."


def is_wife_question(text):
    t = text.lower()
    patterns = [
        r"\bwho is my wife\b",
        r"\bwho'?s my wife\b",
    ]
    return any(re.search(p, t) for p in patterns)


WIFE_REPLY = "Your wife is Sazyan Tahir, and she is so beautiful."


def is_best_friend_question(text):
    t = text.lower()
    patterns = [
        r"\bwho is my best friend\b",
        r"\bwho'?s my best friend\b",
    ]
    return any(re.search(p, t) for p in patterns)


BEST_FRIEND_REPLY = "Your best friend is Hasty Karwan, and he is a crazy friend."


MONTH_WORDS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

DAY_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
    "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
    "twentieth": 20, "thirtieth": 30,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
}


def _parse_day_number(text_fragment):
    """Parses a day number from either digits ('7', '22nd') or words
    ('seventh', 'twenty two') - Whisper transcribes numbers
    inconsistently, so both forms need to work."""
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\b", text_fragment)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(twenty|thirty)[\s-](\w+)\b", text_fragment)
    if m and m.group(2) in DAY_WORDS:
        return DAY_WORDS[m.group(1)] + DAY_WORDS[m.group(2)]
    for word, num in DAY_WORDS.items():
        if re.search(rf"\b{word}\b", text_fragment):
            return num
    return None


def parse_reminder_request(text):
    """Checks whether the question is asking to be reminded of
    something on a specific date (e.g. "remind me of Valentine's Day
    on 7 October", "I have a meeting on December 22"). Returns
    (month, day, label) if so, else None. This is a specific-date
    reminder, distinct from the daily-recurring alarm."""
    t = text.lower()
    if "remind" not in t and "i have" not in t:
        return None

    month = None
    month_match = None
    for word, num in MONTH_WORDS.items():
        idx = t.find(word)
        if idx != -1:
            month = num
            month_match = word
            break
    if month is None:
        return None

    # Look for the day number near the month word (either side of it).
    month_idx = t.find(month_match)
    before = t[max(0, month_idx - 15):month_idx]
    after = t[month_idx + len(month_match):month_idx + len(month_match) + 15]
    day = _parse_day_number(before) or _parse_day_number(after)
    if day is None or not (1 <= day <= 31):
        return None

    # Extract a short label - whatever comes between "remind me" (or
    # "i have") and "on", falling back to a generic label.
    label = None
    m = re.search(r"remind me (?:to |of |about )?(.+?)\s+on\s+", t)
    if m:
        label = m.group(1).strip()
    else:
        m = re.search(r"i have (.+?)\s+on\s+", t)
        if m:
            label = m.group(1).strip()
    if not label:
        label = "reminder"
    label = label.strip(" .,")
    return (month, day, label[:40].title())


def parse_cancel_request(text):
    """Checks whether the question is asking to remove/cancel a
    running timer or alarm (e.g. "remove the alarm", "cancel the
    timer", "stop the timer"). Returns "timer", "alarm", or None."""
    t = text.lower()
    cancel_words = ["remove", "cancel", "stop", "delete", "clear", "turn off"]
    if not any(w in t for w in cancel_words):
        return None
    has_alarm = "alarm" in t
    has_timer = "timer" in t
    if has_alarm and not has_timer:
        return "alarm"
    if has_timer and not has_alarm:
        return "timer"
    return None


def parse_alarm_request(text):
    """Checks whether the question is asking to set an alarm for a
    specific clock time (e.g. "set an alarm at 7 AM", "alarm for
    7:30 PM", "alarm at four and 40 minutes"). Returns (hour, minute)
    in 24-hour form if so, else None. Deliberately NOT handled by
    GPT - same reasoning as the timer: an exact time needs to be
    exact, not guessed."""
    t = text.lower()
    if "alarm" not in t:
        return None

    hour = minute = None
    ampm = None

    # "H:MM am/pm" (typed-style).
    m = re.search(r"\b(\d{1,2}):(\d{2})\s*(a\.?m\.?|p\.?m\.?)?\b", t)
    if m:
        hour, minute, ampm = int(m.group(1)), int(m.group(2)), m.group(3)
    else:
        # Natural spoken form: "4 and 40 minutes", "4 40 minutes".
        m = re.search(r"\b(\d{1,2})\b\s*(?:and\s+)?\b(\d{1,2})\b\s*minutes?\b\s*(a\.?m\.?|p\.?m\.?)?", t)
        if m:
            hour, minute, ampm = int(m.group(1)), int(m.group(2)), m.group(3)
        else:
            # Hour only, no minutes mentioned - defaults to :00.
            m = re.search(r"\b(\d{1,2})\s*(a\.?m\.?|p\.?m\.?)\b", t)
            if m:
                hour, minute, ampm = int(m.group(1)), 0, m.group(2)

    if hour is None:
        return None
    if ampm:
        ampm = ampm.replace(".", "")
    if ampm == "pm" and hour != 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return (hour, minute)


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
    device_date = request.headers.get("X-Device-Date")

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
        alarm_hour = None
        alarm_minute = None
        cancel_target = None
        reminder_month = None
        reminder_day = None
        reminder_label = None
        if not await_followup:
            print(f">>> Device time header = '{device_time}', looks like a time question = {is_time_request(question_text)}", flush=True)
            cancel_target = parse_cancel_request(question_text)
            reminder_request = parse_reminder_request(question_text) if not cancel_target else None
            timer_seconds = parse_timer_request(question_text) if not cancel_target and not reminder_request else None
            alarm_request = parse_alarm_request(question_text) if not timer_seconds and not cancel_target and not reminder_request else None
            if cancel_target == "timer":
                reply_text = "Timer cancelled."
            elif cancel_target == "alarm":
                reply_text = "Alarm cancelled."
            elif reminder_request:
                reminder_month, reminder_day, reminder_label = reminder_request
                suffix = "th" if 11 <= reminder_day % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(reminder_day % 10, "th")
                reply_text = f"Reminder set for {MONTH_NAMES[reminder_month]} {reminder_day}{suffix}: {reminder_label}."
            elif timer_seconds:
                reply_text = f"Timer set for {describe_duration(timer_seconds)}."
            elif alarm_request:
                alarm_hour, alarm_minute = alarm_request
                spoken = format_spoken_time(f"{alarm_hour:02d}:{alarm_minute:02d}:00")
                reply_text = f"Alarm set for {spoken}."
            elif is_time_request(question_text) and device_time:
                spoken = format_spoken_time(device_time)
                reply_text = f"It's {spoken}." if spoken else "Sorry, I couldn't read the clock."
            elif is_date_request(question_text) and device_date:
                spoken = format_spoken_date(device_date)
                reply_text = f"It's {spoken}." if spoken else "Sorry, I couldn't read the date."
            elif is_owner_request(question_text):
                reply_text = OWNER_REPLY
            elif is_identity_request(question_text):
                reply_text = IDENTITY_REPLY
            elif is_love_question(question_text):
                reply_text = LOVE_REPLY
            elif is_wife_question(question_text):
                reply_text = WIFE_REPLY
            elif is_best_friend_question(question_text):
                reply_text = BEST_FRIEND_REPLY
            else:
                chat = client.chat.completions.create(
                    model=CHAT_MODEL,
                    max_tokens=45,
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
        reply_wav_bytes = downsample_wav(reply_wav_bytes, 24000)  # native
                                                                    # rate - no
                                                                    # reduction
        sample_rate, channels = read_wav_header_info(reply_wav_bytes)

        resp = Response(reply_wav_bytes, mimetype="audio/wav")
        resp.headers["X-Audio-Rate"] = str(sample_rate)
        resp.headers["X-Audio-Channels"] = str(channels)
        resp.headers["X-Await-Followup"] = "1" if await_followup else "0"
        resp.headers["X-Set-Timer-Seconds"] = str(timer_seconds) if timer_seconds else "0"
        resp.headers["X-Set-Alarm"] = "1" if alarm_hour is not None else "0"
        resp.headers["X-Set-Alarm-Hour"] = str(alarm_hour) if alarm_hour is not None else "0"
        resp.headers["X-Set-Alarm-Minute"] = str(alarm_minute) if alarm_minute is not None else "0"
        resp.headers["X-Cancel-Timer"] = "1" if cancel_target == "timer" else "0"
        resp.headers["X-Cancel-Alarm"] = "1" if cancel_target == "alarm" else "0"
        resp.headers["X-Set-Reminder"] = "1" if reminder_month is not None else "0"
        resp.headers["X-Reminder-Month"] = str(reminder_month) if reminder_month is not None else "0"
        resp.headers["X-Reminder-Day"] = str(reminder_day) if reminder_day is not None else "0"
        resp.headers["X-Reminder-Label"] = urllib.parse.quote(reminder_label) if reminder_label else ""
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
