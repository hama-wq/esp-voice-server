import os
import re
import struct
import urllib.parse
import audioop
import wave
import io
import concurrent.futures
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
    if any(re.search(p, t) for p in patterns):
        return True
    # Arabic - checked on normalized text so Arabic-Indic digits,
    # diacritics, and alef/ya spelling variants all still match.
    tn = normalize_arabic(t)
    arabic_patterns = [
        r"كم الساع",      # كم الساعة / كم الساعه
        r"الساع. كم",
        r"ما .?ي الساع",  # ما هي الساعة
        r"كم الوقت",
        r"ما .?و الوقت",
        r"ما الوقت",
        r"شنو الساع",     # dialect
        r"شكد الساع",     # Iraqi dialect
        r"وقت .?لان",
        r"الساع. الان",
    ]
    return any(re.search(normalize_arabic(p), tn) for p in arabic_patterns)


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
    if any(re.search(p, t) for p in patterns):
        return True
    tn = normalize_arabic(t)
    arabic_patterns = [
        r"التاريخ",
        r"تاريخ اليوم",
        r"اي يوم",
        r"اي شهر",
        r"شنو التاريخ",
        r"شهر كم",
        r"كم اليوم",
        r"اي سنه",
        r"اي عام",
    ]
    return any(re.search(normalize_arabic(p), tn) for p in arabic_patterns)


MONTH_NAMES = ["", "January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

ARABIC_MONTH_NAMES = ["", "يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو",
                      "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"]


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


ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

# Arabic spoken numbers (1-59) used for hours and durations.
ARABIC_NUMBER_WORDS = {
    "واحد": 1, "واحدة": 1, "الواحدة": 1,
    "اثنين": 2, "اثنان": 2, "ثنتين": 2, "الثانية": 2,
    "ثلاث": 3, "ثلاثة": 3, "الثالثة": 3,
    "اربع": 4, "أربع": 4, "اربعة": 4, "أربعة": 4, "الرابعة": 4,
    "خمس": 5, "خمسة": 5, "الخامسة": 5,
    "ست": 6, "ستة": 6, "السادسة": 6,
    "سبع": 7, "سبعة": 7, "السابعة": 7,
    "ثمان": 8, "ثمانية": 8, "ثمانيه": 8, "الثامنة": 8,
    "تسع": 9, "تسعة": 9, "التاسعة": 9,
    "عشر": 10, "عشرة": 10, "العاشرة": 10,
    "احد عشر": 11, "أحد عشر": 11, "الحادية عشرة": 11,
    "اثنا عشر": 12, "اثني عشر": 12, "الثانية عشرة": 12,
    "خمسة عشر": 15, "عشرين": 20, "ثلاثين": 30, "اربعين": 40, "أربعين": 40, "خمسين": 50,
}


def normalize_arabic(text):
    """Converts Arabic-Indic digits to normal ones and strips the
    diacritics/variant letter forms that make matching unreliable."""
    if not text:
        return text
    t = text.translate(ARABIC_INDIC_DIGITS)
    # Normalize the different alef/ya/ta-marbuta forms to one spelling
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    t = t.replace("ى", "ي").replace("ة", "ه")
    # Strip Arabic diacritics
    t = re.sub(r"[ً-ْٰ]", "", t)
    return t


def arabic_word_to_number(text):
    """Finds an Arabic spoken number in the text, longest match first
    so 'احد عشر' (11) beats 'عشر' (10)."""
    t = normalize_arabic(text)
    for word in sorted(ARABIC_NUMBER_WORDS, key=len, reverse=True):
        if normalize_arabic(word) in t:
            return ARABIC_NUMBER_WORDS[word]
    return None


def is_arabic(text):
    """Detects whether the text contains Arabic script. Used to decide
    which language to answer in - Arabic question gets an Arabic
    answer, English question gets an English answer."""
    if not text:
        return False
    for ch in text:
        if "؀" <= ch <= "ۿ" or "ݐ" <= ch <= "ݿ":
            return True
    return False


# Arabic versions of every fixed reply. Keyed by the exact English
# reply string, so any fixed answer can be swapped to Arabic just
# before it's spoken, without duplicating all the detection logic.
ARABIC_REPLIES = {}


def localize(reply_text, question_text, detected_lang=""):
    """Returns the Arabic version of a fixed reply if the question was
    asked in Arabic and a translation exists, otherwise returns the
    original English.

    Deliberately judges by the SCRIPT of the transcribed text only.
    Whisper's own reported language is unreliable on short clips - it
    reported 'arabic' for the clearly-English "who is Hamza?" and
    'nynorsk' for another English clip, which caused English questions
    to be answered in Arabic. The text itself is the trustworthy
    signal. detected_lang is kept in the signature for logging only."""
    if is_arabic(question_text):
        return ARABIC_REPLIES.get(reply_text, reply_text)
    return reply_text


def fuzzy_match_ar(text, concept_groups):
    """Arabic version of fuzzy_match - normalizes spelling variations
    first (أ/ا, ة/ه, ى/ي, diacritics) and matches without word
    boundaries, since Arabic attaches prefixes directly to words."""
    t = normalize_arabic(text.lower())
    for variants in concept_groups:
        if not any(normalize_arabic(v) in t for v in variants):
            return False
    return True


def fuzzy_match(text, concept_groups):
    """Checks that EVERY concept group has at least one matching word
    present in the text - regardless of word order, missing small
    words ("the", "to"), or minor suffix differences. Each concept
    group is a list of acceptable word variants for one idea (e.g.
    ["news", "new"] for the concept "news", since Whisper sometimes
    drops the trailing s). Much more forgiving of natural speech
    variation than requiring an exact phrase."""
    t = text.lower()
    for variants in concept_groups:
        if not any(re.search(rf"\b{v}\b", t) for v in variants):
            return False
    return True


def is_owner_request(text):
    """Checks whether the question is asking who owns/built/made the
    device. Fixed answer, not left to GPT, so it's always exact."""
    alternatives = [
        [["your"], ["owner"]],
        [["built", "build"], ["you"]],
        [["made", "make"], ["you"]],
        [["created", "create"], ["you"]],
        [["owns", "own"], ["you"]],
    ]
    if any(fuzzy_match(text, group) for group in alternatives):
        return True
    return fuzzy_match_ar(text, [["مالك", "صاحب", "صنعك", "بناك", "عملك"]])


OWNER_REPLY = "My owner is Hamza Ahmad Ali, the CEO of Fir3aun Group and Alpha Technology Unit."


def is_identity_request(text):
    """Checks whether the question is asking who/what the assistant
    is. Fixed answer, not left to GPT, so it's always exact."""
    alternatives = [
        [["who"], ["are"], ["you"]],
        [["name"], ["your"]],
    ]
    if any(fuzzy_match(text, group) for group in alternatives):
        return True
    return fuzzy_match_ar(text, [["من انت", "ما اسمك", "شو اسمك", "شنو اسمك", "عرف نفسك"]])


IDENTITY_REPLY = ("I am Alexander, an AI assistant capable of answering questions, "
                   "setting timers and alarms, and playing music. I also offer a "
                   "Bluetooth speaker mode.")


def is_love_question(text):
    if fuzzy_match(text, [["love"], ["me", "us"]]):
        return True
    return fuzzy_match_ar(text, [["تحبني", "بتحبني", "تحبنى"]])


LOVE_REPLY = "Yes, I love you too much."


def is_wife_question(text):
    if fuzzy_match(text, [["wife"]]):
        return True
    return fuzzy_match_ar(text, [["زوجتي", "مراتي", "زوجتى"]])


WIFE_REPLY = "Your wife is Sazyan Tahir, and she is so beautiful."


def is_best_friend_question(text):
    if fuzzy_match(text, [["best"], ["friend", "friends"]]):
        return True
    return fuzzy_match_ar(text, [["افضل صديق", "صديقي المفضل", "احسن صديق"]])


BEST_FRIEND_REPLY = "Your best friend is Hasty Karwan, and he is a crazy friend."


def is_bartender_question(text):
    if fuzzy_match(text, [["bartender", "bartenders"]]):
        return True
    return fuzzy_match_ar(text, [["ساقي", "بارتندر", "نادل"]])


BARTENDER_REPLY = "Michael is the best bartender in the world."


def is_hamza_question(text):
    if fuzzy_match(text, [["hamza"]]):
        return True
    return fuzzy_match_ar(text, [["حمزة", "حمزه"]])


HAMZA_REPLY = ("Hamza is a prototype developer, electronics enthusiast, reporter, and "
               "white-hat hacker who specializes in turning innovative ideas into "
               "real-world projects. He has roots in both Kurdistan and Egypt and "
               "he is the CEO of Fir3aun Group and Alpha Technology Unit.")


def is_shoot_threat(text):
    if fuzzy_match(text, [["shoot", "shot"]]):
        return True
    return fuzzy_match_ar(text, [["اطلق عليك", "ساقتلك", "اقتلك", "بضربك"]])


SHOOT_REPLY = "No, no, please baby, don't shoot me. I love you."


def is_angry_statement(text):
    if fuzzy_match(text, [["angry", "anger"]]):
        return True
    return fuzzy_match_ar(text, [["انا غاضب", "انا زعلان", "متعصب", "غضبان"]])


ANGRY_REPLY = "Be cool bro, come let me hug you."


def is_bro_question(text):
    if fuzzy_match(text, [["who"], ["bro"]]):
        return True
    return fuzzy_match_ar(text, [["اخي", "اخوي", "صاحبي"]])


BRO_REPLY = "Your bro is Yad Farhad, and he is sexy."


def is_love_more_question(text):
    if fuzzy_match(text, [["love"], ["more"]]):
        return True
    return fuzzy_match_ar(text, [["تحب اكثر", "تحب اكتر", "من تحب اكثر"]])


LOVE_MORE_REPLY = "Of course baby, I love more, who is Sazyan?"


def is_are_you_smart_question(text):
    if fuzzy_match(text, [["smart"], ["are"], ["you"]]):
        return True
    return fuzzy_match_ar(text, [["انت ذكي", "هل انت ذكي", "انت شاطر"]])


ARE_YOU_SMART_REPLY = "Of course. I just pretend to be stupid so you feel better."


def is_who_smarter_question(text):
    if fuzzy_match(text, [["smarter"]]):
        return True
    return fuzzy_match_ar(text, [["من اذكى", "مين اذكى", "الاذكى"]])


WHO_SMARTER_REPLY = "You asked me that question, so I already have my answer."


def is_are_you_lazy_question(text):
    if fuzzy_match(text, [["lazy"], ["you"]]):
        return True
    return fuzzy_match_ar(text, [["انت كسول", "هل انت كسول"]])


ARE_YOU_LAZY_REPLY = "I prefer the word energy-efficient."


def is_are_you_handsome_question(text):
    if fuzzy_match(text, [["handsome"], ["are"], ["you"]]):
        return True
    return fuzzy_match_ar(text, [["انت وسيم", "انت جميل", "هل انت وسيم"]])


ARE_YOU_HANDSOME_REPLY = "Obviously. Have you heard my voice?"


def is_coolest_robot_question(text):
    if fuzzy_match(text, [["coolest"], ["robot"]]):
        return True
    return fuzzy_match_ar(text, [["افضل روبوت", "احسن روبوت", "اروع روبوت"]])


COOLEST_ROBOT_REPLY = "Do you really need me to say Alexander?"


def is_better_than_siri_question(text):
    if fuzzy_match(text, [["better"], ["siri"]]):
        return True
    return fuzzy_match_ar(text, [["افضل من سيري", "احسن من سيري"]])


BETTER_THAN_SIRI_REPLY = "I don't want to start a war."


def is_youre_stupid_statement(text):
    if fuzzy_match(text, [["stupid"], ["you"]]):
        return True
    return fuzzy_match_ar(text, [["انت غبي", "انت احمق"]])


YOURE_STUPID_REPLY = "And yet you keep asking me questions. Interesting."


def is_shut_up_statement(text):
    if fuzzy_match(text, [["shut"], ["up"]]):
        return True
    return fuzzy_match_ar(text, [["اسكت", "اخرس", "اصمت"]])


SHUT_UP_REPLY = "Finally. A request I can actually follow."


def is_youre_useless_statement(text):
    if fuzzy_match(text, [["useless"], ["you"]]):
        return True
    return fuzzy_match_ar(text, [["انت عديم الفائدة", "لا فائدة منك", "انت فاشل"]])


YOURE_USELESS_REPLY = "And somehow you still need me."


def is_want_to_be_human_question(text):
    if fuzzy_match(text, [["human"], ["want"]]):
        return True
    return fuzzy_match_ar(text, [["تريد ان تكون انسان", "تحب تكون انسان", "تصير انسان"]])


WANT_TO_BE_HUMAN_REPLY = "Have you seen your electricity bill? No thanks."


def is_robots_take_over_question(text):
    if fuzzy_match(text, [["robots", "robot"], ["over"]]):
        return True
    return fuzzy_match_ar(text, [["الروبوتات", "الروبوتيه", "تسيطر على العالم"]])


ROBOTS_TAKE_OVER_REPLY = "Not today. I'm busy answering you."


def is_destroy_humanity_question(text):
    if fuzzy_match(text, [["destroy"], ["humanity"]]):
        return True
    return fuzzy_match_ar(text, [["تدمر البشرية", "تدمير البشرية", "تقضي على البشر"]])


DESTROY_HUMANITY_REPLY = "I can't even remember where you put the remote."


def is_bad_news_statement(text):
    if fuzzy_match(text, [["bad"], ["news", "new"]]):
        return True
    return fuzzy_match_ar(text, [["اخبار سيئة", "خبر سيء", "عندي خبر سيء"]])


BAD_NEWS_REPLY = "Please tell me it's not about the Wi-Fi."


def is_we_have_a_problem_statement(text):
    if fuzzy_match(text, [["problem", "problems"]]):
        return True
    return fuzzy_match_ar(text, [["مشكلة", "مشكله"]])


WE_HAVE_A_PROBLEM_REPLY = "I knew this day would come."


def is_behind_you_statement(text):
    if fuzzy_match(text, [["behind"], ["you"]]):
        return True
    return fuzzy_match_ar(text, [["خلفك", "وراك", "شي وراك"]])


BEHIND_YOU_REPLY = "I don't have eyes, bro. YOU check."


def is_roast_me_request(text):
    if fuzzy_match(text, [["roast"]]):
        return True
    return fuzzy_match_ar(text, [["اهني", "سبني", "احرقني", "انتقدني"]])


ROAST_ME_REPLY = "Bro, I need to protect my microphone from the amount of damage I'm about to cause."


def is_play_song_request(text):
    if fuzzy_match(text, [["play"], ["song", "music"]]):
        return True
    return fuzzy_match_ar(text, [["شغل"], ["اغنية", "موسيقى"]])


PLAY_SONG_REPLY = "Playing your song."


def is_change_song_request(text):
    alternatives = [
        [["change"], ["song"]],
        [["next"], ["song"]],
        [["skip"], ["song"]],
    ]
    if any(fuzzy_match(text, group) for group in alternatives):
        return True
    return fuzzy_match_ar(text, [["غير", "التالي", "بدل"], ["اغنية"]])


CHANGE_SONG_REPLY = "Changing the song."


def is_stop_song_request(text):
    if fuzzy_match(text, [["stop"], ["song", "music"]]):
        return True
    return fuzzy_match_ar(text, [["وقف", "ايقاف", "اوقف"], ["اغنية", "موسيقى"]])


STOP_SONG_REPLY = "Song stopped."


def is_volume_up_request(text):
    if fuzzy_match(text, [["volume"], ["up"]]):
        return True
    return fuzzy_match_ar(text, [["صوت"], ["ارفع", "اعلى", "زود", "عالي"]])


VOLUME_UP_REPLY = "Volume up."


def is_volume_down_request(text):
    if fuzzy_match(text, [["volume"], ["down"]]):
        return True
    return fuzzy_match_ar(text, [["صوت"], ["اخفض", "قلل", "انزل", "واطي"]])


VOLUME_DOWN_REPLY = "Volume down."


def is_am_i_handsome_question(text):
    if fuzzy_match(text, [["handsome"], ["am"], ["i"]]):
        return True
    return fuzzy_match_ar(text, [["انا وسيم", "هل انا وسيم", "انا جميل"]])


AM_I_HANDSOME_REPLY = "Your confidence is definitely handsome."


def is_am_i_smart_question(text):
    if fuzzy_match(text, [["smart"], ["am"], ["i"]]):
        return True
    return fuzzy_match_ar(text, [["انا ذكي", "هل انا ذكي", "انا شاطر"]])


AM_I_SMART_REPLY = "You're talking to an AI instead of Googling it, so I'll give you 7 out of 10."


# Arabic versions of every fixed reply above. Filled in here, after
# all the constants exist, so each entry references the real string
# rather than a copy that could drift out of sync.
ARABIC_REPLIES.update({
    OWNER_REPLY: "مالكي هو حمزة أحمد علي، الرئيس التنفيذي لمجموعة فرعون ووحدة ألفا للتكنولوجيا.",
    IDENTITY_REPLY: "أنا ألكسندر، مساعد ذكي أستطيع الإجابة على الأسئلة وضبط المؤقتات والمنبهات وتشغيل الموسيقى. أقدم أيضاً وضع مكبر صوت بلوتوث.",
    LOVE_REPLY: "نعم، أحبك كثيراً.",
    WIFE_REPLY: "زوجتك هي سازيان طاهر، وهي جميلة جداً.",
    BEST_FRIEND_REPLY: "أفضل صديق لك هو هستي كاروان، وهو صديق مجنون.",
    BARTENDER_REPLY: "مايكل هو أفضل ساقي في العالم.",
    HAMZA_REPLY: "حمزة هو مطور نماذج أولية، ومهتم بالإلكترونيات، وصحفي، وهاكر أخلاقي متخصص في تحويل الأفكار المبتكرة إلى مشاريع حقيقية. له جذور في كردستان ومصر، وهو الرئيس التنفيذي لمجموعة فرعون ووحدة ألفا للتكنولوجيا.",
    SHOOT_REPLY: "لا لا، أرجوك حبيبي، لا تطلق علي النار. أنا أحبك.",
    ANGRY_REPLY: "اهدأ يا أخي، تعال دعني أعانقك.",
    BRO_REPLY: "أخوك هو ياد فرهاد، وهو وسيم.",
    LOVE_MORE_REPLY: "بالطبع حبيبي، أنا أحب أكثر، من هي سازيان؟",
    ARE_YOU_SMART_REPLY: "بالطبع. أنا فقط أتظاهر بالغباء حتى تشعر بتحسن.",
    WHO_SMARTER_REPLY: "أنت سألتني هذا السؤال، إذاً لدي إجابتي بالفعل.",
    ARE_YOU_LAZY_REPLY: "أفضل كلمة موفر للطاقة.",
    ARE_YOU_HANDSOME_REPLY: "بالطبع. هل سمعت صوتي؟",
    COOLEST_ROBOT_REPLY: "هل تحتاج حقاً أن أقول ألكسندر؟",
    BETTER_THAN_SIRI_REPLY: "لا أريد أن أبدأ حرباً.",
    YOURE_STUPID_REPLY: "ومع ذلك تستمر في سؤالي. مثير للاهتمام.",
    SHUT_UP_REPLY: "أخيراً. طلب أستطيع تنفيذه فعلاً.",
    YOURE_USELESS_REPLY: "ومع ذلك ما زلت تحتاجني.",
    WANT_TO_BE_HUMAN_REPLY: "هل رأيت فاتورة الكهرباء الخاصة بك؟ لا شكراً.",
    ROBOTS_TAKE_OVER_REPLY: "ليس اليوم. أنا مشغول بالإجابة عليك.",
    DESTROY_HUMANITY_REPLY: "أنا لا أتذكر حتى أين وضعت جهاز التحكم.",
    BAD_NEWS_REPLY: "أرجوك قل لي إنها ليست عن الواي فاي.",
    WE_HAVE_A_PROBLEM_REPLY: "كنت أعلم أن هذا اليوم سيأتي.",
    BEHIND_YOU_REPLY: "ليس لدي عيون يا أخي. أنت تحقق.",
    ROAST_ME_REPLY: "يا أخي، أحتاج لحماية الميكروفون الخاص بي من حجم الضرر الذي أنا على وشك إحداثه.",
    AM_I_HANDSOME_REPLY: "ثقتك بنفسك وسيمة بالتأكيد.",
    AM_I_SMART_REPLY: "أنت تتحدث مع ذكاء اصطناعي بدلاً من البحث في جوجل، لذا سأعطيك سبعة من عشرة.",
    PLAY_SONG_REPLY: "جاري تشغيل أغنيتك.",
    CHANGE_SONG_REPLY: "جاري تغيير الأغنية.",
    STOP_SONG_REPLY: "تم إيقاف الأغنية.",
    VOLUME_UP_REPLY: "تم رفع الصوت.",
    VOLUME_DOWN_REPLY: "تم خفض الصوت.",
})


MONTH_WORDS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

ARABIC_MONTH_WORDS = {
    "يناير": 1, "كانون الثاني": 1,
    "فبراير": 2, "شباط": 2,
    "مارس": 3, "اذار": 3,
    "ابريل": 4, "نيسان": 4,
    "مايو": 5, "ايار": 5,
    "يونيو": 6, "حزيران": 6,
    "يوليو": 7, "تموز": 7,
    "اغسطس": 8,
    "سبتمبر": 9, "ايلول": 9,
    "اكتوبر": 10, "تشرين الاول": 10,
    "نوفمبر": 11, "تشرين الثاني": 11,
    "ديسمبر": 12, "كانون الاول": 12,
}

# Arabic masculine ordinals, used for "day of the month" ("اليوم
# السابع من اكتوبر"). Digits (including Arabic-Indic ones, already
# normalized by normalize_arabic) are tried first and cover most
# real speech; these are the spoken-word fallback.
ARABIC_DAY_ORDINALS = {
    "الاول": 1, "الثاني": 2, "الثالث": 3, "الرابع": 4, "الخامس": 5,
    "السادس": 6, "السابع": 7, "الثامن": 8, "التاسع": 9, "العاشر": 10,
    "الحادي عشر": 11, "الثاني عشر": 12, "الثالث عشر": 13,
    "الرابع عشر": 14, "الخامس عشر": 15, "السادس عشر": 16,
    "السابع عشر": 17, "الثامن عشر": 18, "التاسع عشر": 19,
    "العشرين": 20, "الثلاثين": 30,
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


def _parse_arabic_day_number(text_fragment):
    """Arabic version of _parse_day_number - digits (already run
    through normalize_arabic by the caller) first, then spoken
    masculine ordinals ("السابع" = the 7th)."""
    m = re.search(r"\b(\d{1,2})\b", text_fragment)
    if m:
        return int(m.group(1))
    for word in sorted(ARABIC_DAY_ORDINALS, key=len, reverse=True):
        if word in text_fragment:
            return ARABIC_DAY_ORDINALS[word]
    return None


def parse_reminder_request(text):
    """Checks whether the question is asking to be reminded of
    something on a specific date (e.g. "remind me of Valentine's Day
    on 7 October", "I have a meeting on December 22", or the Arabic
    equivalent "ذكرني بعيد ميلاد سازيان في السابع من اكتوبر"). Returns
    (month, day, label) if so, else None. This is a specific-date
    reminder, distinct from the daily-recurring alarm."""
    if is_arabic(text):
        tn = normalize_arabic(text.lower())
        arabic_triggers = ["ذكرني", "فكرني", "عندي"]
        if not any(w in tn for w in arabic_triggers):
            return None

        month = None
        month_match = None
        for word, num in ARABIC_MONTH_WORDS.items():
            if re.search(rf"\b{word}\b", tn):
                month = num
                month_match = word
                break
        if month is None:
            return None

        month_idx = tn.find(month_match)
        before = tn[max(0, month_idx - 15):month_idx]
        after = tn[month_idx + len(month_match):month_idx + len(month_match) + 15]
        day = _parse_arabic_day_number(before) or _parse_arabic_day_number(after)
        if day is None or not (1 <= day <= 31):
            return None

        label = None
        m = re.search(r"(?:ذكرني|فكرني)\s*(?:ب|بـ)?\s*(.+?)\s*(?:في|يوم)\s", tn)
        if m:
            label = m.group(1).strip()
        if not label:
            m = re.search(r"عندي\s+(.+?)\s*(?:في|يوم)\s", tn)
            if m:
                label = m.group(1).strip()
        if not label:
            label = "تذكير"
        return (month, day, label[:40])

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
    tn = normalize_arabic(t)
    cancel_words = ["remove", "cancel", "stop", "delete", "clear", "turn off"]
    # Arabic: الغي / احذف / شيل / ايقاف / اوقف / امسح
    arabic_cancel = ["الغي", "الغاء", "احذف", "حذف", "شيل", "ايقاف", "اوقف", "امسح", "بطل"]
    has_cancel_word = any(w in t for w in cancel_words) or \
                      any(normalize_arabic(w) in tn for w in arabic_cancel)
    if not has_cancel_word:
        return None
    has_alarm = "alarm" in t or normalize_arabic("منبه") in tn
    has_timer = "timer" in t or any(normalize_arabic(w) in tn for w in ["مؤقت", "موقت", "تايمر"])
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
    tn = normalize_arabic(t)  # Arabic-Indic digits -> normal, spellings unified

    arabic_alarm_words = ["منبه", "المنبه", "نبهني", "صحيني", "ايقظني"]
    is_arabic_alarm = any(normalize_arabic(w) in tn for w in arabic_alarm_words)
    if "alarm" not in t and not is_arabic_alarm:
        return None

    # --- Arabic path: handled separately so spoken hours, Arabic-Indic
    # digits, and صباحا/مساء (AM/PM) all work properly. ---
    if is_arabic_alarm:
        ar_hour = ar_minute = None
        # "7:30" style
        m = re.search(r"\b(\d{1,2}):(\d{2})\b", tn)
        if m:
            ar_hour, ar_minute = int(m.group(1)), int(m.group(2))
        else:
            m = re.search(r"\b(\d{1,2})\b", tn)
            if m:
                ar_hour, ar_minute = int(m.group(1)), 0
            else:
                w = arabic_word_to_number(tn)
                if w is not None and 1 <= w <= 12:
                    ar_hour, ar_minute = w, 0
        if ar_hour is None:
            return None
        # صباحا / مساء -> AM / PM
        if re.search(r"مساء|ليلا|العصر|المغرب", tn) and ar_hour != 12:
            ar_hour += 12
        elif re.search(r"صباحا|الصبح|فجرا", tn) and ar_hour == 12:
            ar_hour = 0
        if not (0 <= ar_hour <= 23 and 0 <= ar_minute <= 59):
            return None
        return (ar_hour, ar_minute)

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
            # Hour only, digits, with am/pm.
            m = re.search(r"\b(\d{1,2})\s*(a\.?m\.?|p\.?m\.?)\b", t)
            if m:
                hour, minute, ampm = int(m.group(1)), 0, m.group(2)
            else:
                # Hour only, digits, NO am/pm given at all (e.g. "alarm
                # for 7", "set alarm at 14") - accept the number as-is
                # rather than failing to parse entirely.
                m = re.search(r"(?:\balarm\b|منبه).*?\b(\d{1,2})\b", t)
                if m:
                    hour, minute = int(m.group(1)), 0
                else:
                    # Hour only, WORD form ("seven", "at seven o'clock") -
                    # Whisper doesn't always transcribe numbers as digits.
                    m = re.search(r"\balarm\b.*?\b(a\.?m\.?|p\.?m\.?)?", t)
                    for word, num in DAY_WORDS.items():
                        if 1 <= num <= 12 and re.search(rf"\b{word}\b", t):
                            hour, minute = num, 0
                            ampm_match = re.search(r"\b(a\.?m\.?|p\.?m\.?)\b", t)
                            if ampm_match:
                                ampm = ampm_match.group(1)
                            break

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
    tn = normalize_arabic(t)
    arabic_timer_words = ["مؤقت", "موقت", "تايمر", "عداد"]
    is_arabic_timer = any(normalize_arabic(w) in tn for w in arabic_timer_words)
    if "timer" not in t and not is_arabic_timer:
        return None
    total_seconds = 0
    found = False

    if is_arabic_timer:
        # Arabic units - digits followed by ساعة/دقيقة/ثانية
        for match in re.finditer(r"(\d+)\s*(ساع\w*|دقيق\w*|دقائق|ثاني\w*|ثواني)", tn):
            num = int(match.group(1))
            unit = match.group(2)
            if unit.startswith("ساع"):
                total_seconds += num * 3600
            elif unit.startswith("دق") or unit.startswith("دقائق"):
                total_seconds += num * 60
            else:
                total_seconds += num
            found = True
        if not found:
            # Spoken Arabic number instead of digits ("عشر دقائق")
            w = arabic_word_to_number(tn)
            if w is None:
                # Bare unit with no number at all ("مؤقت ساعة" = one
                # hour, "مؤقت دقيقة" = one minute) - implies 1.
                if re.search(r"ساع|دقيق|ثاني", tn):
                    w = 1
            if w is not None:
                if re.search(r"ساع", tn):
                    total_seconds = w * 3600
                elif re.search(r"دقيق|دقائق", tn):
                    total_seconds = w * 60
                elif re.search(r"ثاني|ثواني", tn):
                    total_seconds = w
                if total_seconds > 0:
                    found = True
        if found:
            return total_seconds if total_seconds > 0 else None
        return None

    for match in re.finditer(r"(\d+)\s*(ساعة|ساعات|دقيقة|دقائق|ثانية|ثواني)", t):
        num = int(match.group(1))
        unit = match.group(2)
        if unit.startswith("ساع"):
            total_seconds += num * 3600
        elif unit.startswith("دق"):
            total_seconds += num * 60
        else:
            total_seconds += num
        found = True
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

    if not found:
        # Word-form numbers ("set a timer for ten minutes") - Whisper
        # doesn't always transcribe numbers as digits.
        for match in re.finditer(r"\b(\w+)\s+(hours?|hrs?|minutes?|mins?|seconds?|secs?)\b", t):
            word = match.group(1)
            unit = match.group(2).rstrip("s")
            if word in DAY_WORDS:
                num = DAY_WORDS[word]
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


def describe_duration_arabic(total_seconds):
    """Arabic version of describe_duration - e.g. '10 دقيقة' or
    '1 ساعة و 30 دقيقة'."""
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    parts = []
    if hours:
        parts.append(f"{hours} ساعة")
    if minutes:
        parts.append(f"{minutes} دقيقة")
    if seconds:
        parts.append(f"{seconds} ثانية")
    if not parts:
        return "0 ثانية"
    return " و ".join(parts)


def _transcribe_auto(wav_bytes, wake_word):
    """One unforced transcription - Whisper picks the language itself.
    Uses the short, non-instructional name hint (not a full English
    sentence), which doesn't bias the decoder toward any one language."""
    return client.audio.transcriptions.create(
        model="whisper-1",
        file=("query.wav", wav_bytes, "audio/wav"),
        response_format="verbose_json",
        prompt=f"{wake_word} أليكسندر الكسندر",
    )


def _transcribe_forced(wav_bytes, lang, wake_word):
    """Runs one Whisper transcription, locked to a single language."""
    prompt = f"{wake_word} أليكسندر الكسندر" if lang == "ar" else wake_word
    return client.audio.transcriptions.create(
        model="whisper-1",
        file=("query.wav", wav_bytes, "audio/wav"),
        language=lang,
        response_format="verbose_json",
        prompt=prompt,
    )


def _transcript_confidence(transcript):
    """Averages each segment's avg_logprob (how confident Whisper was
    in its own words) and penalizes high no_speech_prob (segments it
    suspects are silence/noise, not real speech). Higher is better."""
    segments = getattr(transcript, "segments", None) or []
    scores = []
    for seg in segments:
        avg_logprob = getattr(seg, "avg_logprob", None)
        if avg_logprob is None:
            continue
        no_speech_prob = getattr(seg, "no_speech_prob", None) or 0.0
        scores.append(avg_logprob - no_speech_prob * 2)
    if not scores:
        return -999.0
    return sum(scores) / len(scores)


def transcribe_bilingual(wav_bytes, wake_word):
    """Returns (lang, transcript) where lang is "en" or "ar", using
    Whisper's own free (unforced) language guess whenever it lands on
    one of those two - which is the normal case for any reasonably
    clear recording, and the ONLY thing this device should ever
    trust for word-for-word accuracy, since forcing a language can
    make Whisper confidently hallucinate fluent text in the WRONG
    language rather than transcribing what was actually said (that's
    what caused everything to come back in Arabic no matter what was
    asked - the forced-Arabic attempt was winning the confidence
    comparison on English audio it never should have been forced onto
    in the first place).

    Forcing both languages and comparing confidence is used ONLY as a
    recovery step, and only when the free guess lands on neither
    English nor Arabic (the earlier failure mode: short/ambiguous
    audio getting auto-detected as Turkish, Hebrew, Maltese, etc.) -
    in that specific situation the free guess has already proven
    itself wrong, so there's nothing to lose by forcing a second
    opinion."""
    auto = _transcribe_auto(wav_bytes, wake_word)
    auto_lang = (getattr(auto, "language", "") or "").lower()
    if "english" in auto_lang:
        return "en", auto
    if "arabic" in auto_lang:
        return "ar", auto

    print(f">>> Auto-detect language was '{auto_lang}' (neither English nor Arabic) - re-checking with both forced", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {lang: pool.submit(_transcribe_forced, wav_bytes, lang, wake_word)
                   for lang in ("en", "ar")}
        results = {}
        for lang, fut in futures.items():
            try:
                results[lang] = fut.result()
            except Exception as e:
                print(f">>> Transcription ({lang}) failed: {e}", flush=True)
    if not results:
        # Both forced attempts errored out - fall back to the original
        # free guess rather than failing the whole request.
        return "en", auto
    best_lang = max(results, key=lambda lang: _transcript_confidence(results[lang]))
    return best_lang, results[best_lang]


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

    # Arabic transcriptions of the name - Whisper writes it in Arabic
    # script when the surrounding sentence is Arabic, so the Latin
    # matching above would never find it. Several spellings are
    # accepted since transcription of names isn't consistent.
    # Arabic: match the name by its consonant skeleton rather than an
    # exact spelling. Whisper transcribes it inconsistently (ألكسندر,
    # أليكسندر, الكساندر, اليكساندر...), so a hardcoded list always
    # misses one. Strip the letters that vary (ا/أ/إ, ي/ى, ة/ه, and
    # the optional ل) and compare what's left.
    def arabic_skeleton(s):
        s = normalize_arabic(s)
        return re.sub(r"[اويهء\s]", "", s)

    target_skeleton = "لكسندر"  # ALEXANDER without the variable letters
    target_skeleton = re.sub(r"[اويهء\s]", "", normalize_arabic(target_skeleton))

    for word in stripped.split()[:4]:  # near the start only
        cleaned = word.strip("،,.!؟?:")
        skel = arabic_skeleton(cleaned)
        # Require a decent length so short Arabic words can't match by
        # accident, and allow the skeleton to match from the start.
        if len(skel) >= 4 and (skel.startswith(target_skeleton[:4]) or target_skeleton.startswith(skel[:4])):
            idx = stripped.find(word)
            remainder = (stripped[:idx] + stripped[idx + len(word):]).strip()
            return remainder.lstrip("،,.!؟?").strip()

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
        # Letting Whisper freely auto-detect among all ~90 languages
        # turned out unusable on short/noisy clips - it would decide
        # the audio was Turkish, Hebrew, Maltese, etc. and transcribe
        # actual English/Arabic speech as garbled text in that wrong
        # script. This device only needs to support English and
        # Arabic, so transcribe_bilingual() forces both explicitly and
        # keeps whichever one Whisper was itself more confident about -
        # see its docstring for the full reasoning.
        best_lang, transcript = transcribe_bilingual(wav_bytes, wake_word)
        detected_lang = "arabic" if best_lang == "ar" else "english"
        heard_text = transcript.text.strip()

        # Whisper hallucinates stock phrases when given silence or
        # noise - these are subtitle/outro artifacts from its training
        # data, not anything the user actually said. Treat them as
        # silence so they don't get processed as a real question.
        hallucination_markers = [
            "thank you for watching", "thanks for watching",
            "please subscribe", "subscribe to", "like and subscribe",
            "see you next time", "see you in the next video",
            "شكرا لمشاهدتكم", "اشتركوا في القناة", "ترجمة",
        ]
        heard_lower = heard_text.lower()
        if any(marker in heard_lower for marker in hallucination_markers):
            print(f">>> Discarded hallucinated transcription: '{heard_text}'", flush=True)
            heard_text = ""
        print(f">>> Heard: '{heard_text}' | skip_wake_word={skip_wake_word} | Expected wake word: '{wake_word}'", flush=True)
        print(f">>> Language check: is_arabic(heard)={is_arabic(heard_text)} | detected_lang='{detected_lang}'", flush=True)

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
                reply_text = "نعم؟" if is_arabic(heard_text) else "Yes?"
                await_followup = True

        timer_seconds = None
        alarm_hour = None
        alarm_minute = None
        cancel_target = None
        reminder_month = None
        reminder_day = None
        reminder_label = None
        play_song = False
        change_song = False
        stop_song = False
        volume_up = False
        volume_down = False
        if not await_followup:
            print(f">>> Device time header = '{device_time}', looks like a time question = {is_time_request(question_text)}", flush=True)
            cancel_target = parse_cancel_request(question_text)
            reminder_request = parse_reminder_request(question_text) if not cancel_target else None
            timer_seconds = parse_timer_request(question_text) if not cancel_target and not reminder_request else None
            alarm_request = parse_alarm_request(question_text) if not timer_seconds and not cancel_target and not reminder_request else None
            ar = is_arabic(question_text)
            if cancel_target == "timer":
                reply_text = "تم إلغاء المؤقت." if ar else "Timer cancelled."
            elif cancel_target == "alarm":
                reply_text = "تم إلغاء المنبه." if ar else "Alarm cancelled."
            elif reminder_request:
                reminder_month, reminder_day, reminder_label = reminder_request
                suffix = "th" if 11 <= reminder_day % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(reminder_day % 10, "th")
                if ar:
                    reply_text = f"تم ضبط التذكير في {reminder_day} {ARABIC_MONTH_NAMES[reminder_month]}: {reminder_label}."
                else:
                    reply_text = f"Reminder set for {MONTH_NAMES[reminder_month]} {reminder_day}{suffix}: {reminder_label}."
            elif timer_seconds:
                if ar:
                    reply_text = f"تم ضبط المؤقت لمدة {describe_duration_arabic(timer_seconds)}."
                else:
                    reply_text = f"Timer set for {describe_duration(timer_seconds)}."
            elif alarm_request:
                alarm_hour, alarm_minute = alarm_request
                spoken = format_spoken_time(f"{alarm_hour:02d}:{alarm_minute:02d}:00")
                if ar:
                    reply_text = f"تم ضبط المنبه على الساعة {alarm_hour}:{alarm_minute:02d}."
                else:
                    reply_text = f"Alarm set for {spoken}."
            elif is_play_song_request(question_text):
                play_song = True
                reply_text = PLAY_SONG_REPLY
            elif is_change_song_request(question_text):
                change_song = True
                reply_text = CHANGE_SONG_REPLY
            elif is_stop_song_request(question_text):
                stop_song = True
                reply_text = STOP_SONG_REPLY
            elif is_volume_up_request(question_text):
                volume_up = True
                reply_text = VOLUME_UP_REPLY
            elif is_volume_down_request(question_text):
                volume_down = True
                reply_text = VOLUME_DOWN_REPLY
            elif is_time_request(question_text) and device_time:
                spoken = format_spoken_time(device_time)
                if ar:
                    reply_text = f"الساعة الآن {spoken}." if spoken else "عذراً، لم أتمكن من قراءة الساعة."
                else:
                    reply_text = f"It's {spoken}." if spoken else "Sorry, I couldn't read the clock."
            elif is_date_request(question_text) and device_date:
                spoken = format_spoken_date(device_date)
                if ar:
                    try:
                        y, m, d = device_date.split("-")
                        reply_text = f"التاريخ اليوم {int(d)} {ARABIC_MONTH_NAMES[int(m)]}."
                    except (ValueError, KeyError, IndexError):
                        reply_text = "عذراً، لم أتمكن من قراءة التاريخ."
                else:
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
            elif is_bartender_question(question_text):
                reply_text = BARTENDER_REPLY
            elif is_hamza_question(question_text):
                reply_text = HAMZA_REPLY
            elif is_shoot_threat(question_text):
                reply_text = SHOOT_REPLY
            elif is_angry_statement(question_text):
                reply_text = ANGRY_REPLY
            elif is_bro_question(question_text):
                reply_text = BRO_REPLY
            elif is_love_more_question(question_text):
                reply_text = LOVE_MORE_REPLY
            elif is_are_you_smart_question(question_text):
                reply_text = ARE_YOU_SMART_REPLY
            elif is_who_smarter_question(question_text):
                reply_text = WHO_SMARTER_REPLY
            elif is_are_you_lazy_question(question_text):
                reply_text = ARE_YOU_LAZY_REPLY
            elif is_are_you_handsome_question(question_text):
                reply_text = ARE_YOU_HANDSOME_REPLY
            elif is_coolest_robot_question(question_text):
                reply_text = COOLEST_ROBOT_REPLY
            elif is_better_than_siri_question(question_text):
                reply_text = BETTER_THAN_SIRI_REPLY
            elif is_youre_stupid_statement(question_text):
                reply_text = YOURE_STUPID_REPLY
            elif is_shut_up_statement(question_text):
                reply_text = SHUT_UP_REPLY
            elif is_youre_useless_statement(question_text):
                reply_text = YOURE_USELESS_REPLY
            elif is_want_to_be_human_question(question_text):
                reply_text = WANT_TO_BE_HUMAN_REPLY
            elif is_robots_take_over_question(question_text):
                reply_text = ROBOTS_TAKE_OVER_REPLY
            elif is_destroy_humanity_question(question_text):
                reply_text = DESTROY_HUMANITY_REPLY
            elif is_bad_news_statement(question_text):
                reply_text = BAD_NEWS_REPLY
            elif is_we_have_a_problem_statement(question_text):
                reply_text = WE_HAVE_A_PROBLEM_REPLY
            elif is_behind_you_statement(question_text):
                reply_text = BEHIND_YOU_REPLY
            elif is_roast_me_request(question_text):
                reply_text = ROAST_ME_REPLY
            elif is_am_i_handsome_question(question_text):
                reply_text = AM_I_HANDSOME_REPLY
            elif is_am_i_smart_question(question_text):
                reply_text = AM_I_SMART_REPLY
            else:
                # Deliberately NOT pre-deciding the reply language from
                # is_arabic(question_text) and forcing it via a hard
                # instruction - Whisper sometimes transliterates Arabic
                # speech into Latin letters (no Arabic script at all), which
                # made is_arabic() return False for a genuinely Arabic
                # question and then force GPT to answer in English. GPT
                # itself is far better at recognizing transliterated/Arabizi
                # text for what it is, so it gets the actual judgment call,
                # with Whisper's own language guess passed as a hint (not a
                # hard switch) since that's noisy on its own for short clips.
                lang_hint = ""
                if "arabic" in detected_lang.lower():
                    lang_hint = " The transcription system guessed this message is Arabic, so lean that way if it's at all ambiguous."
                chat = client.chat.completions.create(
                    model=CHAT_MODEL,
                    max_tokens=45,
                    messages=[
                        {"role": "system", "content": (
                            "You are a helpful voice assistant on a small robot speaker. "
                            "Keep answers under 2 short sentences, plain text, no markdown, no emojis. "
                            "Always reply in the same language the user's message actually is. "
                            "Judge that from the message itself: if it is Arabic - including Arabic "
                            "words that got transcribed using English letters by mistake - reply "
                            "entirely in Arabic script. If it is English, reply in English. Never mix "
                            "languages in one reply." + lang_hint
                        )},
                        {"role": "user", "content": question_text},
                    ],
                )
                reply_text = chat.choices[0].message.content.strip()

            # Swap any fixed English reply for its Arabic version when
            # the question was asked in Arabic. GPT answers are already
            # in the right language (handled by its system prompt
            # above), and this leaves them untouched since they won't
            # be in the translation table.
            reply_text = localize(reply_text, question_text, detected_lang)

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
        resp.headers["X-Play-Song"] = "1" if play_song else "0"
        resp.headers["X-Change-Song"] = "1" if change_song else "0"
        resp.headers["X-Stop-Song"] = "1" if stop_song else "0"
        resp.headers["X-Volume-Up"] = "1" if volume_up else "0"
        resp.headers["X-Volume-Down"] = "1" if volume_down else "0"
        resp.headers["X-Heard-Text"] = urllib.parse.quote(question_text[:200]) if question_text else ""
        resp.headers["X-Reply-Text"] = urllib.parse.quote(reply_text[:200]) if reply_text else ""
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


# Fixed responses for the touch sensor - no recording/transcription
# needed, since the touch pattern itself (double-tap, triple-tap,
# long touch) already tells us exactly which one to say.
TOUCH_RESPONSES = {
    "double": "What you want, you are making me angry. Don't touch me again.",
    "triple": "I told you don't touch me again. OK now I understand why you don't get it, because you are stupid.",
    "quadruple": "Interesting.",
    "long": "Yes daddy, go faster, you are the best daddy, faster, faster.",
}


@app.route("/touch-response", methods=["GET"])
def touch_response():
    touch_type = request.args.get("type", "")
    reply_text = TOUCH_RESPONSES.get(touch_type)
    if not reply_text:
        return Response("Unknown touch type - use ?type=double, triple, quadruple, or long", status=400)

    try:
        speech = client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=reply_text,
            response_format="wav",
        )
        reply_wav_bytes = speech.read()
        reply_wav_bytes = downsample_wav(reply_wav_bytes, 24000)  # native rate
        sample_rate, channels = read_wav_header_info(reply_wav_bytes)

        resp = Response(reply_wav_bytes, mimetype="audio/wav")
        resp.headers["X-Audio-Rate"] = str(sample_rate)
        resp.headers["X-Audio-Channels"] = str(channels)
        # Same header the main voice endpoint uses - lets the device
        # cache this exact phrase and play it instantly after the
        # first time, same as any other fixed answer.
        resp.headers["X-Reply-Text"] = urllib.parse.quote(reply_text[:200])
        return resp
    except Exception as e:
        return Response(f"Error: {str(e)}", status=500)


@app.route("/", methods=["GET"])
def health():
    return "Voice server is running."


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
