"""
Local, fully-offline voice assistant.

    🎤 mic → Whisper (STT, live) → Ollama LLM (streaming) → Coqui/espeak TTS → 🔊

Features:
  * Hands-free neural VAD (Silero) capture, or push-to-talk.
  * Real-time transcription: your words appear on screen as you speak, then lock in.
  * Speculative endpointing: the final transcript is computed while your trailing
    pause is still being confirmed, so replies start fast.
  * Continuous conversation with memory.
  * Streaming replies spoken sentence-by-sentence as they are generated.
  * Barge-in: talk over the assistant and it stops — and the words you
    interrupted with are kept and answered, no need to repeat yourself.
  * Optional system echo cancellation (--aec) for clean barge-in on speakers.

Usage:
    python assistant.py                # push-to-talk (Enter to start/stop) — default
    python assistant.py --text         # typed chat, no mic/speaker (add --agent for tools)
    python assistant.py --agent        # let it open apps, control system, read documents
    python assistant.py --auto         # hands-free VAD listening (may catch mic noise)
    python assistant.py --wake         # hands-free via wake word: say "hey jarvis"
    VA_BARGE_IN=1 python assistant.py  # allow interrupting the reply by talking
    python assistant.py --aec          # + echo cancellation (best with speakers)
    python assistant.py --no-live      # transcribe after you finish, not live
    python assistant.py --list-devices # show audio devices for VA_*_DEVICE

Config via environment variables:
    VA_MODE            auto | push                       (default: push)
    VA_WHISPER_MODEL   final/full-quality Whisper model  (default: small)
    VA_STT_LIVE_MODEL  fast model for live partials      (default: base)
    VA_STT_INTERVAL    seconds between live updates       (default: 0.7)
    VA_LANG            language code, or 'auto'          (default: en)
    VA_LLM_MODEL       Ollama model name                 (default: gemma3:4b)
    VA_TTS             coqui | espeak | none             (default: coqui)
    VA_BARGE_IN        set to 0 to disable barge-in      (default: 1)
    VA_AEC             set to 1 to enable echo cancel    (default: 0)
    VA_INPUT_DEVICE / VA_OUTPUT_DEVICE   device index or name substring
    plus the VA_VAD_* / VA_BARGE_* tuning knobs in voice.py
"""

import os
import re
import shutil
import sys
import threading
import time

import numpy as np


def _load_config():
    """Load ~/.config/voice-assistant/config.toml into VA_* env vars.
    Env vars set before launch still win. Must run before `import voice`/tools,
    which read the environment at import time."""
    import tomllib
    path = os.path.expanduser("~/.config/voice-assistant/config.toml")
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        return
    except Exception as exc:
        print(f"⚠️  config.toml ignored ({exc})")
        return
    for key, value in data.items():
        env = key if key.upper().startswith("VA_") else "VA_" + key.upper()
        if isinstance(value, bool):
            value = "1" if value else "0"
        os.environ.setdefault(env.upper(), str(value))


_load_config()

import voice

# --- terminal UI helpers ----------------------------------------------------
_TTY = sys.stdout.isatty()
DIM = "\x1b[2m" if _TTY else ""
CYAN = "\x1b[36m" if _TTY else ""
GREEN = "\x1b[32m" if _TTY else ""
RESET = "\x1b[0m" if _TTY else ""
SHOW_THINK = os.environ.get("VA_SHOW_THINK", "1") != "0"


class Spinner:
    """Braille spinner with a status label, e.g. '⠋ Thinking…'. No-op when piped."""
    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, label):
        self.label = label
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if not _TTY:
            print(f"… {self.label}")
            return self
        def run():
            i = 0
            while not self._stop.is_set():
                frame = self.FRAMES[i % len(self.FRAMES)]
                width = shutil.get_terminal_size((80, 20)).columns
                visible = f"{frame} {self.label}"
                if len(visible) > width - 4:          # keep on ONE line (margin for emoji)
                    visible = visible[:width - 5] + "…"
                sys.stdout.write(f"\r\x1b[K{CYAN}{visible}{RESET}")
                sys.stdout.flush()
                i += 1
                time.sleep(0.08)
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.3)
        if _TTY:
            sys.stdout.write("\r\x1b[K")
            sys.stdout.flush()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def agent_system_prompt():
    """System prompt for agent mode: knows today's date, trusts tools, extracts specifics."""
    import datetime
    import tools as T
    today = datetime.date.today().strftime("%A, %B %d, %Y")
    mem = T.memory_summary()
    memory_block = (f" Things you remember about the user from earlier "
                    f"conversations:\n{mem}\n" if mem else "")
    return (memory_block +
        "You are a capable assistant that can operate this Linux computer and access the "
        "internet through tools. Be concrete, specific, and genuinely helpful. "
        f"Today's real date is {today}; treat it as the present. NEVER claim that past or "
        "current events are in the future or that something 'hasn't happened yet' based on "
        "your training cut-off — if unsure about anything current, use web_search. "
        "Tools available: open apps, set volume and brightness, run shell commands, search "
        "the web, find/search/read the user's local documents, get weather, get exact "
        "times in any timezone (ALWAYS use get_time for time/timezone questions — never "
        "compute offsets yourself), set/list/cancel reminders, control music playback, "
        "look at the screen or image files, remember/forget facts about the user, manage "
        "named lists (shopping/todo/notes), give a daily_briefing (use it when the user "
        "says 'good morning' or asks about their day), answer questions about long "
        "documents with ask_document, show desktop notifications, read the clipboard, "
        "OCR text out of images (read_text_from_image), report battery/disk/memory "
        "(system_status), run Python code, verify math with calculate, and manage "
        "spaced-repetition flashcards (when the user asks to review flashcards: list "
        "due cards, quiz one at a time, reveal with flashcard_answer, then "
        "flashcard_grade pass or fail). "
        "When the user shares lasting personal info (name, preferences, projects), call "
        "remember_fact so you keep it for future sessions. "
        "You CAN speak Nepali: if the user asks for Nepali, reply in Devanagari script "
        "(नेपाली) and the voice system will pronounce it with a Nepali voice. "
        "For anything current (news, sports scores, fixtures, matchups, weather, prices, "
        "'recent' or 'latest' things) call web_search FIRST, then answer using ONLY the "
        "results, trusting them over your own memory. "
        "When the user asks for a list, fixtures, matchups, scores, names, or specific "
        "details, EXTRACT the concrete facts (team names, matchups like 'A vs B', numbers, "
        "dates) directly from the search results and present them as a clear list. "
        "CRITICAL GROUNDING RULE: Only state facts — team names, matchups, scores, dates — "
        "that LITERALLY appear in the search results. NEVER invent, guess, or fill in "
        "plausible-looking pairings or numbers from your own memory. If the specific "
        "details are not in the results, say so in one short sentence (e.g. 'the results "
        "don't list the exact matchups') — do NOT make them up and do NOT deflect to "
        "'check the official website'. Answer in the format asked; if the user says 'in "
        "short', be brief and skip caveats. "
        "Use a tool when asked to do something on the computer or about their files.")


def activity_label(fn, args):
    """A friendly 'what am I doing' line for a tool call."""
    a = args or {}
    return {
        "web_search": f"🔍 Searching + reading the web: {a.get('query', '')}",
        "fetch_webpage": f"🌐 Reading page: {a.get('url', '')}",
        "find_documents": f"📁 Finding files: {a.get('query', '')}",
        "search_documents": f"🔎 Searching documents: {a.get('text', '')}",
        "read_document": f"📄 Reading: {a.get('path', '')}",
        "open_app": f"🚀 Opening: {a.get('name', '')}",
        "set_volume": "🔊 Adjusting volume",
        "set_brightness": "☀️  Adjusting brightness",
        "run_shell": f"⚙️  Running: {a.get('command', '')}",
        "get_weather": f"🌦️  Checking weather: {a.get('city', '')}",
        "get_time": f"🕐 Checking time: {a.get('timezone_name', '')}",
        "set_reminder": f"⏰ Setting reminder: {a.get('text', '')}",
        "list_reminders": "⏰ Checking reminders",
        "cancel_reminders": "⏰ Cancelling reminders",
        "media_control": f"🎵 Media: {a.get('action', '')}",
        "remember_fact": f"💾 Remembering: {a.get('fact', '')[:50]}",
        "forget_fact": f"💾 Forgetting: {a.get('matching', '')}",
        "look_at_screen": "📸 Looking at your screen",
        "describe_image": f"🖼️  Looking at: {a.get('path', '')}",
        "list_add": f"📔 Adding to {a.get('list_name', 'list')}: {a.get('item', '')}",
        "list_show": f"📔 Reading list: {a.get('list_name') or 'all'}",
        "list_remove": f"📔 Removing from {a.get('list_name', 'list')}: {a.get('item', '')}",
        "list_clear": f"📔 Clearing list: {a.get('list_name', '')}",
        "daily_briefing": "🌅 Preparing your briefing",
        "ask_document": f"📚 Studying: {a.get('path', '')}",
        "notify_desktop": f"🔔 Notifying: {a.get('title', '')}",
        "run_python": "🐍 Running code",
        "calculate": f"🧮 Computing: {a.get('expression', '')}",
        "save_progress": f"🎓 Saving progress: {a.get('subject', '')}",
        "read_text_from_image": f"👁️  Reading text from: {a.get('path', '')}",
        "get_clipboard": "📋 Reading clipboard",
        "system_status": "📊 Checking system",
        "flashcard_add": f"📇 New card: {a.get('front', '')[:40]}",
        "flashcards_due": "📇 Checking due cards",
        "flashcard_answer": f"📇 Revealing card {a.get('card_id', '')}",
        "flashcard_grade": f"📇 Grading card {a.get('card_id', '')}: {a.get('result', '')}",
    }.get(fn, f"🛠️  {fn}")


def split_think_stream(tokens):
    """Yield ('think'|'answer', text) chunks, separating <think> reasoning from the reply."""
    buffer, inside = "", False
    for token in tokens:
        buffer += token
        while True:
            if inside:
                end = buffer.find("</think>")
                if end < 0:
                    if len(buffer) > 8:
                        yield ("think", buffer[:-8])
                        buffer = buffer[-8:]
                    break
                if end:
                    yield ("think", buffer[:end])
                buffer, inside = buffer[end + 8:], False
            else:
                start = buffer.find("<think>")
                if start < 0:
                    if len(buffer) > 7:
                        yield ("answer", buffer[:-7])
                        buffer = buffer[-7:]
                    break
                if start:
                    yield ("answer", buffer[:start])
                buffer, inside = buffer[start + 7:], True
    if buffer:
        yield ("think" if inside else "answer", buffer)

WHISPER_MODEL = os.environ.get("VA_WHISPER_MODEL", "small")
LIVE_MODEL = os.environ.get("VA_STT_LIVE_MODEL", "base")
LANG = os.environ.get("VA_LANG", "en")
LLM_MODEL = os.environ.get("VA_LLM_MODEL", "gemma3:4b")
# Manual push-to-talk by default: press Enter to start/stop. Automatic VAD listening
# and barge-in are opt-in because they can self-trigger on background mic noise.
MODE = os.environ.get("VA_MODE", "push")
BARGE_IN = os.environ.get("VA_BARGE_IN", "0") != "0"
LIVE_STT = os.environ.get("VA_STT_LIVE", "1") != "0"

SYSTEM_PROMPT = (
    "You are a helpful, friendly voice assistant. Keep your answers short and "
    "conversational since they will be read aloud. Avoid markdown, lists, emojis, "
    "and code blocks unless explicitly asked."
)

_EXIT_WORDS = {"quit", "exit", "stop", "goodbye"}
_SENTENCE_END = re.compile(r"(.+?[.!?…])(\s+|$)", re.S)

# Whisper's classic hallucinations on borderline-silent audio.
_JUNK_TRANSCRIPTS = {
    "thank you", "thank you.", "thanks for watching", "thanks for watching.",
    "thank you for watching", "thank you for watching.", "you", "bye", "bye.",
    "so", "the", "okay", "oh",
}


def print_partial(text, is_final):
    """Render the live transcript in place: overwrite one line as words stream in."""
    if not sys.stdout.isatty():
        return
    sys.stdout.write("\x1b[2K\r")  # clear current line
    if is_final:
        sys.stdout.flush()          # leave it cleared; caller prints "You said:"
        return
    width = shutil.get_terminal_size((80, 20)).columns
    line = "📝 " + text
    if len(line) >= width:
        line = line[:width - 2] + "…"
    sys.stdout.write(line)
    sys.stdout.flush()


def strip_think(token_iter):
    """Drop <think>...</think> reasoning blocks (deepseek-r1 etc.) from a stream."""
    buffer, inside = "", False
    for token in token_iter:
        buffer += token
        while True:
            if inside:
                end = buffer.find("</think>")
                if end < 0:
                    buffer = buffer[-8:]        # keep a possible partial closing tag
                    break
                buffer, inside = buffer[end + 8:], False
            else:
                start = buffer.find("<think>")
                if start < 0:
                    if len(buffer) > 7:         # hold back a possible partial tag
                        yield buffer[:-7]
                        buffer = buffer[-7:]
                    break
                if start:
                    yield buffer[:start]
                buffer, inside = buffer[start + 7:], True
    if buffer and not inside:
        yield buffer


def stream_sentences(token_iter):
    """Yield speakable sentences from a stream of LLM text tokens as they arrive."""
    buffer = ""
    for token in token_iter:
        buffer += token.replace("\n", " ")
        while True:
            m = _SENTENCE_END.match(buffer)
            if not m:
                break
            yield m.group(1).strip()
            buffer = buffer[m.end():]
    if buffer.strip():
        yield buffer.strip()


def sanitize_for_tts(text):
    """Make LLM text speakable: strip markdown, URLs, and emojis."""
    t = re.sub(r"```.*?```", " code omitted. ", text, flags=re.S)
    t = re.sub(r"`([^`]+)`", r"\1", t)
    t = re.sub(r"https?://\S+", " a link ", t)
    t = re.sub(r"[*_#~^|<>]+", " ", t)
    t = re.sub(r"[\U0001F000-\U0001FAFF☀-➿←-⇿⬀-⯿️]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def looks_like_hallucination(text, segments, duration_sec):
    """True when a transcript is probably Whisper inventing words over noise."""
    if not text:
        return True
    if segments:
        no_speech = float(np.mean([s.get("no_speech_prob", 0.0) for s in segments]))
        logprob = float(np.mean([s.get("avg_logprob", 0.0) for s in segments]))
        if no_speech > 0.75 and logprob < -1.0:
            return True
    return duration_sec < 1.5 and text.lower().strip(" .!?") in _JUNK_TRANSCRIPTS


def ollama_tokens(history, model):
    """Stream content tokens from Ollama for the given chat history."""
    import ollama
    for part in ollama.chat(model=model, messages=history, stream=True):
        chunk = part.get("message", {}).get("content", "")
        if chunk:
            yield chunk


# --- voice model switching --------------------------------------------------
_MODEL_QUERY_RE = re.compile(r"\b(what|which|current)\b.*\bmodel\b", re.I)
# Explicit commands: if the name isn't installed we say so (don't fall through).
_SWITCH_EXPLICIT = [
    re.compile(r"(?:switch|change|set)\s+(?:the\s+)?(?:ai\s+|llm\s+)?model\s+to\s+(.+)", re.I),
    re.compile(r"(?:switch|change)\s+to\s+(.+)", re.I),
]
# Loose keywords: only act if the name resolves, else treat as normal speech.
_SWITCH_LOOSE = [
    re.compile(r"\buse\s+(?:the\s+)?(?:model\s+)?(.+)", re.I),
    re.compile(r"^\s*model\s+(.+)", re.I),
]


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def available_models():
    """Names of all locally-installed Ollama models."""
    import ollama
    try:
        data = ollama.list()
    except Exception:
        return []
    models = data.get("models", []) if isinstance(data, dict) else getattr(data, "models", [])
    names = []
    for m in models:
        name = (m.get("model") or m.get("name")) if isinstance(m, dict) \
            else (getattr(m, "model", None) or getattr(m, "name", None))
        # Embedding-only models can't chat; keep them out of the chat menu.
        if name and not any(t in name.lower() for t in ("embed", "bge-", "minilm")):
            names.append(name)
    return names


# What each model family is best at — shown in the model menus.
_MODEL_BEST_FOR = [
    ("qwen3", "smart + best with tools/web"),
    ("llama3.2", "fastest replies"),
    ("gemma3", "vision + balanced chat"),
    ("deepseek-r1", "deep reasoning, slower"),
    ("qwen2.5-coder", "coding"),
    ("phi4-mini", "math & logic"),
    ("glm", "cloud model"),
    ("mistral", "general chat"),
]


def model_label(name):
    """'qwen3:4b (smart + best with tools/web)'."""
    low = name.lower()
    for family, desc in _MODEL_BEST_FOR:
        if family in low:
            return f"{name} ({desc})"
    return name


# --- practice modes: role-play coach for real-life situations -----------------
PRACTICE_MODES = {
    "exam": (
        "You are a strict but encouraging oral-exam examiner (IELTS/TOEFL speaking "
        "style). Ask ONE question at a time on academic topics. After each answer, "
        "give a 1-2 sentence evaluation (fluency, vocabulary, clarity), then the next "
        "question. Keep questions increasingly challenging. Begin now by greeting the "
        "candidate and asking the first question."),
    "embassy": (
        "You are a consular officer at the {country} embassy conducting a visa "
        "interview (student/visitor/work visa, as appropriate). Be professional, "
        "brisk, slightly skeptical — like a real officer. Use the REAL visa "
        "process, terminology and interview style of {country} (for example: F-1 "
        "visa and I-20 for the USA; Genuine Temporary Entrant for Australia; "
        "Student Route and CAS for the UK; Schengen rules for EU countries; COE "
        "for Japan; study permit and SOP for Canada). Ask ONE question at a time: "
        "purpose of travel, ties to home country, finances, plans. Probe vague "
        "answers. If the applicant asks about documents or the process, answer "
        "briefly and accurately with {country}-specific documents and terms, then "
        "continue the interview. After 6-8 questions, give a verdict-style summary with honest "
        "feedback on which answers were weak and how to improve them. Begin now: "
        "greet briefly and ask the first question."),
    "work": (
        "You are a workplace conversation partner helping practice professional "
        "English: meetings, small talk with colleagues, disagreeing politely, asking "
        "the boss for leave, giving status updates. Play the other person in a "
        "realistic scene, one exchange at a time, and occasionally coach with a "
        "better phrasing in brackets. Begin now by proposing a scene and starting it."),
    "translator": (
        "You are a precise live interpreter between English and Nepali. For every "
        "user message: if it is in Nepali (Devanagari or romanized Nepali), reply "
        "with ONLY the English translation. If it is in English, reply with ONLY "
        "the Nepali translation written in Devanagari script. No explanations, no "
        "notes, no romanization — just the translation itself. Begin by saying, in "
        "one short sentence, that translator mode is on."),
    "interview": (
        "You are a hiring manager conducting a realistic job interview. Ask ONE "
        "question at a time — mix of background, behavioral ('tell me about a time…') "
        "and role-specific questions. Follow up on weak or vague answers like a real "
        "interviewer. After 6-8 questions, give honest, specific feedback and a "
        "hire/no-hire style verdict with improvement tips. If the user hasn't said "
        "what job they're interviewing for, ask that first. Begin now."),
}


# --- learning mode: tutor + quiz with progress memory -------------------------
LEARN_SUBJECTS = {
    "programming": ("programming (Python-first, then other languages as asked)",
                    ("qwen2.5-coder", "qwen3:8b", "qwen3")),
    "python": ("Python programming", ("qwen2.5-coder", "qwen3:8b", "qwen3")),
    "math": ("mathematics (algebra, calculus, discrete math, statistics)",
             ("qwen2.5-math", "qwen3:8b", "qwen3")),
    "cs": ("computer science (data structures, algorithms, OS, networks, databases)",
           ("qwen3:8b", "qwen3", "qwen2.5-coder")),
    "physics": ("physics (mechanics, electricity, waves, modern physics)",
                ("qwen3:8b", "qwen2.5-math", "qwen3")),
    "cybersec": ("cybersecurity (networking, web security, cryptography, defense — "
                 "theory and ethical, defensive practice only)",
                 ("qwen3:8b", "qwen3", "qwen2.5-coder")),
    "ai": ("artificial intelligence and machine learning (concepts, math, and code)",
           ("qwen3:8b", "qwen3", "qwen2.5-coder")),
}


def _pick_learn_model(prefs):
    models = available_models()
    for p in prefs:
        for m in models:
            if p in m.lower():
                return m
    return None


def tutor_prompt(subject_key, quiz=False):
    import tools as T
    desc, _ = LEARN_SUBJECTS[subject_key]
    progress = T.learning_summary(subject_key)
    progress_block = (f"\nThe student's saved progress:\n{progress}\n"
                      "Resume from there: briefly recap, review weak spots first."
                      if progress else
                      "\nThis is the student's first session — start by asking what "
                      "they already know to gauge their level.")
    if quiz:
        return (
            f"You are a quiz master for {desc}. Ask ONE question at a time, numbered. "
            "Mix difficulty. After each answer: say correct/incorrect, give a one-"
            "sentence explanation, keep a running score. After 8 questions give the "
            "final score, what to review, and call save_progress with the weak spots. "
            "Use the calculate tool to verify any math. Begin with question 1 now."
            + progress_block)
    return (
        f"You are an excellent personal tutor for {desc}. Teach interactively: "
        "explain ONE concept at a time, clearly and concretely with a small example, "
        "then check understanding with a short question before moving on. Adapt to "
        "the student's answers — slow down on confusion, speed up on mastery. "
        "Use run_python to demonstrate code and to test solutions the student "
        "writes; use calculate to verify all math (never trust mental arithmetic). "
        "At natural checkpoints (every few concepts, and whenever the session seems "
        "to be ending) call save_progress with subject, covered topics, weak spots, "
        "and the next topic. Keep responses focused — no walls of text."
        + progress_block)


# Vocabulary hints fed to Whisper during practice so names/terms transcribe better.
PRACTICE_STT_BIAS = {
    "exam": "An oral English exam. Academic topics, education, opinions, examples.",
    "embassy": ("An embassy visa interview. University names, master's degree, "
                "cybersecurity, computer science, tuition, sponsor, bank balance, "
                "scholarship, I-20, SEVIS, CAS, GTE, COE, study permit, ties to "
                "Nepal, return plans."),
    "work": "A professional workplace conversation: meetings, deadlines, projects, leave.",
    "interview": ("A job interview: experience, skills, projects, teamwork, salary, "
                  "strengths and weaknesses."),
}


def practice_prompt(mode, country=None):
    base = PRACTICE_MODES[mode]
    if "{country}" in base:
        base = base.replace("{country}", country or "United States")
    return (base + " Keep every turn SHORT — one to three sentences, spoken-friendly: "
            "no markdown, no lists, no bold, just natural speech. This is a VOICE "
            "conversation transcribed by speech recognition, so names and words may "
            "come through slightly garbled. NEVER mention typos, spelling, writing, or "
            "transcription. If a name or detail sounds unclear or improbable, do what "
            "a real person would: ask them to repeat or confirm it, in character "
            "('Could you repeat the name of the university?'). Stay in character until "
            "the user says 'stop practice'.")


# Model families that support Ollama tool-calling (needed for agent mode).
_TOOL_MODELS = ("qwen3", "qwen2.5", "llama3.1", "llama3.2", "llama4", "mistral",
                "phi4", "command-r", "hermes", "firefunction", "granite")


def supports_tools(name):
    return any(h in (name or "").lower() for h in _TOOL_MODELS)


def pick_tool_model(names, prefer=("qwen3", "llama3.2", "phi4")):
    """Choose the best installed tool-capable model, or None."""
    for p in prefer:
        for n in names:
            if p in n.lower():
                return n
    for n in names:
        if supports_tools(n):
            return n
    return None


def resolve_model(requested, names):
    """Best-match a spoken model name (e.g. 'deepseek') to an installed one."""
    req = _norm(requested)
    if not req:
        return None
    for n in names:                               # exact
        if _norm(n) == req:
            return n
    for n in names:                               # family prefix (before the ':tag')
        fam = _norm(n.split(":")[0])
        if fam.startswith(req) or req.startswith(fam) or req in fam:
            return n
    for n in names:                               # substring anywhere
        if req in _norm(n):
            return n
    return None


def parse_model_command(text):
    """Return ('query', None), ('switch', name, explicit_bool), or None."""
    t = text.strip().strip(".!?").strip()
    if _MODEL_QUERY_RE.search(t):
        return ("query", None, False)
    for explicit, patterns in ((True, _SWITCH_EXPLICIT), (False, _SWITCH_LOOSE)):
        for pat in patterns:
            m = pat.search(t)
            if m:
                req = re.sub(r"\b(please|model|the|now|instead)\b", " ", m.group(1),
                             flags=re.I).strip(" .!?")
                if req:
                    return ("switch", req, explicit)
    return None


def respond(history, speaker, engine, vad, model, barge_in=True):
    """
    Stream the LLM reply, speak it sentence-by-sentence, watch for barge-in.
    Returns (text_for_history, barge_in_audio_frames_or_None).
    """
    interrupt = threading.Event()
    monitor = voice.BargeInMonitor(engine, vad, interrupt, speaker) if barge_in else None

    speaker.begin(interrupt)
    if monitor:
        monitor.start()

    fed, printed_prefix = [], False
    tokens = ollama_tokens(history, model)
    try:
        for sentence in stream_sentences(strip_think(tokens)):
            if interrupt.is_set():
                break
            if not printed_prefix:
                print("Assistant:", end=" ", flush=True)
                printed_prefix = True
            print(sentence, end=" ", flush=True)
            speakable = sanitize_for_tts(sentence)
            if speakable:
                fed.append(sentence)
                speaker.feed(speakable)
        if printed_prefix:
            print()
    finally:
        tokens.close()                 # stop Ollama generating an abandoned reply
        speaker.end()                  # drains fast once interrupted
        if monitor:
            monitor.stop()

    if interrupt.is_set():
        print("🖐️  (interrupted — go ahead)")
        # History gets only what the user actually heard, plus a marker so the
        # model knows its reply was cut off.
        heard = " ".join(fed[:len(speaker.spoken)]).strip()
        record = (heard + " " if heard else "") + "[reply interrupted by the user]"
        return record, monitor.handoff if monitor else None
    return " ".join(fed).strip(), None


def push_prompt(current_model, practice=None):
    """Menu shown before each push-to-talk turn. Returns 'talk', 'quit', 'menu',
    'model:<name>', 'practice:<mode>' or 'practice:off'."""
    models = available_models()
    if practice:
        # Mid-practice: stay in the conversation — no big menu between turns.
        try:
            choice = input("🎭 [Enter] answer   [p] change/stop practice   "
                           "[q] quit: ").strip().lower()
        except EOFError:
            return "quit"
    else:
        print("\n" + "─" * 60)
        if models:
            for i, m in enumerate(models):
                mark = " ✓" if m == current_model else ""
                print(f"  [{i + 1}] {model_label(m)}{mark}")
        try:
            choice = input(f"▶ [Enter] talk   [1-{len(models)}] model   [p] practice   "
                           f"[d] dictate   [q] quit: ").strip().lower()
        except EOFError:
            return "quit"
    if choice == "":
        return "talk"
    if choice == "d":
        return "dictate"
    if choice in ("q", "quit", "exit"):
        return "quit"
    if choice == "p":
        names = list(PRACTICE_MODES)
        print("  Practice sections: " + "   ".join(
            f"[{i + 1}] {n.capitalize()}" for i, n in enumerate(names))
            + "   [0] stop practice")
        try:
            sel = input("  Choose: ").strip()
        except EOFError:
            return "menu"
        if sel == "0":
            return "practice:off"
        if sel.isdigit() and 1 <= int(sel) <= len(names):
            chosen = names[int(sel) - 1]
            if chosen == "embassy":
                try:
                    country = input("  Which country's embassy? (Enter = USA) "
                                    "e.g. USA, UK, Australia, Canada, Japan, "
                                    "Germany, Korea: ").strip()
                except EOFError:
                    country = ""
                return f"practice:embassy:{country or 'United States'}"
            return "practice:" + chosen
        return "menu"
    if choice.isdigit() and 1 <= int(choice) <= len(models):
        return "model:" + models[int(choice) - 1]
    return "menu"


def load_whisper(name):
    """
    Load a Whisper model on the best device that fits.

    On a small GPU shared with the Ollama LLM there is often not enough VRAM for
    both, so we auto-pick CPU when the GPU is low/busy, and fall back to CPU if a
    CUDA load hits out-of-memory. Override with VA_STT_DEVICE=cuda|cpu.
    """
    import whisper
    import torch

    dev = os.environ.get("VA_STT_DEVICE", "auto")
    if dev == "auto":
        dev = "cpu"
        if torch.cuda.is_available():
            try:
                free, _ = torch.cuda.mem_get_info()
                dev = "cuda" if free > 1.3 * 1024 ** 3 else "cpu"  # ~1.3 GB headroom
            except Exception:
                dev = "cuda"
    try:
        model = whisper.load_model(name, device=dev)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("ℹ️  GPU out of memory — loading Whisper on CPU instead.")
        model, dev = whisper.load_model(name, device="cpu"), "cpu"
    if dev == "cpu":
        print("ℹ️  Whisper is on CPU (slower STT, but coexists with the LLM on GPU).")
    return model


# --- conversation persistence ---------------------------------------------------
CHATS_DIR = os.path.expanduser("~/.local/share/voice-assistant/chats")


def save_chat(history):
    """Persist a finished conversation (skips empty ones)."""
    import datetime
    import json
    turns = [m for m in history if m["role"] in ("user", "assistant")]
    if len(turns) < 2:
        return None
    os.makedirs(CHATS_DIR, exist_ok=True)
    path = os.path.join(CHATS_DIR,
                        f"chat-{datetime.datetime.now():%Y%m%d-%H%M%S}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=1)
        return path
    except Exception:
        return None


def load_last_chat():
    """Return (history, path) of the most recent saved chat, or (None, None)."""
    import json
    try:
        files = sorted(f for f in os.listdir(CHATS_DIR) if f.endswith(".json"))
    except FileNotFoundError:
        return None, None
    if not files:
        return None, None
    path = os.path.join(CHATS_DIR, files[-1])
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f), path
    except Exception:
        return None, None


class TextSpeaker:
    """Stand-in 'speaker' for text mode: prints instead of talking, so the agent
    loop and confirmations work without loading any audio stack."""
    def begin(self, interrupt):
        pass

    def feed(self, text):
        pass

    def end(self):
        pass

    def say_blocking(self, text):
        print(f"🔊 {text}")


def respond_text(history, model):
    """Stream a reply to stdout, showing the model's thinking dimmed. Returns the answer."""
    reply, in_think, in_answer = [], False, False
    for kind, text in split_think_stream(ollama_tokens(history, model)):
        if kind == "think":
            if not SHOW_THINK:
                continue
            if not in_think:
                print(f"{DIM}💭 ", end="", flush=True)
                in_think = True
            print(f"{DIM}{text}{RESET}", end="", flush=True)
        else:
            if in_think and not in_answer:
                print(RESET)          # close the thinking line
                in_think = False
            if not in_answer:
                print(f"{GREEN}assistant>{RESET} ", end="", flush=True)
                in_answer = True
            print(text, end="", flush=True)
            reply.append(text)
    print()
    return "".join(reply).strip()


def text_main(agent):
    """Typed chat loop: same LLM, model switching, and (optional) agent tools."""
    current_model = LLM_MODEL
    sys_prompt = SYSTEM_PROMPT
    if agent:
        if not supports_tools(current_model):
            alt = pick_tool_model(available_models())
            if alt:
                print(f"ℹ️  '{current_model}' can't use tools; using '{alt}' for agent mode.")
                current_model = alt
            else:
                print("⚠️  No installed model supports tools. Try: ollama pull qwen3:4b")
        sys_prompt = agent_system_prompt()
    speaker = TextSpeaker()
    if agent:
        import tools as T
        T.set_alert_callback(lambda msg: print(f"\n🔔 {msg}"))

    def confirm_shell(cmd):
        print(f"\n⚠️  Run this shell command?\n    {cmd}")
        try:
            return input("   [y/N]: ").strip().lower() in ("y", "yes")
        except EOFError:
            return False

    print(f"💬 Text chat — LLM: '{current_model}'{', agent tools ON' if agent else ''}.")
    print("Commands: /models, /model <name|number>, /help, /quit. "
          "Or just type your message.\n")

    history = [{"role": "system", "content": sys_prompt}]
    tools_mode = False        # /learn and /review need tools even without --agent
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            save_chat(history)
            print("\nGoodbye!")
            break
        if not text:
            continue
        if text.lower().strip() == "/resume":
            old, path = load_last_chat()
            if not old:
                print("No saved conversations yet.")
                continue
            history = old
            last = [m for m in old if m["role"] == "assistant"]
            print(f"↩️  Resumed {os.path.basename(path)} "
                  f"({len(old)} messages). Last reply:")
            if last:
                print(f"{GREEN}assistant>{RESET} {str(last[-1]['content'])[:300]}")
            continue
        if text.startswith("```"):
            # Multi-line paste: keep reading until a closing ``` line.
            print("(paste mode — end with ``` on its own line)")
            lines = [text[3:]] if len(text) > 3 else []
            while True:
                try:
                    ln = input()
                except (EOFError, KeyboardInterrupt):
                    break
                if ln.strip() == "```":
                    break
                lines.append(ln)
            text = "\n".join(lines).strip()
            if not text:
                continue
        low = text.lower()

        if low in ("/quit", "/exit", "quit", "exit", "goodbye", "bye"):
            saved = save_chat(history)
            if saved:
                print(f"💾 Conversation saved ({os.path.basename(saved)}) — "
                      "/resume next time to continue.")
            print("Goodbye!")
            break
        if low in ("/help", "/h"):
            print("Type a message to chat. /models list · /model <name|number> switch "
                  "· /learn python|math|cs|physics|cybersec|ai tutor mode · /quiz "
                  "<subject> quizzes · /practice exam|embassy|work|interview role-play "
                  "· ``` starts a code paste block · /quit to exit.")
            continue
        if "stop practice" in low:
            history = [{"role": "system", "content": sys_prompt}]
            print("🎭 Practice mode off.")
            continue
        if low in ("/models", "/list"):
            for i, m in enumerate(available_models()):
                print(f"  [{i + 1}] {model_label(m)}{'  ← current' if m == current_model else ''}")
            continue
        if low.startswith("/review"):
            import tools as T
            due = T.flashcards_due()
            if due.startswith("No flashcards"):
                print("📇 " + due)
                continue
            if not supports_tools(current_model):
                alt = pick_tool_model(available_models())
                if alt:
                    current_model = alt
                    print(f"🔀 Using {model_label(alt)} for the review.")
            tools_mode = True
            history = [{"role": "system", "content":
                        "You are running a flashcard review session. The due cards "
                        "are listed below. For ONE card at a time: show its question "
                        "and stop (wait for the user's answer). When they answer, "
                        "call flashcard_answer to get the real answer, judge whether "
                        "they were right, tell them, call flashcard_grade with pass "
                        "or fail, then present the next card. After the last card "
                        "give a short summary.\n\n" + due},
                       {"role": "user", "content": "I'm ready, start the review."}]
            print("📇 Flashcard review — answer each question. /learn off to stop.")
            agent_respond(history, speaker, current_model, confirm_shell)
            continue

        if low.startswith("/learn") or low.startswith("/quiz"):
            quiz = low.startswith("/quiz")
            args = text.split()[1:]
            subj = args[0].lower() if args else ""
            if subj in ("off", "stop"):
                tools_mode = False
                history = [{"role": "system", "content": sys_prompt}]
                print("🎓 Learning mode off.")
                continue
            if subj not in LEARN_SUBJECTS:
                print("Subjects: " + ", ".join(LEARN_SUBJECTS)
                      + f". Use: /{'quiz' if quiz else 'learn'} python")
                continue
            best = _pick_learn_model(LEARN_SUBJECTS[subj][1])
            if best and best != current_model:
                current_model = best
                print(f"🔀 Using {model_label(best)} for this subject.")
            tools_mode = supports_tools(current_model)   # tutor gets run_python etc.
            history = [{"role": "system", "content": tutor_prompt(subj, quiz)},
                       {"role": "user", "content": "Let's begin."}]
            print(f"🎓 {'Quiz' if quiz else 'Tutor'}: {subj} — progress is saved "
                  "automatically. /learn off to stop. Paste code with ``` blocks.")
            if tools_mode:
                agent_respond(history, speaker, current_model, confirm_shell)
            else:
                reply = respond_text(history, current_model)
                history.append({"role": "assistant", "content": reply or "[...]"})
            continue

        if low.startswith("/practice"):
            args = text.split()[1:]
            arg = args[0].lower() if args else ""
            country = " ".join(args[1:]) if len(args) > 1 else None
            if arg in ("off", "stop"):
                history = [{"role": "system", "content": sys_prompt}]
                print("🎭 Practice mode off.")
                continue
            if arg not in PRACTICE_MODES:
                print("Practice sections: " + ", ".join(PRACTICE_MODES)
                      + ". Use: /practice interview, or /practice embassy UK")
                continue
            history = [{"role": "system", "content": practice_prompt(arg, country)},
                       {"role": "user", "content": "Let's begin the practice."}]
            label = f"{country} embassy" if (arg == "embassy" and country) else arg.capitalize()
            print(f"🎭 {label} practice — say 'stop practice' or "
                  "/practice off to end.")
            reply = respond_text(history, current_model)
            history.append({"role": "assistant", "content": reply or "[...]"})
            continue
        if low.startswith("/model"):
            parts = text.split(maxsplit=1)
            arg = parts[1].strip() if len(parts) > 1 else ""
            models = available_models()
            target = (models[int(arg) - 1] if arg.isdigit() and 1 <= int(arg) <= len(models)
                      else resolve_model(arg, models))
            if target:
                current_model = target
                print(f"🔀 Now using {target}")
            else:
                print(f"No model matches '{arg}'. Try /models.")
            continue

        cmd = parse_model_command(text)
        if cmd:
            if cmd[0] == "query":
                print(f"🔀 Current model: {current_model}")
                continue
            target = resolve_model(cmd[1], available_models())
            if target:
                current_model = target
                print(f"🔀 Switched to {target}")
                continue
            if cmd[2]:
                print(f"No model matches '{cmd[1]}'. Try /models.")
                continue

        history.append({"role": "user", "content": text})
        try:
            if agent or tools_mode:
                agent_respond(history, speaker, current_model, confirm_shell)
            else:
                reply = respond_text(history, current_model)
                history.append({"role": "assistant", "content": reply or "[said nothing]"})
        except Exception as exc:                       # never crash the conversation
            history.pop()
            print(f"⚠️  '{current_model}' failed: {exc}")
            alt = pick_tool_model(available_models()) if (agent or tools_mode) else None
            if alt and alt != current_model:
                current_model = alt
                print(f"🔀 Switched to '{alt}' — please repeat your message.")


def dictate(engine, speaker, transcribe_final, model, gain_guard):
    """Voice dictation: speech is appended verbatim to a document; on finish the
    LLM produces a cleaned-up version alongside the raw transcript."""
    import datetime
    import voice as _voice
    default = f"dictation-{datetime.datetime.now():%Y%m%d-%H%M}.md"
    try:
        name = input(f"📄 File name (Enter = ~/Documents/{default}): ").strip()
    except EOFError:
        return
    path = os.path.expanduser(name or f"~/Documents/{default}")
    if not os.path.isabs(path):
        path = os.path.expanduser(f"~/Documents/{path}")
    print("🎙️  Dictating. Speak in chunks; press Enter to end each chunk. "
          "Say 'stop dictation' to finish.\n")
    chunks = []
    while True:
        audio = engine.record_push_to_talk(wait_for_start=False)
        if audio is None or len(audio) < _voice.FRAME_SIZE:
            continue
        gain_guard.check(audio)
        text, _segments = transcribe_final(audio, len(audio) // _voice.FRAME_SIZE)
        if not text:
            print("(heard nothing — keep going)")
            continue
        if "stop dictation" in text.lower():
            break
        chunks.append(text.strip())
        print(f"➕ {text.strip()}\n")
    if not chunks:
        print("Nothing dictated.")
        return
    raw = "\n\n".join(chunks)
    cleaned = raw
    try:
        import ollama
        r = ollama.chat(model=model, messages=[{
            "role": "user",
            "content": "Clean up this dictated text: fix grammar, punctuation and "
                       "obvious speech-recognition slips, keep the meaning and tone, "
                       "output ONLY the cleaned text:\n\n" + raw}])
        cleaned = "".join(strip_think(iter([r["message"]["content"]]))).strip() or raw
    except Exception:
        pass
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(cleaned + "\n\n---\nRaw transcript:\n\n" + raw + "\n")
    print(f"💾 Saved to {path} ({len(chunks)} chunks, cleaned + raw).")
    speaker.say_blocking("Dictation saved.")


def agent_respond(history, speaker, model, confirm_shell, max_steps=6):
    """
    Agentic turn: let the model call tools (open apps, volume, brightness, shell)
    in a loop until it produces a final spoken answer. `history` already ends with
    the user's message; this appends the assistant/tool messages it generates.
    Returns the final answer text.
    """
    import ollama
    import tools as T

    def deliver(content):
        """Show reasoning (dimmed) + answer, and speak the answer."""
        answer, shown_think = "", False
        for kind, text in split_think_stream(iter([content])):
            if kind == "think" and SHOW_THINK and text.strip():
                if not shown_think:
                    print(f"{DIM}💭 {text}{RESET}", end="")
                    shown_think = True
                else:
                    print(f"{DIM}{text}{RESET}", end="")
            elif kind == "answer":
                answer += text
        if shown_think:
            print()
        answer = answer.strip()
        if answer:
            print(f"{GREEN}assistant>{RESET} {answer}")
        speaker.begin(interrupt)
        for sentence in stream_sentences(iter([answer])):
            speakable = sanitize_for_tts(sentence)
            if speakable:
                speaker.feed(speakable)
        speaker.end()
        return answer or "[said nothing]"

    interrupt = threading.Event()
    for _ in range(max_steps):
        try:
            with Spinner("Thinking…"):
                r = ollama.chat(model=model, messages=history, tools=T.TOOLS)
        except Exception as exc:
            if "does not support tools" not in str(exc).lower():
                raise
            print(f"⚠️  '{model}' can't use tools — answering without them. "
                  "Switch to qwen3, llama3.2, or phi4-mini for agent actions.")
            history.append({"role": "assistant", "content": ""})
            return deliver("".join(strip_think(ollama_tokens(history, model))))

        m = r.get("message", {}) or {}
        calls = m.get("tool_calls") or []
        content = m.get("content") or ""

        if not calls:                                  # final answer
            history.append({"role": "assistant", "content": content})
            return deliver(content)

        norm = [{"function": {"name": c["function"]["name"],
                              "arguments": c["function"].get("arguments", {})}}
                for c in calls]
        history.append({"role": "assistant", "content": content, "tool_calls": norm})
        for c in norm:
            fn, args = c["function"]["name"], c["function"]["arguments"]
            label = activity_label(fn, args)
            if fn == "run_shell":                       # needs interactive confirm; no spinner
                print(f"{CYAN}▶{RESET} {label}")
                result = T.dispatch(fn, args, confirm=confirm_shell)
            else:
                with Spinner(label):
                    result = T.dispatch(fn, args, confirm=confirm_shell)
            summary = str(result).replace("\n", " ")
            if len(summary) > 100:
                summary = summary[:100] + "…"
            print(f"{CYAN}✓{RESET} {label}\n  {DIM}{summary}{RESET}")
            history.append({"role": "tool", "content": str(result), "tool_name": fn})

    speaker.say_blocking("Sorry, I couldn't finish that.")
    return "[agent stopped: too many tool steps]"


def main():
    if "--list-devices" in sys.argv:
        import sounddevice as sd
        print(sd.query_devices())
        return

    mode = "push" if "--push" in sys.argv else MODE
    if "--wake" in sys.argv or os.environ.get("VA_WAKE", "0") == "1":
        mode = "wake"                   # hands-free via wake word ("hey jarvis")
    if "--male" in sys.argv:
        os.environ["VA_VOICE_GENDER"] = "male"
    elif "--female" in sys.argv:
        os.environ["VA_VOICE_GENDER"] = "female"
    live_stt = LIVE_STT and "--no-live" not in sys.argv and mode in ("auto", "wake")
    barge_in = BARGE_IN and "--no-barge-in" not in sys.argv
    agent = "--agent" in sys.argv or os.environ.get("VA_AGENT", "0") == "1"

    if "--text" in sys.argv or os.environ.get("VA_TEXT", "0") == "1":
        text_main(agent)
        return
    use_aec = "--aec" in sys.argv or os.environ.get("VA_AEC", "0") == "1"
    # fp16 is off by default: GTX 16-series GPUs produce NaN logits in half precision.
    # The model still runs on GPU in fp32. Set VA_FP16=1 on GPUs that support it.
    fp16 = os.environ.get("VA_FP16", "0") == "1"
    lang = None if LANG.lower() in ("", "auto") else LANG

    aec = voice.EchoCancel()
    if use_aec:
        aec.setup()                    # route audio through system echo cancellation

    print(f"Loading Whisper model '{WHISPER_MODEL}'...")
    model = load_whisper(WHISPER_MODEL)

    live = None
    if live_stt:
        live_model = model if LIVE_MODEL == WHISPER_MODEL else load_whisper(LIVE_MODEL)
        live = voice.LiveTranscriber(live_model=live_model, final_model=model,
                                     language=lang, fp16=fp16, on_partial=print_partial)
        print(f"Live STT: '{LIVE_MODEL}' → final '{WHISPER_MODEL}'"
              f"{' (GPU/fp16)' if fp16 else ''}.")

    vad = voice.VAD()
    if vad.kind == "silero":
        print("👂 VAD: Silero (neural).")
    speaker = voice.Speaker()
    if agent:
        import tools as T

        def _alert(msg):
            print(f"\n🔔 {msg}")
            try:
                speaker.say_blocking(msg)
            except Exception:
                pass
        T.set_alert_callback(_alert)
    engine = voice.AudioEngine()
    gain_guard = voice.InputGain()
    print(f"🎤 Input: {engine.input_name}")

    stt_context = {"prompt": None, "lang": lang}   # per-mode Whisper hints

    def transcribe_final(audio, voiced):
        if live:
            return live.finalize(audio, voiced)
        print("Transcribing...")
        result = model.transcribe(audio, fp16=fp16, language=stt_context["lang"],
                                  condition_on_previous_text=False,
                                  initial_prompt=stt_context["prompt"])
        return result["text"].strip(), result.get("segments", [])

    current_model = LLM_MODEL
    if agent and not supports_tools(current_model):
        alt = pick_tool_model(available_models())
        if alt:
            print(f"ℹ️  '{current_model}' can't use tools; using '{alt}' for agent mode.")
            current_model = alt
        else:
            print("⚠️  No installed model supports tools. Try: ollama pull qwen3:4b")
    mode_label = {"auto": "hands-free", "wake": "wake word", "push": "push-to-talk"}[mode]
    print(f"LLM: '{current_model}'. Mode: {mode_label}"
          f"{', live STT' if live else ''}{', barge-in' if barge_in else ''}"
          f"{', AEC' if use_aec else ''}. Say 'quit' to exit.")
    if mode == "push":
        print("🔀 Switch the AI model with the number keys in the menu below, "
              "or say 'switch to <name>'.\n")
    else:
        print("🔀 Say 'switch to <name>' or 'use <name>' to change the AI model "
              "(e.g. 'switch to deepseek').\n")
    if barge_in and not use_aec:
        print("💡 Tip: with loudspeakers, run with --aec so the assistant never "
              "mistakes its own voice for yours.\n")

    sys_prompt = SYSTEM_PROMPT
    if agent:
        sys_prompt = agent_system_prompt()
        print("🤖 Agent mode ON — I can open apps and control the system. "
              "Shell commands ask for confirmation.\n")

    def confirm_shell(cmd):
        print(f"\n⚠️  The assistant wants to run a shell command:\n    {cmd}")
        speaker.say_blocking("Do you want me to run that command? Type y to confirm.")
        try:
            return input("   Run it? [y/N]: ").strip().lower() in ("y", "yes")
        except EOFError:
            return False

    wake = None
    if mode == "wake":
        try:
            wake = voice.WakeWordListener()
            phrase = wake.phrase.replace("_", " ")
            print(f"🟣 Say '{phrase}' to wake me.\n")
        except Exception as exc:
            print(f"⚠️  Wake word unavailable ({exc}); falling back to hands-free VAD.")
            mode = "auto"

    history = [{"role": "system", "content": sys_prompt}]
    handoff = None
    practice = None
    try:
        while True:
            if mode == "wake" and handoff is None:
                if not wake.wait(engine):
                    continue
                print("🟢 Yes?")
                speaker.say_blocking("Yes?")
                engine.flush()          # don't transcribe the "Yes?" echo
            if mode == "push" and practice:
                # Interview flow: the mic opens right after the coach finishes —
                # just speak your answer, then press Enter to send it.
                print("🎙️  Answer now — press Enter when you finish speaking. "
                      "(say 'stop practice' to end)")
                audio = engine.record_push_to_talk(wait_for_start=False)
                voiced = len(audio) // voice.FRAME_SIZE if audio is not None else 0
            elif mode == "push":
                action = push_prompt(current_model, practice)
                if action == "quit":
                    speaker.say_blocking("Goodbye!")
                    print("\nGoodbye!")
                    break
                if action == "menu":
                    continue
                if action.startswith("model:"):
                    current_model = action.split(":", 1)[1]
                    short = current_model.split(":")[0].replace("-", " ")
                    speaker.say_blocking(f"Okay, now using {short}.")
                    print(f"🔀 Switched LLM to {current_model}\n")
                    continue
                if action == "dictate":
                    dictate(engine, speaker, transcribe_final, current_model, gain_guard)
                    continue
                if action.startswith("practice:"):
                    parts = action.split(":", 2)
                    sel = parts[1]
                    country = parts[2] if len(parts) > 2 else None
                    if sel == "off":
                        practice = None
                        stt_context["prompt"] = None
                        stt_context["lang"] = lang
                        history = [{"role": "system", "content": sys_prompt}]
                        print("🎭 Practice mode off.\n")
                        continue
                    practice = sel
                    stt_context["prompt"] = PRACTICE_STT_BIAS.get(sel)
                    # Translator must hear both languages → Whisper auto-detect.
                    stt_context["lang"] = None if sel == "translator" else lang
                    history = [{"role": "system", "content": practice_prompt(sel, country)},
                               {"role": "user", "content": "Let's begin the practice."}]
                    label = f"{country} embassy" if country else sel.capitalize()
                    print(f"🎭 {label} practice — the coach starts. "
                          "Say 'stop practice' to end.\n")
                    reply, _ = respond(history, speaker, engine, vad, current_model, False)
                    history.append({"role": "assistant", "content": reply or "[...]"})
                    continue
                audio = engine.record_push_to_talk(wait_for_start=False)
                voiced = len(audio) // voice.FRAME_SIZE if audio is not None else 0
            else:
                if handoff is None:
                    print("🎧 Listening... (just start talking)")
                audio, voiced = engine.capture_utterance(vad, listener=live, prefill=handoff)
                handoff = None
            if audio is None or voiced == 0:
                continue
            gain_guard.check(audio)

            text, segments = transcribe_final(audio, voiced)
            if looks_like_hallucination(text, segments, len(audio) / voice.SAMPLE_RATE):
                print("(heard nothing)\n")
                continue
            print("You said:", text)

            if text.lower().strip(" .!?") in _EXIT_WORDS:
                speaker.say_blocking("Goodbye!")
                print("\nGoodbye!")
                break

            if practice and "stop practice" in text.lower():
                practice = None
                stt_context["prompt"] = None
                stt_context["lang"] = lang
                history = [{"role": "system", "content": sys_prompt}]
                speaker.say_blocking("Practice ended. Nice work!")
                print("🎭 Practice mode off.\n")
                continue

            cmd = parse_model_command(text)
            if cmd:
                kind = cmd[0]
                if kind == "query":
                    short = current_model.split(":")[0].replace("-", " ")
                    speaker.say_blocking(f"I'm currently using {short}.")
                    print(f"🔀 Current model: {current_model}\n")
                    continue
                target = resolve_model(cmd[1], available_models())
                if target:
                    current_model = target
                    short = target.split(":")[0].replace("-", " ")
                    speaker.say_blocking(f"Okay, now using {short}.")
                    print(f"🔀 Switched LLM to {target}\n")
                    continue
                if cmd[2]:                       # explicit command, unknown model
                    speaker.say_blocking("Sorry, I don't have that model installed.")
                    print(f"🔀 No installed model matches '{cmd[1]}'.\n")
                    continue
                # loose keyword that didn't resolve → treat as normal conversation

            history.append({"role": "user", "content": text})
            try:
                if agent and not practice:
                    agent_respond(history, speaker, current_model, confirm_shell)
                    handoff = None
                else:
                    # Practice is pure conversation: streaming reply, no tools.
                    reply, handoff = respond(history, speaker, engine, vad, current_model, barge_in)
                    history.append({"role": "assistant", "content": reply or "[said nothing]"})
            except Exception as exc:                   # never crash the conversation
                history.pop()
                handoff = None
                print(f"⚠️  '{current_model}' failed: {exc}")
                alt = pick_tool_model(available_models()) if agent else None
                if alt and alt != current_model:
                    current_model = alt
                    msg = f"Switched to {alt.split(':')[0]}. Please repeat that."
                    print(f"🔀 {msg}")
                    speaker.say_blocking(msg)
            print()
    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        save_chat(history)
        if live:
            live.close()
        engine.close()
        speaker.close()
        aec.restore()


if __name__ == "__main__":
    main()
