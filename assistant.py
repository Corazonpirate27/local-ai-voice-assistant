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
    python assistant.py                # AI Agent REPL (default): text + /voice, /agent,
                                       #   /chat, /course — everything via / commands
    python assistant.py --push         # classic voice pipeline, push-to-talk
    python assistant.py --agent        # let it open apps, control system, read documents
    python assistant.py --auto         # hands-free VAD listening (may catch mic noise)
    python assistant.py --wake         # hands-free via wake word: say "hey jarvis"
    VA_BARGE_IN=1 python assistant.py  # allow interrupting the reply by talking
    python assistant.py --aec          # + echo cancellation (best with speakers)
    python assistant.py --no-live      # transcribe after you finish, not live
    python assistant.py --list-devices # show audio devices for VA_*_DEVICE
    python assistant.py --doctor       # self-check dependencies and print fixes

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

voice = None                     # heavy (scipy/sounddevice); loaded on demand


def _load_voice():
    global voice
    if voice is None:
        import voice as _v
        voice = _v
    return voice

# --- terminal UI helpers (Claude-Code-style) ----------------------------------
_TTY = sys.stdout.isatty()
DIM = "\x1b[2m" if _TTY else ""
CYAN = "\x1b[36m" if _TTY else ""
GREEN = "\x1b[32m" if _TTY else ""
ORANGE = "\x1b[38;5;209m" if _TTY else ""      # Claude's coral accent
BOLD = "\x1b[1m" if _TTY else ""
RESET = "\x1b[0m" if _TTY else ""
SHOW_THINK = os.environ.get("VA_SHOW_THINK", "1") != "0"

# Whimsical progress verbs, in the Claude Code tradition.
_WHIMSY = ["Pondering", "Mulling", "Noodling", "Brewing", "Scheming", "Computing",
           "Percolating", "Ruminating", "Conjuring", "Cogitating", "Simmering",
           "Untangling", "Deliberating", "Marinating", "Crunching", "Divining"]


def _term_width():
    return shutil.get_terminal_size((80, 20)).columns


class Spinner:
    """'✻ Pondering… (3s)' status line in the coral accent. No-op when piped."""

    def __init__(self, label=None):
        import random
        self.label = label or (random.choice(_WHIMSY) + "…")
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if not _TTY:
            print(f"✻ {self.label}")
            return self

        def run():
            t0 = time.monotonic()
            phase = 0
            glyphs = "✻✼✻✽"
            while not self._stop.is_set():
                elapsed = int(time.monotonic() - t0)
                suffix = f" {DIM}({elapsed}s){RESET}" if elapsed >= 2 else ""
                visible = f"{self.label}"
                width = _term_width()
                if len(visible) > width - 12:
                    visible = visible[:width - 13] + "…"
                sys.stdout.write(f"\r\x1b[K{ORANGE}{glyphs[phase % len(glyphs)]}{RESET} "
                                 f"{visible}{suffix}")
                sys.stdout.flush()
                phase += 1
                time.sleep(0.25)
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.4)
        if _TTY:
            sys.stdout.write("\r\x1b[K")
            sys.stdout.flush()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


SLASH_COMMANDS = ["/agent", "/voice", "/text", "/chat", "/course", "/course start",
                  "/course syllabus", "/course done", "/learn", "/quiz", "/review",
                  "/practice", "/models", "/model", "/resume", "/history", "/memory",
                  "/file", "/help", "/quit"]
_INPUT_HIST_FILE = os.path.expanduser("~/.local/share/voice-assistant/input_history")


def _cmd_desc(cmd):
    for c, d in HELP_ROWS:
        if d and (c == cmd or c.split()[0] == cmd):
            return d
    return ""


def _read_key(fd):
    """Read one key (handles escape sequences and UTF-8 multibyte)."""
    b = os.read(fd, 1)
    if not b:
        return "EOF"
    if b == b"\x1b":                                    # escape sequence
        seq = os.read(fd, 1)
        if seq == b"[":
            code = os.read(fd, 1)
            return {b"A": "UP", b"B": "DOWN", b"C": "RIGHT", b"D": "LEFT"}.get(code, "")
        return "ESC"
    n = 0
    if b[0] >= 0xF0:
        n = 3
    elif b[0] >= 0xE0:
        n = 2
    elif b[0] >= 0xC0:
        n = 1
    for _ in range(n):
        b += os.read(fd, 1)
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return ""


def claude_input():
    """Claude-Code-style input box. Typing '/' pops a live dropdown of commands:
    ↑/↓ select, Tab fills, Enter runs. ↑ recalls input history otherwise."""
    if not _TTY:
        return input("> ")
    try:
        import termios
        import tty
    except ImportError:
        return input("> ")

    width = min(_term_width() - 1, 100)
    print(f"{DIM}╭{'─' * (width - 2)}╮{RESET}")
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    buf, sel, shown = "", 0, 0
    try:
        with open(_INPUT_HIST_FILE, encoding="utf-8") as f:
            hist = [ln.rstrip("\n") for ln in f if ln.strip()][-200:]
    except OSError:
        hist = []
    hidx = len(hist)

    def suggestions():
        if not buf.startswith("/") or " " in buf:
            return []
        return [c for c in SLASH_COMMANDS if c.startswith(buf)][:8]

    def render(sugg):
        nonlocal shown
        for i, c in enumerate(sugg):
            mark = f"{ORANGE}▸{RESET}" if i == sel else " "
            desc = _cmd_desc(c)[:max(10, width - 26)]
            sys.stdout.write(f"\r\n\x1b[K  {mark} {BOLD}{c:<18}{RESET}{DIM}{desc}{RESET}")
        for _ in range(max(0, shown - len(sugg))):        # clear leftovers
            sys.stdout.write("\r\n\x1b[K")
        lines_below = max(shown, len(sugg))
        if lines_below:
            sys.stdout.write(f"\x1b[{lines_below}A")
        shown = len(sugg)
        sys.stdout.write(f"\r\x1b[K{DIM}│{RESET} > {buf}")   # cursor at end of input
        sys.stdout.flush()

    try:
        tty.setcbreak(fd)
        render(suggestions())
        while True:
            key = _read_key(fd)
            sugg = suggestions()
            if key in ("\r", "\n"):
                if sugg and buf != sugg[sel]:
                    buf = sugg[sel]                       # Enter runs the selection
                break
            if key == "EOF" or key == "\x04":
                if not buf:
                    raise EOFError
                break
            if key == "\x03":
                raise KeyboardInterrupt
            if key == "\x7f":                            # backspace
                buf = buf[:-1]
            elif key == "\t" or (key == "RIGHT" and sugg):
                if sugg:
                    buf = sugg[sel] + (" " if sugg[sel] in
                                       ("/model", "/file", "/learn", "/quiz",
                                        "/practice", "/course start") else "")
            elif key == "UP":
                if sugg:
                    sel = (sel - 1) % len(sugg)
                elif hist:
                    hidx = max(0, hidx - 1)
                    buf = hist[hidx]
            elif key == "DOWN":
                if sugg:
                    sel = (sel + 1) % len(sugg)
                elif hist:
                    hidx = min(len(hist), hidx + 1)
                    buf = hist[hidx] if hidx < len(hist) else ""
            elif key == "ESC":
                buf = ""
            elif key and key.isprintable():
                buf += key
                sel = 0
            render(suggestions())
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write("\r\n")
        for _ in range(shown):                            # wipe the dropdown
            sys.stdout.write("\x1b[K\r\n")
        if shown:
            sys.stdout.write(f"\x1b[{shown}A")
        sys.stdout.write(f"\x1b[K{DIM}╰{'─' * (width - 2)}╯{RESET}\n")
        sys.stdout.flush()
    if buf.strip():
        try:
            os.makedirs(os.path.dirname(_INPUT_HIST_FILE), exist_ok=True)
            with open(_INPUT_HIST_FILE, "a", encoding="utf-8") as f:
                f.write(buf.strip() + "\n")
        except OSError:
            pass
    return buf


def with_memory(base_prompt):
    """Append what the assistant knows about the user — memory works in EVERY
    mode, not just agent mode."""
    import tools as T
    mem = T.memory_summary()
    if not mem:
        return base_prompt
    return (base_prompt + "\n\nThings you remember about this user from earlier "
            "sessions (use them naturally, don't recite them):\n" + mem)


AUTO_MEMORY = os.environ.get("VA_AUTO_MEMORY", "1") != "0"


def auto_memorize(history, model):
    """Quietly learn from a finished conversation: extract up to 3 lasting facts
    about the user and store them. Returns the list of new facts."""
    import ollama
    import tools as T
    user_msgs = [m["content"] for m in history
                 if m.get("role") == "user" and not str(m["content"]).startswith("[")]
    if len(user_msgs) < 3:
        return []
    convo = "\n".join(f"- {str(m)[:300]}" for m in user_msgs[-25:])
    known = T.memory_summary()
    try:
        r = ollama.chat(model=model, messages=[{"role": "user", "content":
            "From these messages a user wrote to their assistant, extract at most 3 "
            "LASTING facts about the user worth remembering for future sessions "
            "(identity, preferences, goals, projects, important life details). "
            "Ignore one-off requests, questions, and anything already known.\n"
            + (f"Already known:\n{known}\n" if known else "")
            + f"\nMessages:\n{convo}\n\n"
            "Reply with ONLY the facts, one per line starting with '- ', "
            "or exactly NONE if nothing qualifies."}])
        out = "".join(strip_think(iter([r["message"]["content"]])))
    except Exception:
        return []
    learned = []
    for line in out.splitlines():
        line = line.strip().lstrip("-• ").strip()
        if not line or line.upper() == "NONE" or len(line) < 8 or len(line) > 200:
            continue
        if "already" in line.lower():
            continue
        if "I'll remember" in T.remember_fact(line):
            learned.append(line)
        if len(learned) >= 3:
            break
    return learned


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
        "Tools available: full file management in the home folder (write_file, "
        "move_file, delete_file, list_folder — overwrites and deletions always ask "
        "the user first; key/credential folders are off-limits), "
        "open apps, set volume and brightness, run shell commands, search "
        "the web, find/search/read the user's local documents, get weather, get exact "
        "times in any timezone (ALWAYS use get_time for time/timezone questions — never "
        "compute offsets yourself), set/list/cancel reminders, control music playback, "
        "look at the screen or image files, remember/forget facts about the user, manage "
        "named lists (shopping/todo/notes), give a daily_briefing (use it when the user "
        "says 'good morning' or asks about their day), answer questions about long "
        "documents with ask_document, show desktop notifications, read the clipboard, "
        "OCR text out of images (read_text_from_image), report battery/disk/memory "
        "(system_status), scan overall OS health for problems (system_health_scan), "
        "install or switch your own AI model when asked (install_model / switch_model), "
        "open web pages (open_url) and play music or video (play_media on youtube or "
        "spotify), run Python code, verify math with calculate, and manage "
        "spaced-repetition flashcards, and search past conversations with search_chats when the user references an earlier discussion (when the user asks to review flashcards: list "
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
        "create_project_file": f"📁 Scaffolding: {a.get('path', '')}",
        "review_project": f"🔍 Reviewing project: {a.get('folder', '') or '(list)'}",
        "run_project": f"▶️  Running in {a.get('folder', '')}: {a.get('command', '')}",
        "calculate": f"🧮 Computing: {a.get('expression', '')}",
        "save_progress": f"🎓 Saving progress: {a.get('subject', '')}",
        "read_text_from_image": f"👁️  Reading text from: {a.get('path', '')}",
        "get_clipboard": "📋 Reading clipboard",
        "write_file": f"✏️  Writing: {a.get('path', '')}",
        "delete_file": f"🗑️  Deleting: {a.get('path', '')}",
        "move_file": f"📦 {'Copying' if a.get('copy') else 'Moving'}: {a.get('src', '')} → {a.get('dst', '')}",
        "list_folder": f"📂 Listing: {a.get('path', '')}",
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
    for kind, text in split_think_stream(token_iter):
        if kind == "answer":
            yield text


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


_MODELS_CACHE = {"at": 0.0, "names": []}


def available_models(max_age=5.0):
    """Names of all locally-installed Ollama models (cached briefly — the menu
    renders every turn and each call is an HTTP round-trip)."""
    import ollama
    now = time.monotonic()
    if now - _MODELS_CACHE["at"] < max_age:
        return list(_MODELS_CACHE["names"])
    try:
        data = ollama.list()
    except Exception:
        return list(_MODELS_CACHE["names"])
    models = data.get("models", []) if isinstance(data, dict) else getattr(data, "models", [])
    names = []
    for m in models:
        name = (m.get("model") or m.get("name")) if isinstance(m, dict) \
            else (getattr(m, "model", None) or getattr(m, "name", None))
        # Embedding-only models can't chat; keep them out of the chat menu.
        if name and not any(t in name.lower() for t in ("embed", "bge-", "minilm")):
            names.append(name)
    _MODELS_CACHE["at"], _MODELS_CACHE["names"] = time.monotonic(), names
    return list(names)


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
    "english": ("the English language — grammar, vocabulary, conversation and "
                "writing, with awareness of common mistakes made by Nepali speakers",
                ("qwen3:8b", "qwen3", "gemma3")),
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


def course_tutor_prompt(course_key):
    """System prompt for a structured course session (CS50-style, project-based)."""
    import courses
    import tools as T
    brief = courses.module_brief(course_key)
    progress = T.learning_summary(course_key)
    progress_block = (f"\nSaved study notes:\n{progress}\n" if progress else "")
    return (
        "You are a world-class course instructor, teaching in the spirit of CS50: "
        "rigorous but warm, building intuition before formalism, always hands-on.\n"
        + brief + progress_block +
        "\nRun the module as: (1) teach the topics ONE concept at a time with a "
        "concrete example each, checking understanding with a short question before "
        "moving on; (2) then a problem set of 3-5 exercises, graded honestly — use "
        "run_python to actually run code answers and calculate to verify math; "
        "(3) then the module PROJECT: create the folder under ~/StudyProjects/"
        f"{course_key}/ with create_project_file (write a PROJECT.md spec with "
        "requirements first, plus any starter files), let the STUDENT write the "
        "code themselves in their editor, then review their work with "
        "review_project and run it with run_project — give specific, honest code "
        "review like a senior engineer, and iterate until it meets the spec. "
        "Create flashcards (flashcard_add) for anything they get wrong. Call "
        "save_progress at checkpoints. When the student has completed the project "
        "to spec, tell them to type /course done to unlock the next module. "
        "Keep every message focused — no walls of text.")


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
                print(f"{ORANGE}⏺{RESET} ", end="", flush=True)
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


def push_prompt(current_model, practice=None, conversing=False):
    """Prompt before each push-to-talk turn. Mid-conversation it stays minimal so
    the chat flows like Claude — the full menu only appears on [m].
    Returns 'talk', 'type', 'quit', 'menu', 'dictate', 'notes',
    'model:<name>', 'practice:<mode>' or 'practice:off'."""
    models = available_models()
    if practice:
        # Mid-practice: stay in the conversation — no big menu between turns.
        try:
            choice = input("🎭 [Enter] answer   [p] change/stop practice   "
                           "[q] quit: ").strip().lower()
        except EOFError:
            return "quit"
    elif conversing:
        # Mid-conversation: minimal, the dialogue just continues.
        try:
            choice = input(f"{DIM}▶ [Enter] speak · [t] type · [m] menu · "
                           f"[q] quit:{RESET} ").strip().lower()
        except EOFError:
            return "quit"
        if choice == "m":
            return "menu"
    else:
        print("\n" + "─" * 60)
        if models:
            for i, m in enumerate(models):
                mark = " ✓" if m == current_model else ""
                print(f"  [{i + 1}] {model_label(m)}{mark}")
        try:
            choice = input(f"▶ [Enter] talk   [t] type   [1-{len(models)}] model   "
                           f"[p] practice   [d] dictate   [n] lecture notes   "
                           f"[q] quit: ").strip().lower()
        except EOFError:
            return "quit"
    if choice == "":
        return "talk"
    if choice == "t":
        return "type"
    if choice == "d":
        return "dictate"
    if choice == "n":
        return "notes"
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


HELP_ROWS = [
    ("MODES", ""),
    ("/agent",               "toggle AI Agent tools: apps · web · files · system"),
    ("/voice",               "voice mode — empty Enter speaks, replies are spoken"),
    ("/text",                "back to typing only"),
    ("/chat",                "fun casual chat mode (/chat off to end)"),
    ("STUDY", ""),
    ("/course",              "structured courses — continue where you left off"),
    ("/course list",         "the 9 courses (python, cs, math, physics, ai, cybersec, software-eng, webdev, english)"),
    ("/course start <name>", "enroll and begin module 1"),
    ("/course syllabus",     "your position in each course"),
    ("/course done",         "finish the current module → unlock the next"),
    ("/learn <subject>",     "free-form tutor (no fixed syllabus)"),
    ("/quiz <subject>",      "8 graded questions with a score"),
    ("/review",              "flashcards due today (spaced repetition)"),
    ("PRACTICE", ""),
    ("/practice embassy <country>", "visa interview role-play, any country"),
    ("/practice interview", "job interview (also: exam, work)"),
    ("/practice translator", "live English ↔ नेपाली interpreter"),
    ("/practice off",        "leave practice"),
    ("CHAT", ""),
    ("/models",              "installed models with what each is best for"),
    ("/model <n|name>",      "switch model (or just say 'switch to qwen')"),
    ("/resume",              "continue your last saved conversation"),
    ("/history",             "list saved conversations — resume or delete them"),
    ("/memory",              "see what it remembers about you — delete any of it"),
    ("```",                  "start a multi-line code paste (end with ``` alone)"),
    ("/file <path> [question]", "attach a pdf/docx/html/md/txt — or just drop the path"),
    ("/help",                "this list"),
    ("/quit",                "save the chat and exit"),
]


def help_table():
    """The full command reference (shown by /help)."""
    lines = []
    for cmd, desc in HELP_ROWS:
        if not desc:
            lines.append(f"\n {ORANGE}{cmd}{RESET}")
        else:
            lines.append(f"  {BOLD}{cmd:<29}{RESET}{DIM}{desc}{RESET}")
    return "\n".join(lines) + "\n"


def text_banner(model, agent):
    """Claude-Code-style welcome box."""
    width = min(_term_width() - 1, 74)
    inner = width - 2

    def row(text="", pad=1):
        vis = re.sub(r"\x1b\[[0-9;]*m", "", text)
        return (f"{DIM}│{RESET}" + " " * pad + text
                + " " * max(0, inner - pad - len(vis)) + f"{DIM}│{RESET}")

    cwd = os.getcwd().replace(os.path.expanduser("~"), "~")
    tools = f"{GREEN}on{RESET}" if agent else f"{DIM}off (--agent to enable){RESET}"
    return "\n".join([
        "",
        f"{DIM}╭{'─' * inner}╮{RESET}",
        row(f"{ORANGE}✻{RESET} {BOLD}AI Agent{RESET} {DIM}· voice + text{RESET}"),
        row(),
        row(f"{DIM}model:{RESET} {model_label(model)}", pad=3),
        row(f"{DIM}tools:{RESET} " + tools + f"   {DIM}cwd:{RESET} {cwd}", pad=3),
        row(f"{DIM}/ for everything — /agent · /voice · /chat · /course · /help{RESET}", pad=3),
        f"{DIM}╰{'─' * inner}╯{RESET}",
        "",
    ])


# --- conversation persistence ---------------------------------------------------
CHATS_DIR = os.path.expanduser("~/.local/share/voice-assistant/chats")


_ATTACH_EXTS = {".pdf", ".docx", ".md", ".txt", ".html", ".htm", ".csv", ".json",
                ".py", ".js", ".ts", ".c", ".cpp", ".java", ".sh", ".log", ".rtf"}
ATTACH_MAX_CHARS = int(os.environ.get("VA_ATTACH_MAX", "14000"))


def try_attach(raw):
    """If `raw` is (or starts with) a path to a readable document, extract it.
    Returns (context_message, notice) or (None, error) or (None, None) if it
    isn't a file path at all. Handles quoted drag-and-drop paths."""
    import tools as T
    candidate = raw.strip().strip("'\"")
    p = os.path.expanduser(candidate)
    if not os.path.isfile(p) or os.path.splitext(p)[1].lower() not in _ATTACH_EXTS:
        return None, None
    text = T._extract_text(p).strip()
    if not text:
        return None, f"📎 Couldn't extract any text from {os.path.basename(p)}."
    words = len(text.split())
    clipped = ""
    if len(text) > ATTACH_MAX_CHARS:
        text = text[:ATTACH_MAX_CHARS]
        clipped = (" (long file — first part attached; ask specific questions and "
                   "I'll search the rest with ask_document)")
    msg = (f"[The user attached the file '{p}'. Its content follows — answer "
           f"questions about it from this content.]\n\n{text}")
    notice = (f"📎 Attached {os.path.basename(p)} — {words:,} words{clipped}. "
              "Ask me anything about it.")
    return msg, notice


MAX_HISTORY = int(os.environ.get("VA_MAX_HISTORY", "40"))


def trim_history(history):
    """Keep the system prompt plus the most recent turns. Unbounded history makes
    every Ollama call slower (the whole context is re-read each turn) and can
    overflow the model's window in long sessions."""
    if len(history) > MAX_HISTORY:
        del history[1:len(history) - (MAX_HISTORY - 8)]
    return history


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


def list_chats():
    """Saved conversations, newest first: [(path, when, preview)]."""
    import json
    try:
        files = sorted((f for f in os.listdir(CHATS_DIR) if f.endswith(".json")),
                       reverse=True)
    except FileNotFoundError:
        return []
    out = []
    for f in files[:30]:
        path = os.path.join(CHATS_DIR, f)
        preview = ""
        try:
            with open(path, encoding="utf-8") as fh:
                for m in json.load(fh):
                    if m.get("role") == "user":
                        preview = str(m.get("content", ""))[:60].replace("\n", " ")
                        break
        except Exception:
            pass
        when = f[len("chat-"):-len(".json")]
        out.append((path, when, preview))
    return out


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


def respond_text(history, model, speaker=None):
    """Stream a reply to stdout (thinking dimmed); optionally speak it sentence by
    sentence through `speaker`. Returns the answer text."""
    reply, in_think, in_answer = [], False, False
    interrupt = threading.Event()
    if speaker is not None:
        speaker.begin(interrupt)
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
                print(f"{ORANGE}⏺{RESET} ", end="", flush=True)
                in_answer = True
            print(text, end="", flush=True)
            reply.append(text)
    print()
    answer = "".join(reply).strip()
    if speaker is not None:
        for sentence in stream_sentences(iter([answer])):
            speakable = sanitize_for_tts(sentence)
            if speakable:
                speaker.feed(speakable)
        speaker.end()
    return answer


CHAT_PROMPT = (
    "You are a fun, warm, witty companion. Be playful, use light humor and the "
    "occasional emoji, ask follow-up questions, keep it casual and human — like "
    "chatting with a good friend. No lectures unless asked.")


def _voice_stack():
    """Lazy-load everything needed to hear and speak inside the unified REPL."""
    _load_voice()
    print(f"{DIM}Loading speech stack (Whisper + TTS)…{RESET}")
    model = load_whisper(WHISPER_MODEL)
    engine = voice.AudioEngine()
    speaker = voice.Speaker()
    fp16 = os.environ.get("VA_FP16", "0") == "1"
    lang = None if LANG.lower() in ("", "auto") else LANG
    print(f"🎤 {engine.input_name} — press Enter on an empty line to speak; "
          "type normally for text; /text to leave voice mode.")
    return {"model": model, "engine": engine, "speaker": speaker,
            "fp16": fp16, "lang": lang}


def text_main(agent):
    """Typed chat loop: same LLM, model switching, and (optional) agent tools."""
    current_model = LLM_MODEL
    sys_prompt = with_memory(SYSTEM_PROMPT)
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

    def confirm_shell(action):
        print(f"\n⚠️  The assistant wants to: {action}")
        try:
            return input("   Allow? [y/N]: ").strip().lower() in ("y", "yes")
        except EOFError:
            return False

    _SLASH_COMMANDS = SLASH_COMMANDS
    try:
        import atexit
        import readline
        hist_file = os.path.expanduser("~/.local/share/voice-assistant/input_history")
        os.makedirs(os.path.dirname(hist_file), exist_ok=True)
        try:
            readline.read_history_file(hist_file)
        except (FileNotFoundError, OSError):
            pass
        readline.set_history_length(500)
        atexit.register(lambda: readline.write_history_file(hist_file))

        def _complete(prefix, state):
            opts = [c + " " for c in _SLASH_COMMANDS if c.startswith(prefix)]
            return opts[state] if state < len(opts) else None
        readline.set_completer(_complete)
        readline.set_completer_delims("\n")
        readline.parse_and_bind("tab: complete")
    except ImportError:
        pass

    print(text_banner(current_model, agent))

    history = [{"role": "system", "content": sys_prompt}]
    active_toolset = None     # study/review sessions bring their own tool subset
    vstack = None             # loaded on /voice: whisper + mic + tts
    voice_on = False
    while True:
        try:
            text = claude_input().strip()
        except (EOFError, KeyboardInterrupt):
            save_chat(history)
            print("\nGoodbye!")
            break
        if not text:
            if not (voice_on and vstack):
                continue
            print("🎙️  Speaking… press Enter to stop.")
            audio = vstack["engine"].record_push_to_talk(wait_for_start=False)
            if audio is None or not len(audio):
                continue
            result = vstack["model"].transcribe(
                audio.flatten(), fp16=vstack["fp16"], language=vstack["lang"],
                condition_on_previous_text=False)
            text = result["text"].strip()
            if looks_like_hallucination(text, result.get("segments", []),
                                        len(audio) / 16000):
                print("(heard nothing)")
                continue
            print(f"{DIM}you said:{RESET} {text}")
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
                print(f"{ORANGE}⏺{RESET} {str(last[-1]['content'])[:300]}")
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

        if text == "/":
            in_chat = history and history[0].get("content") == CHAT_PROMPT
            entries = [
                ("/agent",  f"AI Agent tools      {'🟢 on' if agent else '⚪ off'}"),
                ("/voice",  f"voice mode          {'🟢 on' if voice_on else '⚪ off'}"),
                ("/chat",   f"fun chat mode       {'🟢 on' if in_chat else '⚪ off'}"),
                ("/course", "structured courses"),
                ("/learn",  "free-form tutor"),
                ("/quiz",   "graded quiz"),
                ("/review", "flashcards due"),
                ("/practice", "role-play (embassy, interview…)"),
                ("/models", "list models"),
                ("/resume", "continue last conversation"),
                ("/history", "saved conversations · delete"),
                ("/memory", "what I remember about you · forget"),
                ("/file",   "attach a pdf/docx/html/md"),
                ("/help",   "full command reference"),
                ("/quit",   "save and exit"),
            ]
            print(f"\n {ORANGE}✻ Commands{RESET}  {DIM}(number to run, Enter to cancel){RESET}")
            for i, (cmd, desc) in enumerate(entries, 1):
                print(f"  {BOLD}{i:>2}{RESET} {cmd:<11}{DIM}{desc}{RESET}")
            try:
                sel = input("  choose: ").strip()
            except EOFError:
                continue
            if not sel.isdigit() or not 1 <= int(sel) <= len(entries):
                continue
            text = entries[int(sel) - 1][0]
            low = text.lower()

        if low.startswith("/memory"):
            import tools as T
            items = T._load_memory()
            if not items:
                print("🧠 I don't have any memories about you yet. Tell me things, "
                      "or just chat — I learn automatically.")
                continue
            print(f"\n {ORANGE}✻ What I remember about you{RESET}  "
                  f"{DIM}(d n delete · clear = forget ALL · Enter close){RESET}")
            for i, it in enumerate(items, 1):
                print(f"  {BOLD}{i:>2}{RESET} {it.get('fact', '')}  "
                      f"{DIM}({it.get('on', '?')}){RESET}")
            try:
                sel = input("  choose: ").strip().lower()
            except EOFError:
                continue
            if sel == "clear":
                if input(f"  Forget ALL {len(items)} memories? [y/N]: ").strip().lower() in ("y", "yes"):
                    T._save_memory([])
                    print("  🧠 All memories forgotten.")
            elif sel.startswith("d") and sel[1:].strip().isdigit():
                i = int(sel[1:].strip())
                if 1 <= i <= len(items):
                    gone = items.pop(i - 1)
                    T._save_memory(items)
                    print(f"  🧠 Forgot: {gone.get('fact', '')}")
            continue

        if low.startswith("/history"):
            chats = list_chats()
            if not chats:
                print("No saved conversations.")
                continue
            print(f"\n {ORANGE}✻ Saved conversations{RESET}  "
                  f"{DIM}(n resume · d n delete · clear = delete ALL · Enter cancel){RESET}")
            for i, (path, when, preview) in enumerate(chats, 1):
                print(f"  {BOLD}{i:>2}{RESET} {when}  {DIM}{preview}{RESET}")
            try:
                sel = input("  choose: ").strip().lower()
            except EOFError:
                continue
            if sel == "clear":
                if input(f"  Delete ALL {len(chats)} conversations? [y/N]: ").strip().lower() in ("y", "yes"):
                    for path, _, _ in list_chats():
                        try:
                            os.unlink(path)
                        except OSError:
                            pass
                    print("  🗑️  All conversations deleted.")
                continue
            if sel.startswith("d") and sel[1:].strip().isdigit():
                i = int(sel[1:].strip())
                if 1 <= i <= len(chats):
                    try:
                        os.unlink(chats[i - 1][0])
                        print(f"  🗑️  Deleted {chats[i - 1][1]}.")
                    except OSError as exc:
                        print(f"  couldn't delete: {exc}")
                continue
            if sel.isdigit() and 1 <= int(sel) <= len(chats):
                import json
                try:
                    with open(chats[int(sel) - 1][0], encoding="utf-8") as f:
                        history = json.load(f)
                    print(f"↩️  Resumed {chats[int(sel) - 1][1]} ({len(history)} messages).")
                except Exception as exc:
                    print(f"couldn't load: {exc}")
            continue

        if low.startswith(("/file", "/attach")) or (
                len(text) < 300 and "\n" not in text):
            arg = text.split(maxsplit=1)[1] if low.startswith(("/file", "/attach")) \
                and " " in text else text
            # a trailing question after the path is allowed: /file x.pdf summarize it
            ctx, notice = try_attach(arg)
            question = None
            if ctx is None and " " in arg:
                head, rest = arg.split(maxsplit=1)
                ctx, notice = try_attach(head)
                question = rest if ctx else None
            if notice and ctx is None:
                print(notice)
                continue
            if ctx:
                trim_history(history).append({"role": "user", "content": ctx})
                print(notice)
                if question:
                    text = question
                else:
                    continue
            elif low.startswith(("/file", "/attach")):
                print("Usage: /file <path> [question] — pdf, docx, html, md, txt…")
                continue
            low = text.lower()

        if low in ("/voice", "/voice on"):
            if vstack is None:
                vstack = _voice_stack()
            voice_on = True
            speaker = vstack["speaker"]
            print("🎙️  Voice mode ON — empty Enter to speak, replies are spoken. /text to switch back.")
            continue
        if low in ("/text", "/voice off"):
            voice_on = False
            speaker = TextSpeaker()
            print("⌨️  Text mode — replies on screen only.")
            continue
        if low.startswith("/agent"):
            arg = low.split()[-1] if " " in low else ("off" if agent else "on")
            agent = arg != "off"
            if agent:
                if not supports_tools(current_model):
                    alt = pick_tool_model(available_models())
                    if alt:
                        current_model = alt
                sys_prompt = agent_system_prompt()
            else:
                sys_prompt = with_memory(SYSTEM_PROMPT)
            history[0] = {"role": "system", "content": sys_prompt}
            print(f"🤖 AI Agent {'ON — apps · web · files · system' if agent else 'off — plain chat'}"
                  f"{' (model: ' + current_model + ')' if agent else ''}")
            continue
        if low.startswith("/chat"):
            if low.endswith("off"):
                sys_prompt = agent_system_prompt() if agent else SYSTEM_PROMPT
                history = [{"role": "system", "content": sys_prompt}]
                print("🎈 Chat mode off.")
                continue
            history = [{"role": "system", "content": with_memory(CHAT_PROMPT)}]
            print("🎈 Chat mode — just hanging out. /chat off to end.")
            continue

        if low in ("/quit", "/exit", "quit", "exit", "goodbye", "bye"):
            if AUTO_MEMORY:
                with Spinner("Remembering…"):
                    learned = auto_memorize(history, current_model)
                for fact in learned:
                    print(f"🧠 {DIM}learned: {fact}{RESET}")
            saved = save_chat(history)
            if saved:
                print(f"💾 Conversation saved ({os.path.basename(saved)}) — "
                      "/resume next time to continue.")
            print("Goodbye!")
            break
        if low in ("/help", "/h"):
            print(help_table())
            continue
        if "stop practice" in low:
            history = [{"role": "system", "content": sys_prompt}]
            print("🎭 Practice mode off.")
            continue
        if low in ("/models", "/list"):
            for i, m in enumerate(available_models()):
                print(f"  [{i + 1}] {model_label(m)}{'  ← current' if m == current_model else ''}")
            continue
        if low.startswith("/course"):
            import courses
            args = text.split()[1:]
            sub = args[0].lower() if args else ""
            if sub in ("", "list") and not courses.course_state(sub or "x"):
                if sub == "" and any(courses.course_state(k) and not courses.course_state(k).get("done")
                                     for k in courses.CURRICULA):
                    # continue the (first) active course
                    sub = next(k for k in courses.CURRICULA
                               if courses.course_state(k) and not courses.course_state(k).get("done"))
                else:
                    print("📚 Courses:\n" + courses.list_courses())
                    print("\nStart one with: /course start python   "
                          "(or cs, math, physics, ai, cybersec, software-eng, webdev, english)")
                    continue
            if sub == "start":
                key = args[1].lower() if len(args) > 1 else ""
                if not courses.start_course(key):
                    print("Unknown course. " + ", ".join(courses.CURRICULA))
                    continue
                sub = key
            if sub == "done":
                active = [k for k in courses.CURRICULA
                          if courses.course_state(k) and not courses.course_state(k).get("done")]
                if not active:
                    print("No active course.")
                    continue
                st = courses.advance_course(active[0])
                if st.get("done"):
                    print(f"🎉 Course '{active[0]}' completed! /course list to pick another.")
                    continue
                print(f"✅ Module complete — moving to module {st['module'] + 1}.")
                sub = active[0]
            if sub == "syllabus":
                active = [k for k in courses.CURRICULA if courses.course_state(k)]
                for k in active or []:
                    print(courses.syllabus(k))
                if not active:
                    print("No courses started yet. /course list")
                continue
            if sub not in courses.CURRICULA:
                print("Unknown course command. Use /course, /course start <name>, "
                      "/course done, /course syllabus, /course list")
                continue
            courses.start_course(sub)
            if not supports_tools(current_model):
                alt = pick_tool_model(available_models())
                if alt:
                    current_model = alt
                    print(f"🔀 Using {model_label(alt)} for the course.")
            import tools as _T
            active_toolset = _T.STUDY_TOOLS
            print("📚 " + courses.syllabus(sub) + "\n")
            history = [{"role": "system", "content": course_tutor_prompt(sub)},
                       {"role": "user", "content": "I'm ready — let's continue the course."}]
            agent_respond(history, speaker, current_model, confirm_shell,
                          toolset=active_toolset)
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
            import tools as _T
            active_toolset = _T.REVIEW_TOOLS
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
            agent_respond(history, speaker, current_model, confirm_shell,
                          toolset=active_toolset)
            continue

        if low.startswith("/learn") or low.startswith("/quiz"):
            quiz = low.startswith("/quiz")
            args = text.split()[1:]
            subj = args[0].lower() if args else ""
            if subj in ("off", "stop"):
                active_toolset = None
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
            import tools as _T
            active_toolset = _T.STUDY_TOOLS if supports_tools(current_model) else None
            history = [{"role": "system", "content": tutor_prompt(subj, quiz)},
                       {"role": "user", "content": "Let's begin."}]
            print(f"🎓 {'Quiz' if quiz else 'Tutor'}: {subj} — progress is saved "
                  "automatically. /learn off to stop. Paste code with ``` blocks.")
            if active_toolset:
                agent_respond(history, speaker, current_model, confirm_shell,
                              toolset=active_toolset)
            else:
                reply = respond_text(history, current_model, speaker)
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
            reply = respond_text(history, current_model, speaker)
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

        trim_history(history).append({"role": "user", "content": text})
        try:
            if agent or active_toolset:
                agent_respond(history, speaker, current_model, confirm_shell,
                              toolset=active_toolset)
                current_model = apply_pending_switch(current_model)
            else:
                reply = respond_text(history, current_model, speaker)
                history.append({"role": "assistant", "content": reply or "[said nothing]"})
        except Exception as exc:                       # never crash the conversation
            history.pop()
            print(f"⚠️  '{current_model}' failed: {exc}")
            alt = pick_tool_model(available_models()) if (agent or active_toolset) else None
            if alt and alt != current_model:
                current_model = alt
                print(f"🔀 Switched to '{alt}' — please repeat your message.")


def take_lecture_notes(engine, model, fp16, lang, llm_model, speaker):
    """Record a long session (class, lecture video, meeting), transcribe it all,
    then produce structured markdown notes + flashcards."""
    import datetime
    try:
        title = input("📝 Lecture title (Enter = untitled): ").strip() or "untitled"
    except EOFError:
        return
    safe = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "-").lower() or "untitled"
    print("🔴 Recording the lecture — leave this running. Press Enter to STOP "
          "and process. (Recording can be as long as you like; transcription of "
          "a full lecture takes a while afterwards.)")
    audio = engine.record_push_to_talk(wait_for_start=False)
    if audio is None or len(audio) < voice.SAMPLE_RATE * 5:
        print("Too short — nothing saved.")
        return
    minutes = len(audio) / voice.SAMPLE_RATE / 60
    print(f"⏹️  {minutes:.1f} minutes recorded. Transcribing (this can take a while)...")
    result = model.transcribe(audio.flatten().astype(np.float32), fp16=fp16,
                              language=lang, condition_on_previous_text=False)
    transcript = result["text"].strip()
    if not transcript:
        print("Nothing intelligible was transcribed.")
        return
    print(f"🧠 Structuring notes from {len(transcript.split())} words...")
    import ollama
    notes = transcript
    try:
        r = ollama.chat(model=llm_model, messages=[{
            "role": "user",
            "content": ("Turn this lecture transcript into clean markdown study "
                        "notes with: a 3-sentence summary, the main topics as "
                        "sections with key points, any definitions, and an 'Action "
                        "items / follow-ups' section. Also end with a section "
                        "'## Flashcards' containing 5 Q/A pairs formatted exactly "
                        "as 'Q: ...' newline 'A: ...'.\n\nTranscript:\n"
                        + transcript[:24000])}])
        notes = "".join(strip_think(iter([r["message"]["content"]]))).strip() or transcript
    except Exception as exc:
        print(f"⚠️  Couldn't structure the notes ({exc}); saving raw transcript.")
    notes_dir = os.path.expanduser("~/Documents/lecture-notes")
    os.makedirs(notes_dir, exist_ok=True)
    path = os.path.join(notes_dir,
                        f"{datetime.date.today()}-{safe}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n{notes}\n\n---\n## Full transcript\n\n{transcript}\n")
    # Harvest the flashcards into the spaced-repetition deck.
    import tools as T
    added = 0
    for q, a in re.findall(r"Q:\s*(.+?)\s*\nA:\s*(.+?)(?:\n|$)", notes):
        if "already exists" not in T.flashcard_add(q.strip(), a.strip(), safe[:20]):
            added += 1
    print(f"💾 Notes saved to {path}" + (f" — {added} flashcards added." if added else ""))
    speaker.say_blocking(f"Lecture notes saved. {added} flashcards created.")


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


def _wrap_lines(text, width, lines_max=3):
    """Wrap text to at most lines_max lines of the given width."""
    text = " ".join(str(text).split())
    out = []
    while text and len(out) < lines_max:
        if len(text) <= width:
            out.append(text)
            text = ""
            break
        cut = text.rfind(" ", 0, width)
        cut = cut if cut > width // 2 else width
        out.append(text[:cut])
        text = text[cut:].strip()
    if text and out:
        out[-1] = out[-1][:max(0, width - 1)] + "…"
    return out


def print_tool_step(label, result, elapsed):
    """Render one completed tool call, Claude-Code style:
        ⏺ 🔍 Searching the web: … (1.2s)
          ⎿ Search results for '…': …"""
    took = f" {DIM}({elapsed:.1f}s){RESET}" if elapsed >= 0.05 else ""
    print(f"{GREEN}⏺{RESET} {label}{took}")
    width = max(40, shutil.get_terminal_size((100, 20)).columns - 6)
    lines = _wrap_lines(result, width) or ["(no output)"]
    print(f"  {CYAN}⎿{RESET} {DIM}{lines[0]}{RESET}")
    for ln in lines[1:]:
        print(f"    {DIM}{ln}{RESET}")


def apply_pending_switch(current_model):
    """If the agent asked to install/switch its own model, apply it now."""
    import tools as T
    if T.PENDING_MODEL[0]:
        new = T.PENDING_MODEL[0]
        T.PENDING_MODEL[0] = None
        if new != current_model:
            print(f"🔀 {DIM}now running {model_label(new)}{RESET}")
        return new
    return current_model


def agent_respond(history, speaker, model, confirm_shell, max_steps=6, toolset=None):
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
            if steps["n"]:
                print()                      # separate the answer from tool steps
            print(f"{ORANGE}⏺{RESET} {answer}")
        speaker.begin(interrupt)
        for sentence in stream_sentences(iter([answer])):
            speakable = sanitize_for_tts(sentence)
            if speakable:
                speaker.feed(speakable)
        speaker.end()
        return answer or "[said nothing]"

    interrupt = threading.Event()
    steps = {"n": 0}
    for _ in range(max_steps):
        try:
            with Spinner():
                r = ollama.chat(model=model, messages=history,
                                tools=toolset if toolset is not None else T.GENERAL_TOOLS)
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
            t0 = time.monotonic()
            if fn == "run_shell":                       # needs interactive confirm; no spinner
                print(f"{CYAN}▶{RESET} {label}")
                result = T.dispatch(fn, args, confirm=confirm_shell)
            else:
                with Spinner(label):
                    result = T.dispatch(fn, args, confirm=confirm_shell)
            print_tool_step(label, result, time.monotonic() - t0)
            steps["n"] += 1
            history.append({"role": "tool", "content": str(result), "tool_name": fn})

    speaker.say_blocking("Sorry, I couldn't finish that.")
    return "[agent stopped: too many tool steps]"


def doctor():
    """Self-check: verify every dependency the assistant needs and print fixes."""
    ok = lambda: print(f"  {GREEN}✓{RESET}", end=" ")
    bad = lambda: print(f"  ✗", end=" ")
    print(f"{CYAN}🩺 AI Agent self-check{RESET}\n")

    print("Ollama")
    try:
        models = available_models()
        ok(); print(f"server running, {len(models)} chat models")
        if pick_tool_model(models):
            ok(); print(f"tool-capable model: {pick_tool_model(models)}")
        else:
            bad(); print("no tool-capable model — fix: ollama pull qwen3:4b")
        import ollama
        names = [m.model if hasattr(m, 'model') else m.get('model', '') for m in
                 (ollama.list().get('models', []) if isinstance(ollama.list(), dict)
                  else getattr(ollama.list(), 'models', []))]
        if any("embed" in (n or "") for n in names):
            ok(); print("embedding model (document Q&A)")
        else:
            bad(); print("no embedding model — fix: ollama pull nomic-embed-text")
    except Exception as exc:
        bad(); print(f"Ollama unreachable ({exc}) — fix: ollama serve")

    print("Audio")
    try:
        import sounddevice as sd
        ins = [d for d in sd.query_devices() if d["max_input_channels"] > 0]
        outs = [d for d in sd.query_devices() if d["max_output_channels"] > 0]
        (ok() if ins else bad()); print(f"microphone devices: {len(ins)}")
        (ok() if outs else bad()); print(f"output devices: {len(outs)}")
    except Exception as exc:
        bad(); print(f"sounddevice failed: {exc}")
    (ok() if shutil.which("espeak-ng") else bad())
    print("espeak-ng (TTS fallback / Nepali)" if shutil.which("espeak-ng")
          else "espeak-ng missing — fix: sudo dnf install espeak-ng")

    print("Speech & GPU")
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            ok(); print(f"GPU: {torch.cuda.get_device_name(0)} "
                        f"({free // 1024**2} MB free of {total // 1024**2} MB)")
        else:
            bad(); print("no CUDA — Whisper/TTS will run on CPU (works, slower)")
        import whisper  # noqa: F401
        ok(); print("whisper importable")
    except Exception as exc:
        bad(); print(f"torch/whisper problem: {exc}")

    print("Extras")
    for binname, what, fix in [
            ("tesseract", "OCR", "sudo dnf install tesseract"),
            ("wl-paste", "clipboard", "sudo dnf install wl-clipboard"),
            ("notify-send", "desktop notifications", "sudo dnf install libnotify"),
            ("gdbus", "media control / screenshots", "(part of glib2)")]:
        (ok() if shutil.which(binname) else bad())
        print(f"{what} ({binname})" if shutil.which(binname)
              else f"{what} missing — fix: {fix}")
    cfg = os.path.expanduser("~/.config/voice-assistant/config.toml")
    (ok() if os.path.exists(cfg) else bad())
    print("config file" if os.path.exists(cfg)
          else "no config file — optional: cp config.example.toml " + cfg)
    (ok() if os.environ.get("VA_TAVILY_KEY") else bad())
    print("Tavily search key" if os.environ.get("VA_TAVILY_KEY")
          else "no Tavily key — optional, better web search: tavily.com")
    print(f"\n{DIM}Run the test suite with: ./venv/bin/python -m pytest tests/ -q{RESET}")


def main():
    if "--doctor" in sys.argv:
        doctor()
        return
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

    classic_voice = any(f in sys.argv for f in ("--push", "--auto", "--wake")) \
        or os.environ.get("VA_CLASSIC", "0") == "1"
    if not classic_voice:               # default: the unified AI Agent REPL
        text_main(agent)
        return
    _load_voice()                       # classic voice pipeline from here on
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

    sys_prompt = with_memory(SYSTEM_PROMPT)
    if agent:
        sys_prompt = agent_system_prompt()
        print(f"{CYAN}╭──────────────────────────────────────────────────────────╮{RESET}\n"
              f"{CYAN}│{RESET} 🤖 {GREEN}Agent mode{RESET} — apps · system · web · documents · vision {CYAN}│{RESET}\n"
              f"{CYAN}│{RESET} {DIM}Shell commands always ask you before running.{RESET}            {CYAN}│{RESET}\n"
              f"{CYAN}╰──────────────────────────────────────────────────────────╯{RESET}\n")

    def confirm_shell(action):
        print(f"\n⚠️  The assistant wants to: {action}")
        speaker.say_blocking("Do you allow that? Type y to confirm.")
        try:
            return input("   Allow? [y/N]: ").strip().lower() in ("y", "yes")
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
    conversing = False
    try:
        while True:
            if mode == "wake" and handoff is None:
                if not wake.wait(engine):
                    continue
                print("🟢 Yes?")
                speaker.say_blocking("Yes?")
                engine.flush()          # don't transcribe the "Yes?" echo
            typed_text = None
            if mode == "push" and practice:
                # Interview flow: the mic opens right after the coach finishes —
                # just speak your answer, then press Enter to send it.
                print("🎙️  Answer now — press Enter when you finish speaking. "
                      "(say 'stop practice' to end)")
                audio = engine.record_push_to_talk(wait_for_start=False)
                voiced = len(audio) // voice.FRAME_SIZE if audio is not None else 0
            elif mode == "push":
                action = push_prompt(current_model, practice, conversing)
                if action == "quit":
                    speaker.say_blocking("Goodbye!")
                    print("\nGoodbye!")
                    break
                if action == "type":
                    typed_text = claude_input().strip()
                    if not typed_text:
                        continue
                if action == "menu":
                    conversing = False       # show the full menu next round
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
                if action == "notes":
                    take_lecture_notes(engine, model, fp16, stt_context["lang"],
                                       current_model, speaker)
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
                if typed_text is None:
                    audio = engine.record_push_to_talk(wait_for_start=False)
                    voiced = len(audio) // voice.FRAME_SIZE if audio is not None else 0
                else:
                    audio, voiced = None, 0
            else:
                if handoff is None:
                    print("🎧 Listening... (just start talking)")
                audio, voiced = engine.capture_utterance(vad, listener=live, prefill=handoff)
                handoff = None
            if typed_text is not None:
                ctx, notice = try_attach(typed_text)
                if ctx:
                    trim_history(history).append({"role": "user", "content": ctx})
                    print(notice)
                    speaker.say_blocking("File attached. Ask me about it.")
                    conversing = True
                    continue
                text = typed_text
            else:
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

            trim_history(history).append({"role": "user", "content": text})
            try:
                if agent and not practice:
                    agent_respond(history, speaker, current_model, confirm_shell)
                    handoff = None
                else:
                    # Practice is pure conversation: streaming reply, no tools.
                    reply, handoff = respond(history, speaker, engine, vad, current_model, barge_in)
                    history.append({"role": "assistant", "content": reply or "[said nothing]"})
                conversing = True            # keep the dialogue flowing turn to turn
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
