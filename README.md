# Voice Assistant

A fully local, offline voice assistant that feels like a real voice-mode chat:

```
🎤 mic → Silero VAD → Whisper (live STT) → Ollama LLM (streaming) → Coqui/espeak TTS → 🔊
```

It transcribes your speech **in real time** (words appear on screen as you talk),
thinks with a local LLM, and **speaks the reply as it is generated** — sentence by
sentence — so it starts talking almost immediately. It keeps conversation history,
and you can **interrupt it just by talking**: it stops mid-word, and the words you
interrupted with are kept and answered — no need to repeat yourself.

## What makes it feel smooth (the voice-mode checklist)

| Capability | How |
|------------|-----|
| Hears quiet speech, ignores noise | **Silero neural VAD** (not energy thresholds) |
| Doesn't cut you off mid-sentence | Endpointing tolerates natural pauses; short utterances get extra grace |
| Always listening, instantly | One persistent mic stream, no per-turn calibration pauses |
| Replies start fast | **Speculative STT**: the final transcript is computed *while* your trailing pause is still being confirmed; the LLM streams; TTS speaks sentence-by-sentence |
| Barge-in without losing your words | The interrupting audio is handed to the next capture and transcribed |
| Interruption feels instant | Playback stops with a ~50 ms fade, the LLM stream is cancelled |
| Doesn't talk over itself | Echo-level gating, or real system echo cancellation with `--aec` |
| No ghost turns | Whisper-hallucination filter (`no_speech_prob` + junk phrases) |
| Survives a bad mic | Auto gain guard: lowers system input volume when the mic clips |

## Requirements

- Python 3.12 (a `venv/` is already set up in this folder — always use
  `./venv/bin/python`, not the system Python)
- [Ollama](https://ollama.com) running locally with a chat model pulled
  (default: `gemma3:4b` — `ollama pull gemma3:4b`)
- A microphone and audio output (PipeWire/PulseAudio)
- Optional: `espeak-ng` as a TTS fallback (`sudo dnf install espeak-ng`)

Python packages: `pip install -r requirements.txt`

## Usage

```bash
./venv/bin/python assistant.py                # hands-free, live STT, streaming, barge-in
./venv/bin/python assistant.py --wake         # hands-free via wake word: say "hey jarvis"
./venv/bin/python assistant.py --text         # typed chat instead of voice
./venv/bin/python assistant.py --agent        # agent tools (combine with any mode)
./venv/bin/python assistant.py --aec          # + echo cancellation (recommended with speakers)
./venv/bin/python assistant.py --push         # push-to-talk instead of VAD
./venv/bin/python assistant.py --no-barge-in  # don't allow interruption
./venv/bin/python assistant.py --no-live      # transcribe after you finish, not live
./venv/bin/python assistant.py --list-devices # show devices for VA_INPUT/OUTPUT_DEVICE
./venv/bin/python assistant.py --doctor       # self-check: deps, models, audio, GPU
```

- **Hands-free (default):** just start talking; the neural VAD captures your
  utterance when you pause. No calibration pauses between turns.
- **Live transcript:** your words stream onto the `📝` line as you speak.
- **Barge-in:** while the assistant is speaking, start talking. It stops within a
  fraction of a second and answers the words you interrupted with.
- Say "quit", "exit", or "goodbye" (or press Ctrl-C) to leave.

## Agent mode (`--agent`)

Turns the assistant into an agent that can *do things*, with a live activity UI
(spinners like `🔍 Searching the web…`, `✓` results) and visible model thinking.
Works in voice and `--text` mode. Model switching: menu number keys, `/model`,
or say "switch to qwen". Agent mode needs a tool-capable model (qwen3 / llama3.2 /
phi4-mini — it auto-switches if the current one can't use tools).

Things you can ask:

- **Web**: "search the web for …" — DuckDuckGo + dated news + auto-reads top
  pages. For best quality get a free [Tavily](https://tavily.com) key and set
  `VA_TAVILY_KEY`.
- **Computer**: "open spotify" (flatpak/desktop-file aware), "set volume to 40",
  "set brightness to 60", shell commands (asks `y/N` confirmation first).
- **Documents**: "find my resume", "search my documents for the invoice",
  "read notes.txt and summarize" (txt/md/pdf/docx; read-only, home-scoped).
- **Weather**: "what's the weather in Kathmandu?" (Open-Meteo, no key).
- **Time**: "what time is it in Nepal?", "convert 8 pm UK time to Tokyo" —
  exact `zoneinfo` math, the model never guesses offsets.
- **Reminders**: "remind me in 10 minutes to check the oven" — the alert is
  printed and spoken when it fires.
- **Media**: "pause the music", "next song", "what's playing?" — MPRIS
  (Spotify, Firefox, VLC…), no playerctl needed.
- **Memory**: "remember that my favorite team is Argentina" — stored in
  `~/.voice_assistant_memory.json` and recalled in *future sessions*;
  "forget …" to remove.
- **Vision**: "what's on my screen right now?", "describe ~/Pictures/photo.jpg"
  — gemma3:4b multimodal. (On GNOME Wayland the first screenshot may show a
  permission dialog.)
- **Lists & notes**: "add milk to my shopping list", "what's on my todo list?",
  "remove eggs from shopping" — persistent named lists
  (`~/.voice_assistant_lists.json`).
- **Daily briefing**: "good morning" / "brief me" — date, weather, reminders,
  todos, and live headlines in one answer (`VA_CITY` sets the default city).
- **Long documents**: "using report.pdf, what does it say about the budget?" —
  `ask_document` semantically searches the whole file (chunk + embeddings via
  `nomic-embed-text`; `ollama pull nomic-embed-text` once). Handles books and
  long PDFs that `read_document` would truncate.
- **Notifications**: reminders also pop desktop notifications (`notify-send`),
  and the agent can send one on request.
- **Nepali / multilingual speech**: replies containing Devanagari (नेपाली) are
  spoken with espeak-ng's Nepali voice automatically (also Chinese, Arabic,
  Russian scripts). For Nepali *input*, run with `VA_LANG=ne` or `VA_LANG=auto`.

## Voice: male or female

The default voice is now **VCTK** — real human-recorded neural voices with both
genders (GPU-accelerated):

```bash
./venv/bin/python assistant.py --female        # default (speaker p225)
./venv/bin/python assistant.py --male          # male voice (speaker p226)
VA_SPEAKER_ID=p230 ./venv/bin/python assistant.py   # any of VCTK's 109 speakers
```

## Lectures, YouTube & phone

- **📝 Lecture note-taker** — press `[n]` in the voice menu before a class,
  lecture video or meeting. It records as long as you like; on stop it
  transcribes everything, writes structured markdown notes (summary, topics,
  definitions, action items) to `~/Documents/lecture-notes/`, and auto-creates
  5 flashcards from the content.
- **▶️ YouTube summarizer** — paste a YouTube link in chat: "summarize this
  video <url>" → transcript with timestamps → key points; you can ask follow-up
  questions or say "quiz me on it".
- **📱 Telegram bridge** — `./venv/bin/python telegram_bot.py` after a one-time
  setup (see the file's docstring): your assistant on your phone with the same
  memory, lists, reminders and web search. Locked to your Telegram account;
  shell commands are always refused remotely.

## More tools & modes

- **🗣️ Live translator** — practice section `Translator`: speak Nepali, it speaks
  English; speak English, it answers in नेपाली (Whisper auto-language + Nepali TTS).
- **📇 Flashcards (spaced repetition)** — tutors create cards for what you get
  wrong; `/review` quizzes the cards due today (SM-2-style scheduling); the daily
  briefing reminds you how many are due.
- **✍️ Dictation** — press `[d]` in the voice menu: speech is typed into a
  document chunk by chunk; on "stop dictation" it saves a grammar-cleaned version
  plus the raw transcript to `~/Documents/`.
- **👁️ OCR** — "read the text in this screenshot" (tesseract, vision fallback).
- **📋 Clipboard** — "summarize my clipboard", "explain this error I copied".
- **📊 System** — "how's my battery?" (battery/disk/memory/CPU).

## Configuration file

Copy `config.example.toml` to `~/.config/voice-assistant/config.toml` — every key
becomes a `VA_*` setting so you don't have to pass env vars each launch (env vars
still override the file).

## Conversations persist

Text chats are auto-saved on exit to `~/.local/share/voice-assistant/chats/`;
type `/resume` to continue the most recent one. Voice sessions save too.

## Tests

```bash
./venv/bin/python -m pytest tests/ -q     # 21 tests, ~2s, no audio/network needed
```

## Study mode 📚 — structured courses with projects (`/course`)

Full CS50-style courses in nine domains, each 7–8 modules with a hands-on
project per module and a capstone:

```
/course list             # python · cs · math · physics · ai · cybersec · software-eng · webdev
/course start python     # enroll and begin module 1
/course                  # continue where you left off (progress is saved)
/course syllabus         # see your position in the course
/course done             # mark the current module's project complete → next module
```

Each module runs like a real course: **teach** (one concept at a time, checked)
→ **problem set** (graded, code actually run) → **project** (Claude-Code style:
the instructor scaffolds a spec + starter files in `~/StudyProjects/<course>/`,
*you* write the code in your editor, then it reviews your files like a senior
engineer, runs them, and iterates with you until the spec is met). Wrong answers
become flashcards automatically; progress persists between sessions.

## Learning mode 🎓 (`/learn`, `/quiz`)

A personal tutor with real teaching mechanics, in text mode (`--text --agent`):

```
/learn python      # also: programming, math, cs, physics, cybersec, ai
/quiz math         # 8 graded questions with a final score + what to review
/learn off         # stop
```

- **Teaches interactively** — one concept at a time, checks understanding before
  moving on, adapts to your answers.
- **Progress memory** — every session saves covered topics, weak spots, and the
  next topic to `~/.voice_assistant_learning.json`; the next `/learn` resumes and
  reviews weak spots first.
- **Real code execution** — the tutor runs examples and *your* solutions with the
  `run_python` tool; paste multi-line code between ``` lines.
- **Verified math** — a sympy-backed `calculate` tool solves/differentiates/
  integrates exactly, so answers are computed, never guessed.
- **Auto model pick** — programming → `qwen2.5-coder`, everything else →
  `qwen3:8b` (pulled for this; the smartest model that fits the GPU).

## Practice modes 🎭 (Exam · Embassy · Work · Interview)

Role-play coaching for real-life situations — the assistant plays the examiner /
visa officer / colleague / hiring manager, asks one question at a time, probes
weak answers, and gives honest feedback at the end.

- **Voice mode:** press **[p]** in the menu, pick a section. Say "stop practice" to end.
- **Text mode:** `/practice embassy` (or `exam`, `work`, `interview`); `/practice off` to end.

Example — Embassy: *"Good morning. Please state your full name for the record."*
→ your answer → *"Can you elaborate on what specifically you hope to study…?"*

## Always-on background service

Run the assistant permanently — it starts at login and waits for "hey jarvis":

```bash
./install-service.sh                           # installs + enables (done once)
systemctl --user start voice-assistant         # start it right now
tail -f ~/.local/share/voice-assistant.log     # see what it's doing
systemctl --user stop voice-assistant          # stop / 'disable' to remove from login
```

Note: only one program can own the mic pipeline cleanly — stop the service
before running `assistant.py` manually in a terminal.

Agent-related settings: `VA_WAKE_WORD` (`hey_jarvis`/`hey_marvin`/`hey_mycroft`/
`alexa`), `VA_WAKE_THRESHOLD` (`0.5`), `VA_MEMORY_FILE`, `VA_VISION_MODEL`
(`gemma3:4b`), `VA_DOCS_ROOT`, `VA_SEARCH_READ` (pages auto-read per search),
`VA_TAVILY_KEY`, `VA_SHOW_THINK=0` to hide model reasoning.

### Echo cancellation (`--aec`)

With loudspeakers, the mic hears the assistant's own voice. Without AEC the
assistant uses an echo-level energy gate (you must be clearly louder than the
echo — works fine at normal volumes, and headphone users don't need any of
this). With `--aec` it loads the system `module-echo-cancel`
(PipeWire/PulseAudio), routes default audio through it, and **restores your
audio settings on exit** — the assistant's voice is then subtracted from the
mic signal before the app ever sees it, so barge-in is clean at any volume.

### Configuration (environment variables)

| Variable             | Default                        | Meaning                                            |
| -------------------- | ------------------------------ | -------------------------------------------------- |
| `VA_MODE`            | `auto`                         | listening mode: `auto` (VAD) or `push`             |
| `VA_WHISPER_MODEL`   | `small`                        | final/full-quality Whisper model (tiny/base/small/…) |
| `VA_STT_LIVE_MODEL`  | `base`                         | fast Whisper model used for live partial updates   |
| `VA_STT_INTERVAL`    | `0.7`                          | seconds between live transcript updates            |
| `VA_STT_LIVE`        | `1`                            | set to `0` (or `--no-live`) to transcribe only at the end |
| `VA_LANG`            | `en`                           | language code, or `auto` to detect                 |
| `VA_FP16`            | `0`                            | set to `1` for half-precision (NOT on GTX 16-series) |
| `VA_LLM_MODEL`       | `gemma3:4b`                    | Ollama model name (reasoning models OK: `<think>` blocks are stripped) |
| `VA_TTS`             | `coqui`                        | TTS engine: `coqui`, `espeak`, or `none`           |
| `VA_COQUI_MODEL`     | `tts_models/en/ljspeech/vits`  | Coqui model                                        |
| `VA_BARGE_IN`        | `1`                            | set to `0` to disable interruption                 |
| `VA_AEC`             | `0`                            | set to `1` to enable echo cancellation (same as `--aec`) |
| `VA_AUTO_GAIN`       | `1`                            | set to `0` to stop the clipping auto-gain guard    |
| `VA_INPUT_DEVICE`    | system default                 | mic: device index or name substring                |
| `VA_OUTPUT_DEVICE`   | system default                 | speaker: device index or name substring            |
| `VA_VAD_START`       | `0.60`                         | speech probability that starts a capture           |
| `VA_VAD_KEEP`        | `0.35`                         | probability that still counts as "speaking"        |
| `VA_VAD_SILENCE`     | `0.8`                          | seconds of trailing silence that ends an utterance |
| `VA_MAX_UTTERANCE`   | `45`                           | hard cap on one utterance, seconds                 |
| `VA_BARGE_PROB`      | `0.80`                         | speech probability required to barge in            |
| `VA_BARGE_SUSTAIN`   | `250`                          | ms of sustained speech required to barge in        |
| `VA_BARGE_FACTOR`    | `1.6`                          | how much louder than the playback echo you must be |

Tuning quick reference: cut off mid-sentence → raise `VA_VAD_SILENCE` (e.g. `1.1`).
Barge-in too eager → raise `VA_BARGE_SUSTAIN` to `350` or `VA_BARGE_PROB` to `0.9`.
Barge-in not triggering with speakers → use `--aec`, or lower `VA_BARGE_FACTOR`.
Live transcription heavy on the GPU → `VA_STT_LIVE_MODEL=tiny` or `--no-live`.

> **fp16 note:** the default is fp32 because GTX 16-series cards (like the 1660 Ti
> here) emit `NaN` logits in half precision, which makes Whisper crash/garble. The
> model still runs on the GPU. Only set `VA_FP16=1` on a GPU that supports it.

## Architecture

- **`assistant.py`** — orchestration: capture → live/speculative STT → stream LLM
  (`<think>` stripping, sentence chunking, TTS sanitization) → speak, with
  barge-in handoff between turns and the hallucination/gain guards.
- **`voice.py`** — the audio engine:
  - `AudioEngine` — one persistent mic stream; frame queue, rolling noise floor,
    `capture_utterance` endpointing, push-to-talk.
  - `VAD` — Silero neural VAD (energy fallback).
  - `BargeInMonitor` — talk-over detection with echo gating; keeps the
    interrupting audio as a handoff.
  - `LiveTranscriber` — live partials + speculative final transcription.
  - `Speaker` — streaming, interruptible TTS playback (Coqui VITS → espeak-ng).
  - `EchoCancel` / `InputGain` — system AEC routing and the clipping guard.
- **`transcribe.py`** — tiny file-based transcription demo.
- **`courses.py`** — course curricula and progress; **`telegram_bot.py`** — phone bridge.

## Hardware note: the mic that cried wolf

This machine's codec shipped with **+30 dB capture gain and +30 dB "Internal Mic
Boost" stacked (+60 dB total)** — the mic clipped constantly, which is what made
the old energy-based listening so unreliable. The boost has been zeroed
(`amixer -c 2 sset 'Internal Mic Boost' 0`) and the assistant now watches every
captured utterance for clipping and steps the system input volume down
automatically (`VA_AUTO_GAIN=0` disables). If speech ever sounds too quiet in
the transcripts, raise the input volume in system settings; the guard only ever
lowers it when it actually clips.

## Notes on the voice

The default voice is **Coqui VITS**, a natural neural voice that runs on the GPU
and needs no reference audio. If it fails to load, the assistant falls back to
`espeak-ng`, then to text-only. Playback is chunked through `sounddevice`, so an
interruption fades the voice out within ~50 ms.

**XTTS-v2** is *not* enabled: it needs `torchcodec` to decode the speaker
reference clip, and `torchcodec` cannot load against the system **ffmpeg 8** (it
only ships bindings for ffmpeg 4–7). To use it you'd need an ffmpeg ≤ 7 build,
then set `VA_COQUI_MODEL=tts_models/multilingual/multi-dataset/xtts_v2`.
