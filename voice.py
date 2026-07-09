"""
Voice I/O for the assistant: persistent microphone engine, neural VAD capture,
barge-in monitoring, live transcription, and interruptible text-to-speech.

Design (what makes it feel like a commercial voice mode):

  * One persistent input stream for the whole session — no per-turn stream
    open/close and no per-turn "calibration" dead time, so the assistant is
    always listening the instant its reply ends.
  * Silero VAD (a small neural network) instead of RMS energy, so quiet
    syllables and natural mid-sentence pauses are not mistaken for silence.
    Falls back to adaptive energy VAD if Silero is unavailable.
  * Endpointing with a short-utterance grace period, plus a speculative final
    transcription that starts *while* the trailing silence is still being
    confirmed, so the reply starts almost immediately after you stop.
  * Barge-in keeps the audio that triggered it and hands it to the next
    capture, so the words you interrupted with are not lost.
  * Playback through sounddevice in small chunks with a fade-out, so an
    interruption stops the voice within ~50 ms instead of killing a process.
  * Optional system echo cancellation (PipeWire/PulseAudio module-echo-cancel)
    so barge-in works with loudspeakers, not just headphones.

TTS uses Coqui VITS (a natural neural voice, GPU-accelerated when available)
and falls back to the system `espeak-ng` binary if Coqui can't be loaded.
"""

import atexit
import collections
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile

SAMPLE_RATE = 16000            # capture rate (Whisper + Silero native)
FRAME_SIZE = 512               # samples per VAD frame (Silero requires 512 @ 16 kHz)
FRAME_SEC = FRAME_SIZE / SAMPLE_RATE   # 32 ms

# --- VAD / endpointing tuning ------------------------------------------------
VAD_START_PROB = float(os.environ.get("VA_VAD_START", "0.60"))   # speech onset
VAD_KEEP_PROB = float(os.environ.get("VA_VAD_KEEP", "0.35"))     # still-speaking
VAD_SILENCE_SEC = float(os.environ.get("VA_VAD_SILENCE", "0.8")) # end of utterance
GRACE_EXTRA_SEC = 0.4          # extra trailing silence allowed for short utterances
GRACE_UNDER_SEC = 1.5          # ...when voiced speech so far is shorter than this
MIN_SPEECH_SEC = 0.25
MAX_UTTERANCE_SEC = float(os.environ.get("VA_MAX_UTTERANCE", "45"))
PREROLL_SEC = 0.5              # audio kept from before speech onset
ENDPOINT_PAD_SEC = 0.25        # trailing silence kept when trimming for STT

# --- barge-in tuning ----------------------------------------------------------
# The mic also hears the assistant's own voice, so barging in requires high
# speech probability, sustained for a while, and (while audio is actually
# playing) energy clearly above the measured playback-echo level.
BARGE_PROB = float(os.environ.get("VA_BARGE_PROB", "0.80"))
BARGE_SUSTAIN_MS = int(os.environ.get("VA_BARGE_SUSTAIN", "250"))
BARGE_ECHO_FACTOR = float(os.environ.get("VA_BARGE_FACTOR", "1.6"))
BARGE_PREROLL_FRAMES = 6       # ~0.2 s of audio kept from before the barge-in run

VOICE = os.environ.get("VA_VOICE", "en")


def _rms(block):
    return float(np.sqrt(np.mean(np.square(block, dtype=np.float64)) + 1e-12))


def _pick_device(env_var, kind):
    """Resolve a sounddevice device from an env var (index or name substring)."""
    want = os.environ.get(env_var, "").strip()
    if not want:
        return None
    if want.lstrip("-").isdigit():
        return int(want)
    for idx, dev in enumerate(sd.query_devices()):
        channels = dev["max_input_channels" if kind == "input" else "max_output_channels"]
        if channels > 0 and want.lower() in dev["name"].lower():
            return idx
    print(f"⚠️  {env_var}='{want}' matched no {kind} device; using default.")
    return None


# ---------------------------------------------------------------------------
# Voice activity detection
# ---------------------------------------------------------------------------
class VAD:
    """
    Frame-level speech probability. Uses the Silero VAD neural network when
    available (robust to quiet speech, noise, and mid-word dips that fool
    energy thresholds); otherwise falls back to adaptive RMS energy.
    """

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()
        self._torch = None
        try:
            import torch
            from silero_vad import load_silero_vad
            self._torch = torch
            self._model = load_silero_vad()
            self.kind = "silero"
        except Exception as exc:
            print(f"⚠️  Silero VAD unavailable ({type(exc).__name__}: {exc}); "
                  f"using energy VAD. `pip install silero-vad` for better listening.")
            self.kind = "energy"
            self._floor = collections.deque(maxlen=int(6 / FRAME_SEC))  # ~6 s of RMS

    def prob(self, frame):
        """Speech probability (0..1) for one float32 frame of FRAME_SIZE samples."""
        if self._model is not None:
            with self._lock:
                return float(self._model(self._torch.from_numpy(frame), SAMPLE_RATE).item())
        # Energy fallback: adaptive floor from the quietest recent frames.
        energy = _rms(frame)
        self._floor.append(energy)
        floor = float(np.percentile(self._floor, 20)) if len(self._floor) > 8 else 1e-4
        return 1.0 if energy > max(floor * 3.0, floor + 0.005) else 0.0

    def reset(self):
        """Reset streaming state at utterance boundaries."""
        if self._model is not None:
            with self._lock:
                self._model.reset_states()


# ---------------------------------------------------------------------------
# Persistent microphone engine
# ---------------------------------------------------------------------------
class AudioEngine:
    """
    Owns the single, always-open input stream. Consumers (utterance capture,
    the barge-in monitor, push-to-talk) take turns reading frames from the
    queue; they never run concurrently. A rolling RMS history provides an
    always-current noise floor with no calibration pauses.
    """

    QUEUE_MAX = int(20 / FRAME_SEC)  # cap ~20 s; drop oldest beyond that

    def __init__(self, start_stream=True):
        self._q = queue.Queue(maxsize=self.QUEUE_MAX)
        self._pending = np.empty(0, dtype=np.float32)  # accumulator to fixed frames
        self._rms_hist = collections.deque(maxlen=int(8 / FRAME_SEC))
        self._stream = None
        self.input_name = "(no stream)"
        if start_stream:
            device = _pick_device("VA_INPUT_DEVICE", "input")
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                blocksize=FRAME_SIZE, device=device, callback=self._callback)
            self._stream.start()
            self.input_name = sd.query_devices(self._stream.device)["name"]

    def _callback(self, indata, n_frames, t, status):
        # Re-chunk whatever PortAudio delivers into exact FRAME_SIZE frames.
        data = np.concatenate([self._pending, indata[:, 0]])
        pos = 0
        while len(data) - pos >= FRAME_SIZE:
            frame = data[pos:pos + FRAME_SIZE].copy()
            pos += FRAME_SIZE
            self._rms_hist.append(_rms(frame))
            try:
                self._q.put_nowait(frame)
            except queue.Full:
                try:
                    self._q.get_nowait()          # drop oldest
                    self._q.put_nowait(frame)
                except queue.Empty:
                    pass
        self._pending = data[pos:]

    def read(self, timeout=0.5):
        """Next mic frame, or None on timeout."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def flush(self, keep_tail=0):
        """Discard queued audio; return up to keep_tail of the newest frames."""
        tail = collections.deque(maxlen=keep_tail) if keep_tail else None
        while True:
            try:
                frame = self._q.get_nowait()
                if tail is not None:
                    tail.append(frame)
            except queue.Empty:
                return list(tail) if tail else []

    def noise_floor(self):
        """Rolling estimate of ambient level (20th percentile of recent RMS)."""
        if len(self._rms_hist) < 8:
            return 1e-4
        return float(np.percentile(self._rms_hist, 20))

    def close(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    # -- utterance capture ----------------------------------------------------
    def capture_utterance(self, vad, listener=None, prefill=None, stop=None):
        """
        Record one utterance, endpointed by the VAD.

        Waits for speech onset (unless `prefill` frames from a barge-in are
        given, in which case the utterance is already in progress), records
        until trailing silence, and returns (audio_float32, voiced_frames) with
        the silence tail trimmed to a small pad. Returns (None, 0) if `stop`
        gets set while waiting, or on a too-short false trigger... which simply
        rearms — only `stop` truly aborts.

        `listener` (optional) receives non-blocking progress callbacks:
            listener.on_progress(audio_snapshot, voiced_frames)   during speech
            listener.on_endpoint_candidate(trimmed, voiced)       likely endpoint
        """
        preroll_frames = max(1, int(PREROLL_SEC / FRAME_SEC))
        base_silence = max(2, int(VAD_SILENCE_SEC / FRAME_SEC))
        grace_silence = base_silence + int(GRACE_EXTRA_SEC / FRAME_SEC)
        min_speech = max(1, int(MIN_SPEECH_SEC / FRAME_SEC))
        max_frames = int(MAX_UTTERANCE_SEC / FRAME_SEC)
        live_every = max(1, int(float(os.environ.get("VA_STT_INTERVAL", "0.7")) / FRAME_SEC))

        vad.reset()
        preroll = collections.deque(self.flush(keep_tail=preroll_frames) if prefill is None else [],
                                    maxlen=preroll_frames)
        collected, voiced, silence_run, since_live = [], 0, 0, 0
        speculated_at = -1
        in_speech = prefill is not None

        if prefill:
            for frame in prefill:                 # rebuild VAD state over the handoff
                if vad.prob(frame) >= VAD_KEEP_PROB:
                    voiced += 1
            collected = list(prefill)

        while True:
            frame = self.read(timeout=0.5)
            if frame is None:
                if stop is not None and stop.is_set():
                    return None, 0
                continue

            p = vad.prob(frame)

            if not in_speech:
                preroll.append(frame)
                if stop is not None and stop.is_set():
                    return None, 0
                if p >= VAD_START_PROB:
                    in_speech = True
                    collected = list(preroll)
                    voiced, silence_run, since_live = 1, 0, 0
                    print("🎙️  ", end="", flush=True)
                continue

            collected.append(frame)
            if p >= VAD_KEEP_PROB:
                voiced += 1
                silence_run = 0
                speculated_at = -1                # speech resumed; re-arm speculation
            else:
                silence_run += 1

            limit = grace_silence if voiced * FRAME_SEC < GRACE_UNDER_SEC else base_silence

            if silence_run >= limit or len(collected) >= max_frames:
                if voiced < min_speech:           # cough/noise blip: back to waiting
                    in_speech, collected, voiced, silence_run = False, [], 0, 0
                    vad.reset()
                    continue
                return self._trim(collected, silence_run), voiced

            # Fire the speculative final transcription once the pause is likely
            # (but not yet certainly) the end of the utterance.
            if (listener is not None and speculated_at < 0
                    and silence_run >= max(2, int(limit * 0.45)) and voiced >= min_speech):
                speculated_at = silence_run
                listener.on_endpoint_candidate(self._trim(collected, silence_run), voiced)

            since_live += 1
            if listener is not None and since_live >= live_every and voiced >= min_speech:
                since_live = 0
                listener.on_progress(np.concatenate(collected), voiced)

    @staticmethod
    def _trim(frames, silence_run):
        """Drop the trailing silence beyond a small pad (Whisper hallucinates on
        long quiet tails), returning a flat float32 array."""
        pad = int(ENDPOINT_PAD_SEC / FRAME_SEC)
        keep = len(frames) - max(0, silence_run - pad)
        return np.concatenate(frames[:keep] if keep > 0 else frames).astype(np.float32)

    # -- push-to-talk ---------------------------------------------------------
    def record_push_to_talk(self, wait_for_start=True):
        """Record between two Enter presses. Returns float32 audio or None.

        With wait_for_start=False the recording begins immediately (the caller has
        already consumed the starting keypress, e.g. from a menu prompt).
        """
        if wait_for_start:
            print("Press Enter to start recording (or Ctrl-C to quit)...")
            input()
        self.flush()
        print("🎙️  Recording... press Enter to stop.")
        done = threading.Event()
        threading.Thread(target=lambda: (input(), done.set()), daemon=True).start()
        frames = []
        while not done.is_set():
            frame = self.read(timeout=0.2)
            if frame is not None:
                frames.append(frame)
        print("⏹️  Stopped recording.")
        return np.concatenate(frames) if frames else None


class BargeInMonitor:
    """
    Watches the mic while the assistant speaks and sets `interrupt` when the
    user genuinely talks over it. To avoid tripping on the assistant's own
    voice from the speakers it requires, sustained over BARGE_SUSTAIN_MS:
      * high neural speech probability, and
      * while audio is actually playing, RMS well above the measured
        playback-echo level at the mic (with echo cancellation active the echo
        level is near the noise floor, so this costs nothing).
    On trigger it keeps the frames of the interruption (`handoff`) so the next
    capture can transcribe the words that interrupted — nothing is lost.
    """

    def __init__(self, engine, vad, interrupt, speaker=None):
        self.engine = engine
        self.vad = vad
        self.interrupt = interrupt
        self.speaker = speaker
        self.handoff = None
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        sustain = max(3, int(BARGE_SUSTAIN_MS / 1000 / FRAME_SEC))
        recent = collections.deque(maxlen=sustain + BARGE_PREROLL_FRAMES)
        hits = collections.deque(maxlen=sustain)
        echo = collections.deque(maxlen=int(3 / FRAME_SEC))   # RMS while TTS plays
        self.engine.flush()
        self.vad.reset()
        while not self._stop.is_set():
            frame = self.engine.read(timeout=0.2)
            if frame is None:
                continue
            recent.append(frame)
            energy = _rms(frame)
            playing = self.speaker.is_playing() if self.speaker else False
            if playing:
                echo.append(energy)
                if len(echo) < 8:
                    continue        # no echo stats yet: can't tell user from self
                # The user must be clearly louder than the assistant's own voice
                # as heard at the mic (with --aec that echo is near-zero anyway).
                loud_enough = energy > max(
                    float(np.percentile(echo, 80)) * BARGE_ECHO_FACTOR,
                    self.engine.noise_floor() * 2.5)
            else:
                loud_enough = energy > min(self.engine.noise_floor() * 2.5, 0.3)

            hits.append(1 if (self.vad.prob(frame) >= BARGE_PROB and loud_enough) else 0)
            # Windowed vote instead of a consecutive run: real speech has brief
            # consonant dips that must not reset the detector.
            if len(hits) == sustain and sum(hits) >= 0.7 * sustain and hits[-1]:
                self.handoff = list(recent)
                self.interrupt.set()
                return

    def start(self):
        self._stop.clear()
        self.handoff = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Live + speculative transcription
# ---------------------------------------------------------------------------
class LiveTranscriber:
    """
    Turns capture progress callbacks into on-screen live partials and a fast
    final transcript.

    A worker thread transcribes the latest audio snapshot with the fast model
    (live partials) and, when the capture signals a likely endpoint, runs the
    full-quality model *speculatively* — so by the time the endpoint is
    confirmed the final text is usually already done. `finalize()` reuses the
    speculative result when the audio didn't change, else transcribes fresh.

    on_partial(text, is_final) is called for live updates and once at the end.
    """

    def __init__(self, live_model, final_model=None, language="en",
                 fp16=False, on_partial=None):
        self.live_model = live_model
        self.final_model = final_model if final_model is not None else live_model
        self.language = language or None
        self.fp16 = fp16
        self.on_partial = on_partial
        self._model_lock = threading.Lock()   # whisper models are not thread-safe
        self._cv = threading.Condition()
        self._snapshot = None                 # (audio, voiced) for live partials
        self._candidate = None                # (audio, voiced) for speculative final
        self._spec_result = None              # (voiced, text, segments)
        self._spec_busy = None                # voiced count being speculated on
        self._spec_done = threading.Event()
        self._last_partial = ""
        self._closing = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _transcribe(self, model, audio):
        # condition_on_previous_text=False stops Whisper from looping/repeating on
        # noisy or near-silent audio (the classic "thank you. thank you." artifact).
        with self._model_lock:
            result = model.transcribe(audio, fp16=self.fp16, language=self.language,
                                      condition_on_previous_text=False)
        return result["text"].strip(), result.get("segments", [])

    # -- called from the capture loop (must not block) -------------------------
    def on_progress(self, audio, voiced):
        with self._cv:
            self._snapshot = (audio, voiced)
            self._cv.notify()

    def on_endpoint_candidate(self, trimmed, voiced):
        with self._cv:
            self._candidate = (trimmed, voiced)
            self._cv.notify()

    # -- worker ----------------------------------------------------------------
    def _worker(self):
        while True:
            with self._cv:
                while not (self._candidate or self._snapshot or self._closing):
                    self._cv.wait()
                if self._closing:
                    return
                candidate, self._candidate = self._candidate, None
                snapshot, self._snapshot = self._snapshot, None
                if candidate is not None:
                    self._spec_busy = candidate[1]
                    self._spec_done.clear()
            try:
                if candidate is not None:       # speculative final beats live update
                    audio, voiced = candidate
                    try:
                        text, segs = self._transcribe(self.final_model, audio)
                        self._spec_result = (voiced, text, segs)
                    finally:
                        self._spec_busy = None
                        self._spec_done.set()
                elif snapshot is not None:
                    audio, voiced = snapshot
                    text, _ = self._transcribe(self.live_model, audio)
                    if text and text != self._last_partial:
                        self._last_partial = text
                        if self.on_partial:
                            self.on_partial(text, False)
            except Exception as exc:
                print(f"⚠️  live STT error: {exc}")

    # -- called from the main loop after capture returns ------------------------
    def finalize(self, audio, voiced):
        """Final transcript for the captured utterance (reuses speculation)."""
        with self._cv:                          # stop pending live work
            self._snapshot = self._candidate = None
        if self._spec_busy == voiced:           # speculation in flight: await it
            self._spec_done.wait(timeout=15)
        spec = self._spec_result
        self._spec_result = None
        self._last_partial = ""
        if spec is not None and spec[0] == voiced:
            text, segs = spec[1], spec[2]
        else:
            text, segs = self._transcribe(self.final_model, audio)
        if self.on_partial:
            self.on_partial(text, True)
        return text, segs

    def close(self):
        with self._cv:
            self._closing = True
            self._cv.notify()


# ---------------------------------------------------------------------------
# Text-to-speech
# ---------------------------------------------------------------------------
class Speaker:
    """
    Streaming, interruptible text-to-speech.

    `feed(text)` is non-blocking: a synth thread renders sentences to audio
    while a playback thread plays them through sounddevice in ~46 ms chunks,
    checking the interrupt event between chunks and fading out when set. The
    playback queue means sentence N+1 is synthesized while N is playing.

    `spoken` collects the sentences that were actually played to completion,
    so conversation history can reflect what the user really heard.
    """

    PLAY_CHUNK = 1024

    def __init__(self, engine=None):
        self.engine = engine or os.environ.get("VA_TTS", "coqui")
        self._coqui = None
        self._sr = 22050
        if self.engine == "coqui":
            self._try_load_coqui()
        if self.engine == "espeak" and shutil.which("espeak-ng") is None:
            print("⚠️  No TTS backend available; replies will be text-only.")
            self.engine = "none"

        self._synth_q = None
        self._play_q = None
        self._interrupt = None
        self._threads = []
        self._stream = None
        self._stream_sr = None
        self._playing = threading.Event()
        self._out_device = _pick_device("VA_OUTPUT_DEVICE", "output")
        self._tmpdir = tempfile.mkdtemp(prefix="va_tts_")
        atexit.register(shutil.rmtree, self._tmpdir, ignore_errors=True)
        self.spoken = []

    # Curated VCTK speakers that sound natural (multi-speaker model).
    GENDER_SPEAKERS = {"female": "p225", "male": "p226"}

    def _try_load_coqui(self):
        try:
            os.environ.setdefault("COQUI_TOS_AGREED", "1")
            import torch
            from TTS.api import TTS
            gender = os.environ.get("VA_VOICE_GENDER", "female").lower()
            self._speaker_id = os.environ.get(
                "VA_SPEAKER_ID", self.GENDER_SPEAKERS.get(gender, "p225"))
            # VCTK is a multi-speaker human-recorded model: real male & female voices.
            model = os.environ.get("VA_COQUI_MODEL", "tts_models/en/vctk/vits")
            self._coqui = TTS(model_name=model, progress_bar=False)
            if torch.cuda.is_available():
                self._coqui.to("cuda")
            self._multi = bool(getattr(self._coqui, "speakers", None))
            self._sr = getattr(self._coqui.synthesizer, "output_sample_rate", 22050)
            voice_desc = f", voice: {gender} ({self._speaker_id})" if self._multi else ""
            print(f"🗣️  TTS: Coqui '{model}'{voice_desc}"
                  f"{' (GPU)' if torch.cuda.is_available() else ''}")
        except Exception as exc:
            print(f"⚠️  Coqui TTS unavailable ({type(exc).__name__}); "
                  f"falling back to espeak-ng.")
            self.engine = "espeak" if shutil.which("espeak-ng") else "none"

    # --- synthesis -----------------------------------------------------------
    @staticmethod
    def _detect_espeak_voice(text):
        """Route non-Latin scripts to a matching espeak-ng voice (multilingual out)."""
        for ch in text:
            o = ord(ch)
            if 0x0900 <= o <= 0x097F:      # Devanagari (Nepali/Hindi)
                return os.environ.get("VA_VOICE_NE", "ne")
            if 0x4E00 <= o <= 0x9FFF:
                return "cmn"                # Chinese
            if 0x0600 <= o <= 0x06FF:
                return "ar"                 # Arabic
            if 0x0400 <= o <= 0x04FF:
                return "ru"                 # Cyrillic
        return None

    def _espeak_wav(self, text, voice):
        path = os.path.join(self._tmpdir, "espeak.wav")
        subprocess.run(["espeak-ng", "-v", voice, "-w", path, text],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        sr, wav = wavfile.read(path)
        if wav.ndim > 1:
            wav = wav[:, 0]
        return wav.astype(np.float32) / 32768.0, sr

    def _synth(self, text):
        """Render text to (float32 mono audio, sample_rate), or None."""
        # Non-Latin text (e.g. Nepali) goes to espeak-ng in the right language —
        # the English Coqui voice can't pronounce Devanagari at all.
        other = self._detect_espeak_voice(text)
        if other and shutil.which("espeak-ng"):
            return self._espeak_wav(text, other)
        if self.engine == "coqui":
            kwargs = {"speaker": self._speaker_id} if getattr(self, "_multi", False) else {}
            wav = np.asarray(self._coqui.tts(text, **kwargs), dtype=np.float32)
            peak = float(np.max(np.abs(wav))) or 1.0
            return wav * (0.9 / peak), self._sr
        if self.engine == "espeak":
            return self._espeak_wav(text, VOICE)
        return None

    # --- streaming playback ---------------------------------------------------
    def begin(self, interrupt):
        self._interrupt = interrupt
        self._synth_q = queue.Queue()
        self._play_q = queue.Queue(maxsize=4)
        self.spoken = []
        self._threads = [threading.Thread(target=self._synth_loop, daemon=True),
                         threading.Thread(target=self._play_loop, daemon=True)]
        for t in self._threads:
            t.start()

    def feed(self, text):
        """Queue one sentence for speech. Non-blocking."""
        text = text.strip()
        if not text or self.engine == "none" or self._synth_q is None:
            return
        if self._interrupt is not None and self._interrupt.is_set():
            return
        self._synth_q.put(text)

    def end(self):
        """Signal end of reply and wait for playback to drain (interruptibly)."""
        if self._synth_q is not None:
            self._synth_q.put(None)
        for t in self._threads:
            t.join()
        self._threads = []
        self._synth_q = self._play_q = None

    def is_playing(self):
        """True while audio is actually coming out of the speakers."""
        return self._playing.is_set()

    def _interrupted(self):
        return self._interrupt is not None and self._interrupt.is_set()

    def _synth_loop(self):
        while True:
            text = self._synth_q.get()
            if text is None:
                self._play_q.put(None)
                return
            if self._interrupted():
                continue
            try:
                item = self._synth(text)
                if item is not None:
                    self._play_q.put((text, item[0], item[1]))
            except Exception as exc:
                print(f"⚠️  TTS synth failed: {exc}")

    def _ensure_stream(self, sr):
        if self._stream is not None and self._stream_sr != sr:
            self._stream.close()
            self._stream = None
        if self._stream is None:
            self._stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32",
                                           device=self._out_device)
            self._stream_sr = sr
        if self._stream.stopped:
            self._stream.start()

    def _play_loop(self):
        try:
            while True:
                got = self._play_q.get()
                if got is None:
                    break
                text, wav, sr = got
                if self._interrupted():
                    continue                    # drain queue without playing
                self._ensure_stream(sr)
                self._playing.set()
                pos, n = 0, len(wav)
                clean = True
                while pos < n:
                    if self._interrupted():     # fade out instead of clicking off
                        fade = min(self.PLAY_CHUNK, n - pos)
                        self._stream.write(wav[pos:pos + fade]
                                           * np.linspace(1.0, 0.0, fade, dtype=np.float32))
                        clean = False
                        break
                    chunk = wav[pos:pos + self.PLAY_CHUNK]
                    self._stream.write(np.ascontiguousarray(chunk))
                    pos += len(chunk)
                self._playing.clear()
                if clean:
                    self.spoken.append(text)
        except Exception as exc:
            print(f"⚠️  audio playback failed: {exc}")
        finally:
            self._playing.clear()
            if self._stream is not None and not self._stream.stopped:
                try:
                    self._stream.stop()         # drains, then releases the device
                except Exception:
                    pass

    # --- convenience -----------------------------------------------------------
    def say_blocking(self, text, interrupt=None):
        """Speak a single string end-to-end (used for short fixed phrases)."""
        self.begin(interrupt or threading.Event())
        self.feed(text)
        self.end()

    def close(self):
        if self._stream is not None:
            self._stream.close()
            self._stream = None


# ---------------------------------------------------------------------------
# Input gain guard
# ---------------------------------------------------------------------------
class InputGain:
    """
    Protects speech recognition from a clipping microphone. A mic driven into
    the rails (this machine shipped with +60 dB of boost!) turns speech into
    garbage no model can transcribe. When captured audio clips, step the
    system source volume down via pactl. Disable with VA_AUTO_GAIN=0.
    """

    MIN_PERCENT = 20

    def __init__(self):
        self.enabled = (os.environ.get("VA_AUTO_GAIN", "1") != "0"
                        and shutil.which("pactl") is not None)
        self._steps = 0

    @staticmethod
    def _current_percent():
        out = subprocess.run(["pactl", "get-source-volume", "@DEFAULT_SOURCE@"],
                             capture_output=True, text=True, timeout=5).stdout
        m = re.search(r"(\d+)%", out)
        return int(m.group(1)) if m else None

    def check(self, audio):
        """Call with each captured utterance; lowers input volume if it clipped."""
        if not self.enabled or self._steps >= 5:
            return
        clipped = float(np.mean(np.abs(audio) > 0.98))
        if clipped < 0.02:
            return
        try:
            cur = self._current_percent()
            if cur is None or cur <= self.MIN_PERCENT:
                return
            new = max(self.MIN_PERCENT, cur - 15)
            subprocess.run(["pactl", "set-source-volume", "@DEFAULT_SOURCE@", f"{new}%"],
                           capture_output=True, timeout=5)
            self._steps += 1
            print(f"🔉 Mic was clipping ({clipped:.0%} of samples) — "
                  f"lowered input volume {cur}% → {new}%.")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# System echo cancellation (optional, PipeWire/PulseAudio)
# ---------------------------------------------------------------------------
class EchoCancel:
    """
    Loads the system `module-echo-cancel` and routes default input/output
    through it, so the assistant's own voice is subtracted from the mic signal
    before we ever see it — barge-in then works cleanly with loudspeakers.
    Everything is restored on exit.
    """

    def __init__(self):
        self._module_id = None
        self._old_source = None
        self._old_sink = None

    @staticmethod
    def _pactl(*args):
        return subprocess.run(["pactl", *args], capture_output=True, text=True, timeout=10)

    def setup(self):
        if shutil.which("pactl") is None:
            print("⚠️  pactl not found; echo cancellation unavailable.")
            return False
        try:
            self._old_source = self._pactl("get-default-source").stdout.strip()
            self._old_sink = self._pactl("get-default-sink").stdout.strip()
            res = self._pactl("load-module", "module-echo-cancel",
                              "source_name=va_ec_src", "sink_name=va_ec_sink")
            if res.returncode != 0:
                print(f"⚠️  Could not load echo-cancel module: {res.stderr.strip()}")
                return False
            self._module_id = res.stdout.strip()
            self._pactl("set-default-source", "va_ec_src")
            self._pactl("set-default-sink", "va_ec_sink")
            atexit.register(self.restore)
            print("🔇 Echo cancellation active (module-echo-cancel).")
            return True
        except Exception as exc:
            print(f"⚠️  Echo cancellation setup failed: {exc}")
            self.restore()
            return False

    def restore(self):
        try:
            if self._old_source:
                self._pactl("set-default-source", self._old_source)
            if self._old_sink:
                self._pactl("set-default-sink", self._old_sink)
            if self._module_id:
                self._pactl("unload-module", self._module_id)
        except Exception:
            pass
        self._module_id = self._old_source = self._old_sink = None


class WakeWordListener:
    """
    Hands-free activation: listens on the shared AudioEngine for a wake word
    ("hey jarvis" by default, via openWakeWord) and returns when it hears it.

    Frames from the engine are float32 at 16 kHz in FRAME_SIZE chunks; the
    model wants int16 in 1280-sample chunks, so we accumulate and convert.
    """

    def __init__(self, phrase=None, threshold=None):
        import openwakeword as _oww
        from openwakeword.model import Model as _Model
        self.phrase = (phrase or os.environ.get("VA_WAKE_WORD", "hey_jarvis")).replace(" ", "_")
        self.threshold = float(threshold or os.environ.get("VA_WAKE_THRESHOLD", "0.5"))
        base = os.path.join(os.path.dirname(_oww.__file__), "resources", "models")
        path = os.path.join(base, f"{self.phrase}_v0.1.onnx")
        if not os.path.exists(path):
            avail = [f[:-len("_v0.1.onnx")] for f in os.listdir(base) if f.endswith("_v0.1.onnx")]
            raise FileNotFoundError(
                f"No wake-word model '{self.phrase}'. Available: {', '.join(avail)}")
        self.model = _Model(wakeword_model_paths=[path])
        self._buf = np.empty(0, dtype=np.float32)

    def wait(self, engine, stop=None):
        """Block until the wake word is heard. Returns True, or False if `stop` set."""
        self.model.reset()
        self._buf = np.empty(0, dtype=np.float32)
        streak = 0
        while True:
            if stop is not None and stop.is_set():
                return False
            frame = engine.read(timeout=0.5)
            if frame is None:
                continue
            self._buf = np.concatenate([self._buf, frame])
            while len(self._buf) >= 1280:
                chunk, self._buf = self._buf[:1280], self._buf[1280:]
                pcm = np.clip(chunk * 32768, -32768, 32767).astype(np.int16)
                scores = self.model.predict(pcm)
                if max(scores.values()) >= self.threshold:
                    streak += 1
                    if streak >= 1:            # one confident hit is enough
                        engine.flush()          # drop the wake word itself
                        return True
                else:
                    streak = 0
