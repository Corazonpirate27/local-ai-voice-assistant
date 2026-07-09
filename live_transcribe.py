import sounddevice as sd
import scipy.io.wavfile as wavfile
import whisper

SAMPLE_RATE = 16000  # Whisper's native sample rate

def record_audio():
    print("Press Enter to start recording...")
    input()
    print("🎙️  Recording... press Enter to stop.")

    recording = []

    def callback(indata, frames, time, status):
        recording.append(indata.copy())

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback)
    with stream:
        input()  # blocks here until you press Enter again

    print("⏹️  Stopped recording.")

    import numpy as np
    audio = np.concatenate(recording, axis=0)
    wavfile.write("mic_input.wav", SAMPLE_RATE, audio)
    return "mic_input.wav"


def main():
    print("Loading Whisper model...")
    model = whisper.load_model("small")
    print("Model loaded.\n")

    audio_file = record_audio()

    print("Transcribing...")
    result = model.transcribe(audio_file, fp16=False)
    user_text = result["text"]
    print("\nYou said:", user_text)

    print("\nThinking...")
    import ollama
    response = ollama.chat(
        model="gemma3:4b",
        messages=[{"role": "user", "content": user_text}]
    )
    print("\nAssistant:", response["message"]["content"])

if __name__ == "__main__":
    main()


