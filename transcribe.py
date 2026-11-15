"""One-shot transcription demo. Loads on GPU only if there's free VRAM, else CPU,
so it works even while the Ollama LLM is holding the GPU."""
import whisper
import torch

device = "cpu"
if torch.cuda.is_available():
    try:
        free, _ = torch.cuda.mem_get_info()
        device = "cuda" if free > 1.3 * 1024 ** 3 else "cpu"
    except Exception:
        device = "cuda"

print(f"Loading Whisper 'small' on {device}...")
model = whisper.load_model("small", device=device)
result = model.transcribe("test.wav", fp16=False)
print(result["text"])
