#!/usr/bin/env bash
# Official Cambridge 21 Test 4 IELTS Speaking Mock Examiner (David)
# Push-to-Talk + Dynamic Conversational AI (Qwen2.5:3b + Whisper + Edge-TTS)

# Ensure clean direct internet connection for Edge-TTS (bypass any stale local proxies)
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY="*"
export no_proxy="*"
export SSL_CERT_FILE="/home/corazonpc/Projects/voice-assistant/.venv/lib/python3.12/site-packages/certifi/cacert.pem"
export REQUESTS_CA_BUNDLE="/home/corazonpc/Projects/voice-assistant/.venv/lib/python3.12/site-packages/certifi/cacert.pem"

# Check if Ollama is running; if not, launch it
if ! curl -s --noproxy "*" http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "⚡ Launching Ollama server for real-time dynamic Part 3 examiner..."
    nohup /home/corazonpc/.local/bin/ollama serve >/tmp/ollama_mock.log 2>&1 &
    sleep 2
fi

cd /home/corazonpc/Projects/voice-assistant || exit 1
exec /home/corazonpc/Projects/voice-assistant/.venv/bin/python /home/corazonpc/Projects/voice-assistant/ielts_examiner.py
