#!/usr/bin/env bash
# Install the voice assistant as an always-on user service (starts at login,
# listens for "hey jarvis" in the background).
set -e
cd "$(dirname "$0")"

mkdir -p ~/.config/systemd/user ~/.local/share
cp voice-assistant.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable voice-assistant.service

echo "✅ Installed. Commands:"
echo "   systemctl --user start voice-assistant     # start now"
echo "   systemctl --user stop voice-assistant      # stop"
echo "   systemctl --user disable voice-assistant   # don't start at login"
echo "   tail -f ~/.local/share/voice-assistant.log # watch what it's doing"
