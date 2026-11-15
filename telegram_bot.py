"""
Telegram bridge: chat with your assistant from your phone.

Same brain, memory, lists, reminders, web search and tools as the terminal —
over Telegram. Shell commands are always declined (no way to confirm remotely).

Setup (one time):
  1. In Telegram, message @BotFather → /newbot → copy the token.
  2. Put in ~/.config/voice-assistant/config.toml:
        telegram_token = "123456:ABC-..."
  3. Run:  ./venv/bin/python telegram_bot.py
  4. Message your bot. The first message prints your chat id here — add it:
        telegram_chat_id = 123456789
     and restart. From then on only YOUR account is answered.

Run it forever: make a systemd user service whose ExecStart runs this script
(copy voice-assistant.service and change ExecStart to telegram_bot.py).
"""

import os
import time

import assistant as A          # triggers config load; reuses the agent machinery

TOKEN = os.environ.get("VA_TELEGRAM_TOKEN", "")
ALLOWED = os.environ.get("VA_TELEGRAM_CHAT_ID", "")
API = f"https://api.telegram.org/bot{TOKEN}"


class SilentSpeaker:
    def begin(self, ev):
        pass

    def feed(self, t):
        pass

    def end(self):
        pass

    def say_blocking(self, t):
        pass


def send(client, chat_id, text):
    for i in range(0, len(text), 3800):          # Telegram message size limit
        client.post(f"{API}/sendMessage", timeout=30,
                    json={"chat_id": chat_id, "text": text[i:i + 3800]})


def main():
    import httpx

    if not TOKEN:
        print("❌ No token. Put telegram_token = \"...\" in "
              "~/.config/voice-assistant/config.toml (see file docstring).")
        return

    model = A.LLM_MODEL
    if not A.supports_tools(model):
        model = A.pick_tool_model(A.available_models()) or model
    sys_prompt = A.agent_system_prompt() + (
        " You are talking over Telegram: keep replies compact, plain text only.")
    histories = {}
    speaker = SilentSpeaker()
    client = httpx.Client()

    me = client.get(f"{API}/getMe", timeout=30).json()
    if not me.get("ok"):
        print(f"❌ Telegram rejected the token: {me}")
        return
    print(f"📱 Bot @{me['result']['username']} online — model '{model}'. "
          f"{'Locked to chat ' + ALLOWED if ALLOWED else '⚠️  No chat id set: will only pair, not chat.'}")

    offset = 0
    while True:
        try:
            r = client.get(f"{API}/getUpdates", timeout=70,
                           params={"timeout": 60, "offset": offset}).json()
        except Exception:
            time.sleep(3)
            continue
        for upd in r.get("result", []):
            offset = upd["update_id"] + 1
            msg = upd.get("message") or {}
            chat_id = str(msg.get("chat", {}).get("id", ""))
            text = (msg.get("text") or "").strip()
            if not chat_id or not text:
                continue
            if not ALLOWED:
                print(f"👋 Pairing request from chat id {chat_id} "
                      f"({msg.get('chat', {}).get('first_name', '?')}). Add "
                      f"telegram_chat_id = {chat_id} to config.toml and restart.")
                send(client, chat_id,
                     "Hi! Pairing: your chat id was printed on the computer. "
                     "Add it to the config and restart the bot.")
                continue
            if chat_id != ALLOWED:
                continue                          # ignore strangers silently
            print(f"📩 {text[:80]}")
            hist = histories.setdefault(chat_id,
                                        [{"role": "system", "content": sys_prompt}])
            hist.append({"role": "user", "content": text})
            try:
                reply = A.agent_respond(hist, speaker, model,
                                        confirm_shell=lambda cmd: False)
            except Exception as exc:
                reply = f"(error: {exc})"
                hist.pop()
            send(client, chat_id, reply or "…")
            if len(hist) > 40:                    # keep phone chats bounded
                histories[chat_id] = [hist[0]] + hist[-20:]


if __name__ == "__main__":
    main()
