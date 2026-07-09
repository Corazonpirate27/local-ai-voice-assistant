"""
Agent tools the voice assistant can call: open apps, control volume/brightness,
and run shell commands (the last one is gated behind a confirmation callback).

Each tool returns a short human-readable string that is fed back to the LLM (and
can be spoken). Everything degrades gracefully if a system utility is missing.
"""

import glob
import os
import shutil
import subprocess

DEVNULL = subprocess.DEVNULL

# Friendly spoken names → actual launch commands.
_APP_ALIASES = {
    "browser": "firefox", "web browser": "firefox", "internet": "firefox",
    "chrome": "google-chrome", "files": "nautilus", "file manager": "nautilus",
    "terminal": "gnome-terminal", "calculator": "gnome-calculator",
    "text editor": "gedit", "editor": "gedit", "code": "code",
    "settings": "gnome-control-center", "music": "rhythmbox",
}


def _run(cmd, timeout=30):
    """Run a command, return its trimmed output (or a short status)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = (r.stdout or r.stderr or "").strip()
        return out if out else ("done" if r.returncode == 0 else f"exit {r.returncode}")
    except subprocess.TimeoutExpired:
        return "the command timed out"
    except Exception as exc:
        return f"error: {exc}"


_DESKTOP_DIRS = [
    "/usr/share/applications",
    os.path.expanduser("~/.local/share/applications"),
    "/var/lib/flatpak/exports/share/applications",
    os.path.expanduser("~/.local/share/flatpak/exports/share/applications"),
    "/var/lib/snapd/desktop/applications",
]


def _desktop_name(path):
    """Read the Name= field from a .desktop file (lowercased)."""
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("Name="):
                    return line[5:].strip().lower()
    except Exception:
        pass
    return ""


def _find_app_launch(name):
    """Resolve an app name to a real launch command, or (None, None).

    Checks, in order: a binary on PATH, an installed .desktop file (incl. Flatpak
    and Snap exports), and Flatpak application IDs. Returns (argv, label)."""
    key = name.lower().strip()
    if shutil.which(key):
        return [key], key

    best = None  # (score, launch_argv, label)
    for d in _DESKTOP_DIRS:
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.endswith(".desktop"):
                continue
            did = fn[:-len(".desktop")]
            nm = _desktop_name(os.path.join(d, fn))
            exact = key == did.lower() or key == nm
            if exact or key in did.lower() or (nm and key in nm):
                launcher = (["gtk-launch", did] if shutil.which("gtk-launch")
                            else ["gio", "launch", os.path.join(d, fn)])
                score = 0 if exact else 1
                if best is None or score < best[0]:
                    best = (score, launcher, did)
    if best:
        return best[1], best[2]

    if shutil.which("flatpak"):
        try:
            out = subprocess.run(
                ["flatpak", "list", "--app", "--columns=application,name"],
                capture_output=True, text=True, timeout=10).stdout
            for line in out.splitlines():
                parts = line.split("\t")
                appid = parts[0].strip()
                nm = (parts[1].strip().lower() if len(parts) > 1 else "")
                if key in appid.lower() or (nm and key in nm):
                    return ["flatpak", "run", appid], appid
        except Exception:
            pass
    return None, None


def open_app(name):
    name = (name or "").strip()
    if not name:
        return "No app name was given."
    target = _APP_ALIASES.get(name.lower(), name)
    path = os.path.expanduser(target)
    try:
        if target.startswith(("http://", "https://")) or os.path.exists(path):
            subprocess.Popen(["xdg-open", path], stdout=DEVNULL, stderr=DEVNULL,
                             start_new_session=True)
            return f"Opening {name}."
        argv, label = _find_app_launch(target)
        if argv is None:
            return (f"I couldn't find an app called '{name}' installed on this "
                    f"computer, so I did not open anything.")
        subprocess.Popen(argv, stdout=DEVNULL, stderr=DEVNULL, start_new_session=True)
        return f"Opening {name}."
    except Exception as exc:
        return f"I found {name} but couldn't launch it: {exc}"


def set_volume(action="set", level=None):
    sink = "@DEFAULT_SINK@"
    if not shutil.which("pactl"):
        return "Volume control isn't available on this system."
    if action == "set" and level is not None:
        lvl = max(0, min(100, int(level)))
        _run(["pactl", "set-sink-volume", sink, f"{lvl}%"])
        return f"Volume set to {lvl} percent."
    if action == "up":
        _run(["pactl", "set-sink-volume", sink, "+10%"])
        return "Volume up."
    if action == "down":
        _run(["pactl", "set-sink-volume", sink, "-10%"])
        return "Volume down."
    if action == "mute":
        _run(["pactl", "set-sink-mute", sink, "1"])
        return "Muted."
    if action == "unmute":
        _run(["pactl", "set-sink-mute", sink, "0"])
        return "Unmuted."
    return "I couldn't change the volume."


def set_brightness(percent):
    p = max(1, min(100, int(percent)))
    if shutil.which("brightnessctl"):
        _run(["brightnessctl", "set", f"{p}%"])
        return f"Brightness set to {p} percent."
    backlights = glob.glob("/sys/class/backlight/*")
    if backlights:
        base = backlights[0]
        try:
            with open(f"{base}/max_brightness") as f:
                mx = int(f.read())
            with open(f"{base}/brightness", "w") as f:
                f.write(str(max(1, int(mx * p / 100))))
            return f"Brightness set to {p} percent."
        except PermissionError:
            return "I don't have permission to change the brightness."
        except Exception as exc:
            return f"I couldn't change the brightness: {exc}"
    return "Brightness control isn't available on this system."


def run_shell(command):
    if not command or not command.strip():
        return "No command was given."
    return _run(["bash", "-lc", command])


# --- documents (read-only) --------------------------------------------------
DOC_ROOT = os.path.realpath(os.path.expanduser(os.environ.get("VA_DOCS_ROOT", "~")))
_DOC_EXTS = {".txt", ".md", ".pdf", ".docx", ".rtf", ".csv", ".log", ".json",
             ".py", ".js", ".ts", ".html", ".css", ".sh", ".c", ".cpp", ".java"}
_SKIP_DIRS = {"node_modules", "venv", ".venv", "__pycache__", ".cache", ".git",
              ".local", ".config", ".mozilla", "site-packages"}
_MAX_READ_CHARS = 6000


def _doc_root(folder):
    """Resolve a folder inside DOC_ROOT (defends against escaping the home root)."""
    if folder:
        p = os.path.expanduser(folder)
        if not os.path.isabs(p):
            p = os.path.join(DOC_ROOT, folder)
        p = os.path.realpath(p)
        if os.path.isdir(p) and os.path.commonpath([p, DOC_ROOT]) == DOC_ROOT:
            return p
    return DOC_ROOT


def _rel(path):
    try:
        return os.path.relpath(path, DOC_ROOT)
    except ValueError:
        return path


def _extract_text(path):
    """Return the text of a document, or '' if it can't be read."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            if shutil.which("pdftotext"):
                r = subprocess.run(["pdftotext", "-q", path, "-"],
                                   capture_output=True, text=True, timeout=30)
                return r.stdout or ""
            return ""
        if ext == ".docx":
            import re
            import zipfile
            with zipfile.ZipFile(path) as z:
                xml = z.read("word/document.xml").decode("utf-8", "ignore")
            xml = xml.replace("</w:p>", "\n")
            return re.sub(r"<[^>]+>", "", xml)
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def find_documents(query, folder=None, limit=20):
    """Find documents whose file name contains `query`."""
    root, q, hits = _doc_root(folder), (query or "").lower(), []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if not d.startswith(".") and d not in _SKIP_DIRS]
        for fn in sorted(fns):
            if q in fn.lower() and os.path.splitext(fn)[1].lower() in _DOC_EXTS:
                hits.append(_rel(os.path.join(dp, fn)))
                if len(hits) >= limit:
                    break
        if len(hits) >= limit:
            break
    if not hits:
        return f"No documents with '{query}' in the name were found under {_rel(root)}."
    return "Found:\n" + "\n".join(f"- {h}" for h in hits)


def search_documents(text, folder=None, limit=8, max_files=400):
    """Search document *contents* for `text` and return files with a snippet."""
    root, needle, results, scanned = _doc_root(folder), (text or "").lower(), [], 0
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if not d.startswith(".") and d not in _SKIP_DIRS]
        for fn in sorted(fns):
            if os.path.splitext(fn)[1].lower() not in _DOC_EXTS:
                continue
            if scanned >= max_files or len(results) >= limit:
                break
            scanned += 1
            path = os.path.join(dp, fn)
            content = _extract_text(path)
            i = content.lower().find(needle)
            if i >= 0:
                snip = content[max(0, i - 60):i + 100].replace("\n", " ").strip()
                results.append(f"- {_rel(path)}: …{snip}…")
        if scanned >= max_files or len(results) >= limit:
            break
    if not results:
        return f"No documents containing '{text}' were found under {_rel(root)}."
    return f"Documents mentioning '{text}':\n" + "\n".join(results)


_SEARCH_READ_TOP = int(os.environ.get("VA_SEARCH_READ", "1"))  # pages to auto-open


def _tavily_search(query, max_results=6):
    """High-quality search via the Tavily API (returns a synthesized answer +
    clean source content). Used only if VA_TAVILY_KEY is set; else returns None."""
    key = os.environ.get("VA_TAVILY_KEY")
    if not key:
        return None
    try:
        import httpx
        r = httpx.post("https://api.tavily.com/search", timeout=25, json={
            "api_key": key, "query": query, "search_depth": "advanced",
            "include_answer": True, "max_results": max_results})
        data = r.json()
    except Exception:
        return None
    parts = []
    if data.get("answer"):
        parts.append("Answer: " + data["answer"])
    lines = []
    for it in data.get("results", []):
        content = (it.get("content", "") or "")[:400]
        lines.append(f"- {it.get('title', '')}: {content} ({it.get('url', '')})")
    if lines:
        parts.append("Sources:\n" + "\n".join(lines))
    if not parts:
        return None
    return (f"Search results for '{query}' (current, as of today):\n\n"
            + "\n\n".join(parts))


def web_search(query, max_results=6, read_top=None):
    """Search the web, then auto-open the top result pages and include their text so
    the model can answer from real content. Uses the Tavily API when VA_TAVILY_KEY is
    set (best quality), otherwise DuckDuckGo."""
    if not query or not query.strip():
        return "No search query was given."
    tavily = _tavily_search(query, max_results)
    if tavily:
        return tavily
    try:
        from ddgs import DDGS
    except ImportError:
        return "Web search isn't available (the 'ddgs' package is not installed)."
    try:
        results = list(DDGS().text(query, max_results=max_results))
    except Exception as exc:
        results = []

    read_top = _SEARCH_READ_TOP if read_top is None else read_top
    parts = []

    # Dated news results are the best signal for "recent/latest" questions.
    try:
        news = list(DDGS().news(query, max_results=8))
    except Exception:
        news = []
    if not results and not news:
        return f"No web results found for '{query}'."
    if news:
        nlines = []
        for n in news:
            date = (n.get("date", "") or "")[:10]
            nlines.append(f"- [{date}] {n.get('title', '').strip()} — "
                          f"{n.get('body', '').strip()} ({n.get('source', '')})")
        parts.append("Recent news (newest first):\n" + "\n".join(nlines))

    lines, pages = [], []
    for i, r in enumerate(results):
        title = r.get("title", "").strip()
        body = r.get("body", "").strip()
        href = r.get("href", "").strip()
        lines.append(f"- {title}: {body} ({href})")
        if i < read_top and href:
            content = fetch_webpage(href, max_chars=1500)
            if content and not content.startswith(
                    ("Couldn't fetch", "Please provide", "The page had")):
                pages.append(f"\n--- Page text: {title} ({href}) ---\n{content}")
    if lines:
        parts.append("Web results:\n" + "\n".join(lines))
    if pages:
        parts.append("".join(pages))

    return (f"Search results for '{query}' (these are current, as of today):\n\n"
            + "\n\n".join(parts))


def fetch_webpage(url, max_chars=1500):
    """Fetch a web page and return its readable main text (nav/menus removed)."""
    if not url or not url.startswith(("http://", "https://")):
        return "Please provide a full http(s) URL."
    try:
        import httpx
        r = httpx.get(url, timeout=12, follow_redirects=True,
                      headers={"User-Agent": "Mozilla/5.0"})
        raw = r.text
    except Exception as exc:
        return f"Couldn't fetch the page: {exc}"
    import html as _html
    import re
    # Drop non-content regions entirely (this is what removes the menu junk).
    raw = re.sub(r"(?is)<(script|style|head|nav|header|footer|aside|form|noscript)\b.*?</\1>",
                 " ", raw)
    # Turn block-level ends into line breaks so we can filter line-by-line.
    raw = re.sub(r"(?i)</(p|div|li|h[1-6]|tr|section|article)\s*>|<br\s*/?>", "\n", raw)
    text = _html.unescape(re.sub(r"(?s)<[^>]+>", " ", raw))
    lines = []
    for ln in text.splitlines():
        ln = re.sub(r"\s+", " ", ln).strip()
        # Keep sentence-like/content lines; drop short nav labels.
        if len(ln) >= 40 or (len(ln) > 15 and ln.endswith((".", "!", "?", ":"))):
            lines.append(ln)
    cleaned = "\n".join(lines).strip()
    if not cleaned:
        return "The page had no readable text."
    return cleaned[:max_chars] + ("…(truncated)" if len(cleaned) > max_chars else "")


def read_document(path):
    """Read a document's text content (truncated) so the model can answer about it."""
    p = os.path.expanduser(path)
    if not os.path.isabs(p):
        p = os.path.join(DOC_ROOT, path)
    p = os.path.realpath(p)
    if os.path.commonpath([p, DOC_ROOT]) != DOC_ROOT:
        return "That file is outside the allowed documents folder."
    if not os.path.isfile(p):
        return f"No file found at {path}."
    text = _extract_text(p).strip()
    if not text:
        return f"I couldn't extract any text from {_rel(p)}."
    if len(text) > _MAX_READ_CHARS:
        text = text[:_MAX_READ_CHARS] + "\n…(truncated)"
    return f"Content of {_rel(p)}:\n{text}"


# --- weather (Open-Meteo, free, no API key) ---------------------------------
_WEATHER_CODES = {
    0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "rime fog", 51: "light drizzle", 53: "drizzle",
    55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "heavy freezing rain", 71: "light snow",
    73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light showers",
    81: "showers", 82: "violent showers", 85: "snow showers",
    86: "heavy snow showers", 95: "thunderstorm", 96: "thunderstorm with hail",
    99: "thunderstorm with heavy hail",
}


def get_weather(city):
    """Current weather + today's range for a city, via Open-Meteo (no key)."""
    if not city or not city.strip():
        return "No city was given."
    try:
        import httpx
        geo = httpx.get("https://geocoding-api.open-meteo.com/v1/search",
                        params={"name": city, "count": 1}, timeout=15).json()
        hits = geo.get("results") or []
        if not hits:
            return f"I couldn't find a city called '{city}'."
        g = hits[0]
        wx = httpx.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": g["latitude"], "longitude": g["longitude"],
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                       "weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": "auto", "forecast_days": 1}, timeout=15).json()
        c, d = wx.get("current", {}), wx.get("daily", {})
        cond = _WEATHER_CODES.get(c.get("weather_code", -1), "unknown conditions")
        place = f"{g['name']}, {g.get('country', '')}".strip(", ")
        out = (f"Weather in {place}: {c.get('temperature_2m')}°C "
               f"(feels like {c.get('apparent_temperature')}°C), {cond}, "
               f"humidity {c.get('relative_humidity_2m')}%, "
               f"wind {c.get('wind_speed_10m')} km/h.")
        if d.get("temperature_2m_max"):
            out += (f" Today: {d['temperature_2m_min'][0]}–{d['temperature_2m_max'][0]}°C, "
                    f"{d['precipitation_probability_max'][0]}% chance of rain.")
        return out
    except Exception as exc:
        return f"Couldn't get the weather: {exc}"


# --- time & timezone (exact math in Python — never let the LLM guess) --------
_TZ_ALIASES = {
    "nepal": "Asia/Kathmandu", "kathmandu": "Asia/Kathmandu",
    "india": "Asia/Kolkata", "new york": "America/New_York",
    "usa": "America/New_York", "us eastern": "America/New_York",
    "us pacific": "America/Los_Angeles", "los angeles": "America/Los_Angeles",
    "london": "Europe/London", "uk": "Europe/London", "japan": "Asia/Tokyo",
    "tokyo": "Asia/Tokyo", "china": "Asia/Shanghai", "germany": "Europe/Berlin",
    "france": "Europe/Paris", "australia": "Australia/Sydney",
    "sydney": "Australia/Sydney", "dubai": "Asia/Dubai", "utc": "UTC",
}


def _resolve_tz(name):
    from zoneinfo import ZoneInfo, available_timezones
    key = (name or "").strip().lower()
    if not key or key == "local":
        return None  # local time
    if key in _TZ_ALIASES:
        return ZoneInfo(_TZ_ALIASES[key])
    for tz in available_timezones():
        if key.replace(" ", "_") in tz.lower():
            return ZoneInfo(tz)
    return "unknown"


def get_time(timezone_name="local", convert_time_str=None, from_timezone=None):
    """Current time in a timezone, or convert a clock time between timezones."""
    from datetime import datetime
    tz = _resolve_tz(timezone_name)
    if tz == "unknown":
        return f"I don't know the timezone '{timezone_name}'."
    if convert_time_str:
        src = _resolve_tz(from_timezone or "local")
        if src == "unknown":
            return f"I don't know the timezone '{from_timezone}'."
        try:
            import re as _re
            m = _re.match(r"(\d{1,2}):(\d{2})\s*(am|pm)?", convert_time_str.strip().lower())
            if not m:
                return "Please give the time as HH:MM (optionally am/pm)."
            h, mnt = int(m.group(1)), int(m.group(2))
            if m.group(3) == "pm" and h < 12:
                h += 12
            if m.group(3) == "am" and h == 12:
                h = 0
            base = datetime.now(src).replace(hour=h, minute=mnt, second=0)
            conv = base.astimezone(tz)
            return (f"{convert_time_str} in {from_timezone or 'local time'} is "
                    f"{conv.strftime('%I:%M %p on %A, %B %d')} in {timezone_name} "
                    f"({conv.tzname()}).")
        except Exception as exc:
            return f"Couldn't convert that time: {exc}"
    now = datetime.now(tz)
    label = timezone_name if timezone_name != "local" else "local time"
    return (f"It is {now.strftime('%I:%M %p')} ({now.tzname()}, UTC{now.strftime('%z')[:3]}:"
            f"{now.strftime('%z')[3:]}) on {now.strftime('%A, %B %d, %Y')} in {label}.")


# --- reminders & timers -------------------------------------------------------
import threading as _threading
import time as _time

_reminders = []            # [{"at": epoch, "text": str, "timer": Timer}]
_alert_cb = [None]         # set by the assistant: called with the reminder text


def set_alert_callback(cb):
    """The assistant registers how alerts are delivered (spoken + printed)."""
    _alert_cb[0] = cb


def set_reminder(minutes=None, text="", seconds=None):
    """Set a timer/reminder that alerts (spoken) when it fires."""
    total = (float(minutes or 0)) * 60 + float(seconds or 0)
    if total <= 0:
        return "Please give a duration, like 10 minutes."
    text = text.strip() or "your reminder"
    fire_at = _time.time() + total

    def fire():
        _reminders[:] = [r for r in _reminders if r["at"] != fire_at]
        msg = f"Reminder: {text}"
        try:
            notify_desktop("⏰ Reminder", text)      # desktop popup
        except Exception:
            pass
        cb = _alert_cb[0]
        if cb:
            cb(msg)
        else:
            print(f"\n🔔 {msg}")

    t = _threading.Timer(total, fire)
    t.daemon = True
    t.start()
    _reminders.append({"at": fire_at, "text": text, "timer": t})
    mins, secs = divmod(int(total), 60)
    dur = f"{mins} minute{'s' if mins != 1 else ''}" if mins else f"{secs} seconds"
    if mins and secs:
        dur += f" {secs} seconds"
    return f"Okay — I'll remind you about '{text}' in {dur}."


def list_reminders():
    if not _reminders:
        return "There are no active reminders."
    now = _time.time()
    lines = []
    for r in sorted(_reminders, key=lambda r: r["at"]):
        left = max(0, int(r["at"] - now))
        lines.append(f"- '{r['text']}' in {left // 60}m {left % 60}s")
    return "Active reminders:\n" + "\n".join(lines)


def cancel_reminders():
    n = len(_reminders)
    for r in _reminders:
        r["timer"].cancel()
    _reminders.clear()
    return f"Cancelled {n} reminder{'s' if n != 1 else ''}."


# --- media control via MPRIS (works with Spotify, Firefox, VLC, ...) ----------
def _mpris_players():
    out = _run(["gdbus", "call", "--session", "--dest", "org.freedesktop.DBus",
                "--object-path", "/org/freedesktop/DBus",
                "--method", "org.freedesktop.DBus.ListNames"], timeout=10)
    import re as _re
    return _re.findall(r"org\.mpris\.MediaPlayer2\.[\w.]+", out)


def _mpris_pick(player=None):
    players = _mpris_players()
    if not players:
        return None
    if player:
        for p in players:
            if player.lower() in p.lower():
                return p
    for p in players:                      # prefer spotify when present
        if "spotify" in p.lower():
            return p
    return players[0]


def media_control(action, player=None):
    """play/pause/next/previous/stop, or 'nowplaying' to see the current track."""
    dest = _mpris_pick(player)
    if not dest:
        return "No media player is running."
    label = dest.rsplit(".", 2)[-2] if dest.count(".") > 3 else dest.split(".")[-1]
    if action == "nowplaying":
        out = _run(["gdbus", "call", "--session", "--dest", dest,
                    "--object-path", "/org/mpris/MediaPlayer2",
                    "--method", "org.freedesktop.DBus.Properties.Get",
                    "org.mpris.MediaPlayer2.Player", "Metadata"], timeout=10)
        import re as _re
        title = _re.search(r"'xesam:title': <'([^']*)'>", out)
        artist = _re.search(r"'xesam:artist': <\['([^']*)'", out)
        if not title:
            return f"Nothing seems to be playing on {label}."
        s = f"Now playing on {label}: {title.group(1)}"
        if artist:
            s += f" by {artist.group(1)}"
        return s + "."
    method = {"play": "Play", "pause": "Pause", "toggle": "PlayPause",
              "next": "Next", "previous": "Previous", "stop": "Stop"}.get(action)
    if not method:
        return f"Unknown media action '{action}'."
    _run(["gdbus", "call", "--session", "--dest", dest,
          "--object-path", "/org/mpris/MediaPlayer2",
          "--method", f"org.mpris.MediaPlayer2.Player.{method}"], timeout=10)
    return f"{method} sent to {label}."


# --- persistent memory ---------------------------------------------------------
MEMORY_FILE = os.path.expanduser(os.environ.get("VA_MEMORY_FILE", "~/.voice_assistant_memory.json"))


def _load_memory():
    import json
    try:
        with open(MEMORY_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_memory(items):
    import json
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=1)
    except Exception:
        pass


def remember_fact(fact):
    """Store a fact about the user permanently (survives restarts)."""
    fact = (fact or "").strip()
    if not fact:
        return "Nothing to remember."
    items = _load_memory()
    if any(fact.lower() == i.get("fact", "").lower() for i in items):
        return "I already have that noted."
    from datetime import date
    items.append({"fact": fact, "on": str(date.today())})
    _save_memory(items[-100:])            # keep at most 100 facts
    return f"Got it — I'll remember: {fact}"


def forget_fact(matching):
    """Delete stored facts containing the given text."""
    items = _load_memory()
    keep = [i for i in items if (matching or "").lower() not in i.get("fact", "").lower()]
    removed = len(items) - len(keep)
    _save_memory(keep)
    return f"Forgot {removed} thing{'s' if removed != 1 else ''}." if removed \
        else "I couldn't find anything matching that."


def memory_summary():
    """All stored facts as a block for the system prompt ('' if none)."""
    items = _load_memory()
    if not items:
        return ""
    return "\n".join(f"- {i['fact']} (noted {i.get('on', '?')})" for i in items)


# --- long-document Q&A (RAG: chunk + embed + retrieve) ---------------------------
EMBED_MODEL = os.environ.get("VA_EMBED_MODEL", "nomic-embed-text")
_RAG_CACHE_DIR = os.path.expanduser("~/.cache/voice_assistant_rag")


def _chunk_text(text, size=1200, overlap=200):
    chunks, i = [], 0
    while i < len(text):
        chunk = text[i:i + size]
        # try to break at a sentence/paragraph boundary near the end
        if i + size < len(text):
            cut = max(chunk.rfind("\n\n"), chunk.rfind(". "))
            if cut > size // 2:
                chunk = chunk[:cut + 1]
        chunks.append(chunk.strip())
        i += max(1, len(chunk) - overlap)
    return [c for c in chunks if c]


def _embed_many(texts):
    import ollama
    r = ollama.embed(model=EMBED_MODEL, input=texts)
    import numpy as _np
    v = _np.array(r["embeddings"], dtype=_np.float32)
    return v / (_np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


def _rag_index(path):
    """Chunk + embed a document, cached by (path, mtime)."""
    import hashlib
    import numpy as _np
    import pickle
    os.makedirs(_RAG_CACHE_DIR, exist_ok=True)
    key = hashlib.sha1(f"{path}:{os.path.getmtime(path)}:{EMBED_MODEL}".encode()).hexdigest()
    cache = os.path.join(_RAG_CACHE_DIR, key + ".pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    text = _extract_text(path)
    if not text.strip():
        return None
    chunks = _chunk_text(text)
    if len(chunks) > 400:                 # cap embedding cost on huge files
        chunks = chunks[:400]
    vecs = _embed_many(chunks)
    index = {"chunks": chunks, "vecs": vecs}
    with open(cache, "wb") as f:
        pickle.dump(index, f)
    return index


def ask_document(path, question, top_k=4):
    """Answer questions about a (possibly long) document: retrieves the most
    relevant passages by semantic similarity and returns them as context."""
    p = os.path.expanduser(path or "")
    if not os.path.isabs(p):
        p = os.path.join(DOC_ROOT, path)
    p = os.path.realpath(p)
    if os.path.commonpath([p, DOC_ROOT]) != DOC_ROOT:
        return "That file is outside the allowed documents folder."
    if not os.path.isfile(p):
        return f"No file found at {path}."
    try:
        index = _rag_index(p)
    except Exception as exc:
        return (f"Couldn't index the document ({exc}). Is the embedding model "
                f"installed? Try: ollama pull {EMBED_MODEL}")
    if not index:
        return f"I couldn't extract any text from {_rel(p)}."
    import numpy as _np
    qv = _embed_many([question or "summary of this document"])[0]
    sims = index["vecs"] @ qv
    order = _np.argsort(-sims)[:top_k]
    picked = [f"[passage {int(i) + 1}, relevance {sims[int(i)]:.2f}]\n{index['chunks'][int(i)]}"
              for i in order]
    total = len(index["chunks"])
    return (f"Most relevant passages from {_rel(p)} ({total} chunks indexed) for "
            f"'{question}':\n\n" + "\n\n".join(picked)
            + "\n\nAnswer the user's question using these passages.")


# --- OCR, clipboard, system status ------------------------------------------------
def read_text_from_image(path):
    """Extract the text in an image (screenshot, photo of a page). Uses tesseract
    OCR when available, falling back to the vision model."""
    p = os.path.realpath(os.path.expanduser(path or ""))
    if not os.path.isfile(p):
        return f"No image found at {path}."
    if shutil.which("tesseract"):
        out = _run(["tesseract", p, "-", "--psm", "3"], timeout=30)
        if out and out not in ("done",) and len(out.strip()) > 3:
            return f"Text found in {os.path.basename(p)}:\n{out.strip()[:4000]}"
    try:
        return _describe_with_vision(
            p, "Transcribe ALL text visible in this image, exactly as written.")
    except Exception as exc:
        return f"Couldn't read text from the image: {exc}"


def get_clipboard():
    """Read the current clipboard contents (Wayland/X11)."""
    for cmd in (["wl-paste", "--no-newline"], ["xclip", "-selection", "clipboard", "-o"]):
        if shutil.which(cmd[0]):
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                text = (r.stdout or "").strip()
                if text:
                    return f"Clipboard contents:\n{text[:4000]}"
                return "The clipboard is empty (or holds non-text content)."
            except Exception:
                continue
    return "No clipboard tool available."


def system_status():
    """Battery, disk, memory and CPU load summary."""
    parts = []
    for bat in glob.glob("/sys/class/power_supply/BAT*"):
        try:
            with open(f"{bat}/capacity") as f:
                cap = f.read().strip()
            with open(f"{bat}/status") as f:
                st = f.read().strip()
            parts.append(f"Battery: {cap}% ({st.lower()})")
        except OSError:
            pass
    try:
        du = shutil.disk_usage(os.path.expanduser("~"))
        parts.append(f"Disk: {du.free // 1024**3} GB free of {du.total // 1024**3} GB")
    except OSError:
        pass
    try:
        with open("/proc/meminfo") as f:
            mem = {l.split(":")[0]: int(l.split()[1]) for l in f if ":" in l}
        parts.append(f"Memory: {mem['MemAvailable'] // 1024} MB free of "
                     f"{mem['MemTotal'] // 1024} MB")
    except Exception:
        pass
    try:
        load = os.getloadavg()[0]
        parts.append(f"CPU load: {load:.1f}")
    except OSError:
        pass
    return ". ".join(parts) if parts else "Couldn't read system status."


# --- flashcards with spaced repetition (SM-2-style) --------------------------------
CARDS_FILE = os.path.expanduser(os.environ.get("VA_CARDS_FILE", "~/.voice_assistant_cards.json"))
_INTERVALS = [1, 3, 7, 16, 35, 90]      # days per consecutive success


def _load_cards():
    import json
    try:
        with open(CARDS_FILE, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, list) else []
    except Exception:
        return []


def _save_cards(cards):
    import json
    try:
        with open(CARDS_FILE, "w", encoding="utf-8") as f:
            json.dump(cards, f, ensure_ascii=False, indent=1)
    except Exception:
        pass


def flashcard_add(front, back, subject="general"):
    """Create a flashcard. The tutor calls this for things the student got wrong."""
    from datetime import date
    if not front or not back:
        return "A flashcard needs both a front (question) and a back (answer)."
    cards = _load_cards()
    if any(c["front"].lower() == front.lower() for c in cards):
        return "A card with that question already exists."
    cards.append({"id": max((c["id"] for c in cards), default=0) + 1,
                  "front": front, "back": back, "subject": subject.lower(),
                  "level": 0, "due": str(date.today())})
    _save_cards(cards)
    return f"Flashcard #{cards[-1]['id']} added ({subject})."


def flashcards_due(subject=None):
    """List the cards due for review today (id, subject and question only)."""
    from datetime import date
    today = str(date.today())
    cards = [c for c in _load_cards() if c["due"] <= today
             and (not subject or c["subject"] == subject.lower())]
    if not cards:
        return "No flashcards are due for review. 🎉"
    lines = [f"- card {c['id']} [{c['subject']}]: {c['front']}" for c in cards[:20]]
    return f"{len(cards)} card(s) due:\n" + "\n".join(lines)


def flashcard_grade(card_id, result):
    """Record a review result ('pass' or 'fail') and reschedule the card."""
    from datetime import date, timedelta
    cards = _load_cards()
    for c in cards:
        if c["id"] == int(card_id):
            if str(result).lower() in ("pass", "correct", "good", "easy", "yes"):
                c["level"] = min(c["level"] + 1, len(_INTERVALS) - 1)
            else:
                c["level"] = 0
            days = _INTERVALS[c["level"]]
            c["due"] = str(date.today() + timedelta(days=days))
            _save_cards(cards)
            return (f"Card {card_id}: {'advanced' if c['level'] else 'reset'} — "
                    f"next review in {days} day{'s' if days != 1 else ''}.")
    return f"No card with id {card_id}."


def flashcard_answer(card_id):
    """Reveal the back (answer) of a flashcard."""
    for c in _load_cards():
        if c["id"] == int(card_id):
            return f"Answer to card {card_id}: {c['back']}"
    return f"No card with id {card_id}."


# --- learning: code runner, verified math, progress ------------------------------
import sys as _sys


def run_python(code, timeout=15):
    """Execute a Python snippet in a fresh interpreter and return its output.
    Used for programming practice: exercises actually run, errors are real."""
    if not code or not code.strip():
        return "No code was given."
    try:
        r = subprocess.run(
            [_sys.executable, "-I", "-c", code],   # -I: isolated, no site/user paths
            capture_output=True, text=True, timeout=timeout,
            cwd=tempfile.gettempdir() if (tempfile := __import__("tempfile")) else None)
        out = (r.stdout or "").strip()
        err = (r.stderr or "").strip()
        result = ""
        if out:
            result += f"Output:\n{out[:2000]}"
        if err:
            result += f"\nErrors:\n{err[:1500]}"
        if not result:
            result = "The code ran with no output." if r.returncode == 0 \
                else f"Exited with code {r.returncode} and no output."
        return result
    except subprocess.TimeoutExpired:
        return f"The code did not finish within {timeout} seconds (infinite loop?)."
    except Exception as exc:
        return f"Couldn't run the code: {exc}"


def calculate(expression, operation="evaluate", variable="x"):
    """Exact math via sympy: evaluate, solve, differentiate, integrate, simplify,
    expand, factor, limit. The tutor explains steps; sympy guarantees the answer."""
    if not expression or not expression.strip():
        return "No expression was given."
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                                implicit_multiplication_application)
        trans = standard_transformations + (implicit_multiplication_application,)
        x = sp.Symbol(variable)
        expression = expression.replace("^", "**")
        if "=" in expression and operation in ("solve", "evaluate"):
            lhs, rhs = expression.split("=", 1)
            eq = sp.Eq(parse_expr(lhs, transformations=trans),
                       parse_expr(rhs, transformations=trans))
            sols = sp.solve(eq, x)
            return f"Solutions for {variable}: {sols}"
        expr = parse_expr(expression, transformations=trans)
        if operation == "solve":
            return f"Solutions for {variable}: {sp.solve(expr, x)}"
        if operation == "differentiate":
            return f"d/d{variable} ({expr}) = {sp.diff(expr, x)}"
        if operation == "integrate":
            return f"∫({expr}) d{variable} = {sp.integrate(expr, x)} + C"
        if operation == "simplify":
            return f"Simplified: {sp.simplify(expr)}"
        if operation == "expand":
            return f"Expanded: {sp.expand(expr)}"
        if operation == "factor":
            return f"Factored: {sp.factor(expr)}"
        if operation == "limit":
            return f"Limit as {variable}→0: {sp.limit(expr, x, 0)}; as {variable}→∞: {sp.limit(expr, x, sp.oo)}"
        val = expr.evalf()
        exact = sp.nsimplify(expr, rational=False)
        return f"{expression} = {val}" + (f" (exact: {exact})" if str(exact) != str(val) else "")
    except Exception as exc:
        return f"Couldn't compute that: {exc}"


LEARNING_FILE = os.path.expanduser(
    os.environ.get("VA_LEARNING_FILE", "~/.voice_assistant_learning.json"))


def _load_learning():
    import json
    try:
        with open(LEARNING_FILE, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def save_progress(subject, covered="", weak_spots="", next_topic=""):
    """Record where a study session ended so the next one can resume."""
    import json
    from datetime import date
    subject = (subject or "general").strip().lower()
    data = _load_learning()
    entry = data.setdefault(subject, {})
    if covered:
        entry["covered"] = (entry.get("covered", "") + "; " + covered).strip("; ")[-800:]
    if weak_spots:
        entry["weak"] = weak_spots
    if next_topic:
        entry["next"] = next_topic
    entry["on"] = str(date.today())
    try:
        with open(LEARNING_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
    except Exception:
        pass
    return f"Progress saved for {subject}."


def learning_summary(subject=None):
    """Progress notes for one subject (or all) — injected into the tutor prompt."""
    data = _load_learning()
    if not data:
        return ""
    subjects = {subject.lower(): data[subject.lower()]} if subject and subject.lower() in data else data
    lines = []
    for name, e in subjects.items():
        bits = [f"{name}: last studied {e.get('on', '?')}"]
        if e.get("covered"):
            bits.append(f"covered: {e['covered'][-300:]}")
        if e.get("weak"):
            bits.append(f"weak spots: {e['weak']}")
        if e.get("next"):
            bits.append(f"planned next: {e['next']}")
        lines.append(" — ".join(bits))
    return "\n".join(lines)


# --- desktop notifications ------------------------------------------------------
def notify_desktop(title, body=""):
    """Pop a desktop notification (best-effort; silent no-op if unavailable)."""
    if shutil.which("notify-send"):
        subprocess.run(["notify-send", "-a", "Voice Assistant", "-i",
                        "audio-input-microphone", title, body],
                       capture_output=True, timeout=10)
        return True
    return False


# --- notes, lists & todos --------------------------------------------------------
LISTS_FILE = os.path.expanduser(os.environ.get("VA_LISTS_FILE", "~/.voice_assistant_lists.json"))


def _load_lists():
    import json
    try:
        with open(LISTS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_lists(data):
    import json
    try:
        with open(LISTS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
    except Exception:
        pass


def _norm_list_name(name):
    n = (name or "").strip().lower()
    n = n.replace("my ", "").replace(" list", "").strip()
    return n or "notes"


def list_add(list_name, item):
    """Add an item to a named list (shopping, todo, notes, ...)."""
    item = (item or "").strip()
    if not item:
        return "Nothing to add."
    data = _load_lists()
    key = _norm_list_name(list_name)
    items = data.setdefault(key, [])
    if any(item.lower() == i.lower() for i in items):
        return f"'{item}' is already on the {key} list."
    items.append(item)
    _save_lists(data)
    return f"Added '{item}' to the {key} list ({len(items)} item{'s' if len(items) != 1 else ''})."


def list_show(list_name=None):
    """Show a list's items, or all lists when no name is given."""
    data = _load_lists()
    if not data:
        return "You don't have any lists yet."
    if list_name:
        key = _norm_list_name(list_name)
        items = data.get(key)
        if not items:
            return f"The {key} list is empty or doesn't exist. Lists: {', '.join(data)}."
        return f"{key.capitalize()} list:\n" + "\n".join(
            f"{i + 1}. {it}" for i, it in enumerate(items))
    return "Your lists:\n" + "\n".join(
        f"- {k} ({len(v)} items)" for k, v in data.items())


def list_remove(list_name, item):
    """Remove an item (by text match or number) from a named list."""
    data = _load_lists()
    key = _norm_list_name(list_name)
    items = data.get(key, [])
    if not items:
        return f"The {key} list is empty."
    target = (item or "").strip()
    removed = None
    if target.isdigit() and 1 <= int(target) <= len(items):
        removed = items.pop(int(target) - 1)
    else:
        for i, it in enumerate(items):
            if target.lower() in it.lower():
                removed = items.pop(i)
                break
    if removed is None:
        return f"I couldn't find '{item}' on the {key} list."
    if items:
        data[key] = items
    else:
        data.pop(key, None)
    _save_lists(data)
    return f"Removed '{removed}' from the {key} list."


def list_clear(list_name):
    """Delete a whole list."""
    data = _load_lists()
    key = _norm_list_name(list_name)
    if key not in data:
        return f"There is no {key} list."
    n = len(data.pop(key))
    _save_lists(data)
    return f"Cleared the {key} list ({n} items)."


# --- daily briefing ---------------------------------------------------------------
BRIEFING_CITY = os.environ.get("VA_CITY", "Kathmandu")


def daily_briefing(city=None):
    """Compose a morning briefing: date, weather, reminders, todos, headlines."""
    from datetime import datetime
    parts = [f"Good {'morning' if datetime.now().hour < 12 else 'afternoon' if datetime.now().hour < 17 else 'evening'}! "
             f"It's {datetime.now().strftime('%A, %B %d, %Y, %I:%M %p')}."]

    wx = get_weather(city or BRIEFING_CITY)
    if not wx.startswith(("Couldn't", "I couldn't", "No city")):
        parts.append(wx)

    rem = list_reminders()
    if not rem.startswith("There are no"):
        parts.append(rem)

    todos = _load_lists().get("todo") or _load_lists().get("todos")
    if todos:
        parts.append("Your todo list: " + "; ".join(todos[:6]) +
                     ("…" if len(todos) > 6 else ""))

    due = flashcards_due()
    if not due.startswith("No flashcards"):
        n = due.split(" ")[0]
        parts.append(f"📇 You have {n} flashcards due for review — say 'review my "
                     f"flashcards' when ready.")

    try:
        from ddgs import DDGS
        news = list(DDGS().news("top world news today", max_results=4))
        if news:
            parts.append("Top headlines:\n" + "\n".join(
                f"- {n.get('title', '').strip()}" for n in news))
    except Exception:
        pass

    return "\n\n".join(parts)


# --- vision: screenshots & image description (gemma3 multimodal) ---------------
VISION_MODEL = os.environ.get("VA_VISION_MODEL", "gemma3:4b")


def _portal_screenshot(path, timeout_s=20):
    """Take a screenshot via xdg-desktop-portal (GNOME Wayland-sanctioned).
    May show a one-time permission dialog. Returns True on success."""
    import re as _re
    import uuid as _uuid
    token = "va" + _uuid.uuid4().hex[:8]
    mon = subprocess.Popen(
        ["gdbus", "monitor", "--session", "--dest", "org.freedesktop.portal.Desktop"],
        stdout=subprocess.PIPE, text=True)
    try:
        _time.sleep(0.3)
        subprocess.run(
            ["gdbus", "call", "--session",
             "--dest", "org.freedesktop.portal.Desktop",
             "--object-path", "/org/freedesktop/portal/desktop",
             "--method", "org.freedesktop.portal.Screenshot.Screenshot",
             "", "{'handle_token': <'%s'>, 'interactive': <false>}" % token],
            capture_output=True, text=True, timeout=10)
        import select
        deadline = _time.time() + timeout_s
        while _time.time() < deadline:
            ready, _, _ = select.select([mon.stdout], [], [], 0.5)
            if not ready:
                continue
            line = mon.stdout.readline()
            if "Response" in line and "uri" in line:
                m = _re.search(r"'uri': <'file://([^']+)'>", line)
                if m and os.path.exists(m.group(1)):
                    shutil.copyfile(m.group(1), path)
                    try:
                        os.unlink(m.group(1))
                    except OSError:
                        pass
                    return True
                return False
        return False
    except Exception:
        return False
    finally:
        mon.terminate()


def _take_screenshot(path):
    """Try every available screenshot method; True on success."""
    for cmd in (["gnome-screenshot", "-f", path],
                ["spectacle", "-b", "-n", "-o", path],
                ["grim", path]):
        if shutil.which(cmd[0]):
            subprocess.run(cmd, capture_output=True, timeout=15)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return True
    return _portal_screenshot(path)


def _describe_with_vision(image_path, question):
    import ollama
    r = ollama.chat(model=VISION_MODEL, messages=[{
        "role": "user",
        "content": question or "Describe what you see in this image, concisely.",
        "images": [image_path]}])
    return (r.get("message", {}) or {}).get("content", "").strip() \
        or "The vision model returned nothing."


def look_at_screen(question=""):
    """Screenshot the screen and describe it / answer a question about it."""
    import tempfile as _tempfile
    path = os.path.join(_tempfile.gettempdir(), "va_screen.png")
    if not _take_screenshot(path):
        return ("I couldn't capture the screen. On GNOME Wayland a permission "
                "dialog may need to be accepted, or install gnome-screenshot.")
    try:
        return _describe_with_vision(path, question)
    except Exception as exc:
        return f"Screenshot taken but vision failed: {exc}"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def describe_image(path, question=""):
    """Describe or answer a question about a local image file."""
    p = os.path.realpath(os.path.expanduser(path or ""))
    if not os.path.isfile(p):
        return f"No image found at {path}."
    try:
        return _describe_with_vision(p, question)
    except Exception as exc:
        return f"Couldn't analyze the image: {exc}"


# Tool schemas advertised to the LLM (Ollama function-calling format).
TOOLS = [
    {"type": "function", "function": {
        "name": "open_app",
        "description": "Open or launch an application, file, or website on the computer.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string",
                     "description": "App/file/URL, e.g. 'firefox', 'files', 'calculator'"}},
            "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "set_volume",
        "description": "Change the system audio volume.",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string",
                       "enum": ["set", "up", "down", "mute", "unmute"]},
            "level": {"type": "integer", "description": "Percent 0-100 for action 'set'"}},
            "required": ["action"]}}},
    {"type": "function", "function": {
        "name": "set_brightness",
        "description": "Change the screen brightness.",
        "parameters": {"type": "object", "properties": {
            "percent": {"type": "integer", "description": "Brightness percent 0-100"}},
            "required": ["percent"]}}},
    {"type": "function", "function": {
        "name": "run_shell",
        "description": ("Run a shell command on this Linux computer for tasks the "
                        "other tools don't cover. The user is asked to confirm first."),
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "The shell command to run"}},
            "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "find_documents",
        "description": "Find local documents/files by name (searches the user's folders).",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Part of the file name to look for"},
            "folder": {"type": "string", "description": "Optional subfolder, e.g. 'Documents'"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "search_documents",
        "description": "Search inside local documents for text and return matching files.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "Text/keyword to find inside documents"},
            "folder": {"type": "string", "description": "Optional subfolder, e.g. 'Documents'"}},
            "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "read_document",
        "description": "Read the text content of a local document (txt, md, pdf, docx, code).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path or name of the file to read"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the internet for current information and return top results.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What to search the web for"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "fetch_webpage",
        "description": "Fetch and read the text of a specific web page URL.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "description": "Full http(s) URL to read"}},
            "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather and today's forecast for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name, e.g. 'Kathmandu'"}},
            "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "get_time",
        "description": ("Get the exact current time/date in any timezone or convert a "
                        "clock time between timezones. ALWAYS use this for time or "
                        "timezone questions — do not compute offsets yourself."),
        "parameters": {"type": "object", "properties": {
            "timezone_name": {"type": "string",
                              "description": "Target timezone/city/country, e.g. 'Nepal', 'Tokyo', 'local'"},
            "convert_time_str": {"type": "string",
                                 "description": "Optional clock time to convert, e.g. '20:00' or '8:00 pm'"},
            "from_timezone": {"type": "string",
                              "description": "Source timezone when converting, e.g. 'UK'"}},
            "required": ["timezone_name"]}}},
    {"type": "function", "function": {
        "name": "set_reminder",
        "description": "Set a timer or reminder that alerts the user when it fires.",
        "parameters": {"type": "object", "properties": {
            "minutes": {"type": "number", "description": "Minutes from now"},
            "seconds": {"type": "number", "description": "Extra seconds from now"},
            "text": {"type": "string", "description": "What to remind about"}},
            "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "list_reminders",
        "description": "List the active reminders and timers.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "cancel_reminders",
        "description": "Cancel all active reminders and timers.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "media_control",
        "description": ("Control music/video playback (Spotify, browser, etc.): play, "
                        "pause, toggle, next, previous, stop, or 'nowplaying' to ask "
                        "what's playing."),
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string",
                       "enum": ["play", "pause", "toggle", "next", "previous",
                                "stop", "nowplaying"]},
            "player": {"type": "string", "description": "Optional player name, e.g. 'spotify'"}},
            "required": ["action"]}}},
    {"type": "function", "function": {
        "name": "remember_fact",
        "description": ("Permanently remember a fact about the user (name, preferences, "
                        "projects). Use when the user shares personal info or says "
                        "'remember that ...'."),
        "parameters": {"type": "object", "properties": {
            "fact": {"type": "string", "description": "The fact to store"}},
            "required": ["fact"]}}},
    {"type": "function", "function": {
        "name": "forget_fact",
        "description": "Delete remembered facts that contain the given text.",
        "parameters": {"type": "object", "properties": {
            "matching": {"type": "string", "description": "Text identifying what to forget"}},
            "required": ["matching"]}}},
    {"type": "function", "function": {
        "name": "look_at_screen",
        "description": ("Take a screenshot of the user's screen and describe it or "
                        "answer a question about what is visible."),
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string",
                         "description": "Optional question about the screen contents"}}}}},
    {"type": "function", "function": {
        "name": "describe_image",
        "description": "Look at a local image file and describe it or answer a question about it.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path to the image file"},
            "question": {"type": "string", "description": "Optional question about the image"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "list_add",
        "description": "Add an item to a named list (shopping, todo, notes, ideas...).",
        "parameters": {"type": "object", "properties": {
            "list_name": {"type": "string", "description": "Which list, e.g. 'shopping'"},
            "item": {"type": "string", "description": "The item to add"}},
            "required": ["list_name", "item"]}}},
    {"type": "function", "function": {
        "name": "list_show",
        "description": "Show a named list's items, or all lists if no name given.",
        "parameters": {"type": "object", "properties": {
            "list_name": {"type": "string", "description": "Optional list name"}}}}},
    {"type": "function", "function": {
        "name": "list_remove",
        "description": "Remove an item from a named list (by text or item number).",
        "parameters": {"type": "object", "properties": {
            "list_name": {"type": "string"},
            "item": {"type": "string", "description": "Item text or its number"}},
            "required": ["list_name", "item"]}}},
    {"type": "function", "function": {
        "name": "list_clear",
        "description": "Delete an entire named list.",
        "parameters": {"type": "object", "properties": {
            "list_name": {"type": "string"}}, "required": ["list_name"]}}},
    {"type": "function", "function": {
        "name": "daily_briefing",
        "description": ("Give the user their daily briefing: date/time, weather, "
                        "reminders, todos, and top news headlines. Use when they say "
                        "'good morning', 'brief me', 'what's my day look like'."),
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "Optional city for weather"}}}}},
    {"type": "function", "function": {
        "name": "ask_document",
        "description": ("Answer a question about a specific (possibly long) local "
                        "document using semantic search over its full content. Better "
                        "than read_document for large files like books or long PDFs."),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path to the document"},
            "question": {"type": "string", "description": "The question to answer"}},
            "required": ["path", "question"]}}},
    {"type": "function", "function": {
        "name": "notify_desktop",
        "description": "Show a desktop notification popup to the user.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"}, "body": {"type": "string"}},
            "required": ["title"]}}},
    {"type": "function", "function": {
        "name": "run_python",
        "description": ("Execute Python code and return its real output/errors. Use "
                        "to demonstrate code, test the user's exercise solutions, or "
                        "verify a program actually works."),
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string", "description": "The Python code to run"}},
            "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "calculate",
        "description": ("Exact math via a computer algebra system. ALWAYS use this to "
                        "verify arithmetic, solve equations, differentiate, integrate, "
                        "simplify, expand, factor — never trust mental math."),
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "e.g. '2*x**2-8=0' or 'sin(x)*x^2'"},
            "operation": {"type": "string",
                          "enum": ["evaluate", "solve", "differentiate", "integrate",
                                   "simplify", "expand", "factor", "limit"]},
            "variable": {"type": "string", "description": "Variable name, default x"}},
            "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "read_text_from_image",
        "description": "OCR: extract the exact text from an image or screenshot file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path to the image"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "get_clipboard",
        "description": ("Read the user's current clipboard. Use when they say "
                        "'my clipboard', 'what I copied', 'this error' etc."),
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "system_status",
        "description": "Battery, disk space, memory and CPU load of this computer.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "flashcard_add",
        "description": ("Create a spaced-repetition flashcard. Tutors: call this for "
                        "each concept the student answered wrongly or found hard."),
        "parameters": {"type": "object", "properties": {
            "front": {"type": "string", "description": "The question/prompt"},
            "back": {"type": "string", "description": "The answer"},
            "subject": {"type": "string"}}, "required": ["front", "back"]}}},
    {"type": "function", "function": {
        "name": "flashcards_due",
        "description": "List flashcards due for review today.",
        "parameters": {"type": "object", "properties": {
            "subject": {"type": "string", "description": "Optional subject filter"}}}}},
    {"type": "function", "function": {
        "name": "flashcard_answer",
        "description": "Reveal the correct answer of a flashcard by id.",
        "parameters": {"type": "object", "properties": {
            "card_id": {"type": "integer"}}, "required": ["card_id"]}}},
    {"type": "function", "function": {
        "name": "flashcard_grade",
        "description": ("Record whether the student got a flashcard right ('pass') or "
                        "wrong ('fail'); reschedules it by spaced repetition."),
        "parameters": {"type": "object", "properties": {
            "card_id": {"type": "integer"},
            "result": {"type": "string", "enum": ["pass", "fail"]}},
            "required": ["card_id", "result"]}}},
    {"type": "function", "function": {
        "name": "save_progress",
        "description": ("Save the student's study progress (topics covered, weak "
                        "spots, what to do next) so the next lesson can resume. Call "
                        "at natural checkpoints and when a lesson ends."),
        "parameters": {"type": "object", "properties": {
            "subject": {"type": "string"},
            "covered": {"type": "string", "description": "Topics covered this session"},
            "weak_spots": {"type": "string", "description": "What the student struggles with"},
            "next_topic": {"type": "string", "description": "Where to resume next time"}},
            "required": ["subject"]}}},
]


def dispatch(name, args, confirm=None):
    """Execute a tool by name. `confirm(command)->bool` gates run_shell."""
    args = args or {}
    if name == "open_app":
        return open_app(args.get("name", ""))
    if name == "set_volume":
        return set_volume(args.get("action", "set"), args.get("level"))
    if name == "set_brightness":
        return set_brightness(args.get("percent", 50))
    if name == "run_shell":
        command = args.get("command", "")
        if confirm is not None and not confirm(command):
            return "The user declined to run that command."
        return run_shell(command)
    if name == "find_documents":
        return find_documents(args.get("query", ""), args.get("folder"))
    if name == "search_documents":
        return search_documents(args.get("text", ""), args.get("folder"))
    if name == "read_document":
        return read_document(args.get("path", ""))
    if name == "web_search":
        return web_search(args.get("query", ""))
    if name == "fetch_webpage":
        return fetch_webpage(args.get("url", ""))
    if name == "get_weather":
        return get_weather(args.get("city", ""))
    if name == "get_time":
        return get_time(args.get("timezone_name", "local"),
                        args.get("convert_time_str"), args.get("from_timezone"))
    if name == "set_reminder":
        return set_reminder(args.get("minutes"), args.get("text", ""),
                            args.get("seconds"))
    if name == "list_reminders":
        return list_reminders()
    if name == "cancel_reminders":
        return cancel_reminders()
    if name == "media_control":
        return media_control(args.get("action", ""), args.get("player"))
    if name == "remember_fact":
        return remember_fact(args.get("fact", ""))
    if name == "forget_fact":
        return forget_fact(args.get("matching", ""))
    if name == "look_at_screen":
        return look_at_screen(args.get("question", ""))
    if name == "describe_image":
        return describe_image(args.get("path", ""), args.get("question", ""))
    if name == "list_add":
        return list_add(args.get("list_name", "notes"), args.get("item", ""))
    if name == "list_show":
        return list_show(args.get("list_name"))
    if name == "list_remove":
        return list_remove(args.get("list_name", "notes"), args.get("item", ""))
    if name == "list_clear":
        return list_clear(args.get("list_name", ""))
    if name == "daily_briefing":
        return daily_briefing(args.get("city"))
    if name == "ask_document":
        return ask_document(args.get("path", ""), args.get("question", ""))
    if name == "notify_desktop":
        return ("Notification shown." if notify_desktop(args.get("title", ""),
                                                        args.get("body", ""))
                else "Desktop notifications aren't available.")
    if name == "run_python":
        return run_python(args.get("code", ""))
    if name == "calculate":
        return calculate(args.get("expression", ""), args.get("operation", "evaluate"),
                         args.get("variable", "x") or "x")
    if name == "save_progress":
        return save_progress(args.get("subject", ""), args.get("covered", ""),
                             args.get("weak_spots", ""), args.get("next_topic", ""))
    if name == "read_text_from_image":
        return read_text_from_image(args.get("path", ""))
    if name == "get_clipboard":
        return get_clipboard()
    if name == "system_status":
        return system_status()
    if name == "flashcard_add":
        return flashcard_add(args.get("front", ""), args.get("back", ""),
                             args.get("subject", "general"))
    if name == "flashcards_due":
        return flashcards_due(args.get("subject"))
    if name == "flashcard_answer":
        return flashcard_answer(args.get("card_id", 0))
    if name == "flashcard_grade":
        return flashcard_grade(args.get("card_id", 0), args.get("result", "fail"))
    return f"Unknown tool: {name}."
