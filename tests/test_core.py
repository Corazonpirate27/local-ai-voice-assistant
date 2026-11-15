"""Core-logic tests — no audio, no network, no Ollama.

Run:  ./venv/bin/python -m pytest tests/ -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import assistant as A
import tools as T


# --- streaming text machinery -------------------------------------------------
def test_stream_sentences_reassembles_fragments():
    toks = ["Hel", "lo the", "re. How ", "are you", "? I am ", "fine", "!", " Bye"]
    assert list(A.stream_sentences(iter(toks))) == [
        "Hello there.", "How are you?", "I am fine!", "Bye"]


def test_strip_think_removes_reasoning():
    toks = ["<th", "ink>secret reasoning</think>", "The answer is 4."]
    assert "".join(A.strip_think(iter(toks))) == "The answer is 4."


def test_split_think_stream_separates_channels():
    toks = ["<think>abc</think>", "visible"]
    kinds = {}
    for kind, text in A.split_think_stream(iter(toks)):
        kinds.setdefault(kind, "")
        kinds[kind] += text
    assert kinds.get("think") == "abc"
    assert kinds.get("answer") == "visible"


def test_sanitize_for_tts_strips_markdown_urls_emoji():
    s = A.sanitize_for_tts("**Bold** `code` https://x.io/a and 😊 text")
    assert "*" not in s and "`" not in s and "https" not in s and "😊" not in s
    assert "Bold" in s and "text" in s


# --- model selection ------------------------------------------------------------
NAMES = ["qwen3:4b", "llama3.2:3b", "gemma3:4b", "deepseek-r1:7b", "qwen2.5-coder:7b"]


def test_resolve_model_partial_and_spaced():
    assert A.resolve_model("deepseek", NAMES) == "deepseek-r1:7b"
    assert A.resolve_model("deep seek", NAMES) == "deepseek-r1:7b"
    assert A.resolve_model("gemma", NAMES) == "gemma3:4b"
    assert A.resolve_model("nonexistent", NAMES) is None


def test_parse_model_command():
    assert A.parse_model_command("switch to deepseek")[0] == "switch"
    assert A.parse_model_command("what model are you using?")[0] == "query"
    assert A.parse_model_command("tell me a joke") is None
    kind, req, explicit = A.parse_model_command("switch to llama")
    assert explicit is True


def test_supports_tools_and_pick():
    assert A.supports_tools("qwen3:4b")
    assert not A.supports_tools("gemma3:4b")
    assert A.pick_tool_model(NAMES).startswith("qwen3")


def test_model_label_annotates():
    assert "(fastest replies)" in A.model_label("llama3.2:3b")
    assert A.model_label("unknown:1b") == "unknown:1b"


# --- hallucination filter ---------------------------------------------------------
def test_hallucination_filter():
    assert A.looks_like_hallucination("", [], 2.0)
    assert A.looks_like_hallucination("thank you.", [], 1.0)
    assert not A.looks_like_hallucination("open the browser please", [], 2.0)


# --- practice & learning prompts ---------------------------------------------------
def test_practice_prompt_embassy_country():
    p = A.practice_prompt("embassy", "Japan")
    assert "Japan" in p and "{country}" not in p
    assert "never mention" in p.lower() or "never" in p.lower()


def test_translator_mode_exists():
    p = A.practice_prompt("translator")
    assert "Devanagari" in p


def test_tutor_prompt_injects_progress(tmp_path):
    T.LEARNING_FILE = str(tmp_path / "learn.json")
    T.save_progress("math", covered="algebra", weak_spots="factoring")
    p = A.tutor_prompt("math")
    assert "factoring" in p


# --- tools: lists, memory, flashcards ----------------------------------------------
def test_lists_roundtrip(tmp_path):
    T.LISTS_FILE = str(tmp_path / "lists.json")
    T.list_add("my shopping list", "milk")
    T.list_add("shopping", "eggs")
    out = T.list_show("shopping")
    assert "milk" in out and "eggs" in out
    assert "Removed" in T.list_remove("shopping", "1")
    assert "Cleared" in T.list_clear("shopping")


def test_memory_roundtrip(tmp_path):
    T.MEMORY_FILE = str(tmp_path / "mem.json")
    T.remember_fact("Likes tea")
    assert "Likes tea" in T.memory_summary()
    T.forget_fact("tea")
    assert T.memory_summary() == ""


def test_flashcards_sm2(tmp_path):
    T.CARDS_FILE = str(tmp_path / "cards.json")
    T.flashcard_add("Q1", "A1", "cs")
    assert "card 1" in T.flashcards_due()
    assert "A1" in T.flashcard_answer(1)
    assert "3 day" in T.flashcard_grade(1, "pass") or "1 day" in T.flashcard_grade(1, "pass")
    assert "No flashcards" in T.flashcards_due()


# --- tools: math & code --------------------------------------------------------------
def test_calculate_solve_and_diff():
    assert "[-2, 2]" in T.calculate("2*x**2 - 8 = 0", "solve")
    assert "cos" in T.calculate("sin(x)", "differentiate")
    assert "36" in T.calculate("0.15*240")


def test_run_python_output_and_timeout():
    assert "hello" in T.run_python("print('hello')")
    assert "did not finish" in T.run_python("while True: pass", timeout=1)


# --- documents ------------------------------------------------------------------------
def test_doc_root_escape_blocked():
    assert "outside" in T.read_document("/etc/passwd")


def test_chunk_text_overlap():
    chunks = T._chunk_text("word " * 2000, size=500, overlap=100)
    assert len(chunks) > 3
    assert all(len(c) <= 500 for c in chunks)


# --- misc utilities ---------------------------------------------------------------------
def test_norm_list_name():
    assert T._norm_list_name("my shopping list") == "shopping"
    assert T._norm_list_name("") == "notes"


def test_app_aliases_resolve_missing_app_honestly():
    argv, label = T._find_app_launch("definitely_not_installed_xyz")
    assert argv is None


# --- courses --------------------------------------------------------------------------
def test_course_lifecycle(tmp_path):
    import courses
    courses.COURSES_FILE = str(tmp_path / "courses.json")
    assert len(courses.CURRICULA) == 9
    assert courses.start_course("python")["module"] == 0
    assert "Module 1 of 12" in courses.module_brief("python")
    courses.advance_course("python")
    assert courses.course_state("python")["module"] == 1
    for _ in range(15):
        courses.advance_course("python")
    assert courses.course_state("python")["done"] is True


def test_project_tools_sandboxed(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "PROJECTS_ROOT", str(tmp_path))
    assert "Refused" in T.create_project_file("../evil.py", "x")
    assert "Wrote" in T.create_project_file("p/main.py", "print(1)")
    assert "Output:\n1" in T.run_project("p", "python main.py")
    assert "print(1)" in T.review_project("p")


# --- home file management: security model ----------------------------------------
def test_file_tools_guardrails(tmp_path, monkeypatch):
    home = str(tmp_path)
    monkeypatch.setattr(T, "_HOME", home)
    monkeypatch.setenv("HOME", home)
    # outside home refused
    assert "Refused" in T.write_file("/etc/nope.txt", "x")
    # protected dirs refused
    assert "protected" in T.write_file(f"{home}/.ssh/config", "x")
    # create works without confirm
    assert "Wrote" in T.write_file(f"{home}/a.txt", "hello")
    # overwrite requires confirm; declined
    assert "declined" in T.write_file(f"{home}/a.txt", "new", confirm=lambda a: False)
    # overwrite approved
    assert "Wrote" in T.write_file(f"{home}/a.txt", "new", confirm=lambda a: True)
    # delete always asks; declined then approved
    assert "declined" in T.delete_file(f"{home}/a.txt", confirm=lambda a: False)
    assert "Deleted" in T.delete_file(f"{home}/a.txt", confirm=lambda a: True)
    # move + copy
    T.write_file(f"{home}/b.txt", "data")
    assert "Copied" in T.move_file(f"{home}/b.txt", f"{home}/c.txt", copy=True)
    assert "Moved" in T.move_file(f"{home}/b.txt", f"{home}/d.txt")
    out = T.list_folder(home)
    assert "c.txt" in out and "d.txt" in out


# --- file attach ---------------------------------------------------------------------
def test_try_attach_formats(tmp_path):
    md = tmp_path / "notes.md"
    md.write_text("# Plan\nThe secret budget is 7777 dollars.")
    ctx, notice = A.try_attach(str(md))
    assert ctx and "7777" in ctx and "notes.md" in notice
    html = tmp_path / "page.html"
    html.write_text("<html><head><style>x{}</style></head><body><p>Hello <b>world</b></p></body></html>")
    ctx, _ = A.try_attach(str(html))
    assert "Hello" in ctx and "<p>" not in ctx and "style" not in ctx
    # quoted drag-and-drop path
    ctx, _ = A.try_attach(f"'{md}'")
    assert ctx
    # not a file → (None, None)
    assert A.try_attach("just a normal message") == (None, None)


def test_push_prompt_conversing_minimal(monkeypatch):
    import builtins
    keys = iter(["", "m"])
    monkeypatch.setattr(builtins, "input", lambda *a: next(keys))
    assert A.push_prompt("qwen3:4b", conversing=True) == "talk"
    assert A.push_prompt("qwen3:4b", conversing=True) == "menu"


# --- memory everywhere + past-chat recall ------------------------------------------
def test_with_memory_injection(tmp_path):
    T.MEMORY_FILE = str(tmp_path / "m.json")
    assert A.with_memory("BASE") == "BASE"          # empty memory: unchanged
    T.remember_fact("User's name is Arogya")
    assert "Arogya" in A.with_memory("BASE")


def test_search_chats(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(T, "CHATS_DIR", str(tmp_path))
    (tmp_path / "chat-20260701-100000.json").write_text(json.dumps(
        [{"role": "user", "content": "plan a trip to Pokhara"}]))
    out = T.search_chats("pokhara")
    assert "2026-07-01" in out
    assert "No past conversation" in T.search_chats("zzz-nothing")


# --- self-management: install / switch model, health, media ------------------------
def test_switch_model_sets_pending(monkeypatch):
    monkeypatch.setattr(T, "_installed_models", lambda: ["qwen2.5-coder:7b", "gemma3:4b",
                                                         "nomic-embed-text:latest"])
    T.PENDING_MODEL[0] = None
    assert "Switching" in T.switch_model("coder")
    assert T.PENDING_MODEL[0] == "qwen2.5-coder:7b"
    assert "embedding" in T.switch_model("nomic")           # embed refused
    assert "No installed model" in T.switch_model("llama99")


def test_install_model_guards(monkeypatch):
    monkeypatch.setattr(T, "_installed_models", lambda: ["gemma3:4b"])
    assert "declined" in T.install_model("mistral:7b", confirm=lambda a: False)
    assert "valid model name" in T.install_model("bad name!")
    T.PENDING_MODEL[0] = None
    assert "already installed" in T.install_model("gemma3:4b")   # no download


def test_play_media_search_fallback(monkeypatch):
    calls = []
    monkeypatch.setattr(T.subprocess, "Popen", lambda *a, **k: calls.append(a[0]))
    monkeypatch.setattr(T, "_find_app_launch", lambda n: (["spotify"], "spotify"))
    assert "youtube" in T.play_media("lofi beats", "youtube").lower() or calls
    assert "Spotify" in T.play_media("some song", "spotify")


def test_create_homework_docx(tmp_path, monkeypatch):
    monkeypatch.setattr(T.subprocess, "Popen", lambda *a, **k: None)
    out = T.create_homework_docx("Physics Homework", "## Motion\n- Velocity\n1. Acceleration", "test_hw", str(tmp_path))
    assert "Physics Homework" in out or "created" in out.lower()
    f = tmp_path / "test_hw.docx"
    assert f.exists()


def test_browse_and_read_webpage(monkeypatch):
    monkeypatch.setattr(T.subprocess, "Popen", lambda *a, **k: None)
    monkeypatch.setattr(T, "fetch_webpage", lambda url, max_chars=4000: "Mock Webpage Content")
    res = T.browse_and_read_webpage("https://example.com")
    assert "https://example.com" in res
    assert "Mock Webpage Content" in res

