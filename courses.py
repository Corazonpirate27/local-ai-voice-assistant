"""
Structured, CS50-style courses for study mode.

Each course is a sequence of modules; each module has topics, an exercise theme,
and a hands-on project. The curriculum here is the skeleton — the tutor LLM
expands each module into lessons, problem sets and project guidance at study
time. Course progress lives in ~/.voice_assistant_courses.json.
"""

import json
import os
from datetime import date

COURSES_FILE = os.path.expanduser(
    os.environ.get("VA_COURSES_FILE", "~/.voice_assistant_courses.json"))
PROJECTS_ROOT = os.path.expanduser(
    os.environ.get("VA_PROJECTS_ROOT", "~/StudyProjects"))

CURRICULA = {
    "python": {
        "title": "Python from Zero to Builder",
        "modules": [
            ("Getting started", "values, variables, types, input/output, errors as friends",
             "Mad-libs story generator that asks the user questions"),
            ("Control flow", "if/elif/else, while, for, range, break/continue",
             "Number guessing game with difficulty levels"),
            ("Data structures", "lists, tuples, dicts, sets, slicing, comprehensions",
             "Contact book: add, search, delete, stored in a dict"),
            ("Functions", "parameters, return values, scope, docstrings, unit thinking",
             "Refactor the contact book into clean functions with tests"),
            ("Files & errors", "reading/writing files, JSON, exceptions, with-blocks",
             "Persistent todo app that saves to JSON"),
            ("Object-oriented Python", "classes, methods, __init__, inheritance, dunder methods",
             "Bank account simulator with account types"),
            ("Modules & the ecosystem", "imports, pip, venv, requests, reading docs",
             "Weather CLI that calls a real web API"),
            ("Capstone", "bring it together: design, structure, README, edge cases",
             "Your own expense tracker CLI with categories, reports and JSON storage"),
        ],
    },
    "cs": {
        "title": "Computer Science Fundamentals (CS50-style)",
        "modules": [
            ("How computers think", "binary, ASCII/Unicode, RAM vs disk, CPU basics, algorithms as recipes",
             "Paper + Python: implement binary-to-decimal converter"),
            ("Algorithms", "linear/binary search, bubble/selection/merge sort, big-O intuition",
             "Implement and time linear vs binary search on 1M items"),
            ("Data structures", "arrays, linked lists, stacks, queues, hash tables, trees",
             "Build a hash table with collision handling from scratch"),
            ("Memory & C-level thinking", "pointers conceptually, stack vs heap, why Python hides this",
             "Simulate a stack machine that evaluates expressions"),
            ("Databases & SQL", "tables, SELECT/INSERT/UPDATE, joins, indexes, SQLite",
             "Design and query a school database in SQLite"),
            ("The internet", "IP, TCP, DNS, HTTP, what happens when you load a page",
             "Write an HTTP client with raw sockets, then with requests"),
            ("Web programming", "HTML/CSS basics, forms, a tiny Flask app",
             "Flask guestbook with an SQLite backend"),
            ("Capstone", "pick a real problem and engineer a solution end to end",
             "A URL shortener web app with stats page"),
        ],
    },
    "math": {
        "title": "Mathematics for Builders",
        "modules": [
            ("Algebra refresh", "equations, inequalities, exponents, factoring",
             "Solve and verify 20 mixed problems with the calculate tool"),
            ("Functions & graphs", "linear, quadratic, exponential, transformations",
             "Plot-by-hand exercise set + function analyzer script in Python"),
            ("Trigonometry", "unit circle, sin/cos/tan, identities, applications",
             "Ladder/shadow/wave word-problem set, verified"),
            ("Calculus I: derivatives", "limits, rules, chain rule, optimization",
             "Optimize a real problem (max area / min cost) and verify with sympy"),
            ("Calculus II: integrals", "antiderivatives, area, fundamental theorem",
             "Compute areas/volumes; write a numeric integrator and compare to sympy"),
            ("Linear algebra", "vectors, matrices, dot product, systems of equations",
             "Implement matrix multiply and solve a linear system by hand + numpy"),
            ("Probability & statistics", "counting, distributions, mean/variance, Bayes",
             "Monte-Carlo simulator: verify probability answers by simulation"),
            ("Discrete math", "logic, proofs, sets, induction, graphs",
             "Model a friend network as a graph; shortest-path by hand + code"),
        ],
    },
    "physics": {
        "title": "Physics: Mechanics to Modern",
        "modules": [
            ("Motion", "kinematics, velocity, acceleration, free fall, graphs",
             "Projectile motion simulator in Python with plots of trajectories"),
            ("Forces", "Newton's laws, friction, inclined planes, tension",
             "Solve a pulley/incline problem set; simulate with code"),
            ("Energy & momentum", "work, KE/PE, conservation, collisions",
             "Collision simulator (elastic vs inelastic) that conserves momentum"),
            ("Rotation & gravity", "torque, angular momentum, orbits, Kepler",
             "Two-body orbit simulator"),
            ("Waves & sound", "frequency, wavelength, interference, Doppler",
             "Analyze a real audio file's frequency content (you have Whisper's ffmpeg!)"),
            ("Electricity & magnetism", "charge, fields, circuits, Ohm's law, induction",
             "Circuit solver for series/parallel resistor networks"),
            ("Thermodynamics", "temperature, heat, laws, entropy intuition",
             "Model cooling of a cup of tea; fit against the real formula"),
            ("Modern physics", "relativity intuition, photons, atoms, radioactivity",
             "Half-life decay simulator compared with the analytic curve"),
        ],
    },
    "ai": {
        "title": "AI & Machine Learning, Ground Up",
        "modules": [
            ("What AI actually is", "rules vs learning, datasets, features, train/test split",
             "Hand-build a spam classifier with simple word rules; measure accuracy"),
            ("Math you need", "vectors, gradients, probability for ML — just enough",
             "Implement gradient descent on a parabola, watch it converge"),
            ("Linear models", "linear/logistic regression, loss functions",
             "Predict house prices with linear regression from scratch (numpy)"),
            ("Neural networks", "neurons, layers, activation, backprop intuition",
             "Build a tiny neural net from scratch that learns XOR"),
            ("Training in practice", "overfitting, validation, hyperparameters",
             "Train/validate on a real CSV dataset; plot learning curves"),
            ("Language models", "tokens, embeddings, attention intuition, why LLMs work",
             "Build a bigram/Markov text generator; compare with your local LLM"),
            ("Using LLMs well", "prompting, RAG, tools/agents, evaluation, limits",
             "Build a mini RAG: embed a document set and answer questions (ollama embeddings)"),
            ("Capstone", "an end-to-end ML project done properly",
             "Your choice: classifier, recommender, or a small agent — with a report"),
        ],
    },
    "cybersec": {
        "title": "Cybersecurity Foundations (defensive & ethical)",
        "modules": [
            ("Security mindset", "CIA triad, threat models, attack surfaces, ethics & law",
             "Threat-model your own laptop: assets, risks, mitigations document"),
            ("Networking for security", "TCP/IP, ports, DNS, TLS, what packets reveal",
             "Map your own home network; explain every open port you find"),
            ("Cryptography", "hashing, symmetric/asymmetric, signatures, TLS handshake",
             "Implement Caesar→Vigenère→then break them; use real hashes with hashlib"),
            ("Authentication", "passwords, salting, 2FA, session tokens, OAuth intuition",
             "Build a toy login system with properly salted+hashed passwords"),
            ("Web security", "OWASP top 10: XSS, SQLi, CSRF — how they work, how to prevent",
             "Harden a deliberately-vulnerable Flask app you build yourself"),
            ("System hardening", "permissions, updates, firewalls, least privilege, backups",
             "Audit and harden your own Fedora system; write the checklist"),
            ("Defensive monitoring", "logs, intrusion detection ideas, incident response basics",
             "Write a log-watcher that alerts on suspicious auth failures"),
            ("Capstone", "put on the defender's hat end to end",
             "Full security assessment report of a system you own"),
        ],
    },
    "software-eng": {
        "title": "Software Engineering: Code That Lasts",
        "modules": [
            ("Beyond scripts", "readability, naming, functions, DRY, code smells",
             "Refactor a messy 200-line script (the tutor generates it) into clean code"),
            ("Version control", "git: commits, branches, merges, good messages, .gitignore",
             "Git-ify a project: history with branches and a merge"),
            ("Testing", "unit tests, pytest, TDD, edge cases, regression thinking",
             "Write a test suite for your Python-course contact book"),
            ("Design", "modules, interfaces, coupling/cohesion, simple design patterns",
             "Design then build a plugin-based calculator"),
            ("Debugging & tooling", "reading tracebacks, debuggers, logging, linters",
             "Fix 10 planted bugs in a program using systematic debugging"),
            ("APIs & integration", "REST, JSON, error handling, retries, rate limits",
             "Build a resilient client for a flaky API (the tutor simulates one)"),
            ("Performance", "profiling, big-O in practice, caching, when to optimize",
             "Profile and speed up a slow program 10x"),
            ("Capstone", "ship something real: repo, tests, README, releases",
             "A complete CLI tool published as a proper repository"),
        ],
    },
    "english": {
        "title": "English: Beginner to Advanced",
        "modules": [
            ("Foundations", "sentence structure (S+V+O), parts of speech, a/an/the "
             "basics, present simple & continuous — with common Nepali-speaker slips",
             "Write a 10-sentence self-introduction; the tutor corrects it line by line"),
            ("Everyday English", "questions & negatives, past simple/continuous, "
             "everyday vocabulary: home, food, travel, numbers, time",
             "Describe your daily routine (12+ sentences) and write a short shop dialogue"),
            ("Plans & requests", "future forms (will / going to / present continuous), "
             "polite requests, offers, suggestions, phone English",
             "Write a phone conversation booking an appointment, then role-play it"),
            ("Intermediate grammar", "present perfect vs past, conditionals 0-2, "
             "comparatives, articles & prepositions (the classic hard parts)",
             "Write a 200-word story about a memorable day; grammar-graded"),
            ("Conversation skills", "small talk, giving opinions, agreeing/disagreeing "
             "politely, phrasal verbs, connectors (however, although, therefore)",
             "Write an opinion piece (150 words), then debate the tutor on it"),
            ("Writing well", "paragraph structure, formal vs informal emails, "
             "punctuation, linking ideas, avoiding run-ons",
             "Write one informal email and one formal email/cover letter"),
            ("Advanced English", "passive voice, reported speech, conditionals 3, "
             "idioms, academic vocabulary, tone and nuance",
             "Write a 300-word argumentative essay; revised to publication quality"),
            ("Mastery & real world", "IELTS/TOEFL-style speaking & writing tasks, "
             "presentations, summarizing, interview English",
             "Write and polish a full Statement of Purpose (SOP) for study abroad"),
        ],
    },
    "webdev": {
        "title": "Web Development Full Path",
        "modules": [
            ("HTML & CSS", "structure, semantic tags, box model, flexbox, responsive",
             "Personal portfolio page, mobile-friendly"),
            ("JavaScript basics", "DOM, events, functions, fetch",
             "Interactive quiz app in vanilla JS"),
            ("Backend with Python", "Flask routes, templates, forms, sessions",
             "Notes web app with login"),
            ("Databases on the web", "SQLite/ORMs, models, migrations intuition",
             "Add persistent storage + search to the notes app"),
            ("APIs", "REST design, JSON endpoints, consuming your own API",
             "JSON API for the notes app + a JS frontend that uses it"),
            ("Deployment thinking", "environments, config, HTTPS, what hosting means",
             "Run the app properly with gunicorn behind a local reverse proxy"),
            ("Capstone", "design→build→polish a real site",
             "Your own project: blog, tracker, or portfolio with a backend"),
        ],
    },
}


def _load():
    try:
        with open(COURSES_FILE, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _save(d):
    try:
        with open(COURSES_FILE, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
    except Exception:
        pass


def list_courses():
    lines = []
    state = _load()
    for key, c in CURRICULA.items():
        n = len(c["modules"])
        st = state.get(key)
        prog = f" — module {st['module'] + 1}/{n}" if st else ""
        done = " ✅ completed" if st and st.get("done") else ""
        lines.append(f"  {key:14} {c['title']} ({n} modules){prog}{done}")
    return "\n".join(lines)


def start_course(key):
    key = key.lower()
    if key not in CURRICULA:
        return None
    state = _load()
    if key not in state:
        state[key] = {"module": 0, "started": str(date.today()), "done": False}
        _save(state)
    return state[key]


def course_state(key):
    return _load().get(key.lower())


def advance_course(key):
    key = key.lower()
    state = _load()
    st = state.get(key)
    if not st:
        return None
    n = len(CURRICULA[key]["modules"])
    if st["module"] + 1 >= n:
        st["done"] = True
    else:
        st["module"] += 1
    _save(state)
    return st


def module_brief(key, index=None):
    """One module's syllabus line for the tutor prompt."""
    key = key.lower()
    c = CURRICULA[key]
    st = course_state(key) or {"module": 0}
    i = st["module"] if index is None else index
    i = min(i, len(c["modules"]) - 1)
    name, topics, project = c["modules"][i]
    return (f"Course: {c['title']} — Module {i + 1} of {len(c['modules'])}: "
            f"'{name}'. Topics: {topics}. Module project: {project}.")


def syllabus(key):
    c = CURRICULA[key.lower()]
    st = course_state(key) or {"module": -1}
    out = [c["title"]]
    for i, (name, _topics, project) in enumerate(c["modules"]):
        mark = "✅" if i < st["module"] else ("👉" if i == st["module"] else "  ")
        out.append(f" {mark} {i + 1}. {name} — project: {project}")
    return "\n".join(out)
