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
        "title": "Python: Beginner to Advanced",
        "modules": [
            ("Absolute basics", "what a program is, running Python, print, input, "
             "variables, numbers vs text, simple errors",
             "Mad-libs story generator that asks the user questions"),
            ("Making decisions", "comparisons, if/elif/else, logic (and/or/not), "
             "while loops, break/continue",
             "Number guessing game with difficulty levels"),
            ("Working with data", "lists, for loops, range, strings & methods, "
             "slicing, input validation",
             "To-do list app in the terminal (add, show, remove)"),
            ("Dictionaries & sets", "key/value thinking, nesting, sets, "
             "comprehensions, choosing the right structure",
             "Contact book with search and delete"),
            ("Functions", "parameters, return values, scope, docstrings, "
             "decomposing problems",
             "Refactor the contact book into clean functions"),
            ("Files & errors", "reading/writing files, JSON, exceptions, "
             "with-blocks, defensive coding",
             "Persistent todo app that saves to JSON"),
            ("Object-oriented Python", "classes, methods, __init__, inheritance, "
             "dunder methods, when OOP helps",
             "Bank account simulator with account types"),
            ("The ecosystem", "modules, pip, venv, requests, reading documentation, "
             "debugging real tracebacks",
             "Weather CLI that calls a real web API"),
            ("Intermediate patterns", "generators, iterators, lambda, sorting with "
             "keys, argparse, regular expressions",
             "Log analyzer: parse a log file and report statistics"),
            ("Advanced Python", "decorators, context managers, typing hints, "
             "dataclasses, a taste of async",
             "Build a @timing/@retry decorator library and use it"),
            ("Testing & quality", "pytest, edge cases, refactoring under tests, "
             "project layout, README",
             "Add a test suite to a previous project and harden it"),
            ("Capstone", "design, structure, persistence, polish — bring it together",
             "Your own expense tracker CLI with categories, reports and JSON storage"),
        ],
    },
    "cs": {
        "title": "Computer Science: Beginner to Advanced",
        "modules": [
            ("How computers work", "binary, bits/bytes, ASCII/Unicode, RAM vs disk, "
             "CPU fetch-execute, what an OS is",
             "Binary-to-decimal converter + a one-page 'how my laptop works' note"),
            ("Thinking in algorithms", "algorithms as recipes, pseudocode, "
             "flowcharts, correctness, simple complexity intuition",
             "Write pseudocode + Python for 3 everyday algorithms"),
            ("Search & sort", "linear/binary search, bubble/selection/insertion/merge "
             "sort, big-O for real",
             "Implement and time linear vs binary search on 1M items"),
            ("Data structures I", "arrays, linked lists, stacks, queues — operations "
             "and trade-offs",
             "Build a stack machine that evaluates expressions"),
            ("Data structures II", "hash tables, trees, BSTs, heaps, graphs",
             "Build a hash table with collision handling from scratch"),
            ("Memory & low-level", "pointers conceptually, stack vs heap, compilation "
             "vs interpretation, why C is fast",
             "Simulate memory allocation; explain 3 real segfault causes"),
            ("Databases & SQL", "tables, SELECT/JOIN/INSERT, keys, indexes, "
             "normalization basics, SQLite",
             "Design and query a school database in SQLite"),
            ("The internet", "IP, TCP/UDP, DNS, HTTP(S), what happens when you load "
             "a page, sockets",
             "HTTP client with raw sockets, then with requests"),
            ("Web programming", "HTML/CSS basics, forms, Flask routes and templates",
             "Flask guestbook with an SQLite backend"),
            ("Operating systems", "processes vs threads, scheduling, files & "
             "permissions, concurrency pitfalls",
             "Multi-threaded downloader; demonstrate and fix a race condition"),
            ("Advanced topics", "compilers vs interpreters, caching, distributed "
             "systems intuition, security basics",
             "Build a tiny interpreter for a calculator language"),
            ("Capstone", "engineer a real solution end to end",
             "A URL shortener web app with stats page"),
        ],
    },
    "math": {
        "title": "Mathematics: Beginner to Advanced",
        "modules": [
            ("Number sense", "fractions, decimals, percentages, ratios, negative "
             "numbers, order of operations",
             "Solve 20 real-life problems (discounts, tips, exchange rates), verified"),
            ("Algebra I", "variables, solving linear equations, inequalities, "
             "word problems",
             "Translate 10 word problems into equations and solve them"),
            ("Algebra II", "exponents, polynomials, factoring, quadratic equations, "
             "the quadratic formula",
             "Solve and verify a mixed problem set with the calculate tool"),
            ("Functions & graphs", "what a function is, linear/quadratic/exponential, "
             "transformations, reading graphs",
             "Function analyzer script that tabulates and describes any f(x)"),
            ("Trigonometry", "unit circle, sin/cos/tan, identities, triangles, waves",
             "Ladder/shadow/wave word-problem set, verified"),
            ("Precalculus", "logarithms, sequences & series, limits intuition",
             "Compound interest explorer: savings growth modeled in Python"),
            ("Calculus I: derivatives", "limits, derivative rules, chain rule, "
             "optimization, related rates",
             "Optimize a real problem (max area / min cost), verified with sympy"),
            ("Calculus II: integrals", "antiderivatives, definite integrals, area, "
             "fundamental theorem, applications",
             "Write a numeric integrator and compare against sympy"),
            ("Linear algebra", "vectors, matrices, dot product, systems, "
             "determinants, eigen-intuition",
             "Implement matrix multiply and solve a linear system by hand + numpy"),
            ("Probability", "counting, conditional probability, Bayes, "
             "distributions, expectation",
             "Monte-Carlo simulator: verify 5 probability answers by simulation"),
            ("Statistics", "mean/median/variance, sampling, confidence intuition, "
             "correlation vs causation",
             "Analyze a real CSV dataset and write an honest findings report"),
            ("Discrete & proofs", "logic, sets, induction, graph theory, "
             "reading and writing simple proofs",
             "Model a friend network as a graph; shortest path by hand + code"),
        ],
    },
    "physics": {
        "title": "Physics: Beginner to Advanced",
        "modules": [
            ("Measuring the world", "units, SI, scientific notation, estimation, "
             "measurement error, vectors vs scalars",
             "Fermi estimates: 5 back-of-envelope problems, checked"),
            ("Motion", "position, velocity, acceleration, free fall, motion graphs",
             "Projectile motion simulator with trajectory plots"),
            ("Forces", "Newton's three laws, friction, inclined planes, tension, "
             "common misconceptions",
             "Solve a pulley/incline problem set; simulate one with code"),
            ("Energy & momentum", "work, kinetic/potential energy, conservation "
             "laws, elastic/inelastic collisions",
             "Collision simulator that demonstrably conserves momentum"),
            ("Rotation & gravity", "torque, angular momentum, circular motion, "
             "Newton's gravity, orbits, Kepler",
             "Two-body orbit simulator"),
            ("Oscillations & waves", "springs, pendulums, frequency/wavelength, "
             "interference, resonance, Doppler",
             "Pendulum simulator compared with the small-angle formula"),
            ("Sound & light", "sound waves, intensity, optics basics: reflection, "
             "refraction, lenses",
             "Analyze a real audio file's frequency content"),
            ("Electricity", "charge, fields, potential, circuits, Ohm's law, "
             "series/parallel",
             "Circuit solver for resistor networks"),
            ("Magnetism & induction", "magnetic fields, motors/generators, "
             "induction, AC intuition",
             "Explain + simulate a simple DC motor's torque curve"),
            ("Thermodynamics", "temperature vs heat, gas laws, the laws of "
             "thermodynamics, entropy intuition",
             "Model the cooling of tea; fit against Newton's law of cooling"),
            ("Modern physics", "special relativity intuition, photons, "
             "atomic models, quantum weirdness honestly told",
             "Half-life decay simulator vs the analytic curve"),
            ("Capstone", "pick a phenomenon, model it properly, present it",
             "A full simulation + write-up of a physical system you choose"),
        ],
    },
    "ai": {
        "title": "AI & Machine Learning: Beginner to Advanced",
        "modules": [
            ("What AI actually is", "rules vs learning, data/features/labels, "
             "train/test split, honest limits of AI",
             "Hand-built spam classifier with word rules; measure accuracy"),
            ("Math for ML", "vectors, dot products, gradients, probability — "
             "just enough, concretely",
             "Implement gradient descent on a parabola; watch it converge"),
            ("Linear models", "linear regression, loss functions, logistic "
             "regression, decision boundaries",
             "House-price predictor from scratch with numpy"),
            ("Model evaluation", "overfitting, validation, precision/recall, "
             "cross-validation, data leakage",
             "Train/validate on a real CSV; plot learning curves"),
            ("Classic ML", "k-NN, decision trees, random forests, clustering, "
             "when NOT to use deep learning",
             "Compare 3 classifiers on one dataset; explain the winner"),
            ("Neural networks", "neurons, layers, activations, backprop intuition, "
             "why depth helps",
             "Tiny neural net from scratch that learns XOR"),
            ("Training deep nets", "initialization, learning rates, batching, "
             "regularization, GPUs",
             "Train a small net on MNIST-style digits; tune it honestly"),
            ("Language models", "tokens, embeddings, attention intuition, "
             "how LLMs are trained, why they hallucinate",
             "Bigram/Markov text generator; compare with your local LLM"),
            ("Build a tiny GPT", "the transformer block, self-attention, "
             "training a character-level model on your GPU",
             "Train a mini-GPT on a small corpus (nanoGPT-style) overnight"),
            ("Using LLMs well", "prompting, RAG, embeddings, tool use/agents, "
             "evaluation, cost/latency trade-offs",
             "Build a mini RAG over a document set (ollama embeddings)"),
            ("Fine-tuning", "LoRA/QLoRA intuition, datasets, when fine-tuning "
             "beats prompting, safety basics",
             "Plan (and optionally run) a QLoRA fine-tune of a small model"),
            ("Capstone", "an end-to-end ML project done properly",
             "Your choice: classifier, recommender, or a small agent — with a report"),
        ],
    },
    "cybersec": {
        "title": "Cybersecurity: Beginner to Advanced (defensive & ethical)",
        "modules": [
            ("Security mindset", "CIA triad, threat models, attack surface, "
             "ethics and law — what white-hat means",
             "Threat-model your own laptop: assets, risks, mitigations document"),
            ("How attacks happen", "phishing, malware types, social engineering, "
             "real breach case studies",
             "Analyze 3 real phishing emails; write detection rules of thumb"),
            ("Networking for security", "TCP/IP, ports, DNS, TLS, what packets "
             "reveal, network scanning concepts (on YOUR network only)",
             "Map your own home network; explain every open port"),
            ("Cryptography I", "hashing, salts, symmetric encryption, key exchange "
             "intuition",
             "Implement Caesar→Vigenère, then break them; real hashes with hashlib"),
            ("Cryptography II", "asymmetric crypto, signatures, certificates, "
             "the TLS handshake end-to-end",
             "Verify a real certificate chain and explain each link"),
            ("Authentication", "password storage done right, 2FA, sessions, "
             "tokens, OAuth intuition",
             "Toy login system with properly salted+hashed passwords"),
            ("Web security", "OWASP top 10: XSS, SQL injection, CSRF — mechanics "
             "and prevention",
             "Build, exploit, then harden your own deliberately-vulnerable Flask app"),
            ("System hardening", "least privilege, permissions, updates, firewalls, "
             "backups, disk encryption",
             "Audit and harden your Fedora system; write the checklist"),
            ("Defensive monitoring", "logs, auditd, intrusion detection ideas, "
             "alert fatigue, incident response steps",
             "Log-watcher that alerts on suspicious auth failures"),
            ("Malware & forensics (defensive)", "how malware persists, sandboxing "
             "concepts, disk/memory forensics basics",
             "Forensic timeline exercise on a prepared innocent disk image"),
            ("Privacy & OPSEC", "data minimization, tracking, VPNs honestly "
             "assessed, secure messaging",
             "Personal privacy audit + a realistic improvement plan"),
            ("Capstone", "the defender's hat, end to end",
             "Full security assessment report of a system you own"),
        ],
    },
    "software-eng": {
        "title": "Software Engineering: Beginner to Advanced",
        "modules": [
            ("Beyond scripts", "readability, naming, functions, DRY, code smells",
             "Refactor a messy 200-line script (tutor-generated) into clean code"),
            ("Version control", "git: commits, branches, merges, good messages, "
             ".gitignore, undoing mistakes",
             "Git-ify a project: branching history with a merge and a revert"),
            ("Testing", "unit tests, pytest, TDD, edge cases, regression thinking",
             "Write a test suite for your Python-course contact book"),
            ("Debugging & tooling", "reading tracebacks, debuggers, logging, "
             "linters, bisecting bugs",
             "Fix 10 planted bugs using systematic debugging"),
            ("Design I", "modules, interfaces, coupling/cohesion, composition "
             "over inheritance",
             "Design then build a plugin-based calculator"),
            ("Design II", "common patterns honestly (strategy, factory, observer), "
             "YAGNI, refactoring safely",
             "Refactor the calculator with two patterns — only where they help"),
            ("APIs & integration", "REST, JSON, error handling, retries, rate "
             "limits, pagination",
             "Resilient client for a flaky API (the tutor simulates one)"),
            ("Data & persistence", "SQLite/ORMs, migrations, transactions, "
             "backups, schema evolution",
             "Add versioned persistent storage to an earlier project"),
            ("Performance", "profiling, big-O in practice, caching, memory, "
             "when to optimize (rarely)",
             "Profile and speed up a slow program 10x"),
            ("CI & automation", "GitHub Actions concepts, pre-commit hooks, "
             "makefiles/scripts, reproducible setups",
             "Add an automated test+lint pipeline to a repo"),
            ("Working like a pro", "code review, reading others' code, docs, "
             "issue tracking, estimating",
             "Review a real open-source PR and write a constructive review"),
            ("Capstone", "ship something real: repo, tests, README, releases",
             "A complete CLI tool published as a proper repository"),
        ],
    },
    "webdev": {
        "title": "Web Development: Beginner to Advanced",
        "modules": [
            ("How the web works", "browsers, servers, URLs, HTTP, what happens "
             "when you hit Enter",
             "Diagram + explain a page load; inspect it in DevTools"),
            ("HTML", "semantic structure, forms, links, images, accessibility "
             "basics",
             "Multi-page personal site skeleton"),
            ("CSS", "box model, flexbox, grid, responsive design, dark mode",
             "Style the portfolio: mobile-friendly with a theme toggle"),
            ("JavaScript I", "variables, functions, DOM, events, forms",
             "Interactive quiz app in vanilla JS"),
            ("JavaScript II", "fetch, JSON, async/await, localStorage, modules",
             "Weather widget that calls a real API and remembers your city"),
            ("Backend with Python", "Flask routes, templates, forms, sessions, "
             "project structure",
             "Notes web app with login"),
            ("Databases on the web", "SQLite/ORM models, relationships, "
             "migrations intuition, search",
             "Persistent storage + search for the notes app"),
            ("Building APIs", "REST design, JSON endpoints, status codes, "
             "consuming your own API from JS",
             "JSON API for the notes app + a JS frontend using it"),
            ("Auth & security", "password hashing, sessions vs tokens, XSS/CSRF "
             "defenses, HTTPS",
             "Harden the notes app: proper auth + security headers"),
            ("Deployment thinking", "environments, config, gunicorn, reverse "
             "proxies, monitoring basics",
             "Run the app properly behind a local reverse proxy"),
            ("Capstone", "design→build→polish a real site",
             "Your own project: blog, tracker, or portfolio with a backend"),
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
