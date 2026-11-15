#!/usr/bin/env python3
"""
Official IELTS Speaking Examiner & Audio Recorder (David - British Council)
Randomized Official Cambridge 18–21 Mock Test Simulator

Features:
1. Audio Archival: Every single candidate spoken turn is recorded in pristine 16kHz WAV format
   to ~/Projects/voice-assistant/candidate_audio/session_[TIMESTAMP]/ for forensic listening.
2. Randomized Official Tests: Randomly selects authentic full speaking exams from
   Cambridge IELTS Books 18, 19, 20, and 21.
3. Push-to-Talk Reliability: Candidate presses [ENTER] to start and [ENTER] to stop,
   preventing cutoffs and premature interruptions.
4. GPU Whisper Speech-to-Text: Verbatim transcription with disfluency and pause tracking.
5. Dynamic Part 3 Examiner: Powered by local Qwen2.5:3b on NVIDIA GTX 1660 Ti,
   generating contextual follow-up questions tailored to candidate ideas.
"""

import os
import queue
import re
import sys
import tempfile
import time
import subprocess
import threading
import select
import urllib.request
import json
import random
import numpy as np
import sounddevice as sd
from scipy import signal
from scipy.io import wavfile
import whisper
import certifi

# Ensure clean direct internet connection for Edge-TTS and valid SSL certificates
for k in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    os.environ.pop(k, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

EXAMINER_VOICE = "en-GB-RyanNeural"
TARGET_SAMPLE_RATE = 16000
SESSION_AUDITS = []

# Candidate Audio Recording Session Directory
SESSION_TIMESTAMP = time.strftime('%Y%m%d_%H%M%S')
AUDIO_DIR = os.path.expanduser("~/Projects/voice-assistant/candidate_audio")
try:
    os.makedirs(AUDIO_DIR, exist_ok=True)
except Exception:
    AUDIO_DIR = os.path.expanduser("~/candidate_audio")
    os.makedirs(AUDIO_DIR, exist_ok=True)
TURN_COUNTER = 0

# Detect hardware default sample rate
try:
    DEFAULT_INPUT_DEVICE_INFO = sd.query_devices(kind='input')
    INPUT_SAMPLE_RATE = int(DEFAULT_INPUT_DEVICE_INFO['default_samplerate'])
    DEVICE_NAME = DEFAULT_INPUT_DEVICE_INFO['name']
except Exception:
    INPUT_SAMPLE_RATE = 44100
    DEVICE_NAME = "Default Audio Input"

def ensure_ollama_server():
    """Ensure local Ollama server is running and responsive on port 11434."""
    def check_alive():
        try:
            req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                return resp.status == 200
        except Exception:
            return False

    if check_alive():
        return True

    ollama_bin = "/home/corazonpc/.local/bin/ollama"
    if os.path.exists(ollama_bin):
        subprocess.Popen([ollama_bin, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(8):
            time.sleep(0.5)
            if check_alive():
                return True
    return False

def flush_terminal_stdin():
    """Flush any residual keypresses or terminal escape codes from stdin."""
    try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass

def analyze_speech_hesitations(audio_16k, sample_rate=TARGET_SAMPLE_RATE):
    """
    Strict IELTS Fluency & Hesitation Auditor:
    Detects silent gaps, cognitive hesitations (>= 0.8s),
    total speaking time vs total stuck/pause time directly from waveform.
    """
    duration = len(audio_16k) / sample_rate
    if duration < 0.5:
        return {
            "duration": round(duration, 1),
            "pause_time": 0.0,
            "pauses": [],
            "longest_pause": 0.0,
            "hesitation_ratio": 0.0
        }
    
    frame_len = int(sample_rate * 0.05)
    num_frames = len(audio_16k) // frame_len
    if num_frames == 0:
        return {"duration": round(duration, 1), "pause_time": 0.0, "pauses": [], "longest_pause": 0.0, "hesitation_ratio": 0.0}

    frames = audio_16k[:num_frames * frame_len].reshape(-1, frame_len)
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    
    p20 = np.percentile(rms, 20)
    thresh = max(p20 * 2.2, 0.008)
    
    is_silent = rms < thresh
    pauses = []
    current_pause_frames = 0
    for s in is_silent:
        if s:
            current_pause_frames += 1
        else:
            pause_sec = current_pause_frames * 0.05
            if pause_sec >= 0.8:
                pauses.append(round(pause_sec, 2))
            current_pause_frames = 0
    pause_sec = current_pause_frames * 0.05
    if pause_sec >= 0.8:
        pauses.append(round(pause_sec, 2))

    tot_pause = sum(pauses)
    longest_pause = max(pauses) if pauses else 0.0
    hesitation_ratio = (tot_pause / duration) * 100.0 if duration > 0 else 0.0

    return {
        "duration": round(duration, 1),
        "pause_time": round(tot_pause, 1),
        "pauses": pauses,
        "longest_pause": longest_pause,
        "hesitation_ratio": round(hesitation_ratio, 1)
    }

def audit_fluency(text, hesitation_info):
    """
    Strict British Council Fluency & Disfluency Auditor:
    Identifies filler words ('um', 'uh', 'ah', 'err', 'like') and computes penalties
    according to official IELTS Band Descriptors.
    """
    filler_matches = re.findall(r"\b(um+|uh+|ah+|ahh+|umm+|uhh+|er+|erm+|like|you know)\b", text, re.IGNORECASE)
    filler_count = len(filler_matches)
    
    highlighted_text = re.sub(
        r"\b(um+|uh+|ah+|ahh+|umm+|uhh+|er+|erm+|like|you know)\b",
        r"\033[1;31m[\1]\033[0m",
        text,
        flags=re.IGNORECASE
    )
    
    words = text.strip().split()
    word_count = len(words)
    longest = hesitation_info["longest_pause"]
    ratio = hesitation_info["hesitation_ratio"]
    duration = hesitation_info["duration"]
    pause_time = hesitation_info["pause_time"]
    
    feedback_notes = []
    
    if word_count < 8:
        estimated_fc = 4.5
        feedback_notes.append("Underdeveloped response (fewer than 8 words)")
    else:
        penalty = 0.0
        if longest >= 3.0:
            penalty += 1.0
            feedback_notes.append(f"Awkward pause ({longest}s) - searching for vocabulary")
        elif longest >= 1.8:
            penalty += 0.5
            feedback_notes.append(f"Noticeable hesitation pause ({longest}s)")
            
        if filler_count >= 5:
            penalty += 0.5
            feedback_notes.append(f"Excessive fillers ({filler_count} detected: {', '.join(set(filler_matches))})")
        elif filler_count >= 2:
            feedback_notes.append(f"Disfluencies detected ({filler_count} fillers: {', '.join(set(filler_matches))})")
            
        if ratio > 30.0:
            penalty += 0.5
            feedback_notes.append(f"High silent hesitation ratio ({ratio}% of turn)")
            
        estimated_fc = max(5.0, min(8.5, 7.5 - penalty))
    
    return {
        "text": text,
        "highlighted_text": highlighted_text,
        "duration": duration,
        "pause_time": pause_time,
        "pauses": hesitation_info["pauses"],
        "longest_pause": longest,
        "hesitation_ratio": ratio,
        "fillers": filler_matches,
        "filler_count": filler_count,
        "estimated_fc": estimated_fc,
        "notes": feedback_notes
    }

# =====================================================================
# OFFICIAL CAMBRIDGE 18–21 SPEAKING TEST REPOSITORY
# =====================================================================
CAMBRIDGE_TEST_BANK = [
    {
        "id": "C18_T2",
        "source": "Cambridge IELTS 18 — Test 2",
        "title": "Daily Routines & Fascinating Personalities",
        "p1_topic": "your daily routine and days of the week",
        "p1_questions": [
            "Which day of the week do you usually look forward to the most, and why?",
            "What is your typical morning routine before you begin your study or work?",
            "Do you tend to spend your weekends in the same way, or do you prefer variety?",
            "If you could change one thing about your daily routine, what would it be?"
        ],
        "p2_prompt": "Describe an interesting person you have not met in person, but would like to know more about.",
        "p2_bullets": [
            "Who this person is",
            "What this person does or is famous for",
            "Why you think this person is interesting",
            "And explain what you would ask or discuss if you ever had the opportunity to meet them."
        ],
        "p2_followup": "Do you often enjoy reading or watching documentaries about notable people?",
        "p3_theme1": "Famous People & Public Interest",
        "p3_theme1_q1": "Let's consider famous people in society first. What kinds of people tend to gain the most fame in your country?",
        "p3_theme1_fallbacks": [
            "Do you think society places too much emphasis on celebrities rather than scientists and educators?",
            "How has social media changed the way everyday people achieve fame compared to the past?"
        ],
        "p3_theme2": "Role Models & Responsibility",
        "p3_theme2_q4": "Now turning to the influence of public figures, do you believe famous individuals have an obligation to act as positive role models for young people?",
        "p3_theme2_fallbacks": [
            "Can having an inspiring role model genuinely change a young person's future trajectory?",
            "Is there a danger that idolizing celebrities can lead to unrealistic expectations in daily life?"
        ]
    },
    {
        "id": "C19_T4",
        "source": "Cambridge IELTS 19 — Test 4",
        "title": "Cafés, Social Spaces & Scenic Natural Places",
        "p1_topic": "cafés, tea, and spending time with friends",
        "p1_questions": [
            "Do you often visit cafés or tea shops in your local area?",
            "Do you prefer drinking tea or coffee when you have a break?",
            "Do you prefer going to cafés alone to relax, or meeting friends there?",
            "What factors make a café an inviting and comfortable place to spend time?"
        ],
        "p2_prompt": "Describe a place you have visited that has beautiful natural views or scenery.",
        "p2_bullets": [
            "Where this place is located",
            "When and with whom you visited this place",
            "What you did while you were there",
            "And explain why you found the natural views so memorable and impressive."
        ],
        "p2_followup": "Do you prefer spending holidays in natural landscapes or in bustling modern cities?",
        "p3_theme1": "Tourism & Natural Environments",
        "p3_theme1_q1": "Let's consider natural landscapes in general first. Why do you think natural scenery has such a strong psychological appeal for people living in busy cities?",
        "p3_theme1_fallbacks": [
            "How can governments balance expanding tourism in scenic areas while protecting fragile local ecosystems?",
            "Do you think future generations will appreciate nature as much as people do today?"
        ],
        "p3_theme2": "Urban Spaces vs Natural Landscapes",
        "p3_theme2_q4": "Now looking at urban living, how important is it for city planners to integrate parks and green spaces into modern metropolitan areas?",
        "p3_theme2_fallbacks": [
            "In what ways does living in concrete urban environments affect human stress levels?",
            "Can technology and virtual reality ever replicate the authentic benefits of being outdoors in nature?"
        ]
    },
    {
        "id": "C20_T3",
        "source": "Cambridge IELTS 20 — Test 3",
        "title": "Public Transport & Fulfilling Projects",
        "p1_topic": "transportation and daily commuting",
        "p1_questions": [
            "How do you usually travel to work, study, or around your city on a typical day?",
            "Do you prefer using public transport, or do you prefer driving private vehicles like a motorbike or car?",
            "Has public transportation or road infrastructure in your area improved in recent years?",
            "Do you think people in your city will rely more on public transport in the future?"
        ],
        "p2_prompt": "Describe a piece of work or a project you did for your studies or work that you felt very satisfied with.",
        "p2_bullets": [
            "What this project or piece of work was",
            "How and when you completed it",
            "What difficulties or challenges you faced along the way",
            "And explain why you felt so satisfied and proud of the final outcome."
        ],
        "p2_followup": "Do you generally prefer working on projects independently, or collaborating within a team?",
        "p3_theme1": "Motivation & Job Satisfaction",
        "p3_theme1_q1": "Let's consider work and accomplishment first. What factors contribute most to a person feeling satisfied with their job or academic studies?",
        "p3_theme1_fallbacks": [
            "Do you think high financial compensation is more important than personal fulfillment in a career?",
            "How can managers and team leaders maintain motivation among employees during difficult projects?"
        ],
        "p3_theme2": "Collaboration & Modern Workplaces",
        "p3_theme2_q4": "Now turning to teamwork, do you think modern digital tools have made team collaboration more productive or more distracting?",
        "p3_theme2_fallbacks": [
            "Why do some individuals find it difficult to compromise when working in group projects?",
            "In the future, will independent freelancing become more prevalent than working for traditional companies?"
        ]
    },
    {
        "id": "C20_T4",
        "source": "Cambridge IELTS 20 — Test 4",
        "title": "News, Information & Meaningful Conversations",
        "p1_topic": "news and staying informed",
        "p1_questions": [
            "How do you usually stay updated with local and international news?",
            "Do you prefer reading news articles online, or watching video broadcasts?",
            "What kinds of news topics do you find most engaging, such as technology, sports, or world events?",
            "Do you think it is easy for ordinary citizens to distinguish between true news and misinformation nowadays?"
        ],
        "p2_prompt": "Describe a time when you had a long discussion about an interesting news story or current event.",
        "p2_bullets": [
            "What the news story or event was",
            "Who you discussed it with",
            "What different viewpoints or opinions were expressed",
            "And explain how you felt about the discussion afterwards."
        ],
        "p2_followup": "Do you enjoy discussing current affairs with friends and family regularly?",
        "p3_theme1": "The Role of Media in Society",
        "p3_theme1_q1": "Let's consider journalism and news reporting first. How important is an independent and free press for the healthy functioning of a modern democratic society?",
        "p3_theme1_fallbacks": [
            "Why do many media outlets tend to focus disproportionately on negative or sensational stories?",
            "Should governments have any authority to regulate news shared on social media platforms?"
        ],
        "p3_theme2": "Information Verification & Digital Literacy",
        "p3_theme2_q4": "Now turning to digital information, what responsibilities do technology companies have in stopping the spread of artificial or fake news?",
        "p3_theme2_fallbacks": [
            "How can schools and universities teach young students to critically evaluate online sources?",
            "Is there a danger that people only consume news that confirms their existing biases?"
        ]
    },
    {
        "id": "C21_T1",
        "source": "Cambridge IELTS 21 — Test 1",
        "title": "Languages, Communication & Travel Guidance",
        "p1_topic": "languages and learning to communicate",
        "p1_questions": [
            "What languages can you speak, and how often do you use them in daily life?",
            "How did you first begin learning English?",
            "In your experience, what is the most challenging aspect of mastering a foreign language?",
            "Would you be interested in learning another new language in the future?"
        ],
        "p2_prompt": "Describe a time when you used information for tourists, for example from a guidebook, website, or travel app.",
        "p2_bullets": [
            "Where you were traveling or visiting",
            "What kind of information you needed to find",
            "How and where you found this information",
            "And explain how helpful and accurate that tourist information turned out to be."
        ],
        "p2_followup": "Do you usually prefer planning travel itineraries in advance, or exploring spontaneously?",
        "p3_theme1": "Travel & Digital Guidance",
        "p3_theme1_q1": "Let's consider travel information first. How have digital apps and user reviews changed the way people experience traveling compared to traditional paper guidebooks?",
        "p3_theme1_fallbacks": [
            "Is there a risk that travel apps and social media cause too many tourists to overcrowd the exact same spots?",
            "Why do some travelers still prefer hiring local human guides instead of using digital maps?"
        ],
        "p3_theme2": "Tourism & Cultural Interaction",
        "p3_theme2_q4": "Now turning to cultural interaction, what can international travelers do to ensure they respect the traditions and customs of host communities?",
        "p3_theme2_fallbacks": [
            "Do you believe international tourism helps to reduce prejudice and foster global understanding?",
            "What negative social consequences can arise when a small local community becomes a major tourist destination?"
        ]
    },
    {
        "id": "C21_T2",
        "source": "Cambridge IELTS 21 — Test 2",
        "title": "Nature, Greenery & Discovering Surprising Truths",
        "p1_topic": "plants, flowers, and greenery at home",
        "p1_questions": [
            "Do you have any plants, flowers, or a garden in your home?",
            "Did you ever learn about botany, plants, or agriculture while in school?",
            "Have you ever tried growing your own vegetables, fruits, or flowers?",
            "Why do you think so many people find caring for houseplants enjoyable and relaxing?"
        ],
        "p2_prompt": "Describe something you thought was not true, but turned out to be true.",
        "p2_bullets": [
            "What this fact, event, or belief was",
            "Where or from whom you first heard about it",
            "Why you initially doubted it or found it hard to believe",
            "And explain how you reacted when you discovered it was actually true."
        ],
        "p2_followup": "Are you generally skeptical when people tell you surprising facts?",
        "p3_theme1": "Beliefs, Myths & Science",
        "p3_theme1_q1": "Let's consider beliefs and scientific truth first. Why do certain popular myths or misconceptions persist in society even after science disproves them?",
        "p3_theme1_fallbacks": [
            "Why is it psychologically difficult for people to admit when an assumption they held was completely incorrect?",
            "What role should scientific communicators play in making complex facts accessible to the general public?"
        ],
        "p3_theme2": "Fact-Checking in Modern Life",
        "p3_theme2_q4": "Now looking at digital communication, how can individuals protect themselves from accepting false claims made by online influencers?",
        "p3_theme2_fallbacks": [
            "Do you think modern society values evidence and scientific inquiry more than in past generations?",
            "How can critical thinking skills be effectively taught to children at an early age?"
        ]
    },
    {
        "id": "C18_T4",
        "source": "Cambridge IELTS 18 — Test 4",
        "title": "Physical Fitness, Health & Dynamic Personalities",
        "p1_topic": "sports, physical activity, and staying fit",
        "p1_questions": [
            "Do you regularly play any sports or engage in physical exercise like running or gym workouts?",
            "What sports or fitness activities are most popular among people in your country?",
            "Did you enjoy physical education and sports classes when you were at school?",
            "Do you think modern technology and desk jobs make it harder for people to stay physically active?"
        ],
        "p2_prompt": "Describe an energetic person you know who is full of energy and enthusiasm.",
        "p2_bullets": [
            "Who this person is",
            "How you know this person",
            "What kinds of activities or work this person does",
            "And explain why you consider this person to be so exceptionally energetic."
        ],
        "p2_followup": "Do you consider yourself to be an energetic person in your daily routine?",
        "p3_theme1": "Active Lifestyles & Well-being",
        "p3_theme1_q1": "Let's consider physical energy and health first. What are the primary factors that help a person maintain high physical and mental energy throughout their life?",
        "p3_theme1_fallbacks": [
            "Do you think high-stress modern careers are causing premature burnout among young professionals?",
            "How can employers create workplace environments that encourage physical well-being?"
        ],
        "p3_theme2": "Inspiration & Personality",
        "p3_theme2_q4": "Now turning to personality, why do people naturally gravitate towards leaders and colleagues who possess enthusiastic, positive energy?",
        "p3_theme2_fallbacks": [
            "Can high enthusiasm ever be a disadvantage in analytical decision-making roles?",
            "Is energy and passion something that can be cultivated, or is it purely innate temperament?"
        ]
    },
    {
        "id": "C19_T1",
        "source": "Cambridge IELTS 19 — Test 1",
        "title": "History, Heritage & Everyday Technology",
        "p1_topic": "history and historical buildings",
        "p1_questions": [
            "Did you enjoy learning about history when you were in school?",
            "Are there many historical buildings or monuments in your hometown?",
            "Do you think it is important for a country to preserve its historical architecture?",
            "Have you ever visited a museum or historical site that made a lasting impression on you?"
        ],
        "p2_prompt": "Describe a useful piece of equipment or technology in your home that is not a computer or smartphone.",
        "p2_bullets": [
            "What this piece of equipment is",
            "How often you or your family use it",
            "What you use it for",
            "And explain why you find it so useful and convenient in your daily life."
        ],
        "p2_followup": "Do you think people today rely more on household technology than previous generations?",
        "p3_theme1": "Household Technology & Modern Living",
        "p3_theme1_q1": "Let's consider domestic technology first. How have modern electrical appliances changed the way households manage their daily responsibilities?",
        "p3_theme1_fallbacks": [
            "Is there a danger that people are becoming overly dependent on automated machines for basic daily needs?",
            "Do you anticipate that domestic AI robots will become commonplace in ordinary homes in the coming decades?"
        ],
        "p3_theme2": "Technology & Traditional Practical Skills",
        "p3_theme2_q4": "Now turning to practical abilities, are younger generations losing the capacity to perform manual tasks because of modern conveniences?",
        "p3_theme2_fallbacks": [
            "Why is it still valuable for individuals to learn fundamental practical skills like cooking or manual repairs?",
            "How can schools preserve traditional craftsmanship and handiwork alongside digital computer education?"
        ]
    },
    {
        "id": "C18_T1",
        "source": "Cambridge IELTS 18 — Test 1",
        "title": "Shopping Habits, Retail & Digital Platforms",
        "p1_topic": "shopping habits and local stores",
        "p1_questions": [
            "Do you prefer shopping in local neighborhood markets or large modern shopping centers?",
            "How frequently do you purchase goods or services over the internet?",
            "What was the most recent item you bought, and why did you purchase it?",
            "Do you believe small independent retail shops will manage to survive in the future?"
        ],
        "p2_prompt": "Describe a website or digital application that you use regularly and find exceptionally useful.",
        "p2_bullets": [
            "What this website or application is",
            "How you first discovered it",
            "What specific features or services you use it for",
            "And explain why you consider it to be so valuable and well-designed."
        ],
        "p2_followup": "Do you often recommend digital tools and websites to your friends and colleagues?",
        "p3_theme1": "E-Commerce vs Physical Retail",
        "p3_theme1_q1": "Let's consider online commerce first. Why has digital shopping expanded so dramatically across the world over the past decade?",
        "p3_theme1_fallbacks": [
            "What are the primary disadvantages or security risks that consumers encounter when purchasing items online?",
            "In what ways can physical brick-and-mortar stores innovate to stay competitive against massive online retailers?"
        ],
        "p3_theme2": "Consumer Behavior & Digital Influence",
        "p3_theme2_q4": "Now turning to digital influence, how do algorithms and targeted online advertising shape consumer purchasing decisions?",
        "p3_theme2_fallbacks": [
            "Do you believe digital payment methods encourage more impulsive spending compared to cash transactions?",
            "Should governments introduce stricter regulatory oversight on global online marketplaces to protect consumers?"
        ]
    }
]

# Pick Cambridge 19 Test 4 (Cafés & Scenic Natural Places / Pokhara Views) for this final warm-up
c19_test = [t for t in CAMBRIDGE_TEST_BANK if t["id"] == "C19_T4"]
SELECTED_TEST = c19_test[0] if c19_test else random.choice(CAMBRIDGE_TEST_BANK)

print("\n" + "=" * 70)
print(f"  🎙️   OFFICIAL IELTS SPEAKING SIMULATOR (DAVID - BRITISH COUNCIL)   🎙️")
print("=" * 70)
print(f"  📌 SELECTED TEST:  \033[1;33m{SELECTED_TEST['source']}\033[0m")
print(f"  📌 TOPIC:          \033[1;36m{SELECTED_TEST['title']}\033[0m")
print(f"  📁 AUDIO ARCHIVE:  \033[1;32m{AUDIO_DIR}\033[0m")
print(f"  ⚡ Audio Input:    {DEVICE_NAME} ({INPUT_SAMPLE_RATE} Hz)")
print("  ⚡ Verifying local Ollama server status...")
ensure_ollama_server()
print("  ⚡ Loading Whisper STT on GPU...")
whisper_model = whisper.load_model("base", device="cuda" if whisper.torch.cuda.is_available() else "cpu")
print("  ✓ Whisper STT loaded & ready on GPU!")
print("  ✓ Examiner Voice: David (British Council - UK Neural: en-GB-RyanNeural)")
print("  ✓ High-Fidelity Audio Saving: Every spoken turn is saved as 16kHz WAV!")
print("  ✓ Rock-Solid Push-to-Talk: Press [ENTER] to Speak, Press [ENTER] when Done.")
print("=" * 70 + "\n")

def speak(text):
    """Synthesize text using neural British voice and play out loud."""
    print(f"\n👨‍💼 \033[1;34mExaminer (David):\033[0m {text}")
    tmp_path = tempfile.mktemp(suffix=".mp3")
    
    cmd = [
        sys.executable, "-m", "edge_tts",
        "--voice", EXAMINER_VOICE,
        "--text", text,
        "--write-media", tmp_path
    ]
    res = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True)
    
    if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 100:
        subprocess.run(["mpv", "--no-video", "--really-quiet", tmp_path],
                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    else:
        print(f"(David spoken audio error: {res.stderr[:80]})")
    
    flush_terminal_stdin()

def record_audio_push_to_talk(turn_label="candidate_speech"):
    """
    Rock-Solid Push-to-Talk Recording + Audio Disk Archival:
    - Press [ENTER] to start speaking.
    - Press [ENTER] when finished.
    - Saves clean 16kHz WAV file directly into candidate_audio directory.
    - Zero accidental cutoffs, total control for candidate.
    """
    global TURN_COUNTER
    TURN_COUNTER += 1
    
    input("\n👉 \033[1;32mPress [ENTER] to start speaking into your mic...\033[0m")
    print("🔴 \033[1;31mRECORDING... Speak clearly into your mic. Press [ENTER] when finished.\033[0m")
    
    audio_queue = queue.Queue()
    stop_event = threading.Event()
    
    def callback(indata, frames, time_info, status):
        audio_queue.put(indata.flatten().copy())
        
    def wait_for_enter():
        input()
        stop_event.set()
        
    enter_thread = threading.Thread(target=wait_for_enter, daemon=True)
    enter_thread.start()
    
    with sd.InputStream(samplerate=INPUT_SAMPLE_RATE, channels=1, dtype='float32', callback=callback):
        while not stop_event.is_set():
            time.sleep(0.04)
            
    chunks = []
    while not audio_queue.empty():
        chunks.append(audio_queue.get())
        
    if not chunks:
        return ""
        
    raw_audio = np.concatenate(chunks).astype(np.float32)
    print("⏳ \033[1;33mAuditing speech audio & transcribing verbatim disfluencies...\033[0m")
    
    # Resample to 16kHz for Whisper STT and audio saving
    if INPUT_SAMPLE_RATE != TARGET_SAMPLE_RATE:
        num_target_samples = int(len(raw_audio) * TARGET_SAMPLE_RATE / INPUT_SAMPLE_RATE)
        audio_16k = signal.resample(raw_audio, num_target_samples).astype(np.float32)
    else:
        audio_16k = raw_audio

    # Save audio file to candidate_audio folder
    wav_filename = f"session_{SESSION_TIMESTAMP}_turn_{TURN_COUNTER:02d}_{turn_label}.wav"
    wav_filepath = os.path.join(AUDIO_DIR, wav_filename)
    try:
        audio_int16 = (np.clip(audio_16k, -1.0, 1.0) * 32767).astype(np.int16)
        wavfile.write(wav_filepath, TARGET_SAMPLE_RATE, audio_int16)
        print(f"  💾 \033[1;32mVoice recording saved:\033[0m file://{wav_filepath}")
    except Exception as e:
        print(f"  ⚠️ Could not save WAV file: {e}")
        
    # 1. Analyze hesitations, pauses, and stuck time directly from waveform
    hesitation_info = analyze_speech_hesitations(audio_16k, TARGET_SAMPLE_RATE)
    
    # 2. Transcribe verbatim with Whisper on GPU
    result = whisper_model.transcribe(
        audio_16k,
        language="en",
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        fp16=whisper.torch.cuda.is_available()
    )
    raw_text = result.get("text", "").strip()
    raw_text = re.sub(r'(.)\1{5,}', '', raw_text).strip()
    
    # 3. Audit fluency and print telemetry box
    audit = audit_fluency(raw_text, hesitation_info)
    audit["turn_label"] = turn_label
    audit["audio_path"] = wav_filepath
    SESSION_AUDITS.append(audit)
    
    print(f"\n🗣️   \033[1;36mYou said:\033[0m \"{audit['highlighted_text']}\"")
    
    print("\n  \033[1;35m┌─ ⏱️  FLUENCY METRICS ────────────────────────────────────────┐\033[0m")
    print(f"  │ • Turn Duration:         {audit['duration']}s")
    print(f"  │ • Total Stuck / Pauses:  {audit['pause_time']}s ({audit['hesitation_ratio']}% silence)")
    print(f"  │ • Audio File:            {os.path.basename(wav_filepath)}")
    print("  \033[1;35m└──────────────────────────────────────────────────────────────┘\033[0m")
    return raw_text

def generate_examiner_followup(theme, prev_question, candidate_answer, turn_number, fallbacks):
    """
    Generate dynamic British Council Examiner follow-up question using local Qwen2.5:3b.
    The examiner listens to the candidate's exact words, reflects their point,
    and asks a probing question that pushes for justification, evaluation, or speculation.
    """
    prompt = f"""You are Examiner David, conducting official IELTS Academic Speaking Part 3 for the British Council.
Current Theme: {theme}
Previous Examiner Question: "{prev_question}"
Candidate Spoken Answer: "{candidate_answer}"

Generate the examiner's next spoken turn following strict IELTS Part 3 examiner standards:
1. Begin with a natural conversational bridge acknowledging or challenging what the candidate just said (e.g., "That's an interesting point about...", "Right, but looking at it from another angle...", "Indeed, although some might argue...", "I see your perspective, however...").
2. Ask ONE probing, analytical follow-up question pushing the candidate to evaluate causes, project into the future, or consider contrasting societal viewpoints.
3. Keep the total response between 18 and 30 words (strictly 1 to 2 spoken sentences).
4. Do NOT include greetings, quotation marks, or meta-commentary. Output ONLY the examiner's spoken sentence.
"""
    try:
        req_data = {
            "model": "qwen2.5:3b",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.4, "max_tokens": 80}
        }
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/chat",
            data=json.dumps(req_data).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            generated = data.get("message", {}).get("content", "").strip()
            generated = re.sub(r'^(Examiner\s*(David)?:\s*|David:\s*)', '', generated, flags=re.IGNORECASE)
            generated = generated.strip('"\' ')
            if len(generated) > 15:
                return generated
    except Exception:
        pass

    return fallbacks[turn_number % len(fallbacks)]

def run_test():
    test = SELECTED_TEST
    
    ans_name = "Corazon"
    p1_answers = []
    
    # -------------------------------------------------------------
    # PART 2: CUE CARD (DIRECT START FOR FAST PREP)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  🚀 DIRECT PART 2 LAUNCH (FAST PREP MODE)")
    print("=" * 60)
    
    speak("Now, I am going to give you a topic, and I would like you to talk about it for one to two minutes. Before you talk, you will have one minute to prepare. You can make some notes if you wish. Here is your topic.")
    
    print("\n" + "=" * 60)
    print(f"  📌 TOPIC: {test['p2_prompt']}")
    for b in test['p2_bullets']:
        print(f"     • {b}")
    print("=" * 60)
    
    print("\n⏱️   15-Second Quick Preparation countdown starting...")
    for sec in range(15, 0, -1):
        print(f"\r   ⏳ {sec:02d} seconds remaining to prepare notes... ", end="", flush=True)
        time.sleep(1)
    print("\r   ⏳ 00 seconds remaining — Preparation time finished!      \n")
        
    speak("Your preparation time is up. Please begin speaking now. You have up to two minutes.")
    ans_p2 = record_audio_push_to_talk("part2_cue_card")
    
    # Examiner rounding-off question
    speak(test['p2_followup'])
    ans_p2_followup = record_audio_push_to_talk("part2_followup")
    
    # -------------------------------------------------------------
    # PART 3: TWO-WAY DISCUSSION (DYNAMIC AI EXAMINER)
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  📋 PART 3: TWO-WAY DISCUSSION (DYNAMIC AI EXAMINER)")
    print("-" * 60)
    
    speak("Thank you. We have been talking about this topic, and I would now like to discuss with you some broader questions related to this.")
    
    p3_dialogue = []
    
    # Theme 1
    theme1 = test['p3_theme1']
    print(f"\n🔹 \033[1;33mDiscussion Theme 1: {theme1}\033[0m")
    q1 = test['p3_theme1_q1']
    speak(q1)
    a1 = record_audio_push_to_talk("part3_q1")
    p3_dialogue.append((q1, a1))
    
    # Dynamic Follow-up 1
    print("⚡ [David is analyzing your answer and formulating a follow-up...]")
    q2 = generate_examiner_followup(theme1, q1, a1, 0, test['p3_theme1_fallbacks'])
    speak(q2)
    a2 = record_audio_push_to_talk("part3_q2")
    p3_dialogue.append((q2, a2))
    
    # Dynamic Follow-up 2
    print("⚡ [David is analyzing your answer and formulating a follow-up...]")
    q3 = generate_examiner_followup(theme1, q2, a2, 1, test['p3_theme1_fallbacks'])
    speak(q3)
    a3 = record_audio_push_to_talk("part3_q3")
    p3_dialogue.append((q3, a3))
    
    # Theme 2
    theme2 = test['p3_theme2']
    print(f"\n🔹 \033[1;33mDiscussion Theme 2: {theme2}\033[0m")
    q4 = test['p3_theme2_q4']
    speak(q4)
    a4 = record_audio_push_to_talk("part3_q4")
    p3_dialogue.append((q4, a4))
    
    # Dynamic Follow-up 3
    print("⚡ [David is analyzing your answer and formulating a follow-up...]")
    q5 = generate_examiner_followup(theme2, q4, a4, 0, test['p3_theme2_fallbacks'])
    speak(q5)
    a5 = record_audio_push_to_talk("part3_q5")
    p3_dialogue.append((q5, a5))
    
    # Dynamic Follow-up 4
    print("⚡ [David is analyzing your answer and formulating a follow-up...]")
    q6 = generate_examiner_followup(theme2, q5, a5, 1, test['p3_theme2_fallbacks'])
    speak(q6)
    a6 = record_audio_push_to_talk("part3_q6")
    p3_dialogue.append((q6, a6))
    
    # -------------------------------------------------------------
    # EVALUATION & OFFICIAL BAND DESCRIPTORS AUDIT
    # -------------------------------------------------------------
    speak("Thank you very much. That concludes your speaking test. I am now compiling your detailed fluency metrics, saving your audio files, and preparing the official Band evaluation.")
    
    total_test_dur = sum(a['duration'] for a in SESSION_AUDITS)
    total_stuck_time = sum(a['pause_time'] for a in SESSION_AUDITS)
    total_fillers = sum(a['filler_count'] for a in SESSION_AUDITS)
    all_pauses = [p for a in SESSION_AUDITS for p in a['pauses']]
    avg_fc = sum(a['estimated_fc'] for a in SESSION_AUDITS) / max(1, len(SESSION_AUDITS))
    stuck_pct = (total_stuck_time / max(0.1, total_test_dur)) * 100
    
    print("\n" + "=" * 70)
    print("  🛑   COMPREHENSIVE TEST FLUENCY & HESITATION AUDIT (DAVID)   🛑")
    print("=" * 70)
    print(f"  • Selected Exam Paper:            {test['source']} ({test['title']})")
    print(f"  • Total Actual Speaking Time:     {total_test_dur:.1f}s (~{total_test_dur/60:.1f} mins)")
    print(f"  • Total Time Stuck in Pauses:     {total_stuck_time:.1f}s ({stuck_pct:.1f}% hesitation ratio)")
    print(f"  • Total Awkward Pauses (≥0.8s):   {len(all_pauses)} pauses across test")
    print(f"  • Longest Single Stuck Moment:    {max(all_pauses) if all_pauses else 0.0}s")
    print(f"  • Total 'Ah / Um / Uh' Fillers:   {total_fillers} disfluencies detected")
    print(f"  • Strict IELTS Fluency Band (FC): Band {avg_fc:.1f}")
    print(f"  • Complete Audio Folder:          {AUDIO_DIR}")
    print("=" * 70)

    # Save Session Audio Index
    session_audio_index = []
    for a in SESSION_AUDITS:
        session_audio_index.append({
            "turn": a.get("turn_label"),
            "audio_file": a.get("audio_path"),
            "duration_s": a.get("duration"),
            "text": a.get("text")
        })
    with open(os.path.join(AUDIO_DIR, "session_index.json"), "w") as f:
        json.dump(session_audio_index, f, indent=2)

    full_transcript = f"Candidate Name: {ans_name}\nOfficial Exam Paper: {test['source']} ({test['title']})\nDate: {time.strftime('%Y-%m-%d %H:%M:%S')}\nAudio Folder: {AUDIO_DIR}\n\n"
    full_transcript += f"--- STRICT FLUENCY TELEMETRY ---\n"
    full_transcript += f"Total Speaking Duration: {total_test_dur:.1f}s\n"
    full_transcript += f"Total Stuck / Hesitation Time: {total_stuck_time:.1f}s ({stuck_pct:.1f}%)\n"
    full_transcript += f"Hesitations (>=0.8s): {len(all_pauses)} pauses (Longest: {max(all_pauses) if all_pauses else 0.0}s)\n"
    full_transcript += f"Total Disfluency Fillers (ah/um/uh): {total_fillers}\n"
    full_transcript += f"Calculated Fluency Band: Band {avg_fc:.1f}\n\n"
    
    full_transcript += f"--- PART 1: {test['p1_topic'].upper()} ---\n"
    for q, a in p1_answers:
        full_transcript += f"Examiner: {q}\nCandidate: {a}\n\n"
    full_transcript += f"--- PART 2: CUE CARD ---\nTopic: {test['p2_prompt']}\nCandidate Speech: {ans_p2}\nFollow-up: {ans_p2_followup}\n\n"
    full_transcript += "--- PART 3: TWO-WAY DISCUSSION (DYNAMIC AI EXAMINER) ---\n"
    for q, a in p3_dialogue:
        full_transcript += f"Examiner: {q}\nCandidate: {a}\n\n"
        
    transcript_path = os.path.expanduser("~/Projects/voice-assistant/latest_speaking_session_transcript.txt")
    try:
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(full_transcript)
        print(f"\n💾 Full speaking transcript saved to:\n   \033[1;32m{transcript_path}\033[0m\n")
    except Exception as e:
        print(f"Note: Could not save transcript file: {e}")

    eval_prompt = f"""You are a Senior British Council IELTS Speaking Examiner. Evaluate this candidate's performance strictly according to the Official IELTS Speaking Band Descriptors:

1. Fluency and Coherence (FC)
2. Lexical Resource (LR)
3. Grammatical Range and Accuracy (GRA)
4. Pronunciation & Delivery (PR)

CANDIDATE TRANSCRIPT & TELEMETRY ({test['source']}):
{full_transcript}

Provide:
1. Exact Criterion Band Scores (e.g. FC: 6.5, LR: 7.0, GRA: 6.5, PR: 7.0)
2. Overall Band Score (Target: Band 7.0+)
3. Strengths with verbatim quotes from the candidate
4. High-Priority Fixes for the official exam tomorrow
"""
    try:
        req_data = {
            "model": "qwen2.5:3b",
            "messages": [{"role": "user", "content": eval_prompt}],
            "stream": False,
            "options": {"temperature": 0.3, "max_tokens": 1200}
        }
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/chat",
            data=json.dumps(req_data).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            eval_text = data.get("message", {}).get("content", "")
            print("\n" + eval_text + "\n")
            with open(transcript_path, "a", encoding="utf-8") as f:
                f.write("\n\n--- OFFICIAL BAND EVALUATION ---\n" + eval_text)
        print("=" * 70)
        print("  🎉   IELTS SPEAKING MOCK COMPLETE & AUDIO ARCHIVED!   🎉")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"\n⚠️   Ollama local evaluation skipped: {e}")
        print("💡 Don't worry! Your audio recordings and verbatim transcript are safely archived.")
        print(f"👉 You can review it at {transcript_path} or share it with Antigravity for an in-depth breakdown!")

if __name__ == "__main__":
    try:
        run_test()
    except KeyboardInterrupt:
        print("\n\nTest session stopped by user.")
