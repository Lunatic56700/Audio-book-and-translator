import os
import re
import sys
import time
import asyncio
import threading
import subprocess
from collections import Counter

import fitz  # PyMuPDF
import requests as http_requests
from flask import Flask, Response, render_template, request, jsonify, url_for, send_from_directory, send_file
from werkzeug.utils import secure_filename
from fpdf import FPDF
import edge_tts
from deep_translator import GoogleTranslator
from database import (
    init_db, register_book, save_progress, get_progress, get_books_info,
    add_bookmark, get_bookmarks, delete_bookmark
)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BOOKS_FOLDER = os.path.join(BASE_DIR, 'books')            # Drop PDFs here to add them to the library
AUDIO_FOLDER = os.path.join(BASE_DIR, 'static', 'audio')  # One cache sub-folder per book
os.makedirs(BOOKS_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

VOICE_RE = re.compile(r'^[A-Za-z]{2,3}-[A-Za-z0-9-]+$')
PAGE_NUMBER_RE = re.compile(r'^(page\s+)?\d{1,4}$', re.IGNORECASE)

# Initialize the SQLite database on startup
init_db()

# Supported languages for translation & TTS
SUPPORTED_LANGUAGES = [
    {"code": "en", "name": "English",              "locale": "en-",  "default_voice": "en-US-AriaNeural"},
    {"code": "ur", "name": "Urdu",                 "locale": "ur-",  "default_voice": "ur-PK-AsadNeural"},
    {"code": "es", "name": "Spanish",              "locale": "es-",  "default_voice": "es-ES-ElviraNeural"},
    {"code": "fr", "name": "French",               "locale": "fr-",  "default_voice": "fr-FR-DeniseNeural"},
    {"code": "de", "name": "German",               "locale": "de-",  "default_voice": "de-DE-KatjaNeural"},
    {"code": "hi", "name": "Hindi",                "locale": "hi-",  "default_voice": "hi-IN-SwaraNeural"},
    {"code": "ar", "name": "Arabic",               "locale": "ar-",  "default_voice": "ar-SA-ZariyahNeural"},
    {"code": "zh-CN", "name": "Chinese (Simplified)", "locale": "zh-", "default_voice": "zh-CN-XiaoxiaoNeural"},
    {"code": "pt", "name": "Portuguese",           "locale": "pt-",  "default_voice": "pt-BR-FranciscaNeural"},
    {"code": "ru", "name": "Russian",              "locale": "ru-",  "default_voice": "ru-RU-SvetlanaNeural"},
    {"code": "ja", "name": "Japanese",             "locale": "ja-",  "default_voice": "ja-JP-NanamiNeural"},
    {"code": "ko", "name": "Korean",               "locale": "ko-",  "default_voice": "ko-KR-SunHiNeural"},
    {"code": "tr", "name": "Turkish",              "locale": "tr-",  "default_voice": "tr-TR-EmelNeural"},
    {"code": "it", "name": "Italian",              "locale": "it-",  "default_voice": "it-IT-ElsaNeural"},
]
LANGUAGE_CODES = {lang["code"] for lang in SUPPORTED_LANGUAGES}

# Running headers/footers per book, keyed by (path, mtime) so replaced files are re-scanned
_boilerplate_cache = {}
_boilerplate_lock = threading.Lock()

# Pages currently being synthesized, keyed by their cache path
_jobs = {}
_jobs_lock = threading.Lock()

# Microsoft's voice service can be slow on long text (whole pages sometimes arrive at
# real-time speed, ~3 min per page). Short sections come back in well under a second,
# so pages are cut into sections, a few are made in parallel, and the audio is
# streamed to the browser in order as it arrives.
MAX_TTS_CONNECTIONS = 4   # Too many at once and the service starts refusing requests
SEGMENT_CHARS = 350       # ~20 seconds of speech per section


# =============================================
#  HELPERS
# =============================================
def resolve_book(filename):
    """Returns the full path of a PDF inside BOOKS_FOLDER, or None if it isn't there."""
    if not filename or os.path.basename(filename) != filename or not filename.lower().endswith('.pdf'):
        return None
    path = os.path.join(BOOKS_FOLDER, filename)
    return path if os.path.isfile(path) else None


def book_title(filename):
    """'Crime-and-Punishment.pdf' -> 'Crime and Punishment'."""
    stem = os.path.splitext(filename)[0]
    return " ".join(stem.replace('_', ' ').replace('-', ' ').split()) or stem


def book_slug(filename):
    """Folder-safe name for a book's audio cache."""
    return re.sub(r'[^A-Za-z0-9_-]+', '_', os.path.splitext(filename)[0]).strip('_') or 'book'


def edge_key(line):
    """Ignores digits and spacing so footers like 'Publisher ✦ 13 ✦' match from page to page."""
    return " ".join(re.sub(r'\d+', '#', line).split())


def find_boilerplate(pdf_path):
    """Finds lines repeated at the top/bottom of most pages (running titles, author names...)."""
    key = (pdf_path, os.path.getmtime(pdf_path))
    with _boilerplate_lock:
        if key not in _boilerplate_cache:
            counts = Counter()
            with fitz.open(pdf_path) as doc:
                # A sample of ~40 pages is plenty to spot a running header, and much faster.
                # The step is odd so both left- and right-hand page headers get sampled.
                sample = range(0, len(doc), max(1, len(doc) // 40) | 1)
                for i in sample:
                    text = doc[i].get_text("text", sort=True)
                    lines = [l.strip() for l in text.splitlines() if l.strip()]
                    counts.update({edge_key(l) for l in lines[:2] + lines[-2:]})
            threshold = max(5, len(sample) * 0.3)
            _boilerplate_cache[key] = {line for line, n in counts.items() if n >= threshold}
        return _boilerplate_cache[key]


def clean_text_for_tts(text):
    """Cleans up PDF formatting so Edge-TTS reads it naturally."""
    text = re.sub(r'(\w)-\n(?=[a-z])', r'\1', text)  # Re-join words hyphenated across lines
    text = re.sub(r'(?:\.\s+){5,}\.?', ' ', text)     # Table-of-contents dot leaders
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = " ".join(text.split())
    return text


def page_text_for_tts(pdf_path, page_num):
    boilerplate = find_boilerplate(pdf_path)
    with fitz.open(pdf_path) as doc:
        raw = doc[page_num - 1].get_text("text", sort=True)

    # Strip running headers/footers and bare page numbers so they aren't read aloud every page
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    while lines and (edge_key(lines[0]) in boilerplate or PAGE_NUMBER_RE.match(lines[0])):
        lines.pop(0)
    while lines and (edge_key(lines[-1]) in boilerplate or PAGE_NUMBER_RE.match(lines[-1])):
        lines.pop()
    return clean_text_for_tts("\n".join(lines))


def unique_book_path(filename):
    """Avoids overwriting a different book that happens to share a filename."""
    stem, ext = os.path.splitext(filename)
    candidate, n = filename, 2
    while os.path.exists(os.path.join(BOOKS_FOLDER, candidate)):
        candidate = f"{stem} ({n}){ext}"
        n += 1
    return candidate


# Google Translate allows ~5 requests a second, so requests are spaced out
_translate_lock = threading.Lock()
_last_translate = 0.0
TRANSLATE_GAP = 0.3   # seconds between requests
TRANSLATE_CHUNK = 1800  # characters per request
GOOGLE_TRANSLATE_URL = 'https://translate.googleapis.com/translate_a/single'


class TranslationError(Exception):
    """Raised when a page can't be translated, so English is never read out by mistake."""


def _normalize(text):
    return " ".join(re.sub(r'[^\w]+', ' ', text.lower()).split())


def _google_translate(text, target_lang):
    """Google's JSON translate endpoint. Returns (translation, detected source language)."""
    r = http_requests.post(
        GOOGLE_TRANSLATE_URL,
        params={'client': 'gtx', 'sl': 'auto', 'tl': target_lang, 'dt': 't'},
        data={'q': text},
        timeout=20,
    )
    if r.status_code == 429:
        raise TranslationError("Google Translate is getting too many requests from this computer")
    r.raise_for_status()
    data = r.json()
    translated = ''.join(part[0] for part in (data[0] or []) if part and part[0])
    detected = data[2] if len(data) > 2 and isinstance(data[2], str) else ''
    return translated, detected


def _deep_translate(text, target_lang):
    """Backup: the deep-translator library (scrapes Google's mobile page)."""
    return GoogleTranslator(source='auto', target=target_lang).translate(text) or '', ''


def _translate_chunk(text, target_lang):
    global _last_translate
    problem = None
    for attempt in range(3):
        for translate in (_google_translate, _deep_translate):
            with _translate_lock:
                wait = _last_translate + TRANSLATE_GAP - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                _last_translate = time.monotonic()
            try:
                translated, detected = translate(text, target_lang)
            except Exception as e:
                if translate is _google_translate or problem is None:
                    problem = e  # Google's reason is the one worth showing
                print(f"[Translation error] {translate.__name__}: {e}")
                continue
            # Same text back means it wasn't translated (unless the book is already
            # in that language)
            # (very short bits like names or titles can legitimately stay the same)
            unchanged = (_normalize(translated) == _normalize(text)
                         and len(_normalize(text).split()) >= 8)
            already_in_target = detected and detected.split('-')[0] == target_lang.split('-')[0]
            if translated.strip() and (not unchanged or already_in_target):
                return translated
            if translate is _google_translate or problem is None:
                problem = TranslationError("the translator sent the text back untranslated")
            print(f"[Translation error] {translate.__name__}: came back untranslated")
        if attempt < 2:
            time.sleep(2 * (attempt + 1))  # Back off before trying again
    raise TranslationError(str(problem) if problem else "translation failed")


def translate_text(text, target_lang):
    """Translates a page via Google Translate, a few sentences-aligned chunks at a time.
    Raises TranslationError instead of ever returning untranslated text."""
    if not text.strip():
        return text
    chunks, current = [], ''
    for sentence in re.split(r'(?<=[.!?…])\s+', text):
        while len(sentence) > TRANSLATE_CHUNK:  # Run-on "sentences": cut at a space
            cut = sentence.rfind(' ', 0, TRANSLATE_CHUNK) + 1 or TRANSLATE_CHUNK
            if current:
                chunks.append(current)
                current = ''
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:]
        if current and len(current) + len(sentence) + 1 > TRANSLATE_CHUNK:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return ' '.join(_translate_chunk(chunk, target_lang) for chunk in chunks)


# =============================================
#  TTS ENGINE
# =============================================
def split_into_segments(text, size=SEGMENT_CHARS):
    """Breaks page text into sentence-aligned sections of roughly `size` characters.
    The first section is shorter so playback can start sooner."""
    sentences = re.split(r'(?:(?<=[.!?…])|(?<=[.!?…][”"’)]))\s+', text)
    pieces = []
    for sentence in sentences:
        # Very long "sentences" (contents pages, run-ons) get cut at a space
        while len(sentence) > size * 2:
            cut = sentence.rfind(' ', size // 2, size) + 1 or size
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:]
        pieces.append(sentence.strip())

    segments, current = [], ''
    for piece in filter(None, pieces):
        limit = size if segments else size // 2
        if current and len(current) + 1 + len(piece) > limit:
            segments.append(current)
            current = piece
        else:
            current = f"{current} {piece}".strip()
    if current:
        segments.append(current)
    return segments


class _ConnectionSlots:
    """Caps parallel requests to the voice service. Pages being listened to go first, and
    background preloads get one connection each so they never crowd them out."""

    def __init__(self, limit):
        self.free = limit
        self.waiting = []  # (job, order, future)
        self.order = 0

    async def acquire(self, job):
        self.order += 1
        future = asyncio.get_running_loop().create_future()
        self.waiting.append((job, self.order, future))
        self.dispatch()
        await future

    def release(self, job):
        job.active -= 1
        self.free += 1
        self.dispatch()

    def dispatch(self):
        """Hands free connections to waiting sections, most urgent first."""
        self.waiting = [w for w in self.waiting if not w[2].done()]
        while self.free > 0:
            ready = [w for w in self.waiting if w[0].active < w[0].max_active]
            if not ready:
                return
            nxt = min(ready, key=lambda w: (w[0].priority, w[1]))
            self.waiting.remove(nxt)
            self.free -= 1
            nxt[0].active += 1
            nxt[2].set_result(None)


class PageAudioJob:
    """Synthesizes one page; any number of listeners can stream it while it's being made."""

    def __init__(self, text, voice, audio_path, priority, translate_to=''):
        self.text = text
        self.voice = voice
        self.audio_path = audio_path
        self.priority = priority  # Lower runs sooner; >= 0 means it's only a preload
        self.translate_to = translate_to
        self.active = 0           # Sections currently being synthesized
        # Filled in by run() once the text is ready (after translating, if asked)
        self.segments = None
        self.parts = []
        self.part_done = []
        self.error = None
        self.error_message = None  # What to tell the listener if something went wrong
        self.cond = threading.Condition()

    def _prepare(self, segments):
        with self.cond:
            self.parts = [bytearray() for _ in segments]
            self.part_done = [False] * len(segments)
            self.segments = segments
            self.cond.notify_all()

    @property
    def max_active(self):
        return 1 if self.priority >= 0 else MAX_TTS_CONNECTIONS

    def stream(self):
        """Yields the page's MP3 bytes in order, waiting on sections that aren't ready yet."""
        with self.cond:
            while self.segments is None and not self.error:
                self.cond.wait(timeout=30)
        index, offset = 0, 0
        while index < len(self.parts):
            with self.cond:
                while (len(self.parts[index]) == offset and not self.part_done[index]
                       and not self.error):
                    self.cond.wait(timeout=30)
                chunk = bytes(self.parts[index][offset:])
                finished = self.part_done[index]
                failed = self.error is not None
            if chunk:
                offset += len(chunk)
                yield chunk
            elif finished:
                index, offset = index + 1, 0
            elif failed:
                return

    async def _make_segment(self, i):
        for attempt in range(3):
            if self.error:
                return
            await _slots.acquire(self)
            try:
                async for message in edge_tts.Communicate(self.segments[i], self.voice).stream():
                    if message["type"] == "audio":
                        with self.cond:
                            self.parts[i] += message["data"]
                            self.cond.notify_all()
                with self.cond:
                    self.part_done[i] = True
                    self.cond.notify_all()
                return
            except Exception as e:
                # Only retry if nothing was sent yet, otherwise listeners would hear it twice
                if self.parts[i] or attempt == 2:
                    app.logger.error("TTS failed for %s: %s", self.audio_path, e)
                    with self.cond:
                        self.error = str(e) or type(e).__name__
                        self.cond.notify_all()
                    return
            finally:
                _slots.release(self)
            await asyncio.sleep(2 * (attempt + 1))

    async def run(self):
        try:
            text = self.text
            if self.translate_to:
                text = await asyncio.to_thread(translate_text, text, self.translate_to)
            self._prepare(split_into_segments(text) or [text])
        except Exception as e:
            app.logger.error("Couldn't prepare %s: %s", self.audio_path, e)
            with self.cond:
                self.error = str(e) or type(e).__name__
                if isinstance(e, TranslationError):
                    self.error_message = (f"Couldn't translate this page ({e}). Nothing was saved, "
                                          "so it will be translated properly when you try again.")
                self.cond.notify_all()
            return
        await asyncio.gather(*(self._make_segment(i) for i in range(len(self.segments))))
        if not self.error:
            # Save the finished page so it plays (and seeks) instantly next time
            os.makedirs(os.path.dirname(self.audio_path), exist_ok=True)
            temp_path = self.audio_path + '.part'
            with open(temp_path, 'wb') as f:
                for part in self.parts:
                    f.write(part)
            os.replace(temp_path, self.audio_path)
            with _jobs_lock:
                if _jobs.get(self.audio_path) is self:
                    del _jobs[self.audio_path]


# All synthesis runs on one background event loop
_tts_loop = asyncio.new_event_loop()
threading.Thread(target=_tts_loop.run_forever, daemon=True).start()
_slots = _ConnectionSlots(MAX_TTS_CONNECTIONS)


def audio_paths(filename, page_num, voice, translate_to=''):
    slug = book_slug(filename)
    # Translated pages are tagged "-translated": older versions could save English audio
    # under a translated name, and those files must not be reused
    lang_tag = f"{translate_to}-translated" if translate_to else 'original'
    audio_filename = f"page_{page_num}_{voice}_{lang_tag}.mp3"
    return (os.path.join(AUDIO_FOLDER, slug, audio_filename),
            url_for('static', filename=f'audio/{slug}/{audio_filename}'))


def is_cached(audio_path):
    return os.path.exists(audio_path) and os.path.getsize(audio_path) > 0


def start_job(audio_path, text, voice, priority, translate_to=''):
    """Returns the running job for this page, starting one if needed (failed jobs are retried)."""
    with _jobs_lock:
        job = _jobs.get(audio_path)
        if job and not job.error:
            if priority < job.priority:
                # Someone's listening now: let it use more connections right away
                job.priority = priority
                _tts_loop.call_soon_threadsafe(_slots.dispatch)
            return job
        job = PageAudioJob(text, voice, audio_path, priority, translate_to)
        _jobs[audio_path] = job
    asyncio.run_coroutine_threadsafe(job.run(), _tts_loop)
    return job


def audio_request_args():
    """Reads and validates the book / translation query parameters shared by the audio routes."""
    filename = request.args.get('book', '')
    translate_to = request.args.get('translate_to', '')
    if translate_to not in LANGUAGE_CODES:
        translate_to = ''
    return filename, translate_to


# =============================================
#  PAGES & LIBRARY
# =============================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/list_books')
def list_books():
    """Every PDF in the books folder, most recently read first."""
    info = get_books_info()
    books = []
    for name in os.listdir(BOOKS_FOLDER):
        path = os.path.join(BOOKS_FOLDER, name)
        if not name.lower().endswith('.pdf') or not os.path.isfile(path):
            continue
        saved = info.get(name, {})
        total = saved.get("total_pages")
        if not total:
            try:
                with fitz.open(path) as doc:
                    total = len(doc)
            except Exception:
                continue  # Not a readable PDF
        books.append({
            "filename": name,
            "title": saved.get("title") or book_title(name),
            "total_pages": total,
            "last_page": saved.get("last_page") or 1,
            "last_opened": saved.get("last_opened"),
            "size_mb": round(os.path.getsize(path) / 1_048_576, 1),
        })
    books.sort(key=lambda b: b["title"].lower())
    books.sort(key=lambda b: b["last_opened"] or "", reverse=True)
    return jsonify({"folder": BOOKS_FOLDER, "books": books})


@app.route('/book_file/<path:filename>')
def book_file(filename):
    return send_from_directory(BOOKS_FOLDER, filename, mimetype='application/pdf')


@app.route('/open_book', methods=['POST'])
def open_book():
    filename = (request.get_json(silent=True) or {}).get('book', '')
    pdf_path = resolve_book(filename)
    if not pdf_path:
        return jsonify({"error": "That book isn't in the books folder anymore."}), 404

    try:
        with fitz.open(pdf_path) as doc:
            total = len(doc)
    except Exception:
        return jsonify({"error": "Couldn't read that PDF — it may be damaged."}), 400

    # Keep a title saved earlier (e.g. the real name of a Gutenberg book)
    title = get_books_info().get(filename, {}).get("title") or book_title(filename)
    register_book(filename, title, total)
    last_page, last_position = get_progress(filename)

    return jsonify({
        "filename": filename,
        "title": title,
        "total_pages": total,
        "pdf_url": url_for('book_file', filename=filename),
        "resume_page": min(max(last_page or 1, 1), total),
        "resume_position": last_position or 0
    })


@app.route('/upload', methods=['POST'])
def upload_file():
    """Saves an uploaded PDF into the books folder so it stays in the library."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    name = secure_filename(file.filename) or 'book.pdf'
    if not name.lower().endswith('.pdf'):
        return jsonify({"error": "Only PDF files are supported"}), 400

    # Same name and same size: it's the book we already have, so just reuse it
    existing = os.path.join(BOOKS_FOLDER, name)
    file.stream.seek(0, os.SEEK_END)
    size = file.stream.tell()
    file.stream.seek(0)
    if os.path.isfile(existing) and os.path.getsize(existing) == size:
        return jsonify({"filename": name})

    name = unique_book_path(name)
    file.save(os.path.join(BOOKS_FOLDER, name))
    return jsonify({"filename": name})


@app.route('/open_books_folder', methods=['POST'])
def open_books_folder():
    """Opens the books folder in the file explorer so PDFs can be dropped in."""
    if sys.platform == 'win32':
        os.startfile(BOOKS_FOLDER)
    elif sys.platform == 'darwin':
        subprocess.Popen(['open', BOOKS_FOLDER])
    else:
        subprocess.Popen(['xdg-open', BOOKS_FOLDER])
    return jsonify({"ok": True})


# =============================================
#  AUDIO
# =============================================
@app.route('/get_audio/<int:page_num>/<voice>')
def get_audio(page_num, voice):
    """Returns where to play a page from, starting synthesis in the background if needed.
    Optional query param: ?translate_to=<lang_code> reads the page translated."""
    filename, translate_to = audio_request_args()
    pdf_path = resolve_book(filename)
    if not pdf_path:
        return jsonify({"error": "No book open"}), 400
    if not VOICE_RE.match(voice):
        return jsonify({"error": "Unknown voice"}), 400

    audio_path, cached_url = audio_paths(filename, page_num, voice, translate_to)

    # Serve from cache if already generated (empty files are failed runs, so redo them)
    if is_cached(audio_path):
        return jsonify({"audio_url": cached_url, "complete": True})

    with fitz.open(pdf_path) as doc:
        if not 1 <= page_num <= len(doc):
            return jsonify({"error": f"Page {page_num} doesn't exist"}), 404
    text = page_text_for_tts(pdf_path, page_num)

    # Blank/image-only page detection — skip rather than generating silence
    if len(text.strip()) < 20:
        return jsonify({"skip": True, "reason": "blank_or_image_page"})

    # Pages being listened to now jump ahead of preloads (most recent request first)
    priority = 0 if request.args.get('preload') else -time.time()
    start_job(audio_path, text, voice, priority, translate_to)
    return jsonify({
        "audio_url": url_for('stream_audio', page_num=page_num, voice=voice,
                             book=filename, translate_to=translate_to or None),
        "complete": False
    })


@app.route('/stream_audio/<int:page_num>/<voice>')
def stream_audio(page_num, voice):
    """Plays a page while it's still being synthesized."""
    filename, translate_to = audio_request_args()
    if not resolve_book(filename) or not VOICE_RE.match(voice):
        return "Not found", 404

    audio_path, _ = audio_paths(filename, page_num, voice, translate_to)
    if is_cached(audio_path):
        return send_file(audio_path, mimetype='audio/mpeg', conditional=True)

    with _jobs_lock:
        job = _jobs.get(audio_path)
    if not job:
        return "This page isn't being generated", 404
    return Response(job.stream(), mimetype='audio/mpeg', headers={"Cache-Control": "no-store"})


@app.route('/audio_status/<int:page_num>/<voice>')
def audio_status(page_num, voice):
    """Lets the player check a page was read in full before moving on."""
    filename, translate_to = audio_request_args()
    if not resolve_book(filename) or not VOICE_RE.match(voice):
        return jsonify({"ok": False, "error": "No book open"})

    audio_path, _ = audio_paths(filename, page_num, voice, translate_to)
    with _jobs_lock:
        job = _jobs.get(audio_path)
    if job and job.error:
        return jsonify({
            "ok": False,
            "error": job.error_message or
                     "The voice service stopped responding. Check your internet connection, "
                     "or run: pip install --upgrade edge-tts"
        })
    return jsonify({"ok": True})


@app.route('/get_voices')
def get_voices():
    """
    Returns voices from Edge-TTS, optionally filtered by locale prefix.
    Usage: /get_voices?locale=en-  →  English voices only
           /get_voices              →  all voices
    """
    locale_filter = request.args.get('locale', '')
    voices = asyncio.run(edge_tts.list_voices())
    if locale_filter:
        voices = [v for v in voices if v['Locale'].startswith(locale_filter)]
    return jsonify(voices)


@app.route('/get_supported_languages')
def get_supported_languages():
    """Returns the curated list of supported languages for the UI."""
    return jsonify(SUPPORTED_LANGUAGES)


# =============================================
#  TOC, PROGRESS & BOOKMARKS
# =============================================
@app.route('/get_toc')
def get_toc():
    """Returns the PDF's built-in Table of Contents / outline."""
    pdf_path = resolve_book(request.args.get('book', ''))
    if not pdf_path:
        return jsonify([])
    with fitz.open(pdf_path) as doc:
        toc = doc.get_toc()  # [[level, title, page_number], ...]
    return jsonify([{"level": item[0], "title": item[1], "page": item[2]} for item in toc])


@app.route('/save_progress', methods=['POST'])
def save_progress_route():
    data = request.get_json(silent=True) or {}
    if not resolve_book(data.get('book', '')):
        return jsonify({"ok": False})
    save_progress(data['book'], data['page'], data['position'])
    return jsonify({"ok": True})


@app.route('/add_bookmark', methods=['POST'])
def add_bookmark_route():
    data = request.get_json(silent=True) or {}
    if not resolve_book(data.get('book', '')):
        return jsonify({"ok": False})
    add_bookmark(
        data['book'],
        data['page'],
        data['position'],
        data.get('note', '')
    )
    return jsonify({"ok": True})


@app.route('/get_bookmarks')
def get_bookmarks_route():
    filename = request.args.get('book', '')
    if not resolve_book(filename):
        return jsonify([])
    rows = get_bookmarks(filename)
    return jsonify([
        {"id": r[0], "page": r[1], "position": r[2], "note": r[3], "date": r[4]}
        for r in rows
    ])


@app.route('/delete_bookmark/<int:bookmark_id>', methods=['DELETE'])
def delete_bookmark_route(bookmark_id):
    delete_bookmark(bookmark_id)
    return jsonify({"ok": True})


# =============================================
#  PROJECT GUTENBERG
# =============================================
GUTENDEX_API = 'https://gutendex.com/books'


@app.route('/gutenberg/search')
def gutenberg_search():
    """Proxy search to Gutendex API. Returns book results as JSON."""
    query = request.args.get('q', '').strip()
    page  = request.args.get('page', '1')
    if not query:
        return jsonify({"count": 0, "results": []})

    try:
        resp = http_requests.get(
            GUTENDEX_API,
            params={'search': query, 'page': page},
            timeout=10
        )
        data = resp.json()

        # Slim down the response to only what the frontend needs
        results = []
        for book in data.get('results', []):
            # Only include books that have text format (readable books)
            formats = book.get('formats', {})
            has_text = any(k.startswith('text/plain') for k in formats)
            if not has_text:
                continue

            authors = ', '.join(a['name'] for a in book.get('authors', []))
            cover   = formats.get('image/jpeg', '')
            langs   = book.get('languages', [])

            results.append({
                'id':        book['id'],
                'title':     book.get('title', 'Untitled'),
                'author':    authors or 'Unknown',
                'cover':     cover,
                'languages': langs,
                'downloads': book.get('download_count', 0),
                'summary':   (book.get('summaries') or [''])[0][:200],
            })

        return jsonify({
            'count':    data.get('count', 0),
            'next':     data.get('next'),
            'previous': data.get('previous'),
            'results':  results
        })
    except Exception as e:
        print(f'[Gutenberg search error] {e}')
        return jsonify({'error': str(e)}), 502


def text_to_pdf(text, title, filepath, chars_per_page=3000):
    """
    Convert plain text into a clean, multi-page PDF using fpdf2.
    Returns the total number of pages.
    """
    # 1. Swap fancy Unicode typography for standard ASCII equivalents
    replacements = {
        '—': '--', '–': '-',
        '“': '"', '”': '"',
        '‘': "'", '’': "'",
        '…': '...',
        '​': '',
        '\xad': '-',
        '\t': '    '  # Tabs break fpdf2's width calculations
    }
    for fancy, basic in replacements.items():
        text = text.replace(fancy, basic)
        title = title.replace(fancy, basic)

    # 2. Catch-all: ignore any other unmappable characters
    text = text.encode('latin-1', 'ignore').decode('latin-1')
    title = title.encode('latin-1', 'ignore').decode('latin-1')

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(25, 25, 25)

    pdf.add_page()

    # Protect the title page: at font size 24 even shorter long-words can exceed the page width
    safe_title_words = []
    for word in title.split(' '):
        if len(word) > 20:
            chunks = [word[i:i+20] for i in range(0, len(word), 20)]
            safe_title_words.append(' '.join(chunks))
        else:
            safe_title_words.append(word)
    safe_title = ' '.join(safe_title_words)

    # Title page
    pdf.set_font('Helvetica', 'B', 24)
    pdf.ln(60)
    pdf.multi_cell(0, 12, safe_title, align='C')
    pdf.ln(10)
    pdf.set_font('Helvetica', '', 11)
    pdf.multi_cell(0, 7, 'From Project Gutenberg - Free eBooks', align='C')

    # Body pages
    pdf.set_font('Helvetica', '', 11)

    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            pdf.ln(4)
        else:
            # Chunk huge "words" (URLs etc.) so they fit on a line, keeping paragraph indents
            safe_words = []
            for word in line.split(' '):
                if len(word) > 50:
                    chunks = [word[i:i+50] for i in range(0, len(word), 50)]
                    safe_words.append(' '.join(chunks))
                else:
                    safe_words.append(word)
            pdf.multi_cell(0, 5.5, ' '.join(safe_words))

    pdf.output(filepath)

    with fitz.open(filepath) as doc:
        return len(doc)


@app.route('/gutenberg/download/<int:book_id>')
def gutenberg_download(book_id):
    """
    Download a book from Project Gutenberg, convert it to PDF and save it into the
    books folder, so it stays in the library like any uploaded book.
    """
    try:
        # 1. Get book metadata from Gutendex
        meta_resp = http_requests.get(f'{GUTENDEX_API}/{book_id}', timeout=10)
        meta = meta_resp.json()
        title = meta.get('title', f'Book {book_id}')
        formats = meta.get('formats', {})

        stem = (secure_filename(title) or 'book')[:80]
        filename = f'{stem}-pg{book_id}.pdf'
        if resolve_book(filename):
            # Downloaded before: reuse it (keeps reading progress)
            return jsonify({'filename': filename})

        # 2. Find plain text URL
        text_url = None
        for key, url in formats.items():
            if key.startswith('text/plain') and 'utf-8' in key:
                text_url = url
                break
        if not text_url:
            for key, url in formats.items():
                if key.startswith('text/plain'):
                    text_url = url
                    break
        if not text_url:
            return jsonify({'error': 'No plain text version available for this book'}), 404

        # 3. Download the text
        print(f'[Gutenberg] Downloading text for "{title}" from {text_url}…')
        text_resp = http_requests.get(text_url, timeout=30)
        text_resp.encoding = 'utf-8'
        book_text = text_resp.text

        # 4. Convert to PDF (written to a temp name so a failed run leaves no broken book)
        filepath = os.path.join(BOOKS_FOLDER, filename)
        temp_path = filepath + '.part'
        print(f'[Gutenberg] Converting {len(book_text)} chars to PDF…')
        total = text_to_pdf(book_text, title, temp_path)
        os.replace(temp_path, filepath)
        print(f'[Gutenberg] Done — {total} pages.')

        # 5. Save the real title so the library shows it
        register_book(filename, title, total)
        return jsonify({'filename': filename})

    except Exception as e:
        print(f'[Gutenberg download error] {e}')
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
