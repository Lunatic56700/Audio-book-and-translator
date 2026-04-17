import os
import asyncio
import glob
import textwrap
import fitz  # PyMuPDF
import requests as http_requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, render_template, request, jsonify, url_for
from fpdf import FPDF
import edge_tts
from deep_translator import GoogleTranslator
from database import (
    init_db, register_book, save_progress, get_progress,
    add_bookmark, get_bookmarks, delete_bookmark
)

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
AUDIO_FOLDER  = os.path.join('static', 'audio')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

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

session_data = {
    "pdf_path":     None,
    "total_pages":  0,
    "current_book": None
}


def clean_text_for_tts(text):
    """Clean PDF whitespace/newlines so TTS reads naturally."""
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = " ".join(text.split())
    return text


def translate_text(text, target_lang, chunk_size=4500):
    """
    Translate text to the specified target language via Google Translate.
    Splits into chunks (GoogleTranslator has a ~5000-char limit per call)
    and processes them in parallel for speed.
    """
    if not text.strip():
        return text

    # Split on sentence boundaries to keep chunks coherent
    chunks  = []
    current = ""
    for sentence in text.replace('. ', '.|').split('|'):
        if len(current) + len(sentence) < chunk_size:
            current += sentence + ' '
        else:
            if current:
                chunks.append(current.strip())
            current = sentence + ' '
    if current:
        chunks.append(current.strip())

    # Translate chunks in parallel using a thread pool
    translated_parts = [""] * len(chunks)

    def translate_chunk(index, chunk_text):
        try:
            translator = GoogleTranslator(source='auto', target=target_lang)
            return index, translator.translate(chunk_text)
        except Exception as e:
            print(f"[Translation error] {e} — keeping original chunk")
            return index, chunk_text  # graceful fallback

    with ThreadPoolExecutor(max_workers=min(len(chunks), 6)) as executor:
        futures = [
            executor.submit(translate_chunk, i, chunk)
            for i, chunk in enumerate(chunks)
        ]
        for future in as_completed(futures):
            idx, result = future.result()
            translated_parts[idx] = result

    return ' '.join(translated_parts)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        for f in glob.glob(os.path.join(AUDIO_FOLDER, "*.mp3")):
            os.remove(f)

        filename       = "current_book.pdf"
        original_title = os.path.splitext(file.filename)[0]
        filepath       = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        doc   = fitz.open(filepath)
        total = len(doc)

        session_data["pdf_path"]     = filepath
        session_data["total_pages"]  = total
        session_data["current_book"] = filename

        register_book(filename, original_title, total)
        last_page, last_position = get_progress(filename)

        return jsonify({
            "message":         "Upload successful",
            "total_pages":     total,
            "pdf_url":         url_for('static', filename=f'uploads/{filename}'),
            "title":           original_title,
            "resume_page":     last_page,
            "resume_position": last_position
        })


@app.route('/get_audio/<int:page_num>/<voice>')
def get_audio(page_num, voice):
    """
    Generate TTS audio for a page.
    Optional query param: ?translate_to=<lang_code>
    If translate_to is omitted, reads the original text (fast path).
    """
    if not session_data["pdf_path"]:
        return jsonify({"error": "No PDF uploaded"}), 400

    translate_to = request.args.get('translate_to', '')
    lang_tag = translate_to if translate_to else 'original'

    # Cache key includes the target language so translated and
    # non-translated audio don't collide
    audio_filename = f"page_{page_num}_{voice}_{lang_tag}.mp3"
    audio_path     = os.path.join(AUDIO_FOLDER, audio_filename)

    # Serve cached MP3 immediately
    if os.path.exists(audio_path):
        return jsonify({"audio_url": url_for('static', filename=f'audio/{audio_filename}')})

    doc  = fitz.open(session_data["pdf_path"])
    text = doc[page_num - 1].get_text("text", sort=True)
    text = clean_text_for_tts(text)

    if len(text.strip()) < 20:
        return jsonify({"skip": True, "reason": "blank_or_image_page"})

    # Only translate if a target language was requested
    if translate_to:
        print(f"[Page {page_num}] Translating {len(text)} chars to {translate_to}…")
        text = translate_text(text, translate_to)
        print(f"[Page {page_num}] Done ({len(text)} chars). Generating TTS…")
    else:
        print(f"[Page {page_num}] No translation — generating TTS for original text ({len(text)} chars)…")

    # Generate TTS audio
    async def generate_tts():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(audio_path)

    asyncio.run(generate_tts())

    return jsonify({"audio_url": url_for('static', filename=f'audio/{audio_filename}')})


@app.route('/get_toc')
def get_toc():
    if not session_data["pdf_path"]:
        return jsonify([])
    doc = fitz.open(session_data["pdf_path"])
    toc = doc.get_toc()
    return jsonify([{"level": item[0], "title": item[1], "page": item[2]} for item in toc])


@app.route('/get_voices')
def get_voices():
    """
    Returns voices from Edge-TTS, optionally filtered by locale prefix.
    Usage: /get_voices?locale=en-  →  English voices only
           /get_voices              →  all voices
    """
    locale_filter = request.args.get('locale', '')

    async def fetch():
        return await edge_tts.list_voices()

    voices = asyncio.run(fetch())

    if locale_filter:
        voices = [v for v in voices if v['Locale'].startswith(locale_filter)]

    return jsonify(voices)


@app.route('/get_supported_languages')
def get_supported_languages():
    """Returns the curated list of supported languages for the UI."""
    return jsonify(SUPPORTED_LANGUAGES)


@app.route('/save_progress', methods=['POST'])
def save_progress_route():
    if not session_data["current_book"]:
        return jsonify({"ok": False})
    data = request.json
    save_progress(session_data["current_book"], data['page'], data['position'])
    return jsonify({"ok": True})


@app.route('/add_bookmark', methods=['POST'])
def add_bookmark_route():
    if not session_data["current_book"]:
        return jsonify({"ok": False})
    data = request.json
    add_bookmark(
        session_data["current_book"],
        data['page'],
        data['position'],
        data.get('note', '')
    )
    return jsonify({"ok": True})


@app.route('/get_bookmarks')
def get_bookmarks_route():
    if not session_data["current_book"]:
        return jsonify([])
    rows = get_bookmarks(session_data["current_book"])
    return jsonify([
        {"id": r[0], "page": r[1], "position": r[2], "note": r[3], "date": r[4]}
        for r in rows
    ])


@app.route('/delete_bookmark/<int:bookmark_id>', methods=['DELETE'])
def delete_bookmark_route(bookmark_id):
    delete_bookmark(bookmark_id)
    return jsonify({"ok": True})


# ── Project Gutenberg Integration ─────────────────────────────────────

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
        '\u200B': '',           
        '\xad': '-',
        '\t': '    '  # <-- FIX 1: Convert tabs to spaces. Tabs break fpdf2's width calculations.
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

    # FIX 2: Protect the Title page. 
    # Font size 24 means even shorter long-words can exceed the page width and crash.
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

    lines = text.splitlines()
    for line in lines:
        line = line.rstrip()
        if not line:
            pdf.ln(4)
        else:
            # 3. THE NEW FIX: Now works perfectly because tabs are gone,
            # meaning .split(' ') won't miss anything, while still preserving paragraph indents.
            safe_words = []
            for word in line.split(' '):
                if len(word) > 50:
                    # Chunk the massive word into 50-character bites separated by a space
                    chunks = [word[i:i+50] for i in range(0, len(word), 50)]
                    safe_words.append(' '.join(chunks))
                else:
                    safe_words.append(word)
            
            safe_line = ' '.join(safe_words)
            pdf.multi_cell(0, 5.5, safe_line)

    pdf.output(filepath)

    doc = fitz.open(filepath)
    total = len(doc)
    doc.close()
    return total


@app.route('/gutenberg/download/<int:book_id>')
def gutenberg_download(book_id):
    """
    Download a book from Project Gutenberg, convert to PDF, and load it.
    """
    try:
        # 1. Get book metadata from Gutendex
        meta_resp = http_requests.get(f'{GUTENDEX_API}/{book_id}', timeout=10)
        meta = meta_resp.json()
        title = meta.get('title', f'Book {book_id}')
        formats = meta.get('formats', {})

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

        # 4. Clear old audio cache
        for f in glob.glob(os.path.join(AUDIO_FOLDER, '*.mp3')):
            os.remove(f)

        # 5. Convert to PDF
        filename = 'current_book.pdf'
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        print(f'[Gutenberg] Converting {len(book_text)} chars to PDF…')
        total = text_to_pdf(book_text, title, filepath)
        print(f'[Gutenberg] Done — {total} pages.')

        # 6. Register in session and database
        session_data['pdf_path']     = filepath
        session_data['total_pages']  = total
        session_data['current_book'] = filename

        register_book(filename, title, total)
        last_page, last_position = get_progress(filename)

        return jsonify({
            'message':         'Book loaded from Project Gutenberg',
            'total_pages':     total,
            'pdf_url':         url_for('static', filename=f'uploads/{filename}'),
            'title':           title,
            'resume_page':     last_page,
            'resume_position': last_position
        })

    except Exception as e:
        print(f'[Gutenberg download error] {e}')
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)