import sqlite3

DB_PATH = "audiobook.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS books (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT UNIQUE,
        title TEXT,
        total_pages INTEGER,
        last_page INTEGER DEFAULT 1,
        last_position REAL DEFAULT 0,
        added_date TEXT DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS bookmarks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        book_id INTEGER,
        page INTEGER,
        position REAL,
        note TEXT,
        created_date TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (book_id) REFERENCES books(id)
    )''')
    conn.commit()
    conn.close()

def register_book(filename, title, total_pages):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT OR IGNORE INTO books (filename, title, total_pages) VALUES (?, ?, ?)',
              (filename, title, total_pages))
    c.execute('UPDATE books SET total_pages=?, title=? WHERE filename=?',
              (total_pages, title, filename))
    conn.commit()
    conn.close()

def save_progress(filename, page, position):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE books SET last_page=?, last_position=? WHERE filename=?',
              (page, position, filename))
    conn.commit()
    conn.close()

def get_progress(filename):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT last_page, last_position FROM books WHERE filename=?', (filename,))
    row = c.fetchone()
    conn.close()
    return row if row else (1, 0.0)

def add_bookmark(filename, page, position, note=""):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id FROM books WHERE filename=?', (filename,))
    book = c.fetchone()
    if book:
        c.execute('INSERT INTO bookmarks (book_id, page, position, note) VALUES (?, ?, ?, ?)',
                  (book[0], page, position, note))
    conn.commit()
    conn.close()

def get_bookmarks(filename):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT bm.id, bm.page, bm.position, bm.note, bm.created_date
                 FROM bookmarks bm
                 JOIN books b ON bm.book_id = b.id
                 WHERE b.filename=?
                 ORDER BY bm.page''', (filename,))
    rows = c.fetchall()
    conn.close()
    return rows

def delete_bookmark(bookmark_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM bookmarks WHERE id=?', (bookmark_id,))
    conn.commit()
    conn.close()
