# database.py
import sqlite3

def init_db():
    conn = sqlite3.connect('rankings.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS rankings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            accuracy REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def add_score(name, accuracy):
    conn = sqlite3.connect('rankings.db')
    c = conn.cursor()
    c.execute('INSERT INTO rankings (name, accuracy) VALUES (?, ?)', (name, accuracy))
    conn.commit()
    conn.close()

def get_rankings():
    conn = sqlite3.connect('rankings.db')
    c = conn.cursor()
    c.execute('SELECT name, accuracy FROM rankings ORDER BY accuracy DESC')
    rankings = c.fetchall()
    conn.close()
    return rankings

init_db()
