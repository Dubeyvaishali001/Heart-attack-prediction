# create_db.py
import sqlite3

conn = sqlite3.connect('user_history.db')
cursor = conn.cursor()



# Table for user accounts
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    email TEXT,
    password TEXT
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    age INTEGER,
    sex TEXT,
    cp TEXT,
    trestbps REAL,
    chol REAL,
    fbs TEXT,
    restecg TEXT,
    thalach REAL,
    exang TEXT,
    oldpeak REAL,
    slope TEXT,
    ca TEXT,
    thal TEXT,
    prediction TEXT,
    report_path TEXT,
    timestamp TEXT
)
''')

conn.commit()
conn.close()
