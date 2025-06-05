import sqlite3
import hashlib

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

conn = sqlite3.connect("users.db")
c = conn.cursor()
c.execute("INSERT OR IGNORE INTO users(username, password, is_admin) VALUES (?, ?, ?)", 
          ("admin", hash_password("adminpass"), 1))
conn.commit()
conn.close()

print("Admin created: admin / adminpass")
