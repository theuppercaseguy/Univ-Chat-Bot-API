import sqlite3

# Create a connection to the SQLite database
conn = sqlite3.connect("chatbot_database.db")
cursor = conn.cursor()

# Create a table to store the chat history
cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL
                 )''')

# Create a table to store the embeddings
cursor.execute('''CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    embedding BLOB NOT NULL
                 )''')

# Commit the changes and close the connection
conn.commit()
conn.close()
