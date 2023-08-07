import sqlite3

# Create a connection to the SQLite database
conn = sqlite3.connect("chatbot_database.db")
cursor = conn.cursor()

# Create a table to store the chat history
cursor.execute('''CREATE TABLE IF NOT EXISTS History (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user Text NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL
                 )''')

# Commit the changes and close the connection
conn.commit()
conn.close()
