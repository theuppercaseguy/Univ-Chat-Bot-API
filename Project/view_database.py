import sqlite3

def view_chat_history():
    conn = sqlite3.connect("chatbot_database.db")
    cursor = conn.cursor()

    # Fetch all rows from the chat_history table
    cursor.execute("SELECT * FROM History")
    rows = cursor.fetchall()

    print("Chat History:")
    print("ID\t\tUSER\t\tQuestion\t\tAnswer")
    print("-" * 80)
    for row in rows:
        print(f"{row[0]}\t\t{row[1]}\t\t{row[2]}\t\t{row[3]}")

    conn.close()

if __name__ == "__main__":
    view_chat_history()
