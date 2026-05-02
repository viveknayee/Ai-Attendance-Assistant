import sqlite3
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
DB_PATH = os.getenv("DB_PATH")
# --- Step 1: Fetch attendance data from FaceAttend DB ---
def fetch_attendance():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, date, time FROM attendance ORDER BY date DESC;")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return "No attendance records found."

    # Convert rows into readable text for AI
    text = "Attendance Records:\n"
    for row in rows:
        text += f"- {row[0]} was present on {row[1]} at {row[2]}\n"
    return text

# --- Step 2: Ask AI a question using that data ---
def ask_ai(question):
    attendance_data = fetch_attendance()

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are an attendance assistant. Answer questions using only the attendance data provided."},
            {"role": "user", "content": f"{attendance_data}\n\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content

# --- Step 3: Test it ---
print("AI Attendance Assistant — Ask anything!")
print("-" * 40)

while True:
    question = input("You: ")
    if question.lower() == "quit":
        print("Goodbye!")
        break
    answer = ask_ai(question)
    print(f"AI: {answer}")
    print("-" * 40)