from flask import Flask, request, jsonify
import sqlite3
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Store conversation history
history = [
    {"role": "system", "content": "You are an attendance assistant. Answer questions using only the attendance data provided. Always give short, direct answers. State the conclusion first, then explain briefly."}
]

def fetch_attendance():
    conn = sqlite3.connect(r"D:\PROJECTS\Face_recognition\instance\attendance.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name, date, time FROM attendance ORDER BY date DESC;")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return "No attendance records found."

    text = "Attendance Records:\n"
    for row in rows:
        text += f"- {row[0]} was present on {row[1]} at {row[2]}\n"
    return text

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data["question"]

    attendance_data = fetch_attendance()

    history.append({"role": "user", "content": f"{attendance_data}\n\nQuestion: {question}"})

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=history
    )

    answer = response.choices[0].message.content
    history.append({"role": "assistant", "content": answer})

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)