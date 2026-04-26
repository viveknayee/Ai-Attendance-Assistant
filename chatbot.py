from groq import Groq
import os 
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key = os.getenv("GROQ_API_KEY"))

history = [
    {"role": "system", "content": "You are an AI attendance assistant. You ONLY answer attendance related questions like present, absent, late markings. If anyone asks anything else, politely refuse and bring topic back to attendance."}
]
print("Attendance Assistant Started! Type 'quit' to exit.")
print("-" * 40)

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        print("Goodbye! ")
        break

    history.append({"role": "user", "content": user_input})

    responce = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=history
    )
    replay = responce.choices[0].message.content
    history.append({"role": "assistant", "content": replay})

    print(f"AI: {replay}")
    print("-" * 40)


