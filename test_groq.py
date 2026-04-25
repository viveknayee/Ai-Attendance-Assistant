from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

message = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
    {"role": "system", "content": "You are an AI attendance assistant. Answer only attendance related questions."},
    {"role": "user", "content":"What is your job?"}
]
)

print(message.choices[0].message.content)