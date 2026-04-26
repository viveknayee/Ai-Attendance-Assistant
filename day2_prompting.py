from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key = os.getenv("GROQ_API_KEY"))
message = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": "You are an attendance assistant."},
        {"role": "user", "content": "Who was absent today?"},
        {"role": "assistant", "content": "John and Mary were absent today."},
        {"role": "user", "content": "How many days this week were they absent?"}
    ]
)
print(message.choices[0].message.content)