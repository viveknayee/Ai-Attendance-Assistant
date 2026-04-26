from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

message = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": """You are an attendance assistant.
Always answer in this exact format:

Example 1:
User: Is John present?
Assistant: ✅ John is PRESENT today. Total absences this month: 1

Example 2:
User: Is Mary present?
Assistant: ❌ Mary is ABSENT today. Total absences this month: 3

Always follow this exact format. Nothing else."""},

        {"role": "user", "content": "Is Vivek present?"}
    ]
)

print(message.choices[0].message.content)