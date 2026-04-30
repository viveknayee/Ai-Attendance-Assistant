import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection(name="attendance")

# Search ChromaDB
question = "who was present on 28 april"
question_embedding = model.encode(question).tolist()

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)


# Convert results to text
context = "\n".join(results['documents'][0])
print(f"Context sending to AI:\n{context}\n")

# Send to Groq
response = groq_client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": "You are an attendance assistant. Answer using only the context provided."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
)

print(f"AI Answer: {response.choices[0].message.content}")