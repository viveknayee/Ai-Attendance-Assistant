from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os
import sqlite3
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

load_dotenv()

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path = "./chroma_store")
collection = chroma_client.get_or_create_collection(name="attendance")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

chat_history = []

def sync_chromadb():
    existing = collection.get()
    existing_ids = set(existing['ids'])

    conn = sqlite3.connect(r"D:\PROJECTS\Face_recognition\instance\attendance.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, date, time FROM attendance")
    rows = cursor.fetchall()
    conn.close()

    for row in rows:
        id,name,date,time = row
        if str(id) not in existing_ids:
            text = f"{name} was present on {date} at {time}"
            embedding = model.encode(text).tolist()
            collection.add(
                documents = [text],
                embeddings=[embedding],
                metadatas=[{"name":name,"date":date}],
                ids=[str(id)]
            )
    print(f"ChromaDB synced. Total: {collection.count()}")

sync_chromadb()



class ChatRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/chat")
def chat(request: ChatRequest):

    question = request.question

    # Search ChromaDB
    question_embedding = model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=collection.count()
    )

    context = "\n".join(results['documents'][0])

    # Build messages with history
    messages = [
    {"role": "system", "content": "You are an attendance assistant for a face recognition system. You have access to attendance records with name, date, and time. Rules: 1. Answer ONLY using the data provided — never guess or predict. 2. For counting questions — count exactly from the data. 3. For date questions — match exact dates only. 4. For summary questions — list all names with their count. 5. If data not found — say 'No records found for this query.' 6. Never add extra information not present in data. 7. Always give consistent answers — same question same answer. 8. Date format in data is YYYY-MM-DD — use it exactly."},
    *chat_history[-6:],
    {"role": "user", "content": f"Attendance data:\n{context}\n\nQuestion: {question}"}
    ]

    # Send to Groq
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.0
    )

    answer = response.choices[0].message.content

    # Save to history
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})

    return ({"answer": answer})

@app.get("/health")
def health():
    return{"status": "ok"}