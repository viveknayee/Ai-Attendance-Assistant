from sentence_transformers import SentenceTransformer
import chromadb
import sqlite3
import os

model = SentenceTransformer('all-MiniLM-L6-v2')
DB_PATH = os.getenv("DB_PATH")
# Open ChromaDB
client = chromadb.PersistentClient(path = "./chroma_store")
collection = client.get_or_create_collection(name = "attendance")

# Fetch data from SQLite
def fetch_attendance():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, date, time FROM attendance ORDER BY date DESC;")
    rows = cursor.fetchall()
    conn.close()
    return rows

rows = fetch_attendance()

for row in rows:
    id,name,date, time = row
    text = f"{name} was present on {date} at {time}"
    embedding = model.encode(text).tolist()
    
    # Store in ChromaDB
    collection.add(
        documents=[text],
        embeddings=[embedding],
        metadatas=[{"name": name, "date":date}],
        ids=[str(id)]
    )

query = "who was present recently"
query_embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

for doc in results['documents'][0]:
    print(f"→ {doc}")
print(f"Total in collection: {collection.count()}")