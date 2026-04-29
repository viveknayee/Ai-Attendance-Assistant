from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert text to numbers (embedding)
text = "Vivek was present on 2026-04-28"
embedding = model.encode(text)

print(f"Text: {text}")
print(f"Embedding size: {len(embedding)} numbers")
print(f"First 5 numbers: {embedding[:5]}")

