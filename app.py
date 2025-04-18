from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import torch

# Initialize app
app = FastAPI(title="RAG QA API")

# Globals
retriever = None
documents = []
index = None
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
generator = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# On startup: load retriever and FAISS index
@app.on_event("startup")
def load_everything():
    global retriever, documents, index
    print(">>> Loading retriever")
    retriever = SentenceTransformer("all-MiniLM-L6-v2")
    print(">>> Retriever loaded")

    try:
        with open("docs.txt", "r") as f:
            documents = [line.strip() for line in f if line.strip()]
        print(f">>> Loaded {len(documents)} documents")
    except Exception as e:
        print(f"!!! Failed to load docs.txt: {e}")
        documents = []

    if documents:
        embeddings = retriever.encode(documents)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        print(">>> FAISS index built")

# Request schema
class Question(BaseModel):
    query: str
    top_k: int = 2

# QA logic
def get_answer(query: str, top_k: int):
    if not retriever or not index or not documents:
        return "System not ready"

    q_embed = retriever.encode([query])
    D, I = index.search(q_embed, top_k)  # D: distances, I: indices

    # Heuristic: check if top similarity score is too low
    threshold = 0.75  # adjust this based on your document embedding quality
    best_distance = D[0][0]
    if best_distance > threshold:  # higher L2 distance = less similar
        return "The answer to this question does not exist in the provided document!"

    retrieved = " ".join([documents[i] for i in I[0]])
    prompt = f"Question: {query} Context: {retrieved}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = generator.generate(**inputs, max_new_tokens=64)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# POST endpoint
@app.post("/ask")
def ask_question(q: Question):
    answer = get_answer(q.query, q.top_k)
    return {"question": q.query, "answer": answer}
