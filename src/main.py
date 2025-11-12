from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from call_gemini import call_gemini as cg
# -----------------------------
# 1. Setup
# -----------------------------
app = FastAPI(title="EagleLegal", version="1.0")

# Load documents (you must have saved them earlier)
with open("assets/acts.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

# Load FAISS index
index = faiss.read_index("assets/EagleLegal.faiss")

# Load sentence transformer
encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# -----------------------------
# 2. Request model
# -----------------------------
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

# -----------------------------
# 3. Retrieval function
# -----------------------------
def retrieve(query, top_k=5):
    query_emb = encoder.encode([query], convert_to_numpy=True)
    # Normalize for cosine similarity (optional)
    query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    scores, idxs = index.search(query_emb, top_k)
    return [documents[i] for i in idxs[0]]

# -----------------------------
# 4. Inference route
# -----------------------------
@app.post("/infer")
def infer(req: QueryRequest):
    try:
        # Retrieve context
        top_docs = retrieve(req.query, req.top_k)
        context = "\n\n".join(top_docs)

        # Build a reasoning-style prompt (you can plug in your LLM here)
        prompt = (
            f"Question: {req.query}\n\n"
            f"Context:\n{context}\n\n"
            "Think step by step and provide an answer. "
            "If you cannot get an answer from the context, respond with 'I don't know'."
        )
        response = cg(prompt)

        return {"query": req.query, "answer": response, "retrieved_docs": top_docs}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
