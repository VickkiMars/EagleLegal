from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from call_gemini import call_gemini as cg
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import os

templates = Jinja2Templates(directory="templates")

# -----------------------------
# 1. Setup
# -----------------------------
app = FastAPI(title="EagleLegal", version="1.0")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# Load documents
try:
    acts_path = "assets/acts.json"
    print(f"Loading documents from {acts_path}...")
    with open(acts_path, "r", encoding="utf-8") as f:
        documents = json.load(f)
    print(f"Loaded {len(documents)} documents.")
except FileNotFoundError:
    print(f"ERROR: File not found - {acts_path}")
    documents = []
except json.JSONDecodeError as e:
    print(f"ERROR: Failed to parse JSON - {e}")
    documents = []

# Load FAISS index
try:
    faiss_path = "assets/EagleLegal_mini.faiss"
    print(f"Loading FAISS index from {faiss_path}...")
    if not os.path.exists(faiss_path):
        raise FileNotFoundError(f"{faiss_path} does not exist")
    index = faiss.read_index(faiss_path)
    print("FAISS index loaded successfully.")
except Exception as e:
    print(f"ERROR loading FAISS index: {e}")
    index = None

# Load sentence transformer
try:
    print("Loading SentenceTransformer model...")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
    print("SentenceTransformer loaded successfully.")
except Exception as e:
    print(f"ERROR loading SentenceTransformer: {e}")
    encoder = None

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
    if encoder is None:
        print("ERROR: Encoder is not loaded.")
        raise RuntimeError("SentenceTransformer encoder not loaded")
    if index is None:
        print("ERROR: FAISS index is not loaded.")
        raise RuntimeError("FAISS index not loaded")
    if not documents:
        print("ERROR: Document list is empty.")
        raise RuntimeError("No documents loaded")

    try:
        print(f"Encoding query: {query}")
        query_emb = encoder.encode([query], convert_to_numpy=True)
        # query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
        # print("Query embedding normalized.")

        scores, idxs = index.search(query_emb, top_k)
        print(f"Retrieved top {top_k} document indices: {idxs}")
        retrieved = [documents[i] for i in idxs[0]]
        print(f"Retrieved documents: {retrieved}")
        return retrieved
    except Exception as e:
        print(f"ERROR during retrieval: {e}")
        raise

# -----------------------------
# 4. Multi-hop RAG function
# -----------------------------
def rag_multi_hop(query, num_hops=2, top_k=3):
    all_docs = []
    current_query = query

    if not query:
        print("ERROR: Empty query")
        return {"query": query, "answer": "ERROR: Empty query", "retrieved_docs": []}

    try:
        for hop in range(1, num_hops + 1):
            print(f"[Hop {hop}] Retrieving top {top_k} documents for query: '{current_query}'")
            top_docs = retrieve(current_query, top_k)
            print(f"[Hop {hop}] Retrieved documents: {top_docs}")

            all_docs.extend(top_docs)
            current_query += " " + " ".join(top_docs)

        context = "\n\n".join(all_docs)
        prompt = (
            f"## Question\n\n{query}\n\n"
            f"## Context\n\n{context}\n\n"
            "## Analysis\n\n"
            "Begin direct analysis without transitional phrases.\n\n"
            "## Answer\n\n"
            "Present findings using markdown formatting:\n"
            "- Use **bold** for key terms and titles\n"
            "- Apply bullet points for listed items\n"
            "- Separate sections with blank lines\n"
            "- Eliminate all conversational language\n"
            "- Remove phrases like 'according to the document' or 'based on'\n"
            "- Do not mirror the question's structure or phrasing\n"
            "- If context is insufficient, state: **I don't know**\n"
            "Base response exclusively on provided context without external knowledge."
        )
        print("Calling LLM via call_gemini with multi-hop context...")
        llm_response = cg(prompt)
        print(f"LLM multi-hop response: {llm_response}")

        return {"query": query, "answer": llm_response, "retrieved_docs": all_docs}

    except Exception as e:
        print(f"ERROR in multi-hop RAG: {e}")
        return {"query": query, "answer": "ERROR: Unable to generate response", "retrieved_docs": all_docs}

# -----------------------------
# 5. Inference route (multi-hop)
# -----------------------------
@app.post("/infer")
def infer(req: QueryRequest):
    result = rag_multi_hop(req.query, num_hops=2, top_k=req.top_k)
    if result["answer"].startswith("ERROR"):
        raise HTTPException(status_code=500, detail=result["answer"])
    return result

# -----------------------------
# 6. Static and template routes
# -----------------------------
@app.get("/")
def root():
    try:
        print("Serving static index.html")
        return FileResponse("static/index.html")
    except Exception as e:
        print(f"ERROR serving index.html: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat")
def chat(request: Request):
    try:
        print("Serving chat.html template")
        return templates.TemplateResponse("chat.html", {"request": request})
    except Exception as e:
        print(f"ERROR serving chat.html: {e}")
        raise HTTPException(status_code=500, detail=str(e))
