"""Core RAG pipeline: chunking, embedding, retrieval, and generation."""

import os
import re
import time
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    DOCUMENTS_DIR,
    OPENROUTER_API_URL,
    OPENROUTER_MODEL,
    OLLAMA_MODEL,
    OLLAMA_URL,
)

SYSTEM_PROMPT = (
    "You are an AI assistant for Indecimal, a construction marketplace. "
    "Answer the user's question ONLY using the provided context. "
    "If the context doesn't contain enough info, say so honestly. "
    "Always cite which document the info comes from. "
    "Be helpful, specific, and concise."
)


# ---------------------------------------------------------------------------
# 1. Document Loading
# ---------------------------------------------------------------------------

def load_documents(docs_dir: str = DOCUMENTS_DIR) -> list[dict]:
    """Read all .md files from the documents directory."""
    documents = []
    for fname in sorted(os.listdir(docs_dir)):
        if fname.endswith(".md"):
            fpath = os.path.join(docs_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            documents.append({"filename": fname, "content": content})
    return documents


# ---------------------------------------------------------------------------
# 2. Smart Chunking (markdown-header-aware)
# ---------------------------------------------------------------------------

def _split_with_overlap(text: str, max_size: int, overlap: int) -> list[str]:
    """Split text into chunks with overlap, preferring sentence boundaries."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        # Try to break at sentence boundary (period or newline)
        boundary = text.rfind(".", start, end)
        newline = text.rfind("\n", start, end)
        break_at = max(boundary, newline)
        if break_at <= start:
            break_at = end  # no good boundary found
        else:
            break_at += 1  # include the period/newline
        chunks.append(text[start:break_at].strip())
        start = break_at - overlap
        if start < 0:
            start = 0
    return [c for c in chunks if c]


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split documents into chunks by markdown headers, with overlap for long sections."""
    chunks = []
    header_pattern = re.compile(r"^(#{2,3})\s+(.+)", re.MULTILINE)

    for doc in documents:
        content = doc["content"]
        filename = doc["filename"]

        # Split by headers
        sections: list[tuple[str, str]] = []  # (header, body)
        matches = list(header_pattern.finditer(content))

        if not matches:
            # No headers — treat whole doc as one section
            sections.append(("", content))
        else:
            # Text before the first header — merge into first section
            preamble = content[: matches[0].start()].strip() if matches[0].start() > 0 else ""
            for i, m in enumerate(matches):
                header = m.group(0).strip()
                start = m.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
                body = content[start:end].strip()
                # Merge preamble into first section
                if i == 0 and preamble:
                    body = preamble + "\n\n" + body
                sections.append((header, body))

        for header, body in sections:
            full_text = f"{header}\n{body}".strip() if header else body.strip()
            if not full_text or len(full_text.strip()) < 50:
                continue

            if len(full_text) <= CHUNK_SIZE:
                chunks.append(
                    {"text": full_text, "source": filename, "header": header}
                )
            else:
                sub_chunks = _split_with_overlap(body, CHUNK_SIZE, CHUNK_OVERLAP)
                for sc in sub_chunks:
                    chunk_text = f"{header}\n{sc}".strip() if header else sc.strip()
                    chunks.append(
                        {"text": chunk_text, "source": filename, "header": header}
                    )

    return chunks


# ---------------------------------------------------------------------------
# 3. Embedding + FAISS Indexing
# ---------------------------------------------------------------------------

def load_embedder(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def build_index(chunks: list[dict], embedder: SentenceTransformer):
    """Encode chunks and build a FAISS IndexFlatIP (cosine similarity)."""
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    # Normalize for cosine similarity via inner product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embeddings


# ---------------------------------------------------------------------------
# 4. Retrieval
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    embedder: SentenceTransformer,
    top_k: int = TOP_K,
) -> list[dict]:
    """Retrieve top_k chunks most similar to the query."""
    q_emb = embedder.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    q_emb = q_emb.astype("float32")

    scores, indices = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = dict(chunks[idx])
        chunk["score"] = float(score)
        results.append(chunk)
    return results


# ---------------------------------------------------------------------------
# 5. Answer Generation — OpenRouter
# ---------------------------------------------------------------------------

def _build_user_message(query: str, retrieved_chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(retrieved_chunks, 1):
        parts.append(f"[Context {i} — source: {c['source']}]\n{c['text']}")
    context_block = "\n\n".join(parts)
    return f"{context_block}\n\nQuestion: {query}"


def generate_answer(
    query: str,
    retrieved_chunks: list[dict],
    api_key: str,
    model: str = OPENROUTER_MODEL,
) -> str:
    """Generate an answer using OpenRouter API."""
    if not api_key:
        return "Error: No OpenRouter API key provided. Please add your key in the sidebar."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://indecimal-rag.streamlit.app",
        "X-Title": "Indecimal RAG Assistant",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(query, retrieved_chunks)},
        ],
    }
    # Retry up to 3 times for rate limits
    for attempt in range(3):
        try:
            resp = requests.post(OPENROUTER_API_URL, json=payload, headers=headers, timeout=60)
            if resp.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout:
            if attempt < 2:
                continue
            return "Error: Request to OpenRouter timed out. Please try again."
        except requests.exceptions.HTTPError as e:
            return f"Error from OpenRouter (HTTP {e.response.status_code}): {e.response.text[:300]}"
        except Exception as e:
            return f"Error generating answer: {e}"
    return "Error: Rate limited after 3 retries. Please wait a moment and try again."


# ---------------------------------------------------------------------------
# 6. Answer Generation — Ollama (bonus / local LLM)
# ---------------------------------------------------------------------------

def generate_answer_ollama(
    query: str,
    retrieved_chunks: list[dict],
    model: str = OLLAMA_MODEL,
) -> str:
    """Generate an answer using a local Ollama model."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(query, retrieved_chunks)},
        ],
        "stream": False,
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Make sure it is running (`ollama serve`)."
    except Exception as e:
        return f"Error generating answer from Ollama: {e}"


# ---------------------------------------------------------------------------
# 7. RAGEngine class — wraps everything
# ---------------------------------------------------------------------------

class RAGEngine:
    def __init__(self, api_key: str = "", model: str = OPENROUTER_MODEL):
        self.api_key = api_key
        self.model = model
        self.chunks: list[dict] = []
        self.index = None
        self.embedder = None
        self.ready = False

    def initialize(self):
        """Load docs, chunk, embed, build index."""
        docs = load_documents()
        self.chunks = chunk_documents(docs)
        self.embedder = load_embedder()
        self.index, _ = build_index(self.chunks, self.embedder)
        self.ready = True

    def query(self, user_question: str, use_ollama: bool = False, ollama_model: str = OLLAMA_MODEL) -> dict:
        """Run the full RAG pipeline: retrieve + generate."""
        if not self.ready:
            return {"answer": "RAG engine is not initialized yet.", "retrieved_chunks": [], "query": user_question}

        retrieved = retrieve(user_question, self.index, self.chunks, self.embedder)

        start = time.time()
        if use_ollama:
            answer = generate_answer_ollama(user_question, retrieved, model=ollama_model)
        else:
            answer = generate_answer(user_question, retrieved, self.api_key, self.model)
        elapsed = time.time() - start

        return {
            "answer": answer,
            "retrieved_chunks": retrieved,
            "query": user_question,
            "response_time": round(elapsed, 2),
        }

    def query_both(self, user_question: str, ollama_model: str = OLLAMA_MODEL) -> dict:
        """Run both OpenRouter and Ollama for comparison."""
        if not self.ready:
            return {"query": user_question, "openrouter": {}, "ollama": {}}

        retrieved = retrieve(user_question, self.index, self.chunks, self.embedder)

        t0 = time.time()
        or_answer = generate_answer(user_question, retrieved, self.api_key, self.model)
        or_time = round(time.time() - t0, 2)

        t0 = time.time()
        ol_answer = generate_answer_ollama(user_question, retrieved, model=ollama_model)
        ol_time = round(time.time() - t0, 2)

        return {
            "query": user_question,
            "retrieved_chunks": retrieved,
            "openrouter": {"answer": or_answer, "response_time": or_time},
            "ollama": {"answer": ol_answer, "response_time": ol_time},
        }
