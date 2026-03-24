# Indecimal RAG AI Assistant

A RAG-powered chatbot for **Indecimal**, a construction marketplace. It answers user questions using internal documents (company policies, package specs, quality systems) instead of general knowledge.

## Live Demo

> [HuggingFace Space — coming soon](#)

## Architecture

```
User Query
    │
    ▼
Embedding (all-MiniLM-L6-v2)
    │
    ▼
FAISS Vector Search (Top-5 chunks)
    │
    ▼
LLM Generation (OpenRouter / Ollama)
    │
    ▼
Grounded Answer with Source Citations
```

## Tech Choices & Reasoning

### Embedding Model: `all-MiniLM-L6-v2`
- **Why:** Fast inference, good semantic quality for short passages, 384 dimensions keeps the index small, runs locally with no API dependency.
- **Alternative considered:** OpenAI `ada-002` (better quality but costs money and adds API dependency).

### Vector Store: FAISS (`IndexFlatIP`)
- **Why:** Assignment preferred, exact search is fine for <100 chunks, inner product on normalized vectors = cosine similarity.
- **Alternative considered:** ChromaDB (more features but overkill for this scale).

### Chunking Strategy: Markdown-header-aware + overlap
- **Why:** Documents are structured with `##` headers. Splitting by headers preserves topical coherence. Overlap prevents information loss at boundaries. Sub-chunks get the header prepended for context.
- **Parameters:** 500 char chunks, 100 char overlap.

### LLM: OpenRouter (`google/gemini-2.0-flash-exp:free`)
- **Why:** Free tier, fast responses, good instruction following, reliable grounding to context.
- **Bonus:** Also supports Ollama local models (`phi3:mini`) for side-by-side comparison.

### Grounding Enforcement
- System prompt explicitly instructs: answer **ONLY** from provided context.
- Retrieved chunks are formatted with source labels so the LLM can cite them.
- If context is insufficient, the model is told to say so honestly.

## How to Run Locally

### Prerequisites
- Python 3.10+
- (Optional) [Ollama](https://ollama.ai) installed for local LLM comparison

### Setup
```bash
git clone <repo-url>
cd indecimal-rag
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Add your OpenRouter API key to .env
streamlit run app.py
```

### Running the Evaluation
```bash
python eval/test_questions.py
```
Results are saved to `eval/eval_results.md`.

## Project Structure

```
indecimal-rag/
├── app.py                    # Streamlit frontend (main entry)
├── rag_engine.py             # Core RAG pipeline
├── config.py                 # All config constants
├── requirements.txt          # Dependencies
├── packages.txt              # System packages for HF Spaces
├── .streamlit/
│   └── config.toml           # Streamlit theme config
├── documents/
│   ├── doc1.md               # Company overview & customer journey
│   ├── doc2.md               # Package comparison & specs
│   └── doc3.md               # Policies, quality, guarantees
├── eval/
│   └── test_questions.py     # 15 test questions + evaluation script
├── .env.example              # Template for API keys
├── .gitignore
└── README.md
```

## Evaluation Results

> Run `python eval/test_questions.py` after setup. Results will appear in `eval/eval_results.md`.

## Features

- **Smart chunking** — markdown-header-aware, not naive character splitting
- **Transparency** — every answer shows source chunks + similarity scores
- **Model comparison** — side-by-side OpenRouter vs local Ollama LLM
- **Quantitative evaluation** — 15 test questions with automated scoring
- **Deployed & live** — real working chatbot on HuggingFace Spaces

## Limitations & Future Improvements

- Small document set — would benefit from more docs
- Basic chunking — could use semantic chunking or recursive splitting
- No reranking step — a cross-encoder reranker would improve retrieval precision
- No chat memory — each query is independent (could add conversation context)
- No hybrid search — combining keyword (BM25) + semantic would catch exact term matches better
