"""Configuration constants for the Indecimal RAG chatbot."""

import os

# Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking
CHUNK_SIZE = 500  # characters
CHUNK_OVERLAP = 100  # characters

# Retrieval
TOP_K = 7

# Documents
DOCUMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")

# OpenRouter
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "stepfun/step-3.5-flash:free"

# Ollama (bonus - local LLM)
OLLAMA_MODEL = "phi3:mini"
OLLAMA_URL = "http://localhost:11434/api/chat"
