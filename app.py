"""Streamlit frontend for the Indecimal RAG AI Assistant."""

import os
import streamlit as st
from dotenv import load_dotenv

from rag_engine import RAGEngine
from config import OPENROUTER_MODEL, OLLAMA_MODEL

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indecimal AI Assistant",
    page_icon="🏗️",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Sidebar branding */
    [data-testid="stSidebar"] {
        background-color: #111419;
    }
    [data-testid="stSidebar"] h1 {
        color: #2E86AB;
    }

    /* Chat bubbles */
    .stChatMessage[data-testid="stChatMessage-user"] {
        background-color: #1a2533;
        border-radius: 12px;
    }
    .stChatMessage[data-testid="stChatMessage-assistant"] {
        background-color: #1A1D23;
        border-radius: 12px;
    }

    /* Status badges */
    .status-ok { color: #4CAF50; font-weight: 600; }
    .status-warn { color: #FF9800; font-weight: 600; }

    /* Chunk cards */
    .chunk-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
    }
    .chunk-score {
        color: #2E86AB;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helper: get API key from multiple sources ────────────────────────────────
def _resolve_api_key(sidebar_key: str) -> str:
    """Try HF Spaces secrets → .env → sidebar input."""
    try:
        return st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        pass
    env_key = os.getenv("OPENROUTER_API_KEY", "")
    if env_key and env_key != "your_key_here":
        return env_key
    return sidebar_key


# ── Cached RAG engine ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model & building index…")
def get_rag_engine() -> RAGEngine:
    engine = RAGEngine()
    engine.initialize()
    return engine


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🏗️ Indecimal")
    st.markdown("**AI Construction Assistant**")
    st.divider()

    model_choice = st.selectbox(
        "LLM Provider",
        ["OpenRouter (Cloud)", "Ollama (Local)"],
    )

    api_key_input = st.text_input(
        "OpenRouter API Key",
        value=os.getenv("OPENROUTER_API_KEY", ""),
        type="password",
        help="Get a free key at https://openrouter.ai",
    )

    ollama_model = st.text_input("Ollama Model", value=OLLAMA_MODEL)

    compare_mode = st.toggle("Compare Models", value=False, help="Run both OpenRouter & Ollama side-by-side")

    st.divider()

    # System status
    engine = get_rag_engine()
    st.markdown("### System Status")
    if engine.ready:
        st.markdown(f'<span class="status-ok">✅ Documents loaded: {len(engine.chunks)} chunks</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="status-ok">✅ Embedding model: all-MiniLM-L6-v2</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="status-ok">✅ FAISS index: ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-warn">⏳ Engine loading…</span>', unsafe_allow_html=True)

    current_llm = ollama_model if model_choice == "Ollama (Local)" else OPENROUTER_MODEL
    st.markdown(f"**LLM:** `{current_llm}`")

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# ── Resolve API key ─────────────────────────────────────────────────────────
api_key = _resolve_api_key(api_key_input)
engine.api_key = api_key
engine.model = OPENROUTER_MODEL


# ── Chat state ───────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome message
if not st.session_state.messages:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": (
                "👋 Hi! I'm the Indecimal AI Assistant. I can answer questions about "
                "Indecimal's construction packages, pricing, quality assurance, customer "
                "journey, and policies. Ask me anything!"
            ),
            "chunks": [],
        }
    )


# ── Render chat history ─────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("chunks"):
            with st.expander(f"📄 Retrieved Context ({len(msg['chunks'])} chunks)"):
                for i, c in enumerate(msg["chunks"], 1):
                    st.markdown(
                        f"""<div class="chunk-card">
                        <strong>Chunk {i}</strong> — <em>{c['source']}</em>
                        {'· ' + c['header'] if c.get('header') else ''}
                        · <span class="chunk-score">score: {c.get('score', 0):.3f}</span>
                        <br/><pre style="white-space:pre-wrap;color:#c9d1d9;font-size:0.85em;">{c['text'][:500]}</pre>
                        </div>""",
                        unsafe_allow_html=True,
                    )
        # Show comparison columns if present
        if msg.get("comparison"):
            comp = msg["comparison"]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"### ☁️ OpenRouter ({comp['openrouter']['response_time']}s)")
                st.markdown(comp["openrouter"]["answer"])
            with col2:
                st.markdown(f"### 🖥️ Ollama ({comp['ollama']['response_time']}s)")
                st.markdown(comp["ollama"]["answer"])


# ── Chat input ───────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about Indecimal…"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt, "chunks": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Validate
    use_ollama = model_choice == "Ollama (Local)"
    if not use_ollama and not api_key:
        with st.chat_message("assistant"):
            st.warning("⚠️ Please enter your OpenRouter API key in the sidebar.")
        st.session_state.messages.append(
            {"role": "assistant", "content": "⚠️ Please enter your OpenRouter API key in the sidebar.", "chunks": []}
        )
    else:
        with st.chat_message("assistant"):
            with st.spinner("Searching documents…"):
                if compare_mode:
                    result = engine.query_both(prompt, ollama_model=ollama_model)
                    chunks = result.get("retrieved_chunks", [])

                    # Show comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"### ☁️ OpenRouter ({result['openrouter']['response_time']}s)")
                        st.markdown(result["openrouter"]["answer"])
                    with col2:
                        st.markdown(f"### 🖥️ Ollama ({result['ollama']['response_time']}s)")
                        st.markdown(result["ollama"]["answer"])

                    with st.expander(f"📄 Retrieved Context ({len(chunks)} chunks)"):
                        for i, c in enumerate(chunks, 1):
                            st.markdown(
                                f"""<div class="chunk-card">
                                <strong>Chunk {i}</strong> — <em>{c['source']}</em>
                                {'· ' + c['header'] if c.get('header') else ''}
                                · <span class="chunk-score">score: {c.get('score', 0):.3f}</span>
                                <br/><pre style="white-space:pre-wrap;color:#c9d1d9;font-size:0.85em;">{c['text'][:500]}</pre>
                                </div>""",
                                unsafe_allow_html=True,
                            )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": "*(Comparison mode — see both responses above)*",
                            "chunks": chunks,
                            "comparison": {
                                "openrouter": result["openrouter"],
                                "ollama": result["ollama"],
                            },
                        }
                    )
                else:
                    result = engine.query(prompt, use_ollama=use_ollama, ollama_model=ollama_model)
                    answer = result["answer"]
                    chunks = result.get("retrieved_chunks", [])
                    resp_time = result.get("response_time", 0)

                    st.markdown(answer)
                    st.caption(f"⏱️ Response time: {resp_time}s")

                    if chunks:
                        with st.expander(f"📄 Retrieved Context ({len(chunks)} chunks)"):
                            for i, c in enumerate(chunks, 1):
                                st.markdown(
                                    f"""<div class="chunk-card">
                                    <strong>Chunk {i}</strong> — <em>{c['source']}</em>
                                    {'· ' + c['header'] if c.get('header') else ''}
                                    · <span class="chunk-score">score: {c.get('score', 0):.3f}</span>
                                    <br/><pre style="white-space:pre-wrap;color:#c9d1d9;font-size:0.85em;">{c['text'][:500]}</pre>
                                    </div>""",
                                    unsafe_allow_html=True,
                                )

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "chunks": chunks}
                    )
