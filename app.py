import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv, find_dotenv
from utils.retriever_pipeline import retrieve_documents
from utils.doc_handler import process_documents, reset_documents
from utils.advanced_rag import cosine_similarity
from langchain_ollama import OllamaEmbeddings

try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if not hasattr(torch.classes, '__path__') or not torch.classes.__path__:
            torch.classes.__path__ = []
    except Exception:
        pass
    from sentence_transformers import CrossEncoder
    _torch_available = True
except Exception:
    device = "cpu"
    CrossEncoder = None
    _torch_available = False

load_dotenv(find_dotenv())

st.set_page_config(
    page_title="Cortex RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

OLLAMA_BASE_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/generate"
DEFAULT_MODEL   = os.getenv("MODEL", "llama3.1:8b")
EMBEDDINGS_MODEL    = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text:latest")
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


@st.cache_resource(show_spinner=False)
def load_reranker():
    if not _torch_available or CrossEncoder is None:
        return None
    try:
        return CrossEncoder(CROSS_ENCODER_MODEL, device=device)
    except Exception as e:
        st.warning(f"CrossEncoder not loaded: {e}. Reranking disabled.")
        return None


@st.cache_resource(show_spinner=False)
def load_cache_embeddings():
    """Lightweight embeddings client used only for semantic caching."""
    try:
        return OllamaEmbeddings(model=EMBEDDINGS_MODEL, base_url=OLLAMA_BASE_URL)
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=30)
def get_ollama_models():
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            return [m for m in models if "embed" not in m.lower()]
    except Exception:
        pass
    return [DEFAULT_MODEL]


# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700;12..96,800&family=Manrope:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    :root {
        --bg:        #0a0b12;
        --bg-soft:   #0d1020;
        --panel:     #13162c;
        --panel-2:   #181c38;
        --line:      #23264a;
        --iris:      #8b7bff;
        --iris-2:    #6ea8ff;
        --mint:      #2dd4bf;
        --amber:     #ffb347;
        --text:      #e7e9f5;
        --text-dim:  #8b91b8;
        --text-faint:#5a6090;
    }

    /* ── Base + atmospheric background ── */
    .stApp {
        background:
            radial-gradient(900px 500px at 12% -8%, rgba(139,123,255,0.14), transparent 60%),
            radial-gradient(800px 520px at 100% 0%, rgba(45,212,191,0.10), transparent 55%),
            var(--bg);
        color: var(--text);
        font-family: 'Manrope', 'Segoe UI', sans-serif;
    }
    /* subtle dot grid overlay */
    .stApp::before {
        content: "";
        position: fixed; inset: 0;
        background-image: radial-gradient(rgba(255,255,255,0.025) 1px, transparent 1px);
        background-size: 26px 26px;
        pointer-events: none; z-index: 0;
    }
    .main .block-container { padding-top: 2.2rem; max-width: 880px; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #11142a 0%, #0a0b16 100%);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] * { color: #c2c7e6 !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4, [data-testid="stSidebar"] h5 {
        font-family: 'Bricolage Grotesque', sans-serif !important;
        color: #d8d4ff !important; letter-spacing: -0.01em;
    }
    [data-testid="stSidebar"] .stMarkdown p { font-size: 0.9rem; }

    /* ── Brand header (main) ── */
    .brand-wrap {
        display: flex; align-items: center; justify-content: center;
        gap: 18px; margin: 0 0 2px 0;
    }
    .brand-logo { width: 64px; height: 64px; filter: drop-shadow(0 0 14px rgba(139,123,255,0.4)); }
    @keyframes float-core { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-3px)} }
    .brand-logo { animation: float-core 4s ease-in-out infinite; }
    .rag-title {
        font-family: 'Bricolage Grotesque', sans-serif;
        background: linear-gradient(100deg, #ffffff 0%, #c4b8ff 55%, var(--mint) 100%);
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent; color: transparent;
        font-size: 3.1rem; font-weight: 800; letter-spacing: -0.02em;
        margin: 0; line-height: 1; padding: 0;
    }
    .brand-kicker {
        text-align: center; color: var(--iris);
        font-family: 'JetBrains Mono', monospace; font-weight: 600;
        font-size: 0.72rem; letter-spacing: 0.32em; margin: 10px 0 2px 0;
    }
    .brand-tag {
        text-align: center; color: var(--text-faint);
        font-size: 0.86rem; margin-bottom: 0.4rem;
    }

    /* ── Enterprise badge (fixed top-right) ── */
    .ent-badge {
        position: fixed; top: 12px; right: 18px; z-index: 9999;
        display: flex; align-items: center; gap: 8px;
        background: #0d2e1a; border: 1px solid #1f6f4a;
        border-radius: 999px; padding: 6px 16px 6px 12px;
        text-decoration: none; transition: all .2s;
        box-shadow: 0 2px 12px rgba(45,212,191,0.12);
    }
    .ent-badge:hover {
        background: #112e1e; border-color: #2dd4bf;
        box-shadow: 0 4px 20px rgba(45,212,191,0.22);
        transform: translateY(-1px);
    }
    .ent-dot { width: 7px; height: 7px; border-radius: 50%; background: #2dd4bf; flex-shrink: 0; }
    .ent-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem; font-weight: 600; color: #2dd4bf;
        letter-spacing: 0.06em; white-space: nowrap;
    }

    /* ── Pipeline chips ── */
    .chip-row { display:flex; flex-wrap:wrap; gap:7px; justify-content:center; margin: 12px 0 4px; }
    .chip {
        font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; font-weight: 500;
        color: #aab0e0; background: var(--panel);
        border: 1px solid var(--line); border-radius: 999px;
        padding: 4px 11px; transition: all .2s;
    }
    .chip:hover { border-color: var(--iris); color: #fff; }

    /* ── Chat ── */
    [data-testid="stChatMessageContent"] { font-size: 0.96rem; line-height: 1.65; }
    [data-testid="stChatMessage"] {
        background: rgba(19,22,44,0.45);
        border: 1px solid var(--line);
        border-radius: 14px; padding: 6px 14px; margin: 8px 0;
        backdrop-filter: blur(4px);
    }

    /* ── Thinking console ── */
    .think-box {
        background: linear-gradient(180deg, #0c0f22, #0a0b16);
        border: 1px solid #2a2f5e; border-left: 3px solid var(--iris);
        border-radius: 10px; margin: 8px 0 12px; overflow: hidden;
        box-shadow: 0 0 0 1px rgba(139,123,255,0.04), 0 8px 24px rgba(0,0,0,0.3);
    }
    .think-box summary {
        list-style:none; cursor:pointer; padding: 10px 15px;
        font-family:'JetBrains Mono',monospace; font-size:0.74rem; font-weight:600;
        color: var(--iris); letter-spacing: 0.06em; user-select:none;
        display:flex; align-items:center; gap:7px;
    }
    .think-box summary::-webkit-details-marker { display:none; }
    .think-box[open] summary { border-bottom: 1px solid #23264a; }
    .think-body {
        padding: 13px 17px; font-size: 0.8rem; color: #9aa0c8;
        font-family:'JetBrains Mono',monospace; white-space:pre-wrap;
        line-height:1.7; max-height:340px; overflow-y:auto;
    }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.25} }
    .think-live { animation: pulse 1.1s ease infinite; color: var(--mint); }

    /* ── Source cards ── */
    .source-card {
        background: var(--panel); border: 1px solid var(--line);
        border-left: 3px solid var(--mint); border-radius: 8px;
        padding: 11px 15px; margin: 7px 0; font-size: 0.81rem; color: #aab0d0;
        line-height: 1.55;
    }
    .source-label {
        color: var(--mint); font-family:'JetBrains Mono',monospace;
        font-weight: 600; font-size: 0.68rem; text-transform: uppercase;
        letter-spacing: 0.1em; margin-bottom: 4px; display:block;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(120deg, var(--iris), #5a61e0);
        color: #fff !important; border: none; border-radius: 10px;
        font-family:'Manrope',sans-serif; font-weight: 600; font-size: 0.86rem;
        transition: transform .15s, box-shadow .2s; box-shadow: 0 4px 14px rgba(139,123,255,0.25);
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(139,123,255,0.4); }
    .stButton > button:active { transform: translateY(0); }

    /* ── Inputs / selects ── */
    [data-testid="stChatInput"] textarea,
    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
        background: var(--panel) !important; border-color: var(--line) !important;
        border-radius: 10px !important; color: var(--text) !important;
    }
    [data-testid="stChatInput"] {
        background: var(--panel) !important; border: 1px solid var(--line) !important;
        border-radius: 14px;
    }
    [data-testid="stBottomBlockContainer"], [data-testid="stBottom"] > div,
    [data-testid="stChatFloatingInputContainer"] {
        background: transparent !important;
    }
    [data-testid="stBottom"], .stApp > footer { background: transparent !important; }
    /* kill the white block behind the fixed chat input (Streamlit 1.30) */
    .stChatFloatingInputContainer, .block-container + div { background: transparent !important; }
    section.main > div.block-container ~ div { background: transparent !important; }

    /* ── Status badges ── */
    .status-badge {
        display:inline-flex; align-items:center; gap:7px;
        padding: 5px 13px; border-radius: 999px;
        font-family:'JetBrains Mono',monospace; font-size: 0.73rem; font-weight: 600;
    }
    .status-ok  { background:#0c2a1c; color:var(--mint); border:1px solid #1f6f4a; }
    .status-err { background:#2e0d12; color:#ff6b8a; border:1px solid #7f2540; }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.35} }
    .status-ok .dot, .status-err .dot { width:7px;height:7px;border-radius:50%;background:currentColor;animation:blink 1.6s infinite; }

    /* ── Misc ── */
    hr { border-color: var(--line) !important; margin: 0.8rem 0; }
    [data-testid="stFileUploaderDropzone"] {
        background: var(--panel); border: 1px dashed #3a3f6e; border-radius: 12px;
    }
    .streamlit-expanderHeader, [data-testid="stExpander"] summary {
        color: var(--mint) !important; font-family:'JetBrains Mono',monospace; font-size:0.8rem;
    }
    [data-testid="stExpander"] { border:1px solid var(--line); border-radius:10px; background:rgba(13,16,32,0.5); }
    ::-webkit-scrollbar { width: 7px; height:7px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #2a2f5e; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--iris); }
    #MainMenu, footer { visibility: hidden; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: 12px; }
    [data-testid="stDecoration"] { background: linear-gradient(90deg, var(--iris), var(--mint)); }
</style>
""", unsafe_allow_html=True)


# Inline SVG brand mark (kept small for the header)
LOGO_SVG = """
<svg class="brand-logo" viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="hg" x1="40" y1="40" x2="160" y2="160" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#8b7bff"/><stop offset="0.55" stop-color="#6ea8ff"/><stop offset="1" stop-color="#2dd4bf"/>
    </linearGradient>
    <radialGradient id="hc" cx="0.5" cy="0.5" r="0.5">
      <stop offset="0" stop-color="#c4b8ff"/><stop offset="0.5" stop-color="#8b7bff"/><stop offset="1" stop-color="#2dd4bf"/>
    </radialGradient>
  </defs>
  <circle cx="100" cy="100" r="78" stroke="url(#hg)" stroke-width="1.5" opacity="0.25"/>
  <g stroke="url(#hg)" stroke-width="2" opacity="0.45" stroke-linecap="round">
    <path d="M96 100 L79.5 44"/><path d="M96 100 L40 100"/><path d="M96 100 L79.5 156"/>
    <path d="M96 100 L115.5 42"/><path d="M96 100 L115.5 158"/><path d="M96 100 L51 66"/><path d="M96 100 L51 134"/>
  </g>
  <path d="M146 61 L115.5 42 L79.5 44 L51 66 L40 100 L51 134 L79.5 156 L115.5 158 L146 139"
        stroke="url(#hg)" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="68" cy="100" r="2.6" fill="#2dd4bf"/><circle cx="88" cy="72" r="2.6" fill="#ffb347"/><circle cx="88" cy="128" r="2.6" fill="#8b7bff"/>
  <g fill="#0a0b12" stroke="url(#hg)" stroke-width="3">
    <circle cx="146" cy="61" r="7"/><circle cx="115.5" cy="42" r="7"/><circle cx="79.5" cy="44" r="7"/>
    <circle cx="51" cy="66" r="7"/><circle cx="40" cy="100" r="7"/><circle cx="51" cy="134" r="7"/>
    <circle cx="79.5" cy="156" r="7"/><circle cx="115.5" cy="158" r="7"/><circle cx="146" cy="139" r="7"/>
  </g>
  <circle cx="96" cy="100" r="13" fill="url(#hc)"/><circle cx="96" cy="100" r="5" fill="#0a0b12" opacity="0.85"/>
  <circle cx="96" cy="100" r="2.4" fill="#c4b8ff"/>
</svg>
"""


# ─── Session state ────────────────────────────────────────────────────────────
defaults = {
    "messages": [],
    "retrieval_pipeline": None,
    "rag_enabled": True,
    "documents_loaded": False,
    "processing": False,
    "enable_hyde": True,
    "enable_reranking": True,
    "enable_graph_rag": True,
    "enable_thinking": True,
    "enable_contextual": False,
    "enable_fusion": False,
    "enable_crag": False,
    "enable_cache": False,
    "cache_threshold": 0.92,
    "semantic_cache": [],
    "temperature": 0.3,
    "max_contexts": 3,
    "suggested_questions": [],
    "last_sources": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

reranker = load_reranker()


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="display:flex;align-items:center;gap:11px;margin:-6px 0 4px;">'
        f'<div style="width:38px;height:38px;flex:0 0 38px;">{LOGO_SVG.replace("brand-logo","")}</div>'
        '<div><div style="font-family:\'Bricolage Grotesque\',sans-serif;font-weight:800;'
        'font-size:1.35rem;line-height:1;background:linear-gradient(100deg,#fff,#c4b8ff,#2dd4bf);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;">Cortex RAG</div>'
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.58rem;'
        'letter-spacing:0.18em;color:#6a70a0;margin-top:3px;">RETRIEVAL ENGINE · 2026</div></div></div>',
        unsafe_allow_html=True
    )

    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False

    if ollama_ok:
        st.markdown('<span class="status-badge status-ok"><span class="dot"></span>Ollama connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-err"><span class="dot"></span>Ollama offline</span>', unsafe_allow_html=True)
        st.error("Start Ollama with: `ollama serve`")

    st.markdown("---")
    st.markdown("### 🧠 Model")
    available_models = get_ollama_models()
    default_idx = available_models.index(DEFAULT_MODEL) if DEFAULT_MODEL in available_models else 0
    selected_model = st.selectbox("LLM Model", available_models, index=default_idx, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### 📁 Documents")
    if st.session_state.documents_loaded:
        st.success("Documents loaded ✓")
        if st.button("🔄 Reset Documents", use_container_width=True):
            reset_documents()
            st.rerun()
    else:
        st.session_state.enable_contextual = st.checkbox(
            "Contextual Retrieval ✨",
            value=st.session_state.enable_contextual,
            help="Prepend an LLM-generated context sentence to each chunk before indexing. "
                 "Much better retrieval, but slower at upload time."
        )
        uploaded_files = st.file_uploader(
            "Upload PDF / DOCX / TXT",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        if uploaded_files and not st.session_state.documents_loaded:
            with st.spinner("Processing documents…"):
                process_documents(
                    uploaded_files, reranker, EMBEDDINGS_MODEL, OLLAMA_BASE_URL,
                    llm_model=selected_model,
                    enable_contextual=st.session_state.enable_contextual,
                )
                if st.session_state.documents_loaded:
                    st.success("Documents ready!")
                    st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ RAG Settings")
    st.session_state.rag_enabled       = st.checkbox("Enable RAG",             value=st.session_state.rag_enabled)
    st.session_state.enable_hyde       = st.checkbox("HyDE Query Expansion",   value=st.session_state.enable_hyde)
    st.session_state.enable_reranking  = st.checkbox("Neural Reranking",       value=st.session_state.enable_reranking)
    st.session_state.enable_graph_rag  = st.checkbox("GraphRAG",               value=st.session_state.enable_graph_rag)

    st.markdown("##### 🆕 Advanced")
    st.session_state.enable_fusion = st.checkbox(
        "RAG-Fusion (RRF)", value=st.session_state.enable_fusion,
        help="Generate multiple query variants and merge results with Reciprocal Rank Fusion."
    )
    st.session_state.enable_crag = st.checkbox(
        "Corrective RAG (CRAG)", value=st.session_state.enable_crag,
        help="The model grades each retrieved doc for relevance and drops irrelevant ones."
    )
    st.session_state.enable_cache = st.checkbox(
        "Semantic Cache ⚡", value=st.session_state.enable_cache,
        help="Instantly reuse answers for semantically similar past questions."
    )

    st.markdown("---")
    st.markdown("### 🎛️ Generation")
    st.session_state.enable_thinking = st.checkbox("Show Thinking Process 🧠", value=st.session_state.enable_thinking)
    st.session_state.temperature     = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 0.05)
    st.session_state.max_contexts    = st.slider("Max Contexts", 1, 6, st.session_state.max_contexts)

    st.markdown("---")
    if st.session_state.enable_cache and st.session_state.semantic_cache:
        st.caption(f"⚡ {len(st.session_state.semantic_cache)} answers cached")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.session_state.semantic_cache = []
        st.rerun()

    st.markdown("""
        <div style="margin-top:20px;font-size:11px;color:#555;text-align:center;">
            Cortex RAG &nbsp;|&nbsp; N Sai Akhil © 2026
        </div>
    """, unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _thinking_html(text: str, live: bool) -> str:
    if live:
        label = '🧠 REASONING<span class="think-live"> ●</span>'
    else:
        label = '🧠 THOUGHT PROCESS &nbsp;·&nbsp; <span style="color:#5a6090">click to expand</span> ✓'
    open_attr = " open" if live else ""
    return (
        f'<details class="think-box"{open_attr}>'
        f'<summary>{label}</summary>'
        f'<div class="think-body">{text}</div>'
        f'</details>'
    )


def _source_html(sources: list) -> str:
    if not sources:
        return ""
    cards = "".join(
        f'<div class="source-card"><div class="source-label">Source {i+1}</div>'
        f'{s[:400]}{"…" if len(s)>400 else ""}</div>'
        for i, s in enumerate(sources)
    )
    return cards


# ─── Enterprise badge (fixed top-right) ──────────────────────────────────────
st.markdown(
    '<a href="https://cortex-rag-beta.vercel.app/" target="_blank" class="ent-badge">'
    '<span class="ent-dot"></span>'
    '<span class="ent-text">Enterprise Solutions ↗</span>'
    '</a>',
    unsafe_allow_html=True
)

# ─── Brand header ─────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="brand-wrap">{LOGO_SVG}<div class="rag-title">Cortex RAG</div></div>',
    unsafe_allow_html=True
)
st.markdown('<div class="brand-kicker">AGENTIC RETRIEVAL ENGINE · 2026</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-tag">RAG that thinks before it answers.</div>', unsafe_allow_html=True)

if st.session_state.documents_loaded and st.session_state.rag_enabled:
    chips = ["🔍 BM25+FAISS"]
    if st.session_state.enable_fusion:     chips.append("🔀 RAG-Fusion")
    if st.session_state.enable_graph_rag:  chips.append("🕸️ GraphRAG")
    if st.session_state.enable_reranking:  chips.append("⚡ Reranked")
    if st.session_state.enable_crag:       chips.append("✅ CRAG")
    if st.session_state.enable_hyde and not st.session_state.enable_fusion:
        chips.append("🧬 HyDE")
    if st.session_state.enable_cache:      chips.append("💾 Cache")
    if st.session_state.enable_thinking:   chips.append("🧠 Thinking")
    st.markdown(
        '<div class="chip-row">' + "".join(f'<span class="chip">{c}</span>' for c in chips) + '</div>',
        unsafe_allow_html=True
    )

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

if st.session_state.documents_loaded and not st.session_state.messages and st.session_state.suggested_questions:
    st.markdown("**💡 Suggested questions based on your documents:**")
    cols = st.columns(min(len(st.session_state.suggested_questions), 3))
    for i, q in enumerate(st.session_state.suggested_questions[:3]):
        with cols[i]:
            if st.button(q, key=f"sq_{i}", use_container_width=True):
                st.session_state._pending_question = q
                st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Show collapsed thinking for past assistant messages
        if message["role"] == "assistant" and message.get("thinking"):
            st.markdown(_thinking_html(message["thinking"], live=False), unsafe_allow_html=True)
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"📄 Sources ({len(message['sources'])})", expanded=False):
                st.markdown(_source_html(message["sources"]), unsafe_allow_html=True)


# ─── Response generation ──────────────────────────────────────────────────────
def generate_response(prompt_text: str, model: str):
    chat_history = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}"
        for m in st.session_state.messages[-6:]
    )

    # ── RAG retrieval ──
    context = ""
    sources = []
    if st.session_state.rag_enabled and st.session_state.retrieval_pipeline:
        try:
            docs    = retrieve_documents(prompt_text, OLLAMA_API_URL, model, chat_history)
            sources = [doc.page_content for doc in docs]
            context = "\n\n".join(f"[Source {i+1}]:\n{doc.page_content}" for i, doc in enumerate(docs))
        except Exception as e:
            st.warning(f"Retrieval error: {e}")
    elif st.session_state.rag_enabled:
        st.info("Upload documents in the sidebar to enable RAG retrieval.")

    # ── CRAG relevance feedback ──
    crag = st.session_state.get("_crag_status")
    crag_low = False
    if crag:
        status, kept, total = crag
        if status == "ok":
            st.markdown(
                f'<span class="status-badge status-ok">✅ CRAG: {kept}/{total} sources relevant</span>',
                unsafe_allow_html=True)
        else:
            crag_low = True
            st.markdown(
                '<span class="status-badge status-err">⚠️ CRAG: no clearly relevant sources — answering with low confidence</span>',
                unsafe_allow_html=True)

    # ── Build prompt ──
    ctx_block = f"\nContext:\n{context}\n" if context else ""
    ctx_instruction = (
        "\n- Answer based on the provided context. Cite [Source N] when referencing specific info."
        if context else ""
    )

    show_thinking = st.session_state.get("enable_thinking", True)

    if show_thinking:
        think_instruction = (
            "Before answering, reason through the problem step by step inside <think>...</think> tags. "
            "Then give your final answer outside those tags.\n\n"
        )
    else:
        think_instruction = ""

    system_prompt = (
        f"{think_instruction}"
        f"You are a helpful, thorough AI assistant.\n\n"
        f"Chat History:\n{chat_history}\n"
        f"{ctx_block}"
        f"Question: {prompt_text}\n\n"
        f"Instructions:\n"
        f"- Be concise and well-structured{ctx_instruction}\n"
        f"- If you don't know, say so clearly\n"
    )
    if crag_low:
        system_prompt += "- The retrieved context may not be relevant; if so, state that the documents don't cover this.\n"
    if show_thinking:
        system_prompt += "- Put ALL reasoning inside <think>...</think>; the answer goes after\n"

    # ── Placeholders ──
    think_ph  = st.empty()   # live thinking panel
    answer_ph = st.empty()   # streaming answer

    thinking = ""
    answer   = ""
    buffer   = ""
    in_think = False
    think_done = False

    try:
        resp = requests.post(
            OLLAMA_API_URL,
            json={
                "model": model,
                "prompt": system_prompt,
                "stream": True,
                "options": {"temperature": st.session_state.temperature, "num_ctx": 4096},
            },
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()

        for line in resp.iter_lines():
            if not line:
                continue
            data  = json.loads(line.decode())
            token = data.get("response", "")
            buffer += token

            # ── Parse <think>...</think> from the stream ──
            changed = True
            while changed:
                changed = False
                if not in_think and not think_done:
                    idx = buffer.find("<think>")
                    if idx != -1:
                        pre = buffer[:idx].strip()
                        if pre:
                            answer += pre
                        buffer    = buffer[idx + 7:]
                        in_think  = True
                        changed   = True
                    else:
                        safe = max(0, len(buffer) - 7)
                        answer += buffer[:safe]
                        buffer  = buffer[safe:]
                elif in_think:
                    idx = buffer.find("</think>")
                    if idx != -1:
                        thinking  += buffer[:idx]
                        buffer     = buffer[idx + 8:]
                        in_think   = False
                        think_done = True
                        changed    = True
                    else:
                        safe      = max(0, len(buffer) - 8)
                        thinking += buffer[:safe]
                        buffer    = buffer[safe:]
                else:
                    answer += buffer
                    buffer  = ""

            # ── Update live displays ──
            live_think = thinking + (buffer if in_think else "")
            if show_thinking and (in_think or think_done) and live_think.strip():
                think_ph.markdown(_thinking_html(live_think, live=in_think), unsafe_allow_html=True)

            live_ans = answer + (buffer if not in_think else "")
            if live_ans.strip():
                answer_ph.markdown(live_ans.lstrip() + ("▌" if not data.get("done") else ""))

            if data.get("done"):
                if in_think:
                    thinking += buffer
                else:
                    answer   += buffer
                break

        # ── Final render ──
        thinking = thinking.strip()
        answer   = answer.strip()

        if show_thinking and thinking:
            think_ph.markdown(_thinking_html(thinking, live=False), unsafe_allow_html=True)
        else:
            think_ph.empty()

        answer_ph.markdown(answer)

    except requests.exceptions.ConnectionError:
        answer = "❌ Cannot connect to Ollama. Make sure it's running with `ollama serve`."
        answer_ph.error(answer)
    except Exception as e:
        answer = f"❌ Generation error: {str(e)}"
        answer_ph.error(answer)

    return thinking, answer, sources


# ─── Semantic cache lookup ────────────────────────────────────────────────────
def check_semantic_cache(query: str):
    """Return a cached entry if a semantically similar question was asked before."""
    if not st.session_state.enable_cache or not st.session_state.semantic_cache:
        return None
    emb_client = load_cache_embeddings()
    if emb_client is None:
        return None
    try:
        q_emb = emb_client.embed_query(query)
    except Exception:
        return None
    best, best_sim = None, 0.0
    for entry in st.session_state.semantic_cache:
        sim = cosine_similarity(q_emb, entry["emb"])
        if sim > best_sim:
            best, best_sim = entry, sim
    if best and best_sim >= st.session_state.cache_threshold:
        return {**best, "similarity": best_sim, "query_emb": q_emb}
    return {"query_emb": q_emb} if st.session_state.enable_cache else None


def store_in_cache(query_emb, query, answer, thinking, sources):
    if not st.session_state.enable_cache or query_emb is None:
        return
    st.session_state.semantic_cache.append({
        "emb": query_emb, "query": query, "answer": answer,
        "thinking": thinking, "sources": sources,
    })
    # Keep the cache bounded
    if len(st.session_state.semantic_cache) > 50:
        st.session_state.semantic_cache.pop(0)


# ─── Chat input ───────────────────────────────────────────────────────────────
pending = st.session_state.pop("_pending_question", None)
prompt  = st.chat_input("Ask about your documents…") or pending

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    cache_hit = check_semantic_cache(prompt)
    query_emb = cache_hit.get("query_emb") if cache_hit else None

    if cache_hit and "answer" in cache_hit:
        # ⚡ Serve from semantic cache
        with st.chat_message("assistant"):
            st.markdown(
                f'<span class="status-badge status-ok">⚡ Cached answer '
                f'(similarity {cache_hit["similarity"]:.0%})</span>',
                unsafe_allow_html=True)
            if cache_hit.get("thinking"):
                st.markdown(_thinking_html(cache_hit["thinking"], live=False), unsafe_allow_html=True)
            st.markdown(cache_hit["answer"])
            if cache_hit.get("sources"):
                with st.expander(f"📄 Sources ({len(cache_hit['sources'])})", expanded=False):
                    st.markdown(_source_html(cache_hit["sources"]), unsafe_allow_html=True)
        st.session_state.messages.append({
            "role": "assistant", "content": cache_hit["answer"],
            "thinking": cache_hit.get("thinking", ""), "sources": cache_hit.get("sources", []),
        })
        st.rerun()
    else:
        with st.chat_message("assistant"):
            thinking, answer, sources = generate_response(prompt, selected_model)

        store_in_cache(query_emb, prompt, answer, thinking, sources)
        st.session_state.messages.append({
            "role":    "assistant",
            "content": answer,
            "thinking": thinking,
            "sources":  sources,
        })
        st.rerun()
