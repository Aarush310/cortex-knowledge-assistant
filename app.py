"""
Cortex · Personal AI Knowledge Assistant
A RAG-based app that lets you chat with your own documents.
"""

import html
import os
import streamlit as st
from datetime import datetime
from pathlib import Path

from knowledge_base import KnowledgeBase
from chat_engine import ChatEngine
from document_loader import load_document

# ---------- Page config ----------
st.set_page_config(
    page_title="Cortex · Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Aesthetic: Dark Editorial / Luxury Terminal ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;0,800;1,400&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-deep: #0a0a0c;
        --bg-surface: #111114;
        --bg-elevated: #18181c;
        --bg-hover: #1e1e24;
        --border: rgba(255, 255, 255, 0.06);
        --border-bright: rgba(255, 255, 255, 0.12);
        --text-primary: #f0ece4;
        --text-secondary: #8a8690;
        --text-muted: #5a5660;
        --accent: #d4a853;
        --accent-glow: rgba(212, 168, 83, 0.15);
        --accent-bright: #e8c476;
        --accent-dim: #a07d3a;
        --teal: #5fb8a0;
        --teal-glow: rgba(95, 184, 160, 0.12);
        --rose: #c47a7a;
        --serif: 'Playfair Display', 'Georgia', serif;
        --sans: 'DM Sans', -apple-system, sans-serif;
        --mono: 'JetBrains Mono', 'SF Mono', monospace;
    }

    html, body, [class*="css"] {
        font-family: var(--sans) !important;
        color: var(--text-primary);
    }

    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    .stApp {
        background: var(--bg-deep);
        color: var(--text-primary);
    }

    /* Animated grain overlay */
    .stApp::before {
        content: '';
        position: fixed;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
        pointer-events: none;
        z-index: 0;
        animation: grain 8s steps(10) infinite;
    }

    @keyframes grain {
        0%, 100% { transform: translate(0, 0); }
        10% { transform: translate(-5%, -10%); }
        30% { transform: translate(3%, -15%); }
        50% { transform: translate(12%, 9%); }
        70% { transform: translate(9%, 4%); }
        90% { transform: translate(-1%, 7%); }
    }

    /* Subtle gold radial glow */
    .stApp::after {
        content: '';
        position: fixed;
        top: -20%;
        right: -10%;
        width: 60%;
        height: 60%;
        background: radial-gradient(ellipse at center, rgba(212, 168, 83, 0.04) 0%, transparent 70%);
        pointer-events: none;
        z-index: 0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(14, 14, 18, 0.92) !important;
        backdrop-filter: blur(30px) saturate(1.2);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--text-secondary);
        font-size: 0.88rem;
    }

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-family: var(--serif) !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em;
    }

    [data-testid="stSidebar"] hr {
        border-color: var(--border) !important;
        margin: 1.2rem 0 !important;
    }

    h1, h2, h3, h4 {
        font-family: var(--serif) !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }

    /* Brand header */
    .cortex-header {
        padding: 1.8rem 0 0.6rem 0;
        position: relative;
    }

    .cortex-logotype {
        font-family: var(--serif);
        font-size: 2.6rem;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: -0.03em;
        line-height: 1;
        margin: 0;
    }

    .cortex-logotype span {
        color: var(--accent);
    }

    .cortex-tagline {
        font-family: var(--sans);
        font-size: 0.9rem;
        color: var(--text-muted);
        margin-top: 0.5rem;
        font-weight: 400;
        letter-spacing: 0.04em;
    }

    .cortex-divider {
        height: 1px;
        background: linear-gradient(90deg, var(--accent) 0%, var(--border) 40%, transparent 100%);
        margin: 1.2rem 0 1.5rem 0;
        border: none;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 1.1rem 1.4rem !important;
        margin-bottom: 0.75rem !important;
        transition: border-color 0.3s ease;
    }

    [data-testid="stChatMessage"]:hover {
        border-color: var(--border-bright) !important;
    }

    [data-testid="stChatMessage"] p {
        color: var(--text-primary) !important;
        font-size: 0.92rem;
        line-height: 1.7;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border-bright) !important;
        border-radius: 12px !important;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: var(--accent-dim) !important;
        box-shadow: 0 0 0 1px var(--accent-glow), 0 8px 32px rgba(0,0,0,0.3) !important;
    }

    [data-testid="stChatInput"] textarea {
        color: var(--text-primary) !important;
        font-family: var(--sans) !important;
        font-size: 0.92rem !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--text-muted) !important;
    }

    /* Buttons */
    .stButton > button {
        background: var(--bg-elevated) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-bright) !important;
        border-radius: 8px !important;
        padding: 0.55rem 1.2rem !important;
        font-family: var(--sans) !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.02em;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    .stButton > button:hover {
        background: var(--bg-hover) !important;
        border-color: var(--accent-dim) !important;
        color: var(--accent-bright) !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3) !important;
        transform: translateY(-1px);
    }

    .stButton > button:active {
        transform: translateY(0px);
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: var(--bg-surface) !important;
        border: 1px dashed var(--border-bright) !important;
        border-radius: 10px !important;
        padding: 0.8rem !important;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent-dim) !important;
    }

    [data-testid="stFileUploader"] small {
        color: var(--text-muted) !important;
    }

    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--accent-dim), var(--accent)) !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: transparent !important;
        color: var(--text-secondary) !important;
        font-size: 0.82rem !important;
        font-family: var(--mono) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }

    .streamlit-expanderContent {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border) !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
    }

    /* Source cards */
    .source-card {
        background: var(--bg-elevated);
        border: 1px solid var(--border);
        border-left: 3px solid var(--teal);
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.82rem;
        color: var(--text-secondary);
        line-height: 1.6;
        transition: border-color 0.2s ease;
    }

    .source-card:hover {
        border-color: var(--border-bright);
        border-left-color: var(--accent);
    }

    .source-title {
        font-family: var(--mono);
        font-weight: 500;
        font-size: 0.78rem;
        color: var(--teal);
        margin-bottom: 0.4rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    .source-relevance {
        display: inline-block;
        font-family: var(--mono);
        font-size: 0.7rem;
        color: var(--accent);
        background: var(--accent-glow);
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        margin-left: 0.5rem;
    }

    /* Stats */
    .stat-row {
        display: flex;
        gap: 0.6rem;
        margin: 0.6rem 0;
    }

    .stat-card {
        flex: 1;
        background: var(--bg-elevated);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.7rem 0.8rem;
        text-align: center;
    }

    .stat-number {
        font-family: var(--serif);
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--accent);
        line-height: 1;
    }

    .stat-label {
        font-family: var(--mono);
        font-size: 0.65rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.25rem;
    }

    /* Document list */
    .doc-item {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        background: var(--bg-elevated);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.55rem 0.75rem;
        margin: 0.3rem 0;
        font-size: 0.82rem;
        color: var(--text-secondary);
        transition: all 0.2s ease;
    }

    .doc-item:hover {
        border-color: var(--border-bright);
        color: var(--text-primary);
    }

    .doc-icon {
        width: 28px;
        height: 28px;
        background: var(--accent-glow);
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.65rem;
        font-family: var(--mono);
        font-weight: 500;
        color: var(--accent);
        flex-shrink: 0;
    }

    .doc-name {
        font-family: var(--mono);
        font-size: 0.78rem;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        position: relative;
    }

    .empty-glyph {
        font-size: 3.5rem;
        margin-bottom: 1.2rem;
        opacity: 0.25;
    }

    .empty-heading {
        font-family: var(--serif);
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--text-secondary);
        margin-bottom: 0.6rem;
    }

    .empty-body {
        font-family: var(--sans);
        font-size: 0.88rem;
        color: var(--text-muted);
        line-height: 1.7;
        max-width: 420px;
        margin: 0 auto;
    }

    .empty-body code {
        font-family: var(--mono);
        font-size: 0.82rem;
        color: var(--accent);
        background: var(--accent-glow);
        padding: 0.15rem 0.4rem;
        border-radius: 4px;
    }

    .sidebar-section {
        font-family: var(--mono);
        font-size: 0.7rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 1rem 0 0.5rem 0;
    }

    .stSpinner > div {
        border-top-color: var(--accent) !important;
    }

    .stSuccess, .stWarning, .stError {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
    }

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .cortex-header { animation: fadeInUp 0.6s cubic-bezier(0.4, 0, 0.2, 1) both; }
    [data-testid="stChatMessage"] { animation: fadeInUp 0.35s cubic-bezier(0.4, 0, 0.2, 1) both; }

    /* Footer */
    .cortex-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        text-align: center;
        padding: 0.6rem 0;
        font-family: var(--mono);
        font-size: 0.7rem;
        color: var(--text-muted);
        letter-spacing: 0.06em;
        background: linear-gradient(to top, var(--bg-deep) 60%, transparent 100%);
        z-index: 999;
        pointer-events: none;
    }

    .cortex-footer a {
        color: var(--accent);
        text-decoration: none;
        pointer-events: auto;
    }

    .cortex-footer a:hover {
        color: var(--accent-bright);
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)


# ---------- Initialize session state ----------
if "kb" not in st.session_state:
    st.session_state.kb = KnowledgeBase(persist_dir="./chroma_db")

if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = ChatEngine(knowledge_base=st.session_state.kb)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 0.2rem 0;">
        <div style="font-family: var(--serif); font-size: 1.3rem; font-weight: 700; color: var(--text-primary);">
            🧠 Cortex
        </div>
        <div style="font-family: var(--mono); font-size: 0.65rem; color: var(--text-muted);
                    text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.2rem;">
            Knowledge Assistant
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="sidebar-section">Ingest Documents</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("⚡ Ingest to Knowledge Base", use_container_width=True, type="primary"):
            progress = st.progress(0, text="Initializing...")
            for i, file in enumerate(uploaded_files):
                progress.progress(
                    (i + 1) / len(uploaded_files),
                    text=f"Embedding {file.name}...",
                )
                temp_path = Path("./temp_uploads") / file.name
                temp_path.parent.mkdir(exist_ok=True)
                temp_path.write_bytes(file.getvalue())

                try:
                    text = load_document(str(temp_path))
                    st.session_state.kb.add_document(
                        doc_id=file.name,
                        text=text,
                        metadata={
                            "filename": file.name,
                            "uploaded_at": datetime.now().isoformat(),
                        },
                    )
                    if file.name not in st.session_state.uploaded_files_list:
                        st.session_state.uploaded_files_list.append(file.name)
                except Exception as e:
                    st.error(f"Failed: {file.name} — {e}")
                finally:
                    temp_path.unlink(missing_ok=True)
            progress.empty()
            st.success(f"✓ {len(uploaded_files)} document(s) ingested")
            st.rerun()

    # Stats
    st.markdown("---")
    stats = st.session_state.kb.get_stats()
    st.markdown('<div class="sidebar-section">Knowledge Base</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="stat-row">
            <div class="stat-card">
                <div class="stat-number">{stats["doc_count"]}</div>
                <div class="stat-label">Documents</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats["chunk_count"]}</div>
                <div class="stat-label">Chunks</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Indexed documents
    if st.session_state.uploaded_files_list:
        st.markdown('<div class="sidebar-section">Indexed Files</div>', unsafe_allow_html=True)
        for fname in st.session_state.uploaded_files_list:
            ext = Path(fname).suffix.lower()
            icon_map = {".pdf": "PDF", ".txt": "TXT", ".md": "MD", ".docx": "DOC"}
            icon_label = icon_map.get(ext, "FILE")
            st.markdown(
                f"""<div class="doc-item">
                    <div class="doc-icon">{icon_label}</div>
                    <div class="doc-name">{fname}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Clear KB", use_container_width=True):
            st.session_state.kb.clear()
            st.session_state.uploaded_files_list = []
            st.session_state.messages = []
            st.success("Cleared")
            st.rerun()
    with col2:
        if st.button("💬 New Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ---------- Main Area ----------
st.markdown("""
<div class="cortex-header">
    <div class="cortex-logotype">Corte<span>x</span></div>
    <div class="cortex-tagline">Personal AI Knowledge Assistant — ask anything across your documents</div>
    <div class="cortex-divider"></div>
</div>
""", unsafe_allow_html=True)

# Empty state
if not st.session_state.messages and st.session_state.kb.get_stats()["doc_count"] == 0:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-glyph">📚</div>
        <div class="empty-heading">Your knowledge base is empty</div>
        <div class="empty-body">
            Upload documents from the sidebar to begin.<br>
            Supports <code>PDF</code> <code>DOCX</code> <code>TXT</code> <code>Markdown</code>
        </div>
    </div>
    """, unsafe_allow_html=True)
elif not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-glyph">💬</div>
        <div class="empty-heading">Ready to explore your knowledge</div>
        <div class="empty-body">
            Ask questions like<br>
            <code>What are the key takeaways from that design book?</code><br>
            <code>Compare the arguments across my uploaded papers</code><br>
            <code>Summarize the main concepts</code>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_agent_details(steps, sources):
    """Show what the agent did (its tool calls) and the chunks it retrieved."""
    if steps:
        with st.expander(f"🔎 {len(steps)} agent step(s)"):
            for step in steps:
                if step["tool"] == "search_knowledge_base":
                    st.markdown(f'Searched for: `{step["input"].get("query", "")}`')
                else:
                    st.markdown(f'Called `{step["tool"]}`')
    if sources:
        with st.expander(f"📎 {len(sources)} source(s) referenced"):
            for src in sources:
                score_pct = int(src.get("score", 0) * 100)
                st.markdown(
                    f"""<div class="source-card">
                        <div class="source-title">
                            {html.escape(src["filename"])}
                            <span class="source-relevance">{score_pct}% match</span>
                        </div>
                        {html.escape(src["text"][:280])}...
                    </div>""",
                    unsafe_allow_html=True,
                )


# Chat history
for msg in st.session_state.messages:
    avatar = "🧠" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        render_agent_details(msg.get("steps"), msg.get("sources"))

# Chat input
if prompt := st.chat_input("Ask your knowledge base anything..."):
    if st.session_state.kb.get_stats()["doc_count"] == 0:
        st.warning("Upload some documents first to start chatting.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("Searching & reasoning..."):
                response, sources, steps = st.session_state.chat_engine.ask(
                    prompt,
                    history=st.session_state.messages[:-1],
                )
            st.markdown(response)
            render_agent_details(steps, sources)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources,
            "steps": steps,
        })

# ---------- Footer ----------
st.markdown("""
<div class="cortex-footer">
    Built by <a href="https://github.com/Aarush310" target="_blank">Aarush Prasad</a>
</div>
""", unsafe_allow_html=True)