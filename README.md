# 🧠 Cortex — Agentic Knowledge Assistant on the Claude API

[![CI](https://github.com/Aarush310/cortex-knowledge-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/Aarush310/cortex-knowledge-assistant/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Claude API](https://img.shields.io/badge/Anthropic-Claude_API-D97757)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-green)
![Container](https://img.shields.io/badge/Container-UBI_9-EE0000?logo=redhat&logoColor=white)

Upload your documents (lecture notes, papers, books) and have a cited, multi-turn conversation with them. Cortex is an **agentic RAG** app: instead of running one fixed retrieval per question, **Claude decides what to search for** using tools over your knowledge base, then answers with citations that are checked against what it actually retrieved.

---

## What makes it agentic

Claude gets two tools and runs in a loop (up to 5 steps per question):

| Tool | What it does |
|---|---|
| `search_knowledge_base(query, filename?, n_results?)` | Semantic search over your documents, optionally inside one file |
| `list_documents()` | Lists the files in the knowledge base |

That lets Claude:
- **Rewrite follow-ups** — "what about the second one?" becomes a standalone query using the conversation history.
- **Decompose questions** — "compare what A and B say about microservices" becomes one filtered search per document.
- **Stop when it has enough** — or say plainly that your documents don't contain the answer.

On the final step the engine sets `tool_choice: none`, so the loop always ends in a written answer.

### Cited, verified answers
Every passage a tool returns gets a source ID (`[S1]`, `[S2]`, …). Claude must cite those IDs, and the engine checks each one: citations that don't point at a passage retrieved in this turn are stripped and flagged. The UI shows both the **agent's tool calls** and the **cited passages**.

---

## Architecture

```
┌──────────────────────── INGESTION ────────────────────────┐
│ PDF / DOCX / TXT / MD ─▶ sentence-aware chunker (600 chars,│
│                          100-char word-boundary overlap)   │
│                     ─▶ ONNX MiniLM-L6-v2 embeddings (local)│
│                     ─▶ ChromaDB, cosine HNSW (persistent)  │
└────────────────────────────────────────────────────────────┘
┌──────────────────────── AGENT LOOP ───────────────────────┐
│ question + last 3 turns ─▶ Claude (Messages API + tools)  │
│        ▲                         │ tool_use               │
│        │ tool_result ([S#] passages)                      │
│        └──── search_knowledge_base / list_documents ◀─────┘
│                                  │ end_turn               │
│                     answer ─▶ citation check ─▶ UI        │
└────────────────────────────────────────────────────────────┘
```

| Component | Technology |
|---|---|
| LLM / agent | Anthropic Claude API (`claude-sonnet-5-5` by default, set `CLAUDE_MODEL` to change) |
| Vector DB | ChromaDB, persistent on disk |
| Embeddings | ONNX all-MiniLM-L6-v2, on CPU — documents never leave your machine for embedding |
| UI | Streamlit |
| Container | Red Hat UBI 9 Python 3.11, non-root, OpenShift-compatible |
| CI | GitHub Actions: tests on 3.11/3.12, image build, container health-check smoke test |

---

## Run it locally

```bash
git clone https://github.com/Aarush310/cortex-knowledge-assistant.git
cd cortex-knowledge-assistant
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then paste your ANTHROPIC_API_KEY
streamlit run app.py          # http://localhost:8501
```

## Run it in a container

```bash
docker build -t cortex .      # or: podman build -t cortex .
docker run -p 8501:8501 \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -v cortex-data:/opt/app-root/data \
  cortex
```

The image pre-downloads the embedding model at build time, runs as a non-root user, stores the vector DB in a volume at `/opt/app-root/data`, and exposes a health check at `/_stcore/health`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

The tests run offline: the agent loop is driven by a scripted fake Claude client (tool calls, multi-search, hallucinated-citation removal, forced final answer, history trimming), the vector store uses a small deterministic embedding function, and the Streamlit app gets a boot smoke test.

---

## Project structure

```
cortex-knowledge-assistant/
├── app.py               # Streamlit UI (dark editorial theme), shows tool calls + cited sources
├── chat_engine.py       # Agent loop on the Claude API: tools, citations, history
├── knowledge_base.py    # Chunking + ChromaDB storage and search
├── document_loader.py   # PDF, DOCX, TXT, MD text extraction
├── tests/               # pytest suite (no network needed)
├── Dockerfile           # UBI 9 image, OpenShift-friendly
└── .github/workflows/   # CI: tests, image build, container smoke test
```

## Privacy

- Embeddings run locally on CPU; the vector DB stays on your disk.
- Only the passages Claude retrieves with its search tool are sent to the Claude API.

## License

MIT

---

Built by [Aarush Prasad](https://github.com/Aarush310)
