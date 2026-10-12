# 🧠 Cortex — Personal AI Knowledge Assistant

An **agentic Retrieval Augmented Generation (RAG)** application: upload your personal documents and have cited conversations with your own knowledge base. Instead of running one fixed search, Claude is given tools and decides how to search — rephrasing, searching several times, or comparing documents — before it answers.

Built with Python, Streamlit, ChromaDB, and the Anthropic Claude API.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-FF4B4B?logo=streamlit&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-green)
![Claude](https://img.shields.io/badge/Anthropic-Claude_API-D97757)
![Docker](https://img.shields.io/badge/Docker-non--root-2496ED?logo=docker&logoColor=white)

---

## What It Does

Upload your documents (lecture notes, research papers, books, personal notes) and chat with them. Claude searches your knowledge base with tools, reads the most relevant passages and writes an answer that cites the files it used.

**Example queries:**
- *"What are the key takeaways from my system design notes?"*
- *"Compare what document A and document B say about microservices"* (the agent searches for each side separately)
- *"What's in my knowledge base?"* (the agent lists your documents)
- Follow-ups like *"and what about chapter 3?"* (the agent writes a standalone search query from the conversation)

---

## Features

- **Agentic retrieval with Claude tool use** — Claude decides when and how to search, with up to 5 tool-use steps per question
- **Multi-format document ingestion** — PDF, DOCX, TXT, Markdown
- **Local vector embeddings** — documents are embedded on your machine using ONNX MiniLM (no data leaves your computer for embedding)
- **Persistent vector store** — ChromaDB stores everything locally on disk between sessions
- **Cited answers + visible reasoning** — every answer shows which searches the agent ran and which chunks it used, with relevance scores
- **Conversation memory** — multi-turn chat with the last 6 messages as context
- **Containerised** — Dockerfile that runs as a non-root user (OpenShift-compatible), verified in CI
- **Tested** — the agent loop is unit-tested with a fake Claude client, so CI needs no API key
- **Modern UI** — dark editorial aesthetic with Playfair Display typography, animated grain texture, and amber/gold accents

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    DOCUMENT INGESTION                     │
│                                                          │
│  PDF/DOCX/TXT/MD ──▶ Sentence-Aware Chunker             │
│                          │                               │
│                          ▼                               │
│                   ONNX MiniLM Embeddings (local)         │
│                          │                               │
│                          ▼                               │
│                   ChromaDB Vector Store (persistent)      │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                 AGENT LOOP (chat_engine.py)              │
│                                                          │
│  Question + chat history + tool definitions              │
│                          │                               │
│                          ▼                               │
│                    Claude (Messages API) ◀──────────┐    │
│                          │                          │    │
│        stop_reason == "tool_use"?                   │    │
│            │ yes                         │ no       │    │
│            ▼                             ▼          │    │
│   Run tool locally:              Cited answer +     │    │
│   • search_knowledge_base        agent steps +      │    │
│   • list_documents               source cards       │    │
│            │                                        │    │
│            └──── tool_result ───────────────────────┘    │
│                  (max 5 steps)                           │
└──────────────────────────────────────────────────────────┘
```

### How the tool-use loop works

1. Cortex sends Claude the conversation, a system prompt and two tool definitions (name, description, JSON schema).
2. If Claude wants information, it replies with `stop_reason: "tool_use"` and one or more `tool_use` blocks, e.g. `search_knowledge_base({"query": "Raft leader election"})`.
3. Cortex runs the search against ChromaDB and sends the chunks back as `tool_result` blocks.
4. Claude either searches again or writes the final answer (`stop_reason: "end_turn"`), citing files like `[source: notes.pdf]`.

The loop is capped at 5 steps so a confused agent can't loop forever, and retrieved document text is treated as data, never as instructions (a basic prompt-injection guard).

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Frontend | Streamlit | Chat interface and document management UI |
| LLM / Agent | Anthropic Claude API (tool use) | Decides how to search, then answers with citations |
| Vector DB | ChromaDB | Persistent local vector storage and similarity search |
| Embeddings | ONNX MiniLM-L6-v2 | Local document embedding (no external API needed) |
| Document Parsing | PyPDF, python-docx | PDF and Word document text extraction |
| Container | Docker (Python 3.12 slim) | Non-root, OpenShift-compatible image |
| CI | GitHub Actions | Tests on Python 3.11–3.13, image build + non-root run check |

---

## Setup

### Prerequisites
- Python 3.11+
- An Anthropic API key ([console.anthropic.com](https://console.anthropic.com/))

### 1. Clone the repo

```bash
git clone https://github.com/Aarush310/cortex-knowledge-assistant.git
cd cortex-knowledge-assistant
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API key

```bash
cp .env.example .env
```

Open `.env` and paste your key:
```
ANTHROPIC_API_KEY=your-key-here
```

Optionally set `CLAUDE_MODEL` (default `claude-sonnet-5-5`; `claude-haiku-5-5` is cheaper and faster).

### 5. Run the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

### Or run with Docker

```bash
docker build -t cortex .
docker run -p 8501:8501 -e ANTHROPIC_API_KEY=your-key-here cortex
```

Mount a volume at `/app/chroma_db` to keep your knowledge base between container restarts. The image runs as a non-root user and every writable folder belongs to group 0, so it also runs under OpenShift's random-UID policy; CI checks this by starting it with `--user 1000870000:0`.

### Run the tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

---

## Usage

1. **Upload documents** using the sidebar file uploader
2. Click **⚡ Ingest to Knowledge Base** to chunk, embed, and store them
3. **Ask questions** in the chat input
4. Open **🔎 agent steps** under an answer to see the searches Claude ran, and **📎 sources** to see the retrieved chunks with relevance scores

---

## Project Structure

```
cortex-knowledge-assistant/
├── app.py                 # Streamlit UI with dark editorial theme
├── chat_engine.py         # Agent loop: Claude tool use over the knowledge base
├── knowledge_base.py      # ChromaDB vector store, chunking, retrieval, document list
├── document_loader.py     # PDF, DOCX, TXT, MD text extraction
├── tests/                 # Agent loop (fake Claude client) and chunking tests
├── Dockerfile             # Non-root, OpenShift-compatible image
├── .github/workflows/     # CI: tests + Docker build and non-root run check
├── requirements.txt       # Python dependencies
├── requirements-dev.txt   # Test dependencies
├── .env.example           # API key template
└── README.md
```

---

## Key Concepts Demonstrated

- **Agentic RAG** — the model chooses its own search queries and how many searches to run, via tool use
- **Retrieval Augmented Generation** — grounding answers in retrieved documents, with citations
- **Vector databases** — embedding, storing, and querying document vectors with cosine similarity
- **Document processing pipelines** — sentence-aware chunking with overlap
- **Testing LLM code** — a fake client makes the agent loop deterministic and testable without an API key
- **Containerisation** — non-root, OpenShift-compatible image built and smoke-tested in CI
- **Privacy-conscious design** — embeddings computed locally, only retrieved chunks sent to the LLM

---

## Privacy

- Embeddings run locally on your CPU (ONNX MiniLM)
- The vector database is stored locally in `./chroma_db`
- Only the chunks the agent retrieves (up to 10 per search) are sent to the Claude API, along with your question and recent chat history

---

## License

MIT

---

Built by [Aarush Prasad](https://github.com/Aarush310)
