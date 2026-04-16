# 🧠 Cortex — Personal AI Knowledge Assistant

A **Retrieval Augmented Generation (RAG)** application that lets you upload your personal documents and have intelligent, cited conversations with your own knowledge base. Ask questions in natural language and get accurate answers with source citations.

Built with Python, Streamlit, ChromaDB, and Google Gemini.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-FF4B4B?logo=streamlit&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-green)
![Gemini](https://img.shields.io/badge/Google_Gemini-2.5_Flash-4285F4?logo=google&logoColor=white)

---

## What It Does

Upload your documents (lecture notes, research papers, books, personal notes) and chat with them. Cortex retrieves the most relevant passages from your knowledge base and uses AI to synthesize accurate, cited answers.

**Example queries:**
- *"What are the key takeaways from my system design notes?"*
- *"Compare what document A and document B say about microservices"*
- *"Summarize the main concepts from chapter 3"*

---

## Features

- **Multi-format document ingestion** — PDF, DOCX, TXT, Markdown
- **Local vector embeddings** — documents are embedded on your machine using ONNX MiniLM (no data leaves your computer for embedding)
- **Persistent vector store** — ChromaDB stores everything locally on disk between sessions
- **Cited answers** — every response shows which document chunks were used, with relevance scores
- **Conversation memory** — multi-turn chat that maintains context
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
│                     QUERY PIPELINE                        │
│                                                          │
│  User Question ──▶ Semantic Search (cosine similarity)   │
│                          │                               │
│                          ▼                               │
│                   Top-K Relevant Chunks                   │
│                          │                               │
│                          ▼                               │
│              Google Gemini 2.5 Flash (with RAG context)  │
│                          │                               │
│                          ▼                               │
│              Cited Answer + Source Cards in UI            │
└──────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Frontend | Streamlit | Chat interface and document management UI |
| Vector DB | ChromaDB | Persistent local vector storage and similarity search |
| Embeddings | ONNX MiniLM-L6-v2 | Local document embedding (no external API needed) |
| LLM | Google Gemini 2.5 Flash | Answer generation with RAG context |
| Document Parsing | PyPDF, python-docx | PDF and Word document text extraction |

---

## Setup

### Prerequisites
- Python 3.11+
- A free Google Gemini API key

### 1. Clone the repo

```bash
git clone https://github.com/aarushprasad/cortex-knowledge-assistant.git
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

### 4. Get your free Gemini API key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Click **Create API Key**
4. Copy the key

No credit card required.

### 5. Configure the API key

```bash
cp .env.example .env
```

Open `.env` and paste your key:
```
GEMINI_API_KEY=your-key-here
```

### 6. Run the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## Usage

1. **Upload documents** using the sidebar file uploader
2. Click **⚡ Ingest to Knowledge Base** to chunk, embed, and store them
3. **Ask questions** in the chat input
4. Click the **📎 sources** expander under any answer to see the retrieved chunks with relevance scores

---

## Project Structure

```
cortex-knowledge-assistant/
├── app.py                 # Streamlit UI with dark editorial theme
├── knowledge_base.py      # ChromaDB vector store, chunking, and retrieval
├── chat_engine.py         # RAG pipeline and Gemini API integration
├── document_loader.py     # PDF, DOCX, TXT, MD text extraction
├── requirements.txt       # Python dependencies
├── .env.example           # API key template
├── .gitignore             # Git ignore rules
└── README.md
```

---

## Key Concepts Demonstrated

- **Retrieval Augmented Generation (RAG)** — combining retrieval with generative AI for grounded answers
- **Vector databases** — embedding, storing, and querying document vectors with cosine similarity
- **Document processing pipelines** — chunking strategies with sentence-aware splitting and overlap
- **API integration** — working with the Google Gemini API for LLM inference
- **Privacy-conscious design** — embeddings computed locally, only retrieved chunks sent to the LLM

---

## Privacy

- Embeddings run locally on your CPU (ONNX MiniLM)
- Only the top-5 retrieved chunks (~5 short passages) are sent to Google Gemini per query
- The vector database is stored locally in `./chroma_db`
- Free tier note: Google may use free-tier API data to improve their models

---

## License

MIT

---

Built by [Aarush Prasad](https://github.com/aarushprasad)
