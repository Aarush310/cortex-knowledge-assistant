"""
Chat Engine — an agentic RAG loop on the Anthropic Claude API.

Instead of always doing one fixed retrieval, Claude is given tools over the
knowledge base and decides what to search for:

  1. The user's question (plus recent conversation) goes to Claude with two
     tools: ``search_knowledge_base`` and ``list_documents``.
  2. Claude calls the tools as many times as it needs — e.g. rewriting a
     follow-up like "what about the second one?" into a standalone query, or
     running one search per document when asked to compare them.
  3. Every retrieved passage gets a source ID ([S1], [S2], ...). Claude must
     cite those IDs, and the engine checks that every cited ID was actually
     retrieved in this turn before showing the answer's sources.

Set ANTHROPIC_API_KEY in your environment or a .env file.
Optionally set CLAUDE_MODEL (default: claude-sonnet-5-5).
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
MAX_AGENT_STEPS = int(os.getenv("CORTEX_MAX_STEPS", "5"))
HISTORY_MESSAGES = 6  # last 3 user/assistant exchanges

SYSTEM_PROMPT = """You are Cortex, a personal knowledge assistant. You answer questions using ONLY the user's own documents, which you can reach through your tools.

How to work:
- Before answering any question about the documents, call search_knowledge_base. Never answer from memory.
- Write each search query as a standalone question. For follow-ups ("what about the second one?"), use the conversation to rewrite the query so it makes sense on its own.
- For comparisons or multi-part questions, run a separate, focused search for each part or document (use the filename filter when helpful).
- Call list_documents if you need to know what files exist.
- Stop searching once you have enough evidence. Do not repeat a search that already returned the same passages.

How to answer:
- Every factual sentence must cite the passage(s) it comes from using their source IDs in square brackets, e.g. [S1] or [S2][S4]. Only cite IDs that your tools returned.
- If the passages don't contain the answer, say so plainly and do not guess.
- Synthesize across sources rather than repeating one passage. Be concise, clear and friendly."""

TOOLS = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Semantic search over the user's uploaded documents. Returns the most "
            "relevant passages, each with a source ID (e.g. S3), filename, chunk "
            "number and similarity score. Cite passages by their source ID."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A standalone, specific search query.",
                },
                "filename": {
                    "type": "string",
                    "description": "Optional. Only search inside this document (exact filename from list_documents).",
                },
                "n_results": {
                    "type": "integer",
                    "description": "How many passages to return (1-10). Default 5.",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_documents",
        "description": "List the documents in the knowledge base and how many chunks each has.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

CITATION_RE = re.compile(r"\[(S\d+)\]")


@dataclass
class AgentStep:
    tool: str
    input: Dict
    result_count: int


@dataclass
class AgentResponse:
    answer: str
    sources: List[Dict] = field(default_factory=list)  # passages the answer cites
    steps: List[AgentStep] = field(default_factory=list)  # tool calls Claude made
    invalid_citations: List[str] = field(default_factory=list)  # cited IDs never retrieved
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


class ChatEngine:
    def __init__(self, knowledge_base, model: str = DEFAULT_MODEL, client=None):
        self.kb = knowledge_base
        self.model = model
        if client is None:
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise ValueError(
                    "ANTHROPIC_API_KEY not found. Add it to a .env file in the project root "
                    "(see .env.example) or export it in your shell."
                )
            client = anthropic.Anthropic()
        self.client = client

    # ---------- public API ----------

    def ask(self, question: str, history: Optional[List[Dict]] = None) -> AgentResponse:
        """Run the agent loop for one user turn."""
        messages = self._build_history(history) + [{"role": "user", "content": question}]
        retrieved: Dict[str, Dict] = {}  # source ID -> passage, for this turn
        response = AgentResponse(answer="", model=self.model)

        try:
            for step in range(MAX_AGENT_STEPS):
                last_step = step == MAX_AGENT_STEPS - 1
                result = self.client.messages.create(
                    model=self.model,
                    max_tokens=1500,
                    temperature=0.2,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    # On the final step, force a written answer from what was found.
                    tool_choice={"type": "none"} if last_step else {"type": "auto"},
                    messages=messages,
                )
                response.input_tokens += result.usage.input_tokens
                response.output_tokens += result.usage.output_tokens
                messages.append({"role": "assistant", "content": result.content})

                tool_calls = [b for b in result.content if b.type == "tool_use"]
                if result.stop_reason != "tool_use" or not tool_calls:
                    response.answer = "".join(
                        b.text for b in result.content if b.type == "text"
                    ).strip()
                    break

                tool_results = []
                for call in tool_calls:
                    content, count = self._run_tool(call.name, call.input or {}, retrieved)
                    response.steps.append(AgentStep(call.name, dict(call.input or {}), count))
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": content,
                    })
                messages.append({"role": "user", "content": tool_results})
        except anthropic.APIError as e:
            response.answer = f"⚠️ Error calling the Claude API: {e}"
            response.sources = list(retrieved.values())
            return response

        if not response.answer:
            response.answer = "I wasn't able to finish answering that — please try rephrasing."

        response.sources, response.invalid_citations = self._resolve_citations(
            response.answer, retrieved
        )
        # Never show the user a citation that doesn't point at a retrieved passage.
        for sid in response.invalid_citations:
            response.answer = response.answer.replace(f"[{sid}]", "")
        return response

    # ---------- tools ----------

    def _run_tool(self, name: str, args: Dict, retrieved: Dict[str, Dict]):
        """Execute a tool call. Returns (text for Claude, number of results)."""
        if name == "list_documents":
            docs = self.kb.list_documents()
            if not docs:
                return "The knowledge base is empty.", 0
            lines = [f"- {fname} ({n} chunks)" for fname, n in docs.items()]
            return "Documents in the knowledge base:\n" + "\n".join(lines), len(docs)

        if name == "search_knowledge_base":
            query = str(args.get("query", "")).strip()
            n = int(args.get("n_results") or 5)
            hits = self.kb.search(query, n_results=max(1, min(n, 10)), filename=args.get("filename"))
            if not hits:
                return f"No passages found for: {query!r}", 0

            blocks = []
            for hit in hits:
                sid = self._source_id_for(hit, retrieved)
                blocks.append(
                    f"[{sid}] {hit['filename']} (chunk {hit['chunk_index']}, "
                    f"similarity {hit['score']:.2f})\n{hit['text']}"
                )
            return "\n\n---\n\n".join(blocks), len(hits)

        return f"Unknown tool: {name}", 0

    @staticmethod
    def _source_id_for(hit: Dict, retrieved: Dict[str, Dict]) -> str:
        """Give each distinct passage one stable source ID for the turn."""
        for sid, existing in retrieved.items():
            if existing["doc_id"] == hit["doc_id"] and existing["chunk_index"] == hit["chunk_index"]:
                return sid
        sid = f"S{len(retrieved) + 1}"
        retrieved[sid] = {**hit, "source_id": sid}
        return sid

    # ---------- helpers ----------

    @staticmethod
    def _resolve_citations(answer: str, retrieved: Dict[str, Dict]):
        """Return (cited passages in citation order, cited IDs that were never retrieved)."""
        cited, invalid = [], []
        for sid in dict.fromkeys(CITATION_RE.findall(answer)):
            if sid in retrieved:
                cited.append(retrieved[sid])
            else:
                invalid.append(sid)
        return cited, invalid

    @staticmethod
    def _build_history(history: Optional[List[Dict]]) -> List[Dict]:
        """Recent turns as plain text, starting with a user message and alternating roles."""
        turns = []
        for msg in (history or [])[-HISTORY_MESSAGES:]:
            role = "user" if msg.get("role") == "user" else "assistant"
            text = str(msg.get("content", "")).strip()
            if not text:
                continue
            if turns and turns[-1]["role"] == role:
                turns[-1]["content"] += "\n\n" + text
            else:
                turns.append({"role": role, "content": text})
        while turns and turns[0]["role"] != "user":
            turns.pop(0)
        # The new question is appended as a user turn, so history must end on assistant.
        while turns and turns[-1]["role"] != "assistant":
            turns.pop()
        return turns


def summarize_steps(steps: List[AgentStep]) -> List[str]:
    """Human-readable trace of the agent's tool calls, for the UI."""
    lines = []
    for s in steps:
        if s.tool == "search_knowledge_base":
            scope = f" in {s.input['filename']}" if s.input.get("filename") else ""
            lines.append(f"searched{scope}: “{s.input.get('query', '')}” → {s.result_count} passages")
        else:
            lines.append(f"{s.tool} → {s.result_count} results")
    return lines


if __name__ == "__main__":  # quick manual check: python chat_engine.py "question"
    import sys
    from knowledge_base import KnowledgeBase

    engine = ChatEngine(KnowledgeBase())
    out = engine.ask(" ".join(sys.argv[1:]) or "What documents do I have?")
    print(out.answer)
    print(json.dumps([s.__dict__ for s in out.steps], indent=2))
