"""
Tests for the agent loop in chat_engine.py.

They use a fake Claude client and a fake knowledge base, so they run
without an API key or a vector database.
Run: pytest -v
"""

from types import SimpleNamespace

from chat_engine import MAX_AGENT_STEPS, ChatEngine


# ---------- fakes ----------

def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(name, tool_input, block_id="toolu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=block_id)


def reply(stop_reason, *blocks):
    return SimpleNamespace(stop_reason=stop_reason, content=list(blocks))


class FakeMessages:
    """Returns the scripted replies in order and records every request."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, **kwargs):
        # Copy the list, because the engine keeps appending to it after the call.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.replies.pop(0)


class FakeClient:
    def __init__(self, replies):
        self.messages = FakeMessages(replies)


class FakeKB:
    def __init__(self):
        self.queries = []

    def search(self, query, n_results=5):
        self.queries.append((query, n_results))
        return [
            {"text": "Raft elects a leader by majority vote.", "filename": "notes.pdf", "doc_id": "notes.pdf", "score": 0.82},
            {"text": "Paxos is harder to understand than Raft.", "filename": "paper.pdf", "doc_id": "paper.pdf", "score": 0.61},
        ][:n_results]

    def list_documents(self):
        return [{"filename": "notes.pdf", "chunks": 4}, {"filename": "paper.pdf", "chunks": 9}]


def make_engine(replies):
    kb = FakeKB()
    client = FakeClient(replies)
    return ChatEngine(knowledge_base=kb, client=client), kb, client


# ---------- tests ----------

def test_answers_directly_without_tools():
    engine, kb, _ = make_engine([reply("end_turn", text_block("Hello!"))])
    answer, sources, steps = engine.ask("hi")
    assert answer == "Hello!"
    assert sources == [] and steps == []
    assert kb.queries == []


def test_runs_search_tool_then_answers():
    engine, kb, client = make_engine([
        reply("tool_use", tool_block("search_knowledge_base", {"query": "how does Raft elect a leader", "n_results": 3})),
        reply("end_turn", text_block("By majority vote [source: notes.pdf].")),
    ])
    answer, sources, steps = engine.ask("How does Raft pick a leader?")

    assert answer == "By majority vote [source: notes.pdf]."
    # The agent chose its own search query and result count
    assert kb.queries == [("how does Raft elect a leader", 3)]
    assert steps == [{"tool": "search_knowledge_base", "input": {"query": "how does Raft elect a leader", "n_results": 3}}]
    # Sources are returned best first
    assert [s["filename"] for s in sources] == ["notes.pdf", "paper.pdf"]

    # The second request carried the tool result back to Claude
    second = client.messages.calls[1]["messages"]
    tool_result = second[-1]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "toolu_1"
    assert "majority vote" in tool_result["content"]


def test_list_documents_tool():
    engine, _, client = make_engine([
        reply("tool_use", tool_block("list_documents", {})),
        reply("end_turn", text_block("You have notes.pdf and paper.pdf.")),
    ])
    _, _, steps = engine.ask("What's in my knowledge base?")
    assert steps[0]["tool"] == "list_documents"
    assert "paper.pdf" in client.messages.calls[1]["messages"][-1]["content"][0]["content"]


def test_n_results_is_clamped():
    engine, kb, _ = make_engine([
        reply("tool_use", tool_block("search_knowledge_base", {"query": "x", "n_results": 500})),
        reply("end_turn", text_block("done")),
    ])
    engine.ask("q")
    assert kb.queries == [("x", 10)]


def test_stops_after_max_steps():
    looping = [reply("tool_use", tool_block("search_knowledge_base", {"query": "again"})) for _ in range(MAX_AGENT_STEPS)]
    engine, kb, _ = make_engine(looping)
    answer, _, steps = engine.ask("q")
    assert len(steps) == MAX_AGENT_STEPS
    assert len(kb.queries) == MAX_AGENT_STEPS
    assert "couldn't reach a confident answer" in answer


def test_history_is_sent_and_starts_with_user():
    engine, _, client = make_engine([reply("end_turn", text_block("ok"))])
    history = [
        {"role": "assistant", "content": "Welcome!"},
        {"role": "user", "content": "Tell me about Raft"},
        {"role": "assistant", "content": "Raft is a consensus algorithm."},
    ]
    engine.ask("And Paxos?", history=history)
    sent = client.messages.calls[0]["messages"]
    assert sent[0]["role"] == "user"
    assert [m["content"] for m in sent] == ["Tell me about Raft", "Raft is a consensus algorithm.", "And Paxos?"]
