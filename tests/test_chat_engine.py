"""Agent-loop tests with a scripted fake Claude client (no network, no API key)."""

import copy
from types import SimpleNamespace

from chat_engine import ChatEngine, summarize_steps


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(id_, name, input_):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=input_)


def reply(blocks, stop_reason):
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
    )


class FakeClient:
    """Returns pre-scripted responses and records every request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        # Snapshot: the engine keeps appending to the same messages list.
        self.requests.append(copy.deepcopy(kwargs))
        return self.responses.pop(0)


class FakeKB:
    def __init__(self):
        self.queries = []

    def search(self, query, n_results=5, filename=None):
        self.queries.append((query, filename))
        return [
            {"text": f"Passage about {query}", "filename": filename or "notes.pdf",
             "doc_id": filename or "notes.pdf", "chunk_index": 0, "score": 0.81},
        ]

    def list_documents(self):
        return {"notes.pdf": 3, "slides.pdf": 2}


def test_agent_searches_then_answers_with_citations():
    client = FakeClient([
        reply([tool_block("t1", "search_knowledge_base", {"query": "CAP theorem"})], "tool_use"),
        reply([text_block("The CAP theorem trades consistency for availability [S1].")], "end_turn"),
    ])
    kb = FakeKB()
    out = ChatEngine(kb, client=client).ask("Explain CAP")

    assert out.answer.endswith("[S1].")
    assert [s["source_id"] for s in out.sources] == ["S1"]
    assert kb.queries == [("CAP theorem", None)]
    assert out.input_tokens == 200 and out.output_tokens == 40
    # The tool result was sent back, tied to the right tool_use id.
    tool_result = client.requests[1]["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "t1"
    assert "[S1] notes.pdf" in tool_result["content"]


def test_multiple_searches_get_distinct_source_ids():
    client = FakeClient([
        reply([
            tool_block("t1", "search_knowledge_base", {"query": "x", "filename": "a.pdf"}),
            tool_block("t2", "search_knowledge_base", {"query": "y", "filename": "b.pdf"}),
        ], "tool_use"),
        reply([text_block("A says x [S1]; B says y [S2].")], "end_turn"),
    ])
    out = ChatEngine(FakeKB(), client=client).ask("Compare a and b")
    assert [s["filename"] for s in out.sources] == ["a.pdf", "b.pdf"]
    assert summarize_steps(out.steps) == [
        "searched in a.pdf: “x” → 1 passages",
        "searched in b.pdf: “y” → 1 passages",
    ]


def test_hallucinated_citations_are_removed():
    client = FakeClient([
        reply([tool_block("t1", "search_knowledge_base", {"query": "q"})], "tool_use"),
        reply([text_block("Real [S1]. Made up [S9].")], "end_turn"),
    ])
    out = ChatEngine(FakeKB(), client=client).ask("q")
    assert out.invalid_citations == ["S9"]
    assert "[S9]" not in out.answer and "[S1]" in out.answer


def test_final_step_forces_an_answer():
    import chat_engine

    steps = chat_engine.MAX_AGENT_STEPS
    looping = [reply([tool_block(f"t{i}", "list_documents", {})], "tool_use") for i in range(steps - 1)]
    client = FakeClient(looping + [reply([text_block("Here is what I found.")], "end_turn")])
    out = ChatEngine(FakeKB(), client=client).ask("q")
    assert out.answer == "Here is what I found."
    assert client.requests[-1]["tool_choice"] == {"type": "none"}
    assert all(r["tool_choice"] == {"type": "auto"} for r in client.requests[:-1])


def test_history_is_trimmed_and_alternates():
    client = FakeClient([reply([text_block("ok")], "end_turn")])
    history = [{"role": "assistant", "content": "welcome"}] + [
        {"role": r, "content": f"{r}{i}"} for i in range(5) for r in ("user", "assistant")
    ]
    ChatEngine(FakeKB(), client=client).ask("new question", history=history)
    sent = client.requests[0]["messages"]
    roles = [m["role"] for m in sent]
    assert roles[0] == "user" and roles[-1] == "user"
    assert all(a != b for a, b in zip(roles, roles[1:]))
    assert sent[-1]["content"] == "new question"
    assert len(sent) <= 7


def test_missing_api_key_gives_clear_error(monkeypatch):
    import pytest

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        ChatEngine(FakeKB())
