"""Smoke test: the Streamlit app boots and renders its empty state."""

from streamlit.testing.v1 import AppTest


def test_app_renders_empty_state(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("CORTEX_DATA_DIR", str(tmp_path))
    at = AppTest.from_file("../app.py", default_timeout=60).run()
    assert not at.exception
    page = " ".join(m.value for m in at.markdown)
    assert "Your knowledge base is empty" in page


def test_app_explains_missing_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CORTEX_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    at = AppTest.from_file("../app.py", default_timeout=60).run()
    assert any("ANTHROPIC_API_KEY" in e.value for e in at.error)
