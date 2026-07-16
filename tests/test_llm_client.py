import importlib
import sys
import types


def _import_llm_client(monkeypatch, load_dotenv_impl=None):
    fake_dotenv = types.ModuleType("dotenv")

    if load_dotenv_impl is None:
        def load_dotenv(*args, **kwargs):
            return None
    else:
        load_dotenv = load_dotenv_impl

    fake_dotenv.load_dotenv = load_dotenv
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    monkeypatch.delitem(sys.modules, "outils.llm_client", raising=False)
    return importlib.import_module("outils.llm_client")


def test_import_does_not_call_load_dotenv(monkeypatch):
    calls = {"count": 0}

    def load_dotenv(*args, **kwargs):
        calls["count"] += 1
        return None

    module = _import_llm_client(monkeypatch, load_dotenv)
    assert calls["count"] == 0
    assert hasattr(module, "create_client")


def test_create_client_ignores_blank_openai_base_url(monkeypatch):
    class FakeOpenAI:
        last_kwargs = None

        def __init__(self, **kwargs):
            FakeOpenAI.last_kwargs = kwargs
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **_kwargs: None)
            )

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = FakeOpenAI

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "   ")
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    module = _import_llm_client(monkeypatch)
    client, provider = module.create_client()

    assert provider == "openai"
    assert isinstance(client, FakeOpenAI)
    assert FakeOpenAI.last_kwargs == {"api_key": "test-key"}
