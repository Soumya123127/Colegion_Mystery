import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("project_server", ROOT / "scripts" / "server.py")
SERVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVER)


def test_runtime_bind_defaults_localhost(monkeypatch):
    monkeypatch.delenv("PORT", raising=False)
    host, port = SERVER.resolve_runtime_bind(default_port=8000)
    assert (host, port) == ("127.0.0.1", 8000)


def test_runtime_bind_uses_render_port(monkeypatch):
    monkeypatch.setenv("PORT", "10000")
    host, port = SERVER.resolve_runtime_bind(default_port=8000)
    assert (host, port) == ("0.0.0.0", 10000)
