"""CLI tests: `qwen-studio serve` must fail with a message, not a traceback.

No network and no upstream session is required - the client constructor is
stubbed wherever a real one would be built.
"""

import json

import pytest

from qwen_studio import cli


class DummyClient:
    """Minimal stand-in for QwenStudio on the serve path."""

    cookie_source = "test:stub"

    def _cookies(self):
        return {"token": "t"}

    def list_model_ids(self):
        return ["qwen3-max"]


def serve_args(*argv):
    return cli.build_parser().parse_args(["serve", *argv])


@pytest.fixture
def stub_client(monkeypatch):
    """Never build a real (networked) client."""
    monkeypatch.setattr(cli, "_build_client", lambda args: DummyClient())
    return DummyClient


def write_auth(tmp_path, payload):
    if isinstance(payload, str):
        text = payload
    else:
        text = json.dumps(payload)
    p = tmp_path / "qwen.auth.json"
    p.write_text(text, encoding="utf-8")
    return p


def test_serve_missing_auth_file_is_a_clean_error(tmp_path, capsys):
    args = serve_args("--auth-file", str(tmp_path / "nope.json"))
    with pytest.raises(SystemExit) as e:
        cli._build_client(args)
    assert e.value.code == 2
    assert "no auth file" in capsys.readouterr().err


def test_serve_corrupt_auth_file_is_a_clean_error(tmp_path, capsys):
    """Truncated/hand-edited auth file -> message + exit 2, no traceback."""
    path = write_auth(tmp_path, '{"version": 1, "session_tok')
    args = serve_args("--auth-file", str(path))
    with pytest.raises(SystemExit) as e:
        cli._build_client(args)
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "cannot read the auth file" in err
    assert "qwen-studio login" in err


def test_serve_auth_file_without_object_is_a_clean_error(tmp_path, capsys):
    path = write_auth(tmp_path, "[1, 2, 3]")
    args = serve_args("--auth-file", str(path))
    with pytest.raises(SystemExit) as e:
        cli._build_client(args)
    assert e.value.code == 2
    assert "does not contain a JSON object" in capsys.readouterr().err


def test_serve_reports_a_taken_port(capsys, monkeypatch, stub_client):
    """Bind failure (port in use) -> message + exit code, no traceback."""
    import qwen_studio.openai_server as server_mod

    class Boom:
        def __init__(self, *a, **k):
            raise OSError(98, "Address already in use")

    monkeypatch.setattr(server_mod, "OpenAIProxyServer", Boom)
    assert cli.cmd_serve(serve_args("--port", "8931")) == 2
    err = capsys.readouterr().err
    assert "cannot bind" in err and "already listening" in err


def test_serve_swallows_a_failing_catalogue_check(capsys, monkeypatch,
                                                  stub_client):
    """A cold/blocked upstream catalogue must not stop the server starting."""
    import qwen_studio.openai_server as server_mod

    class DummyServer:
        instance = None

        def __init__(self, addr, service, api_key=None):
            DummyServer.instance = self
            self.addr, self.service = addr, service

        def serve_forever(self, poll_interval=0.5):
            raise KeyboardInterrupt

        def server_close(self):
            pass

    class BadCatalogue(DummyClient):
        def list_model_ids(self):
            raise RuntimeError("upstream unreachable")

    monkeypatch.setattr(server_mod, "OpenAIProxyServer", DummyServer)
    monkeypatch.setattr(cli, "_build_client", lambda args: BadCatalogue())
    assert cli.cmd_serve(serve_args()) == 0
    captured = capsys.readouterr()
    assert "catalogue check failed" in captured.err
    assert "OpenAI-compatible API on" in captured.out
