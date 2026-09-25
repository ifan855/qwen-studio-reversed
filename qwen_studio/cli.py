"""The ``qwen-studio`` command line interface.

Two commands do all the work::

    # 1. Pull the browser's Qwen cookies into the current directory
    $ qwen-studio login                      # auto-detects Firefox/Chrome/...
    $ ls qwen.auth.json                      # <- ready for `serve`

    # 2. Serve a well-behaved OpenAI-compatible API wrapping Qwen Studio
    $ qwen-studio serve --port 8080
    [proxy] OpenAI-compatible API on http://127.0.0.1:8080/v1

    $ curl http://127.0.0.1:8080/v1/chat/completions -d '{...}'

``login`` never needs the password when a local browser profile is logged
in to chat.qwen.ai; ``--email/--password`` (or the ``QWEN_EMAIL`` /
``QWEN_PASSWORD`` env vars) is the fallback path. When no browser profile
exists and env vars are set, ``login`` automatically enables the
headless-browser warmup (``--warmup``) so the resulting auth file carries
the full anti-bot cookie jar — without it, the four-cookie jar from
``/auths/signin`` alone is eventually punished on ``/chat/completions``
(see docs/anti-bot.md). ``--save-password`` optionally persists the
credentials so ``serve`` can re-auth forever.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

AUTH_FILENAME = "qwen.auth.json"
AUTH_VERSION = 1


def _mask(value: Optional[str], keep: int = 6) -> str:
    if not value:
        return "(none)"
    return f"{value[:keep]}…({len(value)} chars)" if len(value) > keep else "…"


# ------------------------------------------------------------------- login
def cmd_login(args: argparse.Namespace) -> int:
    from . import exceptions as qe

    auth: Dict[str, Any] = {"version": AUTH_VERSION,
                            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "session_token": None,
                            "extra_cookies": {},
                            "cookie_source": None,
                            "email": None,
                            "password": None,
                            "warmup": False,
                            "warmup_backend": "auto"}
    jar: Dict[str, str] = {}
    source = None

    # env-var fallback: QWEN_EMAIL/QWEN_PASSWORD (the same vars `serve` reads)
    email = args.email or os.environ.get("QWEN_EMAIL")
    password = args.password or os.environ.get("QWEN_PASSWORD")

    if not email:
        # primary path: read the complete jar straight from the browser
        from . import browser_cookies as bc
        try:
            prof, cookies = bc.find_qwen_jar(args.browser, args.profile,
                                             domain=args.domain)
        except bc.BrowserCookieError as e:
            print(f"error: {e}", file=sys.stderr)
            if not password:
                print("hint: set QWEN_EMAIL+QWEN_PASSWORD env vars, or pass "
                      "--email/--password, to sign in without a local "
                      "browser profile (--warmup will mint the anti-bot "
                      "cookies via a headless browser)", file=sys.stderr)
                return 2
        else:
            jar = {c.name: c.value for c in cookies}
            source = f"{prof.browser}:{prof.name} ({prof.cookie_db})"
            auth["session_token"] = jar.pop("token", None)
            if not auth["session_token"]:
                print(f"error: no session token for {args.domain!r} in "
                      f"{source}; log in to chat.qwen.ai in that browser first",
                      file=sys.stderr)
                return 2

    if email and password:
        from .client import QwenStudio
        warmup_on = args.warmup
        # when there's no browser profile, warmup is mandatory (otherwise
        # the resulting session would carry only the four signin cookies
        # and get punished on /chat/completions)
        if not jar and not warmup_on:
            print("note: --warmup not set but no browser profile found; "
                  "enabling warmup automatically (otherwise the session "
                  "would have a thin anti-bot jar)", file=sys.stderr)
            warmup_on = True
        try:
            q = QwenStudio.from_credentials(
                email, password,
                extra_cookies=jar,
                min_interval=args.min_interval,
                warmup=warmup_on,
                warmup_backend=args.warmup_backend)
        except qe.QwenStudioError as e:
            print(f"error: sign-in failed: {e}", file=sys.stderr)
            return 2
        auth["session_token"] = q.session_token or auth["session_token"]
        # merge both the absorbed Set-Cookie values AND the warmup jar
        jar = dict(q.extra_cookies) or jar
        source = source or (f"credentials+warmup({args.warmup_backend})"
                            if warmup_on else "credentials")
        if args.save_password:
            auth["email"], auth["password"] = email, password
        auth["warmup"] = warmup_on
        auth["warmup_backend"] = args.warmup_backend

    if not auth["session_token"]:
        print("error: nothing to save (no token obtained)", file=sys.stderr)
        return 2

    auth["extra_cookies"] = jar
    auth["cookie_source"] = source
    path = os.path.abspath(args.file)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(auth, f, indent=2, ensure_ascii=False)
    try:
        os.chmod(path, 0o600)
    except OSError:  # pragma: no cover
        pass

    print(f"auth file written : {path}")
    print(f"  cookie source   : {source}")
    print(f"  session token   : {_mask(auth['session_token'])}")
    print(f"  anti-bot cookies: {len(jar)} "
          f"({', '.join(sorted(jar)[:6])}{'…' if len(jar) > 6 else ''})")
    print(f"  warmup          : {'on' if auth['warmup'] else 'off'} "
          f"(backend={auth['warmup_backend']})")
    print(f"  password saved  : {'yes' if auth['password'] else 'no'}")
    print("next: qwen-studio serve   (serves an OpenAI-compatible API)")
    return 0


# ------------------------------------------------------------------- serve
def _build_client(args: argparse.Namespace):
    from . import exceptions as qe
    from .client import QwenStudio

    email = args.email or os.environ.get("QWEN_EMAIL")
    password = args.password or os.environ.get("QWEN_PASSWORD")
    path = os.path.abspath(args.auth_file)
    auth: Dict[str, Any] = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            auth = json.load(f)
    elif not (email and password):
        print(f"error: no auth file at {path} and no --email/--password.\n"
              f"run `qwen-studio login` in this directory first (or pass "
              f"--auth-file / QWEN_EMAIL+QWEN_PASSWORD).", file=sys.stderr)
        raise SystemExit(2)

    jar = auth.get("extra_cookies") or {}
    email = email or auth.get("email")
    password = password or auth.get("password")
    # warmup is on by default for credential-based auth (the thin jar from
    # signin alone gets punished on /chat/completions - see docs/anti-bot.md);
    # honour an explicit --no-warmup, and honour whatever the auth file recorded
    warmup_on = args.warmup if args.warmup is not None else auth.get("warmup", False)
    warmup_backend = args.warmup_backend or auth.get("warmup_backend", "auto")
    min_iv = {"min_interval": args.min_interval}
    try:
        if email and password:
            q = QwenStudio.from_credentials(
                email, password,
                extra_cookies=jar, warmup=warmup_on,
                warmup_backend=warmup_backend, **min_iv)
        elif auth.get("session_token"):
            q = QwenStudio.from_session_token(
                auth["session_token"],
                extra_cookies=jar, warmup=warmup_on,
                warmup_backend=warmup_backend, **min_iv)
        else:
            print("error: auth file has neither credentials nor a session "
                  "token; run `qwen-studio login` again", file=sys.stderr)
            raise SystemExit(2)
    except qe.QwenStudioError as e:
        print(f"error: could not establish the upstream session: {e}",
              file=sys.stderr)
        raise SystemExit(2)
    print(f"upstream session   : {q.cookie_source}")
    print(f"  cookie jar size  : {len(q._cookies())}")
    if getattr(q, "_warmed_up", False):
        print(f"  warmup           : on (backend={warmup_backend})")
    return q


def cmd_serve(args: argparse.Namespace) -> int:
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    from .openai_api import OpenAICompatService, QwenBackend
    from .openai_server import OpenAIProxyServer

    client = _build_client(args)

    # startup self-check (non-fatal: the risk engine may simply be cold)
    backend = QwenBackend(client, default_model=args.default_model)
    try:
        ids = backend.model_ids()
        print(f"upstream catalogue : {len(ids)} models "
              f"({', '.join(ids[:3])}{'…' if len(ids) > 3 else ''})")
    except Exception as e:  # noqa: BLE001
        print(f"warning: catalogue check failed ({e}); continuing and will "
              f"retry per request", file=sys.stderr)

    service = OpenAICompatService(
        backend, ttl=args.ttl, max_sessions=args.max_sessions,
        replay_mode=args.replay, thinking=args.thinking,
        oneshot_ttl=None if args.oneshot_ttl < 0 else args.oneshot_ttl)
    service.start_sweeper()
    server = OpenAIProxyServer((args.host, args.port), service,
                               api_key=args.api_key)

    print(f"OpenAI-compatible API on http://{args.host}:{args.port}/v1")
    print(f"  history memory   : {int(args.ttl)}s TTL, "
          f"max {args.max_sessions} conversations")
    print("  one-shot cleanup : " + (
        "off" if args.oneshot_ttl < 0 else
        f"single-turn conversations' chat + project deleted after "
        f"{int(args.oneshot_ttl)}s idle (history kept; follow-ups revive)"))
    print(f"  replay mode      : {args.replay} (unseen histories are "
          f"prompt-engineered in)")
    print(f"  api key          : {'required' if args.api_key else 'not set'}")
    print("Ctrl-C to stop (expires and cleans up conversations).", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            service.shutdown()
        except Exception:  # noqa: BLE001
            pass
        server.server_close()
        print("stopped; conversations cleaned up upstream (best effort).")
    return 0


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qwen-studio",
        description="Browser-cookie login + an OpenAI-compatible API server "
                    "wrapping Qwen Studio (chat.qwen.ai).")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("login", help="pull cookies from a local browser "
                                      "(or sign in) into an auth file in the "
                                      "current directory")
    pl.add_argument("--browser", choices=["firefox", "chrome", "chromium",
                                          "brave", "edge"],
                    help="browser to pull cookies from (default: auto-detect)")
    pl.add_argument("--profile", help="specific browser profile name")
    pl.add_argument("--domain", default="qwen.ai",
                    help="cookie domain filter (default qwen.ai)")
    pl.add_argument("--email", help="sign in with credentials instead of "
                                    "pulling a browser profile (or set "
                                    "QWEN_EMAIL+QWEN_PASSWORD)")
    pl.add_argument("--password", help="password for --email (or set "
                                       "QWEN_PASSWORD)")
    pl.add_argument("--save-password", action="store_true",
                    help="persist the password in the auth file so `serve` "
                         "can re-authenticate forever")
    pl.add_argument("--file", default=AUTH_FILENAME,
                    help=f"auth file path (default ./{AUTH_FILENAME})")
    pl.add_argument("--min-interval", type=float, default=4.0)
    pl.add_argument("--warmup", action="store_true", default=False,
                    help="run a headless-browser warmup after sign-in to "
                         "mint the complete anti-bot cookie jar "
                         "(auto-enabled when no browser profile is found "
                         "and QWEN_EMAIL/QWEN_PASSWORD are set)")
    pl.add_argument("--warmup-backend", default="auto",
                    choices=["auto", "playwright", "agent-browser"],
                    help="which headless-browser backend to use for the "
                         "warmup (default: auto - Playwright if installed, "
                         "else the agent-browser CLI)")
    pl.set_defaults(fn=cmd_login)

    ps = sub.add_parser("serve", help="serve the OpenAI-compatible API")
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=8080)
    ps.add_argument("--auth-file", default=AUTH_FILENAME,
                    help=f"auth file (default ./{AUTH_FILENAME})")
    ps.add_argument("--email", help="override/replacement credentials "
                                    "(or set QWEN_EMAIL + QWEN_PASSWORD)")
    ps.add_argument("--password")
    ps.add_argument("--warmup", dest="warmup", action="store_true",
                    default=None,
                    help="run a headless-browser warmup after sign-in to "
                         "mint the complete anti-bot cookie jar (default: "
                         "on when credentials are used, off when a browser "
                         "profile / session token supplies the jar)")
    ps.add_argument("--no-warmup", dest="warmup", action="store_false",
                    default=None,
                    help="skip the warmup even when credentials are used "
                         "(the session will have only the four signin "
                         "cookies and may be punished on /chat/completions)")
    ps.add_argument("--warmup-backend", default="auto",
                    choices=["auto", "playwright", "agent-browser"],
                    help="which headless-browser backend to use for the "
                         "warmup (default: auto)")
    ps.add_argument("--api-key", help="require clients to send "
                                      "'Authorization: Bearer <key>'")
    ps.add_argument("--ttl", type=float, default=3600.0,
                    help="conversation memory window in seconds (default 3600)")
    ps.add_argument("--max-sessions", type=int, default=256)
    ps.add_argument("--oneshot-ttl", type=float, default=-1.0,
                    help="optional early deletion of a single-turn "
                         "conversation's upstream chat + project; disabled "
                         "by default because the normal session TTL controls "
                         "cleanup (-1 disables)")
    ps.add_argument("--replay", choices=["both", "file", "inline"],
                    default="both",
                    help="how unseen histories are replayed into a fresh "
                         "conversation (default: upload a history file AND "
                         "inline the transcript)")
    ps.add_argument("--thinking", action="store_true", default=None,
                    help="enable Qwen thinking (exposed as "
                         "'reasoning_content' deltas)")
    ps.add_argument("--default-model",
                    help="model id used when the client's model is unknown "
                         "(default: first catalogue entry)")
    ps.add_argument("--min-interval", type=float, default=4.0)
    ps.set_defaults(fn=cmd_serve)

    pv = sub.add_parser("version", help="print the version")
    pv.set_defaults(fn=lambda a: (_print_version(), 0)[1])
    return p


def _print_version() -> None:
    from . import __version__
    print(f"qwen-studio {__version__}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
