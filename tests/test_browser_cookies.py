"""Offline tests for browser-profile cookie extraction (no network, no browser).

Builds throwaway fake HOMEs containing Firefox and Chrome-family profiles -
including properly encrypted cookie rows (v10 CBC + v11 GCM) - and pins the
extraction, auto-pick, error and masking behaviour of
qwen_studio.browser_cookies.
"""
import base64
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes  # noqa: E402
from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: E402

from qwen_studio import QwenStudio, browser_cookies as bc  # noqa: E402
from qwen_studio.exceptions import AuthError  # noqa: E402

TOKEN = "sess-token-abcdef1234567890"
EXPIRY = 4102444800  # 2100-01-01


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


# ------------------------------------------------------------------ fixtures
def _make_firefox(home: Path, name: str = "default-release", with_token: bool = True,
                  mtime: float = 0.0) -> Path:
    root = home / ".mozilla/firefox"
    prof = root / "Profiles" / f"abc123.{name}"
    prof.mkdir(parents=True)
    (root / "profiles.ini").write_text(
        "[General]\nStartWithLastProfile=1\nVersion=2\n\n"
        f"[Profile0]\nName={name}\nIsRelative=1\n"
        f"Path=Profiles/abc123.{name}\nDefault=1\n", encoding="utf-8")
    con = sqlite3.connect(prof / "cookies.sqlite")
    con.execute(
        "CREATE TABLE moz_cookies (id INTEGER PRIMARY KEY, originAttributes TEXT,"
        " baseDomain TEXT, name TEXT, value TEXT, host TEXT, path TEXT,"
        " expiry INTEGER, lastAccessed INTEGER, creationTime INTEGER,"
        " isSecure INTEGER, isHttpOnly INTEGER, inBrowserElement INTEGER,"
        " sameSite INTEGER, rawSameSite INTEGER, schemeMap INTEGER)")
    rows = [("token", TOKEN, ".qwen.ai"), ("acw_tc", "ACW1", "chat.qwen.ai"),
            ("tfstk", "TFK1", ".qwen.ai"), ("sid", "X", ".example.com")]
    con.executemany(
        "INSERT INTO moz_cookies (name, value, host, path, expiry, isSecure,"
        " isHttpOnly, sameSite) VALUES (?,?,?,?,?,1,1,0)",
        [(n, v, h, "/", EXPIRY) for n, v, h in rows if with_token or n != "token"])
    con.commit()
    con.close()
    db = prof / "cookies.sqlite"
    if mtime:
        import os
        os.utime(db, (mtime, mtime))
    return db


def _v10_blob(host: str, value: str) -> bytes:
    key = hashlib.pbkdf2_hmac("sha1", b"peanuts", b"saltysalt", 1, dklen=16)
    plain = hashlib.sha256(host.lstrip(".").encode()).digest() + value.encode()
    pad = 16 - len(plain) % 16
    plain += bytes([pad]) * pad
    enc = Cipher(algorithms.AES(key), modes.CBC(b" " * 16)).encryptor()
    return b"v10" + enc.update(plain) + enc.finalize()


def _v11_blob(host: str, value: str, key: bytes) -> bytes:
    nonce = b"0123456789ab"
    return b"v11" + nonce + AESGCM(key).encrypt(nonce, value.encode(), None)


def _make_chromium(home: Path, root_name: str = "chromium",
                   profile: str = "Default", v11: bool = False) -> Path:
    root = home / ".config" / root_name
    prof = root / profile / "Network"
    prof.mkdir(parents=True)
    db = prof / "Cookies"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE cookies (creation_utc INTEGER NOT NULL, host_key TEXT NOT NULL,"
        " name TEXT NOT NULL, value TEXT NOT NULL, encrypted_value BLOB NOT NULL,"
        " path TEXT NOT NULL, expires_utc INTEGER NOT NULL, is_secure INTEGER NOT NULL,"
        " is_httponly INTEGER NOT NULL, last_access_utc INTEGER NOT NULL)")
    host = "chat.qwen.ai"
    if v11:
        key = bytes(range(32))
        (root / "Local State").write_text(json.dumps(
            {"os_crypt": {"encrypted_key": base64.b64encode(b"DPAPI" + key).decode()}}))
        blob = _v11_blob(host, "V11SECRET", key)
    else:
        blob = _v10_blob(host, "V10SECRET")
    con.execute("INSERT INTO cookies VALUES (?,?,?,?,?,?,?,?,?,?)",
                (13370000000000000, host, "token", "", blob, "/", EXPIRY * 1_000_000, 1, 1, 13370000000000000))
    con.execute("INSERT INTO cookies VALUES (?,?,?,?,?,?,?,?,?,?)",
                (13370000000000001, ".qwen.ai", "acw_tc", "ACWPLAIN", b"", "/", 0, 0, 0, 13370000000000001))
    con.execute("INSERT INTO cookies VALUES (?,?,?,?,?,?,?,?,?,?)",
                (13370000000000002, ".example.com", "other", "", _v10_blob(".example.com", "NOPE"), "/", 0, 0, 0, 13370000000000002))
    con.commit()
    con.close()
    return db


# ------------------------------------------------------------------- firefox
def test_firefox_extraction(fake_home):
    _make_firefox(fake_home)
    cookies = bc.load_browser_cookies("firefox")
    names = {c.name: c.value for c in cookies}
    assert names["token"] == TOKEN
    assert names["acw_tc"] == "ACW1" and names["tfstk"] == "TFK1"
    assert "sid" not in names                       # foreign domain filtered
    assert cookies[0].source.startswith("firefox:")


def test_firefox_found_via_autodetect(fake_home):
    _make_firefox(fake_home)
    prof, jar = bc.find_qwen_jar()                  # no browser given
    assert prof.browser == "firefox"
    assert {c.name for c in jar} >= {"token", "acw_tc"}


# ------------------------------------------------------------- chrome family
def test_chromium_v10_extraction(fake_home):
    _make_chromium(fake_home, "chromium")
    names = {c.name: c.value for c in bc.load_browser_cookies("chromium")}
    assert names["token"] == "V10SECRET"            # v10 CBC + peanuts key
    assert names["acw_tc"] == "ACWPLAIN"            # plaintext value column
    assert "other" not in names


def test_chrome_brand_v11_extraction(fake_home):
    _make_chromium(fake_home, "google-chrome", profile="Profile 1", v11=True)
    prof, jar = bc.find_qwen_jar("chrome")
    assert prof.name == "Profile 1"
    names = {c.name: c.value for c in jar}
    assert names["token"] == "V11SECRET"            # v11 GCM via Local State key
    assert names["acw_tc"] == "ACWPLAIN"


def test_brave_and_edge_roots_seen(fake_home):
    _make_chromium(fake_home, "BraveSoftware/Brave-Browser")
    profs = bc.list_profiles("brave")
    assert len(profs) == 1 and profs[0].browser == "brave"
    with pytest.raises(bc.BrowserCookieError):
        bc.list_profiles("netscape")


# ----------------------------------------------------------------- selection
def test_autopick_prefers_newest_profile(fake_home):
    _make_firefox(fake_home, "old", with_token=True, mtime=1000)
    _make_firefox(fake_home, "new", with_token=True, mtime=9999)
    prof, _ = bc.find_qwen_jar("firefox")
    assert prof.name == "new"


def test_missing_token_is_actionable(fake_home):
    _make_firefox(fake_home, with_token=False)
    with pytest.raises(bc.BrowserCookieError, match="session token"):
        bc.find_qwen_jar("firefox")


def test_no_profiles_is_actionable(fake_home):
    with pytest.raises(bc.BrowserCookieError, match="no supported browser"):
        bc.find_qwen_jar()


def test_explicit_profile_not_found(fake_home):
    _make_firefox(fake_home)
    with pytest.raises(bc.BrowserCookieError, match="available"):
        bc.find_qwen_jar("firefox", profile="nope")


def test_qwen_cookie_dict_shape(fake_home):
    _make_firefox(fake_home)
    jar = bc.qwen_cookie_dict("firefox")
    assert jar["token"] == TOKEN and set(jar) == {"token", "acw_tc", "tfstk"}


# ------------------------------------------------------------ client wiring
def test_from_browser_wiring(fake_home, monkeypatch):
    from qwen_studio.browser_cookies import BrowserCookie, BrowserProfile
    fake = (BrowserProfile("firefox", "default-release",
                           Path("/fake/cookies.sqlite"), "firefox"),
            [BrowserCookie("token", TOKEN, ".qwen.ai"),
             BrowserCookie("acw_tc", "A", "chat.qwen.ai")])
    monkeypatch.setattr(bc, "find_qwen_jar", lambda *a, **k: fake)
    q = QwenStudio.from_browser(auto_refresh=False)
    assert q.session_token == TOKEN
    assert q._cookies() == {"token": TOKEN, "acw_tc": "A"}
    assert q.cookie_source.startswith("firefox:default-release")
    assert q.http._is_curl_cffi is True              # hardened transport default


def test_from_browser_without_token_raises(fake_home, monkeypatch):
    from qwen_studio.browser_cookies import BrowserCookie, BrowserProfile
    fake = (BrowserProfile("firefox", "p", Path("/fake/cookies.sqlite"), "firefox"),
            [BrowserCookie("acw_tc", "A", ".qwen.ai")])
    monkeypatch.setattr(bc, "find_qwen_jar", lambda *a, **k: fake)
    with pytest.raises(AuthError, match="session token"):
        QwenStudio.from_browser(auto_refresh=False)


# ----------------------------------------------------------------------- CLI
def test_cli_masks_values(fake_home, capsys):
    _make_firefox(fake_home)
    assert bc.main(["--browser", "firefox"]) == 0
    out = capsys.readouterr().out
    assert "token" in out
    assert TOKEN not in out                          # never print the secret
    assert "firefox" in out


def test_cli_error_exit(fake_home, capsys):
    assert bc.main(["--browser", "firefox"]) == 1
    assert "error:" in capsys.readouterr().err


# ----------------------------------------------------- transport hard-wiring
def test_impersonate_required():
    # curl_cffi present in CI: impersonate must yield its session
    q = QwenStudio.from_access_token("t")
    assert q.http._is_curl_cffi is True
    # and a missing-transport request is a loud ImportError, not silent fallback
    import qwen_studio.client as cl
    saved = cl.crequests
    cl.crequests = None
    try:
        with pytest.raises(ImportError, match="curl_cffi"):
            QwenStudio.from_access_token("t")
    finally:
        cl.crequests = saved
