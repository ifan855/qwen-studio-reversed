"""Pull live cookies straight out of local browser profiles (Linux).

The anti-bot experiments in docs/anti-bot.md proved that ``/chat/completions``
is gated on the *complete* browser cookie jar - the session ``token`` plus
the edge cookies the browser earned (``acw_tc``, ``tfstk``, ``isg``,
``ssxmod_itna*``, ``cna``, ...). This module reads that jar directly from
the machine's browser profiles so a script can present exactly the session
material the real browser presents, with no manual export step.

Supported browsers (Linux locations, incl. snap and flatpak):

- **Firefox** - plain SQLite (``cookies.sqlite``), no decryption needed.
- **Chrome / Chromium / Brave / Edge** - SQLite cookie store encrypted with
  the browser's profile key:
    - ``v10`` cookies: AES-128-CBC, key = PBKDF2-HMAC-SHA1(keyring password
      or the well-known ``"peanuts"`` fallback, salt ``b"saltysalt"``, 1
      iteration, 16 bytes), IV = 16 space characters.
    - ``v11`` cookies (newer Chrome builds): AES-256-GCM with the key stored
      in ``Local State`` (``os_crypt.encrypted_key``).
    - The libsecret / kwallet password ("... Safe Storage") is looked up
      best-effort; locked keyrings are skipped, never prompted.

Privacy: cookies are read from a temporary copy of the database (works even
while the browser is running), and only the requested domain's rows are
returned. Nothing is written back, nothing leaves the machine except to
whichever host the caller sends them.

CLI debug view: ``python -m qwen_studio.browser_cookies`` (values masked).
"""

from __future__ import annotations

import base64
import configparser
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

__all__ = [
    "BrowserCookie", "BrowserProfile", "BrowserCookieError",
    "list_profiles", "find_qwen_jar", "load_browser_cookies",
    "qwen_cookie_dict", "main",
]

#: default domain filter: every *.qwen.ai host
QWEN_DOMAIN = "qwen.ai"

_FIREFOX_ROOTS = (
    ".mozilla/firefox",                                # native
    "snap/firefox/common/.mozilla/firefox",            # snap
    ".var/app/org.mozilla.firefox/.mozilla/firefox",   # flatpak
)

#: browser name -> user-data roots to scan
_CHROMIUM_ROOTS = {
    "chrome": (
        ".config/google-chrome",
        ".config/google-chrome-beta",
    ),
    "chromium": (
        ".config/chromium",
        "snap/chromium/common/chromium",
        ".var/app/org.chromium.Chromium/config/chromium",
    ),
    "brave": (
        ".config/BraveSoftware/Brave-Browser",
    ),
    "edge": (
        ".config/microsoft-edge",
    ),
}

#: libsecret/kwallet label keywords per browser (plus the " Safe Storage" suffix)
_SAFE_STORAGE_LABELS = {
    "chrome": ("Chrome Safe Storage",),
    "chromium": ("Chromium Safe Storage",),
    "brave": ("Brave Safe Storage",),
    "edge": ("Microsoft Edge Safe Storage", "Edge Safe Storage"),
}

_CHROME_PBKDF2_SALT = b"saltysalt"
_CHROME_IV = b" " * 16                     # 16 space characters, as Chromium does
_WEBKIT_EPOCH_OFFSET = 11644473600         # seconds between 1601-01-01 and unix epoch


class BrowserCookieError(Exception):
    """Cookie extraction failed (no profile, no token, undecryptable store)."""


@dataclass
class BrowserCookie:
    """One cookie row pulled from a browser profile."""
    name: str
    value: str
    domain: str = ""
    path: str = ""
    expires: int = 0          # unix seconds; 0 = session cookie
    secure: bool = False
    http_only: bool = False
    source: str = ""          # e.g. "firefox:default-release"


@dataclass
class BrowserProfile:
    """A browser profile that owns a cookie database."""
    browser: str              # firefox | chrome | chromium | brave | edge
    name: str                 # profile name, e.g. "default-release" / "Default"
    cookie_db: Path
    kind: str                 # "firefox" | "chromium"

    def tag(self, cookie_name: str = "") -> str:
        base = f"{self.browser}:{self.name}"
        return f"{base}#{cookie_name}" if cookie_name else base


# --------------------------------------------------------------------- paths

def _home() -> Path:
    return Path.home()


def _firefox_profiles() -> List[BrowserProfile]:
    out: List[BrowserProfile] = []
    seen: set = set()
    for rel in _FIREFOX_ROOTS:
        root = _home() / rel
        if not root.is_dir():
            continue
        found: List[BrowserProfile] = []
        ini = root / "profiles.ini"
        if ini.exists():
            cp = configparser.ConfigParser()
            try:
                cp.read(ini, encoding="utf-8")
            except configparser.Error:
                cp = configparser.ConfigParser()
            for sec in cp.sections():
                rel_path = cp[sec].get("Path")
                if not rel_path:
                    continue
                pdir = root / rel_path if cp[sec].get("IsRelative", "1") == "1" else Path(rel_path)
                db = pdir / "cookies.sqlite"
                if db.exists() and db not in seen:
                    found.append(BrowserProfile(
                        "firefox", cp[sec].get("Name") or pdir.name, db, "firefox"))
        if not found:  # glob fallback (portable / unpacked installs)
            for db in sorted(root.glob("*/cookies.sqlite")):
                found.append(BrowserProfile("firefox", db.parent.name, db, "firefox"))
        for p in found:
            if p.cookie_db not in seen:
                seen.add(p.cookie_db)
                out.append(p)
    return out


def _chromium_profiles(browser: str) -> List[BrowserProfile]:
    out: List[BrowserProfile] = []
    for rel in _CHROMIUM_ROOTS.get(browser, ()):
        root = _home() / rel
        if not root.is_dir():
            continue
        found: List[BrowserProfile] = []
        for db in list(root.glob("*/Network/Cookies")) + list(root.glob("*/Cookies")):
            prof_dir = db.parent if db.parent.name != "Network" else db.parent.parent
            found.append(BrowserProfile(browser, prof_dir.name, db, "chromium"))
        for cand in (root / "Network" / "Cookies", root / "Cookies"):
            if cand.exists() and all(p.cookie_db != cand for p in found):
                found.append(BrowserProfile(browser, "(default)", cand, "chromium"))
        out.extend(found)
    return out


def list_profiles(browser: Optional[str] = None) -> List[BrowserProfile]:
    """Enumerate usable cookie databases.

    ``browser`` may be ``"firefox"``, ``"chrome"``, ``"chromium"``,
    ``"brave"``, ``"edge"`` (case-insensitive) or ``None`` for all.
    """
    if browser:
        b = browser.strip().lower()
        if b in ("google-chrome", "googlechrome"):
            b = "chrome"
        if b == "firefox":
            return _firefox_profiles()
        if b in _CHROMIUM_ROOTS:
            return _chromium_profiles(b)
        raise BrowserCookieError(
            f"unsupported browser {browser!r}; supported: firefox, "
            "chrome, chromium, brave, edge (Linux)")
    return _firefox_profiles() + sum((_chromium_profiles(b) for b in _CHROMIUM_ROOTS), [])


# ------------------------------------------------------------- sqlite access

@contextmanager
def _copied_db(db: Path) -> Iterator[Path]:
    """Copy the cookie DB (plus WAL/SHM sidecars) to a temp dir and open that.

    Reading the live file while the browser runs can hit mid-write states;
    a private copy keeps us consistent and never touches the profile.
    """
    tmp = Path(tempfile.mkdtemp(prefix="qwen-studio-cookies-"))
    try:
        dst = tmp / db.name
        shutil.copy2(db, dst)
        for suffix in ("-wal", "-shm"):
            side = db.with_name(db.name + suffix)
            if side.exists():
                shutil.copy2(side, tmp / (db.name + suffix))
        yield dst
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _host_matches(host: str, domain: str) -> bool:
    if domain == "*":
        return True
    d = domain.lstrip(".").lower()
    h = host.lstrip(".").lower()
    return h == d or h.endswith("." + d)


def _read_firefox(profile: BrowserProfile, domain: str) -> List[BrowserCookie]:
    with _copied_db(profile.cookie_db) as db:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            rows = con.execute(
                "SELECT name, value, host, path, expiry, isSecure, isHttpOnly "
                "FROM moz_cookies").fetchall()
        finally:
            con.close()
    return [BrowserCookie(name=r[0], value=r[1] or "", domain=r[2], path=r[3] or "",
                          expires=int(r[4] or 0), secure=bool(r[5]),
                          http_only=bool(r[6]), source=profile.tag(r[0]))
            for r in rows if _host_matches(r[2], domain)]


# ------------------------------------------------- chrome-family decryption

def _pbkdf2(password: bytes, dklen: int) -> bytes:
    return hashlib.pbkdf2_hmac("sha1", password, _CHROME_PBKDF2_SALT, 1, dklen=dklen)


def _local_state_key(root: Path) -> Optional[bytes]:
    """v11 AES-256-GCM key from ``Local State`` (os_crypt.encrypted_key)."""
    for cand in (root / "Local State", root.parent / "Local State"):
        try:
            data = json.loads(cand.read_text(encoding="utf-8"))
            raw = base64.b64decode(data["os_crypt"]["encrypted_key"])
            if raw[:5] == b"DPAPI":       # prefix kept even on Linux builds
                raw = raw[5:]
            if len(raw) == 32:
                return raw
        except Exception:  # noqa: BLE001 - missing/corrupt Local State
            continue
    return None


def _safe_storage_passwords(browser: str) -> List[bytes]:
    """Best-effort keyring passwords; never prompts, never blocks."""
    labels = list(_SAFE_STORAGE_LABELS.get(browser, ()))
    out: List[bytes] = []
    try:  # libsecret (GNOME and friends)
        import secretstorage  # type: ignore

        bus = secretstorage.dbus_init()
        for coll in secretstorage.get_all_collections(bus):
            if coll.is_locked():
                continue                  # skip locked collections silently
            for item in coll.get_all_items():
                try:
                    label = item.get_label() or ""
                except Exception:  # noqa: BLE001
                    continue
                if label in labels or ("safe storage" in label.lower()
                                       and browser in label.lower()):
                    secret = item.get_secret()
                    if secret:
                        out.append(bytes(secret))
    except Exception:  # noqa: BLE001 - no dbus / no secretstorage / no entry
        pass
    kw = shutil.which("kwallet-query")
    if kw:  # KDE
        for label in labels:
            try:
                r = subprocess.run([kw, "-r", label, "kdewallet"],
                                   capture_output=True, timeout=5)
                if r.returncode == 0 and r.stdout.strip():
                    out.append(r.stdout.strip())
            except Exception:  # noqa: BLE001
                pass
    return out


def _chrome_key_candidates(profile: BrowserProfile) -> Tuple[List[bytes], List[bytes]]:
    """(16-byte v10 CBC keys, 32-byte v11 GCM keys) in preference order."""
    root = profile.cookie_db.parent          # .../Network -> profile dir
    if root.name == "Network":
        root = root.parent
    passwords = [pw for pw in _safe_storage_passwords(profile.browser)]
    cbc: List[bytes] = [_pbkdf2(pw, 16) for pw in passwords]
    gcm: List[bytes] = []
    ls = _local_state_key(root)
    if ls:
        gcm.append(ls)
    gcm += [_pbkdf2(pw, 32) for pw in passwords]
    cbc.append(_pbkdf2(b"peanuts", 16))      # well-known "basic" store fallback
    return cbc, gcm


def _cbc_decrypt(blob: bytes, key: bytes) -> bytes:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    dec = Cipher(algorithms.AES(key), modes.CBC(_CHROME_IV)).decryptor()
    plain = dec.update(blob) + dec.finalize()
    if plain and 1 <= plain[-1] <= 16:       # PKCS7-style padding
        plain = plain[:-plain[-1]]
    return plain


def _gcm_decrypt(blob: bytes, keys: List[bytes]) -> bytes:
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce, ct = blob[3:15], blob[15:]
    last_err: Optional[Exception] = None
    for key in keys:
        if len(key) != 32:
            continue
        try:
            return AESGCM(key).decrypt(nonce, ct, None)
        except InvalidTag as e:              # wrong key - try the next one
            last_err = e
    raise BrowserCookieError("v11 cookie: no valid AES-256-GCM key") from last_err


def _decrypt_chromium_value(blob: bytes, host: str, cbc_keys: List[bytes],
                            gcm_keys: List[bytes]) -> str:
    version, body = blob[:3], blob[3:]
    if version == b"v11":
        try:
            plain = _gcm_decrypt(blob, gcm_keys)
        except BrowserCookieError:
            plain = _cbc_decrypt(body, cbc_keys[0])   # last-ditch
    elif version == b"v10":
        plain = _cbc_decrypt(body, cbc_keys[0])
    else:
        raise BrowserCookieError(f"unsupported cookie encryption {version!r}")
    digest = hashlib.sha256(host.lstrip(".").encode()).digest()
    if plain[:32] == digest:                 # Chromium prepends sha256(host)
        plain = plain[32:]
    return plain.decode("utf-8", "replace")


def _read_chromium(profile: BrowserProfile, domain: str) -> List[BrowserCookie]:
    cbc_keys, gcm_keys = _chrome_key_candidates(profile)
    with _copied_db(profile.cookie_db) as db:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            rows = con.execute(
                "SELECT name, value, encrypted_value, host_key, path, "
                "expires_utc, is_secure, is_httponly FROM cookies").fetchall()
        finally:
            con.close()
    out: List[BrowserCookie] = []
    failures = 0
    for name, value, enc, host, path, exp, secure, httponly in rows:
        if not _host_matches(host, domain):
            continue
        if value:
            out.append(BrowserCookie(name=name, value=value, domain=host,
                                     path=path or "",
                                     expires=int(exp / 1_000_000 - _WEBKIT_EPOCH_OFFSET) if exp else 0,
                                     secure=bool(secure), http_only=bool(httponly),
                                     source=profile.tag(name)))
            continue
        if not enc:
            continue
        try:
            val = _decrypt_chromium_value(bytes(enc), host, cbc_keys, gcm_keys)
        except Exception:  # noqa: BLE001 - wrong key / unsupported version
            failures += 1
            continue
        out.append(BrowserCookie(name=name, value=val, domain=host, path=path or "",
                                 expires=int(exp / 1_000_000 - _WEBKIT_EPOCH_OFFSET) if exp else 0,
                                 secure=bool(secure), http_only=bool(httponly),
                                 source=profile.tag(name)))
    if failures and not out:
        raise BrowserCookieError(
            f"{profile.browser}:{profile.name}: {failures} encrypted cookie(s) "
            "could not be decrypted (keyring locked or empty). Unlock your "
            "desktop keyring, or note that v10 cookies fall back to the "
            "'peanuts' key only when Chrome uses the basic password store.")
    return out


# ----------------------------------------------------------------- top level

def _read_profile(profile: BrowserProfile, domain: str) -> List[BrowserCookie]:
    if profile.kind == "firefox":
        return _read_firefox(profile, domain)
    return _read_chromium(profile, domain)


def find_qwen_jar(browser: Optional[str] = None, profile: Optional[str] = None,
                  domain: str = QWEN_DOMAIN) -> Tuple[BrowserProfile, List[BrowserCookie]]:
    """Pick the best local profile for ``domain`` and read its cookies.

    - explicit ``profile``: that profile is used as-is (token not required).
    - auto: among every profile holding a ``token`` cookie for the domain,
      the most recently used (newest cookie-db mtime) wins.
    Raises :class:`BrowserCookieError` with actionable hints otherwise.
    """
    profiles = list_profiles(browser)
    if profile:
        profiles = [p for p in profiles if p.name == profile]
        if not profiles:
            names = [p.name for p in list_profiles(browser)]
            raise BrowserCookieError(
                f"profile {profile!r} not found; available: {names or 'none'}")
        if len(profiles) == 1:
            return profiles[0], _read_profile(profiles[0], domain)
    if not profiles:
        raise BrowserCookieError(
            "no supported browser profile found on this machine (looked for "
            "Firefox, Chrome, Chromium, Brave and Edge under ~/.config, "
            "~/.mozilla and their snap/flatpak equivalents); log in to "
            "chat.qwen.ai in a browser first, or pass session_token=...")
    winners: List[Tuple[BrowserProfile, List[BrowserCookie]]] = []
    no_token: List[str] = []
    for p in sorted(profiles, key=lambda p: p.cookie_db.stat().st_mtime, reverse=True):
        try:
            cookies = _read_profile(p, domain)
        except BrowserCookieError:
            no_token.append(f"{p.browser}:{p.name} (undecryptable)")
            continue
        if any(c.name == "token" for c in cookies):
            winners.append((p, cookies))
        else:
            no_token.append(f"{p.browser}:{p.name}")
    if not winners:
        raise BrowserCookieError(
            "found browser profile(s) [" + ", ".join(no_token) + "] but none "
            f"holds a '{domain}' session token; log in to chat.qwen.ai in "
            "your browser first")
    return winners[0]


def load_browser_cookies(browser: Optional[str] = None, profile: Optional[str] = None,
                         domain: str = QWEN_DOMAIN) -> List[BrowserCookie]:
    """Read cookies for ``domain`` from the best local browser profile."""
    return find_qwen_jar(browser, profile, domain=domain)[1]


def qwen_cookie_dict(browser: Optional[str] = None, profile: Optional[str] = None,
                     domain: str = QWEN_DOMAIN) -> Dict[str, str]:
    """Flat ``{name: value}`` cookie jar - the shape ``extra_cookies`` wants."""
    return {c.name: c.value for c in load_browser_cookies(browser, profile, domain)}


# ---------------------------------------------------------------------- CLI

def _mask(value: str) -> str:
    if len(value) > 8:
        return f"{value[:4]}…{value[-2:]}"
    return "•" * max(len(value), 1)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m qwen_studio.browser_cookies",
        description="List browser profiles holding qwen.ai session cookies "
                    "(values masked by default; nothing is sent anywhere).")
    ap.add_argument("--browser", help="firefox | chrome | chromium | brave | edge")
    ap.add_argument("--profile", help="exact profile name")
    ap.add_argument("--domain", default=QWEN_DOMAIN, help="domain filter")
    ap.add_argument("--all-domains", action="store_true", help="list every cookie")
    ap.add_argument("--show-values", action="store_true",
                    help="print full cookie values (secret!)")
    args = ap.parse_args(argv)
    domain = "*" if args.all_domains else args.domain
    try:
        prof, cookies = find_qwen_jar(args.browser, args.profile, domain=domain)
    except BrowserCookieError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"profile: {prof.browser} [{prof.name}]")
    print(f"db:      {prof.cookie_db}")
    print(f"cookies matching {domain!r}: {len(cookies)}")
    for c in cookies:
        val = c.value if args.show_values else _mask(c.value)
        print(f"  {c.name:<20} {val}")
    if not args.show_values:
        print("\n(values masked; re-run with --show-values to print secrets)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
