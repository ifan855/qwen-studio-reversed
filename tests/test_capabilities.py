"""Offline tests for the v0.3.0 capability additions (no network).

Pins the wire shapes proven in the capability study (docs/capabilities.md):
file entries, import records, OSS signing, message-nodes-with-files, and
the new error-code mappings.
"""
import base64
import hashlib
import hmac
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from qwen_studio import QwenStudio  # noqa: E402
from qwen_studio.chats import ChatService  # noqa: E402
from qwen_studio.chat import ChatCompletion, DEFAULT_FEATURE_CONFIG  # noqa: E402
from qwen_studio.files import FileRef, FileService  # noqa: E402
from qwen_studio.projects import Project  # noqa: E402


# ----------------------------------------------------------------- file refs
def test_file_ref_entry_shape():
    ref = FileRef(id="f-1", url="https://oss/x.png", name="x.png",
                  content_type="image/png")
    e = ref.entry()
    assert e == {"type": "image", "name": "x.png", "file_type": "image/png",
                 "showType": "image", "status": "uploaded",
                 "file_class": "vision", "id": "f-1", "url": "https://oss/x.png"}


def test_user_message_carries_files():
    msg = ChatCompletion.user_message(
        "look", "m1", feature_config=dict(DEFAULT_FEATURE_CONFIG),
        files=[{"type": "image", "id": "f-1"}])
    assert msg["files"] == [{"type": "image", "id": "f-1"}]
    assert msg["role"] == "user" and msg["chat_type"] == "t2t"


def test_oss_string_to_sign():
    s = FileService._oss_string_to_sign(
        "PUT", "image/png", "Mon, 01 Jan 2035 00:00:00 GMT",
        "STS-TOKEN", "bucket", "dir/key.png")
    # canonical: VERB \n md5 \n ctype \n date \n x-oss headers \n resource
    assert s == ("PUT\n\nimage/png\nMon, 01 Jan 2035 00:00:00 GMT\n"
                 "x-oss-security-token:STS-TOKEN\n/bucket/dir/key.png")
    sig = base64.b64encode(hmac.new(b"secret", s.encode(), hashlib.sha1).digest())
    assert isinstance(sig, bytes)


# ------------------------------------------------------------- import records
def test_build_import_record_shape():
    f1, f2 = "11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222"
    u = {"id": f1, "fid": f1, "parentId": None, "childrenIds": [f2],
         "role": "user", "content": "hi", "files": []}
    a = {"id": f2, "fid": f2, "parentId": f1, "childrenIds": [],
         "role": "assistant", "content": "hello", "files": []}
    rec = ChatService.build_import_record("c-1", "t", [u, a], "user-1",
                                          model="m1", created_at=123)
    assert rec["user_id"] == "user-1" and rec["archived"] is False
    assert rec["chat"]["history"]["currentId"] == f2
    assert rec["chat"]["history"]["messages"][f1]["content"] == "hi"
    assert rec["chat"]["messages"][1]["role"] == "assistant"
    assert set(rec) >= {"id", "user_id", "title", "chat", "created_at",
                        "updated_at", "archived", "pinned", "folder_id"}


def test_build_import_record_json_safe():
    f1 = "33333333-3333-3333-3333-333333333333"
    u = {"id": f1, "fid": f1, "parentId": None, "childrenIds": [],
         "role": "user", "content": "x", "files": [], "weird": {"s": 1}}
    rec = ChatService.build_import_record("c", "t", [u], "u", created_at=1)
    import json
    json.dumps(rec)  # must not raise


# ------------------------------------------------------------------ projects
def test_project_dataclass():
    p = Project(id="p1", name="n", custom_instruction="be terse")
    assert p.custom_instruction == "be terse"


# ------------------------------------------------------------ error mapping
def test_error_code_mappings():
    from qwen_studio.exceptions import BadRequestError, NotFoundError
    q = QwenStudio.from_access_token("t")
    for code, cls in (("PARENT_NOT_FOUND", BadRequestError),
                      ("RequestValidationError", BadRequestError),
                      ("CHAT_NOT_FOUND", NotFoundError),
                      ("Bad_Request", BadRequestError),
                      ("Not_Found", NotFoundError)):
        try:
            q._check_app_error({"success": False}, {"success": False}) if False else None
            r = type("R", (), {"status_code": 200})()
            q._check_app_error(r, {"success": False,
                                   "data": {"code": code, "details": "d"}})
            raise AssertionError(f"{code} did not raise")
        except cls:
            pass
