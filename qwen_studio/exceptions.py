"""Exception hierarchy for the qwen-studio library.

Every exception carries whatever context the server returned so callers can
log, retry or back off with full information. The hierarchy mirrors the two
rejection layers observed on the wire:

1. The application layer (JSON ``{"success": false, "data": {"code": ...}}``
   responses and SSE ``error`` events) -> :class:`APIError` subclasses.
2. The edge / anti-bot layer (Alibaba "SM" risk engine, RGV587 punish pages)
   -> :class:`PunishedError`.
"""

from __future__ import annotations

from typing import Any, Optional


class QwenStudioError(Exception):
    """Base class for every error raised by this library."""


class AuthError(QwenStudioError):
    """Sign-in or token refresh failed.

    Raised when ``/auths/signin`` returns a non-success payload, when the
    refresh exchange is rejected (expired/rotated session cookie), or when a
    request that requires a Bearer token is made before any token is known.
    """

    def __init__(self, message: str, *, status: Optional[int] = None,
                 payload: Optional[Any] = None) -> None:
        super().__init__(message)
        self.status = status
        self.payload = payload


class TokenExpiredError(AuthError):
    """The access token has expired and no refresh material is available.

    The library refreshes automatically whenever it can; this is raised only
    when automatic renewal is impossible (e.g. bearer-only construction with
    no session cookie and no credentials).
    """


class APIError(QwenStudioError):
    """Application-level error response (``success: false``).

    Attributes:
        code: server error code, e.g. ``Bad_Request``, ``Not_Found``,
            ``Internal_Server_Error``.
        details: human readable server message, when present.
        status: HTTP status code of the response.
        payload: the parsed JSON body, for callers that need everything.
    """

    def __init__(self, message: str, *, code: Optional[str] = None,
                 details: Optional[str] = None, status: Optional[int] = None,
                 payload: Optional[Any] = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details
        self.status = status
        self.payload = payload


class NotFoundError(APIError):
    """The requested chat / resource does not exist (e.g. after deletion)."""


class BadRequestError(APIError):
    """The server rejected the request shape (``Bad_Request``).

    Observed live when a ``role:"function"`` tool-result continuation is sent
    from an unrecognised client session - see docs/local-tools.md, section
    "Server-side gating of continuations".
    """


class QuotaError(APIError):
    """Credit / quota enforcement fired.

    The web client recognises the error codes ``quotaLimited``,
    ``ExceedLimit``, ``quota_exhausted`` and shows a paywall toast for them.
    """

    QUOTA_CODES = {"quotaLimited", "ExceedLimit", "quota_exhausted"}

    @classmethod
    def matches(cls, code: Optional[str]) -> bool:
        return code in cls.QUOTA_CODES


class RateLimitedError(APIError):
    """Rate / concurrency limit signalled by the server.

    Covers the client-side taxonomy ``RateLimited``, ``ParallelLimited`` and
    ``Too_Many_Requests``. Back off and retry later; do not hammer.
    """

    RATE_CODES = {"RateLimited", "ParallelLimited", "Too_Many_Requests"}

    @classmethod
    def matches(cls, code: Optional[str]) -> bool:
        return code in cls.RATE_CODES


class PunishedError(QwenStudioError):
    """The Alibaba edge risk engine intercepted the request.

    Identified by ``FAIL_SYS_USER_VALIDATE`` / ``RGV587_ERROR`` payloads and
    ``_____tmd_____/punish`` URLs. This is an anti-abuse control, not an
    application error. The correct reaction is to stop, wait a long time
    (minutes), and reduce request volume - repeated retries deepen the block.

    When ``auto_solve_captcha=True`` is set on the client, the punish URL
    is extracted from ``body`` and passed to
    :func:`qwen_studio.captcha.solve_punish` which launches a headless
    browser to drag the Baxia slider and obtain the ``x5sec`` cookie.
    """

    def __init__(self, message: str, *, body: str = "",
                 punish_url: Optional[str] = None) -> None:
        super().__init__(message)
        self.body = body
        # the punish URL extracted from body (or None if not a punish response)
        self.punish_url = punish_url


class ContinuationBlockedError(PunishedError):
    """A tool-result continuation was refused by a rejection layer.

    Raised by :meth:`qwen_studio.local_tools.ToolSession.send_tool_results`
    when the server answers the ``role:"function"`` follow-up with either the
    anti-bot punish page (web surface) or an application-level rejection
    (desktop surface). The tool call itself succeeded and the locally executed
    results are attached; see docs/local-tools.md for the full discussion.
    """

    def __init__(self, message: str, *, body: str = "",
                 results: Optional[dict] = None) -> None:
        super().__init__(message, body=body)
        self.results = results or {}


class StreamInterruptedError(QwenStudioError):
    """The SSE stream ended abnormally (timeout, connection reset, bad frame)."""


class ToolExecutionError(QwenStudioError):
    """A locally registered tool callable raised an exception."""
