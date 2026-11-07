# -*- coding: utf-8 -*-
"""Apply the SPQ12 default output limit at the shared API boundary.

The project has several OpenAI-compatible call paths.  Keeping this adapter
next to ``api_profiles.py`` lets those paths retain their original algorithm
code.  An existing explicit output limit always wins, including the dynamic
limit used by S5-7 context compression.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar


_PATCH_FLAG = "_spq_qwen36_output_request_patch"
_OUTPUT_TOKEN_KEYS = (
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
)
_MODEL_MAX_TOKENS: dict[str, int] = {}
DEFAULT_OUTPUT_TOKENS = 8_192
_BYPASS_DEFAULT = ContextVar("spq_bypass_default_output_limit", default=False)


@contextmanager
def without_default_output_limit():
    """Public initial answers retain provider defaults; isolate each request."""
    token = _BYPASS_DEFAULT.set(True)
    try:
        yield
    finally:
        _BYPASS_DEFAULT.reset(token)


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _has_output_limit(payload: Mapping | None) -> bool:
    if not isinstance(payload, Mapping):
        return False
    if any(key in payload for key in _OUTPUT_TOKEN_KEYS):
        return True
    extra_body = payload.get("extra_body")
    return isinstance(extra_body, Mapping) and any(
        key in extra_body for key in _OUTPUT_TOKEN_KEYS
    )


def apply_qwen36_output_request_options(payload: dict, *, responses: bool = False) -> dict:
    """Add the configured default only when the request has no output limit."""
    if not isinstance(payload, dict):
        return payload
    if _BYPASS_DEFAULT.get() or _has_output_limit(payload):
        return payload
    model = str(payload.get("model") or "")
    max_tokens = _MODEL_MAX_TOKENS.get(model, DEFAULT_OUTPUT_TOKENS)
    key = "max_tokens"
    if responses:
        key = "max_output_tokens"
    elif model.lower().startswith(("gpt-5", "o1", "o3", "o4")):
        key = "max_completion_tokens"
    payload[key] = int(max_tokens)
    return payload


def _is_chat_completions_url(url) -> bool:
    if isinstance(url, urllib.request.Request):
        value = getattr(url, "full_url", "")
    else:
        value = str(url or "")
    return any(part in value.lower() for part in ("/chat/completions", "/responses"))


def _decode_json_data(data):
    if isinstance(data, bytes):
        raw = data
    elif isinstance(data, bytearray):
        raw = bytes(data)
    else:
        return None
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _encode_json_data(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _patch_request_body(request: urllib.request.Request) -> bool:
    if not _is_chat_completions_url(request):
        return False
    payload = _decode_json_data(getattr(request, "data", None))
    if payload is None:
        return False
    before = dict(payload)
    apply_qwen36_output_request_options(payload, responses="/responses" in request.full_url.lower())
    if payload == before:
        return False
    body = _encode_json_data(payload)
    request.data = body
    request.add_header("Content-Length", str(len(body)))
    return True


def _patch_raw_body(url, data):
    if not _is_chat_completions_url(url):
        return False, data
    payload = _decode_json_data(data)
    if payload is None:
        return False, data
    before = dict(payload)
    apply_qwen36_output_request_options(payload, responses="/responses" in str(url).lower())
    if payload == before:
        return False, data
    return True, _encode_json_data(payload)


def _install_urllib_patch() -> None:
    original = urllib.request.urlopen
    if getattr(original, _PATCH_FLAG, False):
        return

    def patched_urlopen(url, data=None, *args, **kwargs):
        if isinstance(url, urllib.request.Request):
            _patch_request_body(url)
        elif data is not None:
            _, data = _patch_raw_body(url, data)
        return original(url, data, *args, **kwargs)

    setattr(patched_urlopen, _PATCH_FLAG, True)
    setattr(patched_urlopen, "_spq_original", original)
    urllib.request.urlopen = patched_urlopen


def _install_openai_sdk_patch() -> None:
    try:
        from openai.resources.chat.completions.completions import (
            AsyncCompletions,
            Completions,
        )
    except Exception:
        return

    original_sync = getattr(Completions, "create", None)
    if callable(original_sync) and not getattr(original_sync, _PATCH_FLAG, False):

        def patched_sync_create(self, *args, **kwargs):
            apply_qwen36_output_request_options(kwargs)
            return original_sync(self, *args, **kwargs)

        setattr(patched_sync_create, _PATCH_FLAG, True)
        setattr(patched_sync_create, "_spq_original", original_sync)
        Completions.create = patched_sync_create

    original_async = getattr(AsyncCompletions, "create", None)
    if callable(original_async) and not getattr(original_async, _PATCH_FLAG, False):

        async def patched_async_create(self, *args, **kwargs):
            apply_qwen36_output_request_options(kwargs)
            return await original_async(self, *args, **kwargs)

        setattr(patched_async_create, _PATCH_FLAG, True)
        setattr(patched_async_create, "_spq_original", original_async)
        AsyncCompletions.create = patched_async_create


def _install_openai_legacy_patch() -> None:
    try:
        import openai
    except Exception:
        return

    target = getattr(openai, "ChatCompletion", None)
    original = getattr(target, "create", None) if target is not None else None
    if not callable(original) or getattr(original, _PATCH_FLAG, False):
        return

    def patched_chat_completion_create(*args, **kwargs):
        apply_qwen36_output_request_options(kwargs)
        return original(*args, **kwargs)

    setattr(patched_chat_completion_create, _PATCH_FLAG, True)
    setattr(patched_chat_completion_create, "_spq_original", original)
    try:
        target.create = patched_chat_completion_create
    except Exception:
        return


def _install_responses_patch() -> None:
    try:
        from openai.resources.responses.responses import AsyncResponses, Responses
    except ImportError:
        return
    original_sync = Responses.create
    if not getattr(original_sync, _PATCH_FLAG, False):
        def patched_sync(self, *args, **kwargs):
            apply_qwen36_output_request_options(kwargs, responses=True)
            return original_sync(self, *args, **kwargs)
        setattr(patched_sync, _PATCH_FLAG, True)
        Responses.create = patched_sync
    original_async = AsyncResponses.create
    if not getattr(original_async, _PATCH_FLAG, False):
        async def patched_async(self, *args, **kwargs):
            apply_qwen36_output_request_options(kwargs, responses=True)
            return await original_async(self, *args, **kwargs)
        setattr(patched_async, _PATCH_FLAG, True)
        AsyncResponses.create = patched_async


def install_qwen36_output_request_patch(model_max_tokens: Mapping[str, int]) -> None:
    if _env_flag("SPQ_DISABLE_QWEN36_OUTPUT_REQUEST_PATCH"):
        return
    for model, max_tokens in model_max_tokens.items():
        parsed = int(max_tokens)
        if parsed <= 0:
            raise ValueError(f"Output token limit must be positive: {model}={max_tokens}")
        _MODEL_MAX_TOKENS[str(model)] = parsed
    _install_openai_sdk_patch()
    _install_openai_legacy_patch()
    _install_responses_patch()
    _install_urllib_patch()
