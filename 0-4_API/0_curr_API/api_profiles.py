"""DeepSeek Flash generation and provider-reported model metadata."""

from datetime import datetime, timezone
from importlib.util import module_from_spec, spec_from_file_location
import json
import sys
import os
from pathlib import Path
import threading
from urllib.parse import urlsplit


MODEL_ALIAS = "dso_DSv41Flash"
MODEL_ID = "deepseek-flash"
DEFAULT_BASE = "https://api.deepseek.com"
_LOG_LOCK = threading.Lock()


def get_api_config(profile="deepseek_official"):
    if profile != "deepseek_official":
        raise ValueError(f"Unsupported generation profile: {profile}")
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Set DEEPSEEK_API_KEY to run BiVerify generation.")
    endpoint = DEFAULT_BASE
    mock = os.environ.get("BIVERIFY_LOCAL_TEST_URL")
    if mock:
        address = urlsplit(mock)
        if (api_key != "local-placeholder" or address.scheme != "http" or
                address.hostname != "127.0.0.1" or not address.port or
                address.username is not None or address.password is not None or
                address.path != "/v1" or address.query or address.fragment):
            raise ValueError("The local test requires a loopback fixture and a dummy credential.")
        endpoint = mock
    return {
        "profile": profile,
        "api_key": api_key,
        "base_url": endpoint,
        "model_map": {MODEL_ALIAS: MODEL_ID},
    }


def api_model_name(alias, config):
    return config["model_map"][alias]


def extra_body_for_model(alias, config, api_model=None):
    if api_model_name(alias, config) != MODEL_ID or api_model not in (None, MODEL_ID):
        raise ValueError("The public generator uses the official deepseek-flash alias.")
    return {"thinking": {"type": "disabled"}}


def is_openai_compatible_model(alias):
    return alias == MODEL_ALIAS


def api_profile_tag():
    return "dso"


def model_file_tag(alias):
    if alias != MODEL_ALIAS:
        raise ValueError(f"Unsupported generation model: {alias}")
    return "DSv41Flash"


def record_response_metadata(response, *, stage, model=MODEL_ID):
    """Record what the provider returned without guessing a release date."""
    path = os.environ.get("BIVERIFY_RESPONSE_METADATA")
    if not path:
        return
    record = {
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "requested_model": model,
        "returned_model": getattr(response, "model", None),
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "response_id": getattr(response, "id", None),
        "thinking": "disabled",
        "endpoint": "local-test" if os.environ.get("BIVERIFY_LOCAL_TEST_URL") else DEFAULT_BASE,
    }
    with _LOG_LOCK, Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")

# The generation workers import this profile in their own process. The initial
# response generator and the evaluator run separately and retain their budgets.
def _install_generation_budget():
    name = "spq_qwen36_output_request_patch"
    if name not in sys.modules:
        spec = spec_from_file_location(name, Path(__file__).with_name("qwen36_output_request_patch.py"))
        adapter = module_from_spec(spec)
        spec.loader.exec_module(adapter)
        sys.modules[name] = adapter
    sys.modules[name].install_qwen36_output_request_patch({MODEL_ID: 8192})


_install_generation_budget()
