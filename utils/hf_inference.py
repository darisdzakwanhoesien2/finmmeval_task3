from __future__ import annotations

import inspect
import os
from collections.abc import Generator
from typing import Any, Optional

import requests


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LOCAL_LLM_CHAT_URL = os.getenv(
    "LOCAL_LLM_CHAT_URL",
    "http://43.156.67.61:1234/v1/chat/completions",
)
LM_STUDIO_BASE_URL = os.getenv(
    "LOCAL_LLM_BASE_URL",
    os.getenv("LM_STUDIO_BASE_URL", LOCAL_LLM_CHAT_URL.rsplit("/chat/completions", 1)[0]),
)


def make_client(api_key: str | None):
    from huggingface_hub import InferenceClient

    return InferenceClient(api_key=api_key)


def validate_hf_token(api_key: str) -> tuple[bool, str]:
    """Validate a Hugging Face token via whoami. Returns (is_valid, message)."""
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=api_key)
        user_info = api.whoami()
        username = user_info.get("name", "Unknown")
        return True, f"✅ Authenticated as **{username}**"
    except Exception as exc:
        return False, f"❌ Authentication failed: {exc}"


def stream_chat_completion(
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.9,
    provider: Optional[str] = None,
) -> Generator[str, None, None]:
    """Stream chat completion tokens from a Hugging Face hosted model."""
    from huggingface_hub import InferenceClient

    client = InferenceClient(model=model, token=api_key)

    supported_params = inspect.signature(client.chat_completion).parameters
    kwargs: dict[str, Any] = dict(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
    )
    if "provider" in supported_params and provider and provider != "auto":
        kwargs["provider"] = provider

    stream = client.chat_completion(**kwargs)
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def collect_chat_completion(
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_tokens: int = 700,
    provider: str = "huggingface",
    hf_provider_hint: str = "auto",
    pre_call_delay: float | None = None,
) -> str:
    """
    Route to the correct backend based on *provider*.
    OpenRouter model IDs (contain ':') must never reach the HuggingFace client.
    """
    _is_openrouter = (
        provider == "openrouter"
        or ":" in model
    )
    _is_lmstudio = provider == "lmstudio"

    if _is_lmstudio:
        response = requests.post(
            LOCAL_LLM_CHAT_URL,
            headers={"Content-Type": "application/json"},
            json={
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            },
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    if _is_openrouter:
        import sys
        import inspect as _inspect
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from shared.model_provider import call_chat_completion as _or_call

        kwargs: dict[str, Any] = dict(
            messages=messages,
            model_id=model,
            provider="openrouter",
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        if "pre_call_delay" in _inspect.signature(_or_call).parameters and pre_call_delay is not None:
            kwargs["pre_call_delay"] = pre_call_delay
        return _or_call(**kwargs)

    # --- HuggingFace path ---
    return "".join(
        stream_chat_completion(
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            provider=hf_provider_hint,
        )
    )


def evaluate_record(
    api_key: str,
    model: str,
    provider: str,
    hf_provider_hint: str = "auto",
    record: dict | None = None,
    rerun_id: int = 0,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_tokens: int = 700,
) -> dict:
    """Evaluate a single normalised trading record and return a result dict."""
    system_prompt = default_system_prompt()
    user_prompt = build_user_prompt(record)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    started = time.perf_counter()

    # Build kwargs conditionally — only pass hf_provider_hint if the
    # underlying collect_chat_completion actually accepts it.
    import inspect as _inspect

    _cc_sig = _inspect.signature(collect_chat_completion)
    _extra: dict = {}
    if "hf_provider_hint" in _cc_sig.parameters:
        _extra["hf_provider_hint"] = hf_provider_hint
    if "provider" in _cc_sig.parameters:
        _extra["provider"] = provider

    response = collect_chat_completion(
        api_key=api_key,
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        **_extra,
    )
    latency_seconds = time.perf_counter() - started

    # ...existing code...
