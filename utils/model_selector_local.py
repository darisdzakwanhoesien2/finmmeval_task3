from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st

from utils.hf_inference import validate_hf_token as validate_hf_token_detailed

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

OPENROUTER_FALLBACK_MODELS = [
    {"id": "openai/gpt-4.1-mini", "label": "GPT-4.1 Mini", "free": False, "vision": True, "notes": "stable"},
    {"id": "google/gemini-2.0-flash-exp:free", "label": "Gemini 2.0 Flash", "free": True, "vision": True, "notes": "free"},
    {"id": "meta-llama/llama-3.3-70b-instruct:free", "label": "Llama 3.3 70B Instruct", "free": True, "vision": False, "notes": "free"},
    {"id": "qwen/qwen-2.5-72b-instruct:free", "label": "Qwen 2.5 72B Instruct", "free": True, "vision": False, "notes": "free"},
]

HF_FALLBACK_MODELS = [
    {"id": "meta-llama/Llama-3.1-8B-Instruct", "label": "Llama 3.1 8B Instruct", "free": True, "vision": False, "notes": "default"},
    {"id": "Qwen/Qwen2.5-72B-Instruct", "label": "Qwen 2.5 72B Instruct", "free": True, "vision": False, "notes": "large"},
    {"id": "mistralai/Mistral-Nemo-Instruct-2407", "label": "Mistral Nemo Instruct", "free": True, "vision": False, "notes": "fast"},
]

LOCAL_LLM_CHAT_URL = os.getenv(
    "LOCAL_LLM_CHAT_URL",
    "http://43.156.67.61:1234/v1/chat/completions",
)
LM_STUDIO_BASE_URL = os.getenv(
    "LOCAL_LLM_BASE_URL",
    os.getenv("LM_STUDIO_BASE_URL", LOCAL_LLM_CHAT_URL.rsplit("/chat/completions", 1)[0]),
)
LM_STUDIO_MODELS_DIR = Path(
    os.getenv(
        "LOCAL_LLM_MODELS_DIR",
        os.getenv("LM_STUDIO_MODELS_DIR", str(Path.home() / ".lmstudio" / "models")),
    )
)


def _scan_lmstudio_models_dir() -> list[dict[str, Any]]:
    if not LM_STUDIO_MODELS_DIR.exists():
        return []
    models: list[dict[str, Any]] = []
    for model_dir in sorted(LM_STUDIO_MODELS_DIR.glob("*/*")):
        if not model_dir.is_dir():
            continue
        label = model_dir.name.replace("-GGUF", "").replace("-gguf", "")
        lower_name = model_dir.name.lower()
        models.append(
            {
                "id": model_dir.name,
                "label": label,
                "free": True,
                "vision": any(token in lower_name for token in ("vision", "vl")),
                "notes": "local disk",
            }
        )
    return models


LM_STUDIO_FALLBACK_MODELS = _scan_lmstudio_models_dir() or [
    {"id": "local-model", "label": "Local LM Studio Model", "free": True, "vision": False, "notes": "local"},
]


def _openrouter_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {(api_key or '').strip()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/finmmeval",
        "X-Title": "FinMMEval",
    }


def validate_openrouter_key(api_key: str) -> tuple[bool, str]:
    api_key = (api_key or "").strip()
    if not api_key:
        return False, "Provide an OpenRouter API key."
    try:
        response = requests.get(
            f"{OPENROUTER_BASE_URL}/models",
            headers=_openrouter_headers(api_key),
            timeout=15,
        )
        if response.status_code == 200:
            return True, "OpenRouter key verified."
        return False, f"OpenRouter validation failed ({response.status_code})."
    except Exception as exc:
        return False, f"OpenRouter validation failed: {exc}"


def validate_hf_token(api_key: str) -> tuple[bool, str]:
    api_key = (api_key or "").strip()
    if not api_key:
        return False, "Provide a Hugging Face token."
    return validate_hf_token_detailed(api_key)


def validate_lmstudio_endpoint(_: str = "") -> tuple[bool, str]:
    try:
        response = requests.post(
            LOCAL_LLM_CHAT_URL,
            json={
                "messages": [{"role": "user", "content": "ping"}],
                "temperature": 0.0,
            },
            timeout=15,
        )
        if response.status_code == 200:
            return True, "Local LLM chat endpoint is reachable."
        return False, f"Local LLM responded with status {response.status_code}."
    except Exception as exc:
        return False, f"Local LLM is not reachable at {LOCAL_LLM_CHAT_URL}: {exc}"


def fetch_openrouter_models(api_key: str | None) -> list[dict[str, Any]]:
    api_key = (api_key or "").strip()
    if not api_key:
        return OPENROUTER_FALLBACK_MODELS
    try:
        response = requests.get(
            f"{OPENROUTER_BASE_URL}/models",
            headers=_openrouter_headers(api_key),
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json().get("data", [])
        models: list[dict[str, Any]] = []
        for item in payload:
            model_id = str(item.get("id", "")).strip()
            if not model_id:
                continue
            lower_id = model_id.lower()
            models.append(
                {
                    "id": model_id,
                    "label": model_id.split("/")[-1],
                    "free": ":free" in lower_id or lower_id.endswith("free"),
                    "vision": any(token in lower_id for token in ("vision", "vl", "gpt-4o", "gemini")),
                    "notes": "live",
                }
            )
        return models or OPENROUTER_FALLBACK_MODELS
    except Exception:
        return OPENROUTER_FALLBACK_MODELS


def fetch_huggingface_models(_: str | None) -> list[dict[str, Any]]:
    return HF_FALLBACK_MODELS


def fetch_lmstudio_models(_: str | None) -> list[dict[str, Any]]:
    try:
        response = requests.get(f"{LM_STUDIO_BASE_URL}/models", timeout=10)
        response.raise_for_status()
        payload = response.json().get("data", [])
        models: list[dict[str, Any]] = []
        for item in payload:
            model_id = str(item.get("id", "")).strip()
            if not model_id:
                continue
            lower_id = model_id.lower()
            models.append(
                {
                    "id": model_id,
                    "label": model_id,
                    "free": True,
                    "vision": any(token in lower_id for token in ("vision", "vl")),
                    "notes": "lmstudio api",
                }
            )
        return models or LM_STUDIO_FALLBACK_MODELS
    except Exception:
        return LM_STUDIO_FALLBACK_MODELS


def render_model_selector(
    key_prefix: str = "ms",
    show_vision_filter: bool = True,
) -> tuple[str, str, str]:
    st.subheader("API Keys")

    key_col1, key_col2 = st.columns(2)
    with key_col1:
        or_key = st.text_input(
            "OpenRouter API key",
            type="password",
            value=st.session_state.get(f"{key_prefix}_or_key", os.getenv("OPENROUTER_API_KEY", "")),
            key=f"{key_prefix}_or_key_input",
        )
        if or_key:
            st.session_state[f"{key_prefix}_or_key"] = or_key
    with key_col2:
        hf_key = st.text_input(
            "Hugging Face token",
            type="password",
            value=st.session_state.get(f"{key_prefix}_hf_key", os.getenv("HF_TOKEN", "")),
            key=f"{key_prefix}_hf_key_input",
        )
        if hf_key:
            st.session_state[f"{key_prefix}_hf_key"] = hf_key

    st.caption(f"Local model endpoint: `{LOCAL_LLM_CHAT_URL}`")
    st.caption(f"Local model directory: `{LM_STUDIO_MODELS_DIR}`")

    val_col1, val_col2, val_col3 = st.columns(3)
    with val_col1:
        if st.button("Validate OpenRouter key", key=f"{key_prefix}_val_or", use_container_width=True):
            ok, msg = validate_openrouter_key(or_key)
            st.session_state[f"{key_prefix}_or_valid"] = ok
            (st.success if ok else st.error)(msg)
    with val_col2:
        if st.button("Validate HF token", key=f"{key_prefix}_val_hf", use_container_width=True):
            ok, msg = validate_hf_token(hf_key)
            st.session_state[f"{key_prefix}_hf_valid"] = ok
            (st.success if ok else st.error)(msg)
    with val_col3:
        if st.button("Validate Local Server", key=f"{key_prefix}_val_lmstudio", use_container_width=True):
            ok, msg = validate_lmstudio_endpoint()
            st.session_state[f"{key_prefix}_lmstudio_valid"] = ok
            (st.success if ok else st.error)(msg)

    st.divider()
    provider = st.radio(
        "Provider",
        options=["openrouter", "huggingface", "lmstudio"],
        horizontal=True,
        key=f"{key_prefix}_provider",
    )
    active_key = or_key if provider == "openrouter" else hf_key if provider == "huggingface" else "lm-studio"

    if st.button("Fetch models", key=f"{key_prefix}_fetch", use_container_width=True):
        with st.spinner("Fetching models..."):
            if provider == "openrouter":
                fetched = fetch_openrouter_models(or_key)
            elif provider == "huggingface":
                fetched = fetch_huggingface_models(hf_key)
            else:
                fetched = fetch_lmstudio_models(None)
        st.session_state[f"{key_prefix}_model_list_{provider}"] = fetched

    model_list = st.session_state.get(
        f"{key_prefix}_model_list_{provider}",
        OPENROUTER_FALLBACK_MODELS if provider == "openrouter" else HF_FALLBACK_MODELS if provider == "huggingface" else LM_STUDIO_FALLBACK_MODELS,
    )

    filter_cols = st.columns(3)
    with filter_cols[0]:
        tier_filter = st.radio("Tier", ["All", "Free only", "Paid only"], horizontal=True, key=f"{key_prefix}_tier")
    with filter_cols[1]:
        vision_filter = "All"
        if show_vision_filter:
            vision_filter = st.radio("Capability", ["All", "Vision only", "Text only"], horizontal=True, key=f"{key_prefix}_vision")
    with filter_cols[2]:
        search = st.text_input("Search model", key=f"{key_prefix}_search")

    visible = model_list
    if tier_filter == "Free only":
        visible = [item for item in visible if item["free"]]
    elif tier_filter == "Paid only":
        visible = [item for item in visible if not item["free"]]
    if vision_filter == "Vision only":
        visible = [item for item in visible if item.get("vision")]
    elif vision_filter == "Text only":
        visible = [item for item in visible if not item.get("vision")]
    if search.strip():
        query = search.lower()
        visible = [item for item in visible if query in item["id"].lower() or query in item["label"].lower()]
    if not visible:
        visible = model_list[:1]

    labels = [f"{item['label']} [{item['notes']}]" for item in visible]
    selected_label = st.selectbox("Select model", labels, key=f"{key_prefix}_model_select")
    selected_model = visible[labels.index(selected_label)]
    st.caption(f"`{selected_model['id']}`")
    return selected_model["id"], provider, active_key
