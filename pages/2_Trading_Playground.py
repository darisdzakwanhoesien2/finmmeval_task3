from __future__ import annotations

import json
import os

import streamlit as st

from utils.data import build_asset_catalog, load_asset_frame, load_normalized_asset_record
from utils.hf_inference import DEFAULT_MODEL, stream_chat_completion, validate_hf_token


st.set_page_config(page_title="Trading Playground", page_icon=":robot_face:")
st.title("Trading Playground")
st.caption("Generate BUY / HOLD / SELL decisions with a hosted Hugging Face model in streaming mode")

st.markdown(
    """
Add your Hugging Face token in `HF_TOKEN` or Streamlit secrets as `HF_TOKEN`.
Inference is remote-only through Hugging Face, so model weights are not stored locally.
"""
)

# Safe secrets access
def _safe_secrets_get(key: str, default=None):
    try:
        return st.secrets.get(key, default)
    except FileNotFoundError:
        return default

api_key = _safe_secrets_get("HF_TOKEN", None) or st.session_state.get("hf_token") or os.getenv("HF_TOKEN")

with st.sidebar:
    st.subheader("Inference Settings")
    manual_key = st.text_input(
        "Hugging Face token",
        type="password",
        help="Optional if `HF_TOKEN` is already configured.",
    )
    if manual_key:
        st.session_state["hf_token"] = manual_key
        api_key = manual_key

    # ── Token validation ──────────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])
    with col1:
        validate_btn = st.button("Validate Token", use_container_width=True)
    with col2:
        token_ok = st.session_state.get("token_valid", False)
        st.markdown(f"{'🟢' if token_ok else '🔴'} {'Valid' if token_ok else 'Not verified'}")

    if validate_btn:
        if not api_key:
            st.error("Enter a token first.")
            st.session_state["token_valid"] = False
        else:
            with st.spinner("Validating token…"):
                is_valid, msg = validate_hf_token(api_key)
            st.session_state["token_valid"] = is_valid
            if is_valid:
                st.success(msg)
            else:
                st.error(msg)

    st.divider()

    model_name = st.text_input("Model", value=DEFAULT_MODEL)
    provider = st.selectbox(
        "Provider",
        options=["auto", "hf-inference", "nebius", "together"],
        help="'auto' lets the library pick. Switch if your installed huggingface_hub version doesn't support a specific provider.",
    )
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.2, step=0.1)
    top_p = st.slider("Top-p", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
    max_tokens = st.slider("Max new tokens", min_value=128, max_value=2048, value=700, step=64)

catalog = build_asset_catalog()
selected_asset = st.selectbox("Reference asset", catalog["asset"].tolist())
frame = load_asset_frame(selected_asset)
row_index = st.number_input("Reference row", min_value=0, max_value=max(len(frame) - 1, 0), value=0, step=1)
record = load_normalized_asset_record(selected_asset, int(row_index))

system_prompt = st.text_area(
    "System prompt",
    value=(
        "You are a financial trading research assistant working on FinMMEval Task 3. "
        "Use only the provided bundle. Decide between BUY, HOLD, and SELL. "
        "Respond with concise reasoning followed by a JSON object containing "
        "`recommended_action`, `confidence`, `key_drivers`, and `risk_flags`."
    ),
    height=140,
)

user_prompt = st.text_area(
    "User prompt",
    value=(
        "Evaluate this daily trading bundle and recommend exactly one action.\n\n"
        f"{record['prompt_context']}\n\n"
        "Return a short explanation first, then valid JSON."
    ),
    height=320,
)

meta_col_1, meta_col_2, meta_col_3 = st.columns(3)
meta_col_1.metric("Date", record["date"])
meta_col_2.metric("Asset", record["asset"])
meta_col_3.metric("Price", f"{record['prices']:.2f}")

preview_tab_1, preview_tab_2 = st.tabs(["Request payload", "Raw row"])
with preview_tab_1:
    st.code(json.dumps(record["request_payload"], ensure_ascii=False, indent=2, default=str), language="json")
with preview_tab_2:
    st.code(json.dumps(record["raw"], ensure_ascii=False, indent=2, default=str), language="json")

run = st.button("Stream response", type="primary", use_container_width=True)

if run:
    if not api_key:
        st.error("Provide a Hugging Face token first via the sidebar, `HF_TOKEN`, or Streamlit secrets.")
    elif not st.session_state.get("token_valid", False):
        with st.spinner("Validating token before streaming…"):
            is_valid, msg = validate_hf_token(api_key)
        st.session_state["token_valid"] = is_valid
        if not is_valid:
            st.error(f"Token validation failed — {msg}")
        else:
            st.info(msg)

    if api_key and st.session_state.get("token_valid", False):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        st.subheader("Streaming Output")
        try:
            streamed_text = st.write_stream(
                stream_chat_completion(
                    api_key=api_key,
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    provider=provider,
                )
            )
            st.session_state["last_model_response"] = streamed_text
        except Exception as exc:
            st.exception(exc)

if st.session_state.get("last_model_response"):
    st.subheader("Last Response")
    st.write(st.session_state["last_model_response"])
