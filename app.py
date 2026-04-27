from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data import (
    CORE_PAYLOAD_FIELDS,
    ENRICHMENT_PAYLOAD_FIELDS,
    build_asset_catalog,
    build_normalized_dataset,
    build_price_history,
    load_normalized_asset_record,
    load_asset_frame,
    load_notes_sections,
    read_notes,
)
from utils.evaluation import (
    build_dataset_analysis_frame,
    evaluate_record,
    generate_results_discussion,
    list_saved_runs,
    load_run_summary,
    load_saved_results,
    save_run_report,
    save_experiment_results,
    summarize_performance_metrics,
)
from utils.hf_inference import collect_chat_completion
from utils.model_selector_local import (
    LM_STUDIO_BASE_URL,
    LOCAL_LLM_CHAT_URL,
    render_model_selector,
    validate_hf_token,
    validate_lmstudio_endpoint,
    validate_openrouter_key,
)


st.set_page_config(
    page_title="FinMMEval Task 3 Lab",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

st.title("FinMMEval Task 3 Lab")
st.caption("Trading dataset explorer and hosted Hugging Face decision playground")

catalog = build_asset_catalog()
price_history = build_price_history()
note_sections = load_notes_sections()

total_rows = int(catalog["rows"].sum()) if not catalog.empty else 0
asset_count = int(catalog["asset"].nunique()) if not catalog.empty else 0
start_date = price_history["date"].min().date() if not price_history.empty else "N/A"
end_date = price_history["date"].max().date() if not price_history.empty else "N/A"

metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
metric_col_1.metric("Assets", asset_count)
metric_col_2.metric("Rows", f"{total_rows:,}")
metric_col_3.metric("Date Range Start", str(start_date))
metric_col_4.metric("Date Range End", str(end_date))

st.markdown(
    """
This app is built directly from the Task 3 trading files in `data/clef-task3-trading/` and the workflow notes in `notes.md`.
Use the sidebar to inspect the dataset, review the benchmark workflow, and test a hosted Hugging Face model in streaming mode.
No model weights are downloaded into this workspace.
Batch evaluations can be saved under `results/` and reviewed in the Evaluation Results page.
"""
)

left, right = st.columns([1.15, 1])

with left:
    st.subheader("Dataset Inventory")
    inventory = catalog.rename(
        columns={
            "asset": "Asset",
            "rows": "Rows",
            "columns": "Columns",
            "start_date": "Start Date",
            "end_date": "End Date",
            "path": "Path",
            "formats": "Formats",
        }
    )
    st.dataframe(inventory, use_container_width=True, hide_index=True)

with right:
    st.subheader("Research Notes Snapshot")
    if note_sections:
        first_section = note_sections[0]
        st.markdown(f"### {first_section['title']}")
        st.markdown(first_section["body"])
    else:
        st.info("`notes.md` was not found.")

st.subheader("Asset Price History")
if price_history.empty:
    st.warning("No pricing history was found under `data/clef-task3-trading/`.")
else:
    chart_frame = (
        price_history.pivot(index="date", columns="asset", values="prices")
        .sort_index()
        .rename_axis(index="date", columns="asset")
    )
    st.line_chart(chart_frame)

st.subheader("Task 3 Workflow at a Glance")
workflow_points = [
    "Data & submission combines historical training/backtesting data, Agent Market Arena endpoint registration, and the FastAPI reference shape.",
    "Scheduling starts at 00:00 UTC and progressively dispatches one request per registered endpoint with a 3-minute timeout.",
    "Request payloads include core fields plus enrichment fields; `10k` and `10q` are `null` for non-equity assets such as BTC.",
    "Valid JSON actions execute directly; invalid responses, failures, and timeouts default to `HOLD` for execution.",
    "Daily actions fully replace the prior close-price position: BUY = long, HOLD = flat, SELL = short.",
    "Evaluation prioritizes Cumulative Return, with Sharpe Ratio, Max Drawdown, Daily Volatility, and Annualized Volatility as secondary metrics.",
]
for point in workflow_points:
    st.write(f"- {point}")

if not price_history.empty:
    st.subheader("Recent Daily Bundles")
    recent_rows = (
        price_history.sort_values(["date", "asset"], ascending=[False, True])
        .head(10)
        .assign(date=lambda frame: frame["date"].dt.date.astype(str))
    )
    st.dataframe(recent_rows.rename(columns={"date": "Date", "asset": "Asset", "prices": "Price"}), hide_index=True)

st.divider()
st.subheader("Research Workflow")
st.caption("Run the full process from this page: inspect the dataset, test prompts, launch saved evaluations, and review stored results.")


def _safe_secrets_get(key: str, default=None):
    try:
        return st.secrets.get(key, default)
    except FileNotFoundError:
        return default


def _run_evaluation_script(
    *,
    experiment_name: str,
    model_id: str,
    provider: str,
    hf_provider_hint: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    selected_assets: list[str],
    limit: int,
    reruns: int,
    api_key: str,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "scripts" / "run_evaluation.py"),
        "--experiment-name",
        experiment_name,
        "--model",
        model_id,
        "--provider",
        provider,
        "--hf-provider-hint",
        hf_provider_hint,
        "--temperature",
        str(float(temperature)),
        "--top-p",
        str(float(top_p)),
        "--max-tokens",
        str(int(max_tokens)),
        "--reruns",
        str(int(reruns)),
    ]
    if int(limit) > 0:
        command.extend(["--limit", str(int(limit))])
    for asset in selected_assets:
        command.extend(["--asset", asset])

    env = os.environ.copy()
    if provider == "openrouter":
        env["OPENROUTER_API_KEY"] = api_key
    elif provider == "huggingface":
        env["HF_TOKEN"] = api_key
    elif provider == "lmstudio":
        env["LM_STUDIO_BASE_URL"] = LM_STUDIO_BASE_URL
        env["LOCAL_LLM_BASE_URL"] = LM_STUDIO_BASE_URL
        env["LOCAL_LLM_CHAT_URL"] = LOCAL_LLM_CHAT_URL

    return subprocess.run(
        command,
        cwd=Path(__file__).resolve().parent,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


api_key = _safe_secrets_get("HF_TOKEN", None) or st.session_state.get("hf_token") or os.getenv("HF_TOKEN")
normalized_dataset = build_normalized_dataset()
asset_options = sorted(normalized_dataset["asset"].unique().tolist()) if not normalized_dataset.empty else []

manual_key = st.text_input(
    "Hugging Face token",
    type="password",
    value="",
    help="Optional if `HF_TOKEN` is already configured in your environment or Streamlit secrets.",
)
if manual_key:
    st.session_state["hf_token"] = manual_key
    api_key = manual_key

# ── Replace the top-level API key block ───────────────────────────────────────
with st.expander("⚙️ Model & API Key Settings", expanded=True):
    model_id, provider, api_key = render_model_selector(
        key_prefix="task3",
        show_vision_filter=False,
    )
    hf_provider_hint = st.selectbox(
        "HF provider hint (only used when provider = huggingface)",
        options=["auto", "hf-inference", "nebius", "together"],
        key="task3_hf_hint",
    )

# Keep existing token_col validate block but extend it:
token_col_1, token_col_2 = st.columns([1, 3])
with token_col_1:
    validate_clicked = st.button("Validate Key / Token", use_container_width=True)
with token_col_2:
    token_status = st.session_state.get("home_token_valid")
    if token_status is True:
        st.success("Key/Token verified for hosted inference.")
    elif token_status is False:
        st.error("Key/Token not verified yet.")
    else:
        st.info("Validation has not been run in this session.")

if validate_clicked:
    if not api_key:
        st.session_state["home_token_valid"] = False
        st.error("Provide an API key or token first.")
    else:
        with st.spinner("Validating..."):
            if provider == "openrouter":
                is_valid, message = validate_openrouter_key(api_key)
            elif provider == "huggingface":
                is_valid, message = validate_hf_token(api_key)
            else:
                is_valid, message = validate_lmstudio_endpoint()
        st.session_state["home_token_valid"] = is_valid
        (st.success if is_valid else st.error)(message)

tab_overview, tab_dataset, tab_playground, tab_batch, tab_results, tab_notes = st.tabs(
    ["Overview", "Dataset Explorer", "Model Playground", "Batch Evaluation", "Saved Results", "Notes"]
)

with tab_overview:
    st.markdown(
        """
This single-page entrypoint now covers the complete Task 3 workflow:
- inspect bundle structure and temporal coverage
- probe individual rows and prompts
- test live hosted-model responses
- run full stored evaluations across the dataset
- review saved experiment outputs for paper-style analysis
"""
    )
    st.markdown("**Arena Payload Contract**")
    payload_contract = pd.DataFrame(
        [
            {"Layer": "Core", "Field": field, "Color in diagram": "Blue"}
            for field in CORE_PAYLOAD_FIELDS
        ]
        + [
            {"Layer": "Enrichment", "Field": field, "Color in diagram": "Teal"}
            for field in ENRICHMENT_PAYLOAD_FIELDS
        ]
    )
    st.dataframe(payload_contract, hide_index=True, use_container_width=True)
    if not normalized_dataset.empty:
        dataset_frame = build_dataset_analysis_frame()
        summary = (
            dataset_frame.groupby("asset", as_index=False)
            .agg(
                rows=("dataset_row_id", "count"),
                start_date=("date", "min"),
                end_date=("date", "max"),
                mean_news_items=("news_count", "mean"),
                mean_news_chars=("news_length", "mean"),
            )
        )
        summary["start_date"] = summary["start_date"].dt.date.astype(str)
        summary["end_date"] = summary["end_date"].dt.date.astype(str)
        st.dataframe(summary, hide_index=True, use_container_width=True)

with tab_dataset:
    if not asset_options:
        st.warning("No dataset rows are available.")
    else:
        selected_asset = st.selectbox("Asset", asset_options, key="app_dataset_asset")
        frame = load_asset_frame(selected_asset)
        normalized = normalized_dataset.loc[normalized_dataset["asset"] == selected_asset].reset_index(drop=True)

        stat_col_1, stat_col_2, stat_col_3, stat_col_4 = st.columns(4)
        stat_col_1.metric("Rows", len(frame))
        stat_col_2.metric("Columns", len(frame.columns))
        stat_col_3.metric("First Date", str(pd.to_datetime(frame["date"]).min().date()))
        stat_col_4.metric("Last Date", str(pd.to_datetime(frame["date"]).max().date()))

        schema_frame = pd.DataFrame(
            {
                "column": frame.columns,
                "dtype": [str(dtype) for dtype in frame.dtypes],
                "non_null": [int(frame[column].notna().sum()) for column in frame.columns],
                "sample_value": [str(frame.iloc[0][column])[:120] for column in frame.columns],
            }
        )
        st.markdown("**Schema**")
        st.dataframe(schema_frame, use_container_width=True, hide_index=True)

        asset_history = price_history.loc[price_history["asset"] == selected_asset].set_index("date")[["prices"]]
        st.markdown("**Price Trend**")
        st.line_chart(asset_history.rename(columns={"prices": f"{selected_asset} price"}))

        row_index = st.slider("Row index", min_value=0, max_value=max(len(frame) - 1, 0), value=0, key="app_dataset_row")
        record = load_normalized_asset_record(selected_asset, row_index)

        left_col, right_col = st.columns([0.95, 1.05])
        with left_col:
            st.markdown("**Prompt-Ready Context**")
            st.text_area("Bundle summary", value=record["prompt_context"], height=280, disabled=True, label_visibility="collapsed")
        with right_col:
            st.markdown("**News Narrative**")
            st.text_area("News narrative", value=record["news_text"], height=280, disabled=True, label_visibility="collapsed")

        st.markdown("**Agent Market Arena Request Payload**")
        st.code(json.dumps(record["request_payload"], ensure_ascii=False, indent=2, default=str), language="json")

        timeline = normalized[["date", "prices", "news_count", "news_length"]].rename(
            columns={"date": "Date", "prices": "Price", "news_count": "News Items", "news_length": "News Characters"}
        )
        st.markdown("**Timeline**")
        st.dataframe(timeline, use_container_width=True, hide_index=True)
        st.markdown("**Raw Row**")
        st.code(json.dumps(record["raw"], ensure_ascii=False, indent=2, default=str), language="json")

with tab_playground:
    if not asset_options:
        st.warning("No dataset rows are available.")
    else:
        st.caption("Run a single completion in streaming mode or local mode — OpenRouter, HuggingFace, or LM Studio.")

        # model / provider now come from the shared selector above — show a reminder
        st.info(f"Using **{model_id}** via **{provider}**. Change in the ⚙️ settings above.")

        temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.2, step=0.1, key="app_playground_temp")
        top_p       = st.slider("Top-p", min_value=0.1, max_value=1.0, value=0.9, step=0.05, key="app_playground_top_p")
        max_tokens  = st.slider("Max new tokens", min_value=128, max_value=2048, value=700, step=64, key="app_playground_max_tokens")

        selected_asset = st.selectbox("Reference asset", asset_options, key="app_playground_asset")
        frame = load_asset_frame(selected_asset)
        row_index = st.number_input(
            "Reference row",
            min_value=0,
            max_value=max(len(frame) - 1, 0),
            value=0,
            step=1,
            key="app_playground_row",
        )
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
            key="app_playground_system",
        )
        user_prompt = st.text_area(
            "User prompt",
            value=(
                "Evaluate this daily trading bundle and recommend exactly one action.\n\n"
                f"{record['prompt_context']}\n\n"
                "Return a short explanation first, then valid JSON."
            ),
            height=320,
            key="app_playground_user",
        )
        with st.expander("Preview request payload", expanded=False):
            st.code(json.dumps(record["request_payload"], ensure_ascii=False, indent=2, default=str), language="json")

        if st.button("Stream response", type="primary", key="app_playground_run"):
            if provider != "lmstudio" and not api_key:
                st.error("Provide an API key or token first.")
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ]
                try:
                    streamed_text = collect_chat_completion(
                        api_key=api_key,
                        model=model_id,
                        messages=messages,
                        temperature=float(temperature),
                        top_p=float(top_p),
                        max_tokens=int(max_tokens),
                        provider=provider,
                        hf_provider_hint=hf_provider_hint,
                    )
                    st.session_state["app_last_model_response"] = streamed_text
                except Exception as exc:
                    st.exception(exc)

        if st.session_state.get("app_last_model_response"):
            st.markdown("**Last Response**")
            st.write(st.session_state["app_last_model_response"])

with tab_batch:
    st.caption("Run a full dataset evaluation and store outputs under `results/`.")
    st.info(f"Using **{model_id}** via **{provider}**. Change in the ⚙️ settings above.")

    experiment_name  = st.text_input("Experiment name", value="task3-home-run", key="app_batch_experiment")
    selected_assets  = st.multiselect("Assets", options=asset_options, default=asset_options, key="app_batch_assets")

    batch_col_1, batch_col_2, batch_col_3, batch_col_4 = st.columns(4)
    with batch_col_1:
        limit       = st.number_input("Row limit", min_value=0, value=0, step=1, help="0 = all rows.", key="app_batch_limit")
    with batch_col_2:
        reruns      = st.number_input("Reruns", min_value=1, value=1, step=1, key="app_batch_reruns")
    with batch_col_3:
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.2, step=0.1, key="app_batch_temp")
    with batch_col_4:
        top_p       = st.slider("Top-p", min_value=0.1, max_value=1.0, value=0.9, step=0.05, key="app_batch_top_p")
    max_tokens = st.slider("Max new tokens", min_value=128, max_value=2048, value=700, step=64, key="app_batch_max_tokens")

    preview_dataset = normalized_dataset.copy()
    if selected_assets:
        preview_dataset = preview_dataset.loc[preview_dataset["asset"].isin(selected_assets)].reset_index(drop=True)
    if limit > 0:
        preview_dataset = preview_dataset.head(int(limit)).reset_index(drop=True)

    st.write(
        f"Planned workload: `{len(preview_dataset)}` rows x `{int(reruns)}` reruns = "
        f"`{len(preview_dataset) * int(reruns)}` model calls"
    )
    st.dataframe(preview_dataset[["dataset_row_id", "date", "asset", "prices", "news_count", "news_length"]].head(10), hide_index=True)

    st.markdown("**Execution mode**")
    st.caption("Use the script runner for the complete research flow, or the in-app runner for checkpointed interactive execution.")

    script_col, interactive_col = st.columns(2)
    with script_col:
        script_run_clicked = st.button(
            "Run via scripts/run_evaluation.py",
            type="primary",
            use_container_width=True,
            key="app_batch_run_script",
        )
    with interactive_col:
        interactive_run_clicked = st.button(
            "Run inside app",
            use_container_width=True,
            key="app_batch_run",
        )

    # ── Checkpoint resume ─────────────────────────────────────────────────────
    RESULTS_DIR = Path(__file__).resolve().parent / "results"
    checkpoint_path = RESULTS_DIR / f"{experiment_name}_checkpoint.jsonl"

    def _load_checkpoint(path: Path) -> tuple[list[dict], set[tuple]]:
        """Load completed records and their (dataset_row_id, rerun_id) keys."""
        if not path.exists():
            return [], set()
        records = []
        done_keys = set()
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    records.append(rec)
                    done_keys.add((rec["dataset_row_id"], rec.get("rerun_id", 0)))
                except json.JSONDecodeError:
                    pass
        return records, done_keys

    def _append_checkpoint(path: Path, record: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    existing_records, done_keys = _load_checkpoint(checkpoint_path)
    resume_count = len(existing_records)

    if resume_count > 0:
        st.info(
            f"Checkpoint found: **{resume_count}** records already completed for `{experiment_name}`. "
            f"Clicking **Run** will resume from where it left off."
        )
        if st.button("Clear checkpoint and restart", key="app_batch_clear_checkpoint"):
            checkpoint_path.unlink(missing_ok=True)
            st.success("Checkpoint cleared. Refresh the page to start fresh.")

    if script_run_clicked:
        if normalized_dataset.empty:
            st.error("No dataset rows are available.")
        elif provider != "lmstudio" and not api_key:
            st.error("Provide an API key or token first.")
        else:
            with st.spinner("Running scripts/run_evaluation.py..."):
                completed = _run_evaluation_script(
                    experiment_name=experiment_name,
                    model_id=model_id,
                    provider=provider,
                    hf_provider_hint=hf_provider_hint,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    max_tokens=int(max_tokens),
                    selected_assets=selected_assets or asset_options,
                    limit=int(limit),
                    reruns=int(reruns),
                    api_key=api_key,
                )
            if completed.returncode == 0:
                st.success("Script evaluation finished successfully.")
                if completed.stdout.strip():
                    st.code(completed.stdout, language="text")
            else:
                st.error("Script evaluation failed.")
                if completed.stderr.strip():
                    st.code(completed.stderr, language="text")
                if completed.stdout.strip():
                    st.code(completed.stdout, language="text")

    if interactive_run_clicked:
        if normalized_dataset.empty:
            st.error("No dataset rows are available.")
        elif provider != "lmstudio" and not api_key:
            st.error("Provide an API key or token first.")
        else:
            run_dataset = normalized_dataset.copy()
            if selected_assets:
                run_dataset = run_dataset.loc[run_dataset["asset"].isin(selected_assets)].reset_index(drop=True)
            if limit > 0:
                run_dataset = run_dataset.head(int(limit)).reset_index(drop=True)

            # Reload checkpoint at run-time (may have changed since page load)
            records, done_keys = _load_checkpoint(checkpoint_path)

            total_calls = len(run_dataset) * int(reruns)
            skipped     = len(records)
            current     = skipped

            progress_bar = st.progress(
                skipped / total_calls if total_calls else 1.0,
                text=f"Resuming — {skipped}/{total_calls} already done",
            )
            status_box = st.empty()

            try:
                for rerun_id in range(int(reruns)):
                    for record_row in run_dataset.to_dict(orient="records"):
                        key = (record_row["dataset_row_id"], rerun_id)
                        if key in done_keys:
                            continue  # already completed — skip

                        current += 1
                        status_box.write(
                            f"Running {current}/{total_calls}: "
                            f"asset={record_row['asset']} date={record_row['date']} rerun={rerun_id}"
                        )

                        result = evaluate_record(
                            api_key=api_key,
                            model=model_id,
                            provider=provider,
                            hf_provider_hint=hf_provider_hint,
                            record=record_row,
                            rerun_id=rerun_id,
                            temperature=float(temperature),
                            top_p=float(top_p),
                            max_tokens=int(max_tokens),
                        )
                        result["rerun_id"] = rerun_id  # ensure key field is present

                        # Persist immediately — survives any subsequent crash
                        _append_checkpoint(checkpoint_path, result)
                        records.append(result)
                        done_keys.add(key)

                        progress_bar.progress(
                            current / total_calls,
                            text=f"Completed {current}/{total_calls}",
                        )

                results_df = pd.DataFrame(records)
                run_dir = save_experiment_results(
                    experiment_name=experiment_name,
                    results=results_df,
                    metadata={
                        "model":            model_id,
                        "provider":         provider,
                        "hf_provider_hint": hf_provider_hint,
                        "temperature":      float(temperature),
                        "top_p":            float(top_p),
                        "max_tokens":       int(max_tokens),
                        "assets":           selected_assets or "ALL",
                        "limit":            int(limit),
                        "reruns":           int(reruns),
                        "launched_from":    "app.py",
                        "resumed_from_checkpoint": skipped > 0,
                    },
                )

                # Clean up checkpoint only after a successful full save
                checkpoint_path.unlink(missing_ok=True)

                st.success(f"Saved evaluation run to `{run_dir}` — checkpoint cleared.")
                st.dataframe(
                    results_df[[
                        "date", "asset", "model", "provider",
                        "parsed_action", "confidence",
                        "latency_seconds", "json_parse_success",
                    ]],
                    use_container_width=True,
                    hide_index=True,
                )

            except Exception as exc:
                st.exception(exc)
                st.warning(
                    f"Evaluation interrupted after {current}/{total_calls} calls. "
                    f"**{len(records)} records** have been saved to the checkpoint at "
                    f"`{checkpoint_path}`. Re-click **Run** to resume."
                )

with tab_results:
    runs = list_saved_runs()
    if runs.empty:
        st.info("No saved experiment runs found yet.")
    else:
        st.dataframe(
            runs.rename(
                columns={
                    "run_name": "Run Name",
                    "experiment_name": "Experiment",
                    "saved_at_utc": "Saved At (UTC)",
                    "rows": "Rows",
                    "parse_success_rate": "Parse Success",
                    "mean_strategy_return": "Mean Strategy Return",
                    "cumulative_return": "Cumulative Return",
                    "sharpe_ratio": "Sharpe Ratio",
                    "path": "Path",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        selected_run = st.selectbox("Select run", runs["run_name"].tolist(), key="app_results_run")
        results = load_saved_results(selected_run)
        run_summary = load_run_summary(selected_run)

        performance = summarize_performance_metrics(results)

        metric_col_1, metric_col_2, metric_col_3, metric_col_4, metric_col_5 = st.columns(5)
        metric_col_1.metric("Rows", len(results))
        metric_col_2.metric("Parse Success", f"{results['json_parse_success'].mean():.1%}")
        metric_col_3.metric("Cumulative Return", f"{performance['cumulative_return']:.2%}")
        metric_col_4.metric("Sharpe Ratio", f"{performance['sharpe_ratio']:.2f}")
        metric_col_5.metric("Max Drawdown", f"{performance['max_drawdown']:.2%}")

        vol_col_1, vol_col_2, vol_col_3 = st.columns(3)
        vol_col_1.metric("Daily Volatility", f"{performance['daily_volatility']:.2%}")
        vol_col_2.metric("Annualized Volatility", f"{performance['annualized_volatility']:.2%}")
        vol_col_3.metric("Failure Defaults", f"{results['failure_defaulted_to_hold'].mean():.1%}")

        action_distribution = (
            results.groupby(["asset", "execution_action"], as_index=False)
            .size()
            .pivot(index="asset", columns="execution_action", values="size")
            .fillna(0)
        )
        st.markdown("**Executed Action Distribution by Asset**")
        st.dataframe(action_distribution, use_container_width=True)

        monthly_actions = (
            results.groupby(["month", "execution_action"], as_index=False)
            .size()
            .pivot(index="month", columns="execution_action", values="size")
            .fillna(0)
            .sort_index()
        )
        st.markdown("**Monthly Executed Action Distribution**")
        st.bar_chart(monthly_actions)

        by_asset = (
            results.groupby("asset", as_index=False)
            .agg(
                rows=("dataset_row_id", "count"),
                parse_success=("json_parse_success", "mean"),
                mean_latency=("latency_seconds", "mean"),
                mean_strategy_return=("strategy_return", "mean"),
                mean_buy_hold_return=("buy_hold_return", "mean"),
                hit_rate=("hit", "mean"),
                cumulative_return=("strategy_return", lambda series: float((1 + series.dropna()).prod() - 1) if series.notna().any() else 0.0),
            )
        )
        st.markdown("**Outcome Summary by Asset**")
        st.dataframe(by_asset, use_container_width=True, hide_index=True)

        issues = results.loc[
            (~results["json_parse_success"])
            | ((results["execution_action"] == "SELL") & (results["next_day_return"] > 0.03))
            | ((results["confidence"].fillna(0) >= 0.9) & (results["hit"] == False))
        ][["date", "asset", "parsed_action", "execution_action", "confidence", "next_day_return", "json_parse_error", "model_response"]].copy()
        issues["date"] = pd.to_datetime(issues["date"]).dt.date.astype(str)
        st.markdown("**Failure and Error Analysis Candidates**")
        st.dataframe(issues, use_container_width=True, hide_index=True)

        summary_json_path = Path(runs.loc[runs["run_name"] == selected_run, "path"].iloc[0]) / "summary.json"
        if summary_json_path.exists():
            with st.expander("Summary File", expanded=False):
                st.code(json.dumps(json.loads(summary_json_path.read_text(encoding="utf-8")), indent=2), language="json")

        st.subheader("Results and Discussion")
        report_key = f"app_results_discussion::{selected_run}"
        if st.button("Generate Results and Discussion", key=f"app_generate_discussion::{selected_run}"):
            report_markdown = generate_results_discussion(
                run_name=selected_run,
                results=results,
                run_summary=run_summary,
            )
            report_path = save_run_report(selected_run, report_markdown)
            st.session_state[report_key] = {
                "markdown": report_markdown,
                "path": str(report_path),
            }

        if report_key in st.session_state:
            report = st.session_state[report_key]
            st.caption(f"Saved to `{report['path']}`")
            st.markdown(report["markdown"])
            st.download_button(
                "Download Markdown",
                data=report["markdown"],
                file_name=f"{selected_run}_results_and_discussion.md",
                mime="text/markdown",
                key=f"app_download_discussion::{selected_run}",
            )

with tab_notes:
    notes = read_notes()
    if not notes:
        st.warning("`notes.md` is missing.")
    else:
        for section in note_sections:
            with st.expander(section["title"], expanded=section["title"].lower().startswith("workflow")):
                st.markdown(section["body"])
        st.markdown("**Full Notes**")
        st.markdown(notes)
