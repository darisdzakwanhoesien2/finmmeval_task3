from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.evaluation import (
    build_dataset_analysis_frame,
    generate_results_discussion,
    list_saved_runs,
    load_run_summary,
    load_saved_results,
    save_run_report,
    summarize_performance_metrics,
)


st.set_page_config(page_title="Evaluation Results", page_icon=":bar_chart:")
st.title("Evaluation Results")
st.caption("Review saved experiment runs and compute paper-ready dataset and outcome summaries")

st.subheader("Dataset Analysis")
dataset = build_dataset_analysis_frame()

if dataset.empty:
    st.warning("No normalized dataset rows were available.")
else:
    asset_summary = (
        dataset.groupby("asset", as_index=False)
        .agg(
            rows=("dataset_row_id", "count"),
            start_date=("date", "min"),
            end_date=("date", "max"),
            mean_news_items=("news_count", "mean"),
            mean_news_chars=("news_length", "mean"),
        )
    )
    asset_summary["start_date"] = asset_summary["start_date"].dt.date.astype(str)
    asset_summary["end_date"] = asset_summary["end_date"].dt.date.astype(str)
    st.dataframe(asset_summary, use_container_width=True, hide_index=True)

    month_density = (
        dataset.groupby(["month", "asset"], as_index=False)
        .agg(mean_news_length=("news_length", "mean"), mean_news_count=("news_count", "mean"))
        .pivot(index="month", columns="asset", values="mean_news_length")
        .sort_index()
    )
    st.markdown("**Monthly context density (mean news characters)**")
    st.line_chart(month_density)

st.subheader("Saved Runs")
runs = list_saved_runs()
if runs.empty:
    st.info("No saved experiment runs found yet. Use `scripts/run_evaluation.py` to create one.")
else:
    display_runs = runs.rename(
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
    )
    st.dataframe(display_runs, use_container_width=True, hide_index=True)

    selected_run = st.selectbox("Select run", runs["run_name"].tolist())
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

    st.markdown("**Executed action distribution by asset**")
    action_distribution = (
        results.groupby(["asset", "execution_action"], as_index=False)
        .size()
        .pivot(index="asset", columns="execution_action", values="size")
        .fillna(0)
    )
    st.dataframe(action_distribution, use_container_width=True)

    st.markdown("**Monthly executed action distribution**")
    monthly_actions = (
        results.groupby(["month", "execution_action"], as_index=False)
        .size()
        .pivot(index="month", columns="execution_action", values="size")
        .fillna(0)
        .sort_index()
    )
    st.bar_chart(monthly_actions)

    st.markdown("**Outcome summary by asset**")
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
    st.dataframe(by_asset, use_container_width=True, hide_index=True)

    st.markdown("**Confidence bucket analysis**")
    bucketed = results.copy()
    bucketed["confidence_bucket"] = pd.cut(
        bucketed["confidence"],
        bins=[-0.01, 0.25, 0.5, 0.75, 1.0, 100.0],
        labels=["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0", "1.0+"],
    )
    confidence_summary = (
        bucketed.groupby("confidence_bucket", as_index=False)
        .agg(rows=("dataset_row_id", "count"), mean_strategy_return=("strategy_return", "mean"), hit_rate=("hit", "mean"))
    )
    st.dataframe(confidence_summary, use_container_width=True, hide_index=True)

    if results["rerun_id"].nunique() > 1:
        st.markdown("**Stability across reruns**")
        stability = (
            results.groupby("dataset_row_id")["parsed_action"]
            .nunique()
            .rename("unique_actions")
            .reset_index()
        )
        stable_rate = (stability["unique_actions"] == 1).mean()
        st.metric("Exact action stability", f"{stable_rate:.1%}")
        st.dataframe(stability.sort_values("unique_actions", ascending=False), use_container_width=True, hide_index=True)

    st.markdown("**Schema failures and qualitative review candidates**")
    issues = results.loc[
        (~results["json_parse_success"])
        | ((results["execution_action"] == "SELL") & (results["next_day_return"] > 0.03))
        | ((results["confidence"].fillna(0) >= 0.9) & (results["hit"] == False))
    ][
        ["date", "asset", "parsed_action", "execution_action", "confidence", "next_day_return", "json_parse_error", "model_response"]
    ].copy()
    issues["date"] = pd.to_datetime(issues["date"]).dt.date.astype(str)
    st.dataframe(issues, use_container_width=True, hide_index=True)

    with st.expander("Run summary file", expanded=False):
        summary_path = Path(selected_run).name
        run_path = runs.loc[runs["run_name"] == selected_run, "path"].iloc[0]
        summary_json_path = Path(run_path) / "summary.json"
        if summary_json_path.exists():
            st.code(json.dumps(json.loads(summary_json_path.read_text(encoding="utf-8")), indent=2), language="json")

    st.subheader("Results and Discussion")
    report_key = f"results_discussion::{selected_run}"
    if st.button("Generate Results and Discussion", key=f"generate_discussion::{selected_run}"):
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
            key=f"download_discussion::{selected_run}",
        )
