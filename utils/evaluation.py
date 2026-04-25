from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from utils.data import build_normalized_dataset, build_price_history, ensure_results_dir
from utils.hf_inference import collect_chat_completion


VALID_ACTIONS = ("BUY", "HOLD", "SELL")
ACTION_TO_POSITION = {"BUY": 1.0, "HOLD": 0.0, "SELL": -1.0}
TRADING_DAYS_PER_YEAR = 252


def default_system_prompt() -> str:
    return (
        "You are a financial trading research assistant working on FinMMEval Task 3. "
        "Use only the provided bundle. Decide between BUY, HOLD, and SELL. "
        "Respond with concise reasoning followed by a JSON object containing "
        "`recommended_action`, `confidence`, `key_drivers`, and `risk_flags`."
    )


def build_user_prompt(record: dict[str, Any]) -> str:
    if record.get("request_payload"):
        payload = json.dumps(record["request_payload"], ensure_ascii=False, indent=2, default=str)
        return (
            "Evaluate this daily trading request payload and recommend exactly one action.\n\n"
            f"{payload}\n\n"
            "Return a short explanation first, then valid JSON."
        )

    return (
        "Evaluate this daily trading bundle and recommend exactly one action.\n\n"
        f"{record['prompt_context']}\n\n"
        "Return a short explanation first, then valid JSON."
    )


def execution_action_from_parse(parsed_action: object) -> str:
    action = str(parsed_action or "").upper().strip()
    return action if action in VALID_ACTIONS else "HOLD"


def _slugify(value: str) -> str:
    lowered = value.strip().lower()
    return re.sub(r"[^a-z0-9]+", "-", lowered).strip("-") or "run"


def _extract_json_block(text: object) -> dict | None:
    """
    Robustly extract and parse a JSON object from *text*.
    Accepts str, dict, list, or None. If *text* is a dict-like response (e.g.
    already-decoded JSON from OpenRouter/HF), try common nested paths first,
    otherwise fall back to dumping to a string and regex extraction.
    Returns parsed dict or None if extraction/parsing failed.
    """
    if text is None:
        return None

    # If it's already a dict-like response, try to locate the main content
    if isinstance(text, dict):
        # common response shapes: {"choices":[{"message":{"content": "..."}}, ...]}
        try_paths = [
            ["choices", 0, "message", "content"],
            ["choices", 0, "text"],
            ["content"],
            ["message", "content"],
        ]
        for path in try_paths:
            node = text
            try:
                for key in path:
                    node = node[key]
                if isinstance(node, str) and node.strip():
                    text = node
                    break
            except Exception:
                continue
        else:
            # nothing useful found — stringify the dict for regex extraction
            try:
                text = json.dumps(text, ensure_ascii=False)
            except Exception:
                text = str(text)

    # Non-string objects -> coerce to string
    if not isinstance(text, str):
        text = str(text)

    # Find JSON object blocks and try to parse the most plausible one.
    matches = list(re.finditer(r"\{[\s\S]*?\}", text))
    if not matches:
        return None

    # Prefer the last match (often model outputs include explanation then JSON)
    for m in reversed(matches):
        block = m.group(0)
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            # try to fix common issues (trailing commas)
            cleaned = re.sub(r",\s*([}\]])", r"\1", block)
            try:
                return json.loads(cleaned)
            except Exception:
                continue
    return None


def parse_model_response(response: object) -> dict:
    payload = _extract_json_block(response)
    if payload is None:
        response_text = "" if response is None else str(response)
        return {
            "parsed_action": "UNPARSED",
            "confidence": None,
            "json_parse_success": False,
            "json_parse_error": "No valid JSON object found in model response.",
            "parsed_json": None,
            "rationale_length": len(response_text),
        }

    parsed_action = None
    confidence = None
    parse_error = ""

    if payload is not None:
        candidate = str(payload.get("recommended_action", "")).upper().strip()
        if candidate in VALID_ACTIONS:
            parsed_action = candidate
        else:
            parse_error = "Missing or invalid recommended_action in JSON."

        raw_confidence = payload.get("confidence")
        try:
            if raw_confidence is not None and raw_confidence != "":
                confidence = float(raw_confidence)
        except (TypeError, ValueError):
            if parse_error:
                parse_error += " "
            parse_error += "Confidence was not numeric."
    else:
        parse_error = "No valid JSON object found in model response."

    return {
        "parsed_action": parsed_action or "UNPARSED",
        "confidence": confidence,
        "json_parse_success": payload is not None and parsed_action in VALID_ACTIONS,
        "json_parse_error": parse_error,
        "parsed_json": payload,
        "rationale_length": len(response),
    }


def _looks_like_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "429" in message or "too many requests" in message or "rate limit" in message


def _build_failed_result(
    *,
    record: dict[str, Any],
    model: str,
    provider: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    rerun_id: int,
    system_prompt: str,
    user_prompt: str,
    latency_seconds: float,
    error_message: str,
    attempt_count: int,
) -> dict[str, Any]:
    return {
        "dataset_row_id": record["dataset_row_id"],
        "row_index": record["row_index"],
        "date": record["date"],
        "asset": record["asset"],
        "price": record["prices"],
        "news_count": record["news_count"],
        "news_length": record["news_length"],
        "prompt_context": record["prompt_context"],
        "model": model,
        "provider": provider or "auto",
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "rerun_id": rerun_id,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "model_response": "",
        "latency_seconds": latency_seconds,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "parsed_action": "UNPARSED",
        "execution_action": "HOLD",
        "failure_defaulted_to_hold": True,
        "confidence": None,
        "json_parse_success": False,
        "json_parse_error": error_message,
        "parsed_json": None,
        "rationale_length": 0,
        "request_error": error_message,
        "request_attempts": attempt_count,
    }


def evaluate_record(
    api_key: str,
    model: str,
    provider: str,
    hf_provider_hint: str = "auto",                # ← NEW
    record: dict | None = None,
    rerun_id: int = 0,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_tokens: int = 700,
    max_attempts: int = 8,
    base_retry_delay_seconds: float = 3.0,
    pre_call_delay_seconds: float = 0.75,
    continue_on_error: bool = True,
) -> dict:
    """Evaluate a single normalised trading record and return a result dict."""
    system_prompt = default_system_prompt()
    user_prompt = build_user_prompt(record)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    started = time.perf_counter()
    response = ""
    last_error = ""
    attempt_count = 0

    for attempt in range(1, max_attempts + 1):
        attempt_count = attempt
        try:
            response = collect_chat_completion(
                api_key=api_key,
                model=model,
                provider=provider,
                hf_provider_hint=hf_provider_hint,         # ← forward to provider
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                pre_call_delay=pre_call_delay_seconds,
            )
            last_error = ""
            break
        except Exception as exc:
            last_error = str(exc)
            if attempt >= max_attempts:
                if continue_on_error:
                    latency_seconds = time.perf_counter() - started
                    return _build_failed_result(
                        record=record,
                        model=model,
                        provider=provider,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        rerun_id=rerun_id,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        latency_seconds=latency_seconds,
                        error_message=last_error,
                        attempt_count=attempt_count,
                    )
                raise

            retry_delay = min(base_retry_delay_seconds * (2 ** (attempt - 1)), 90.0)
            if _looks_like_rate_limit_error(exc):
                retry_delay = min(max(retry_delay, 10.0), 120.0)
            time.sleep(retry_delay)

    latency_seconds = time.perf_counter() - started
    parsed = parse_model_response(response)
    execution_action = execution_action_from_parse(parsed["parsed_action"])

    return {
        "dataset_row_id": record["dataset_row_id"],
        "row_index": record["row_index"],
        "date": record["date"],
        "asset": record["asset"],
        "price": record["prices"],
        "news_count": record["news_count"],
        "news_length": record["news_length"],
        "prompt_context": record["prompt_context"],
        "model": model,
        "provider": provider or "auto",
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "rerun_id": rerun_id,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "model_response": response,
        "latency_seconds": latency_seconds,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "request_error": last_error,
        "request_attempts": attempt_count,
        "execution_action": execution_action,
        "failure_defaulted_to_hold": execution_action == "HOLD" and not parsed["json_parse_success"],
        **parsed,
    }


def add_outcome_columns(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results.copy()

    history = build_price_history().copy()
    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values(["asset", "date"]).reset_index(drop=True)
    history["next_price"] = history.groupby("asset")["prices"].shift(-1)
    history["next_date"] = history.groupby("asset")["date"].shift(-1)
    history["next_day_return"] = (history["next_price"] - history["prices"]) / history["prices"]

    enriched = results.copy()
    enriched["date"] = pd.to_datetime(enriched["date"])
    outcome_columns = [
        "next_price",
        "next_date",
        "next_day_return",
        "position",
        "strategy_return",
        "buy_hold_return",
        "hit",
        "month",
    ]
    enriched = enriched.drop(columns=[column for column in outcome_columns if column in enriched.columns])
    enriched = enriched.merge(
        history[["asset", "date", "next_price", "next_date", "next_day_return"]],
        on=["asset", "date"],
        how="left",
    )
    if "execution_action" not in enriched.columns:
        enriched["execution_action"] = enriched["parsed_action"].map(execution_action_from_parse)
    else:
        enriched["execution_action"] = enriched["execution_action"].map(execution_action_from_parse)
    enriched["failure_defaulted_to_hold"] = (
        (enriched["execution_action"] == "HOLD") & (enriched["json_parse_success"] == False)
    )
    enriched["position"] = enriched["execution_action"].map(ACTION_TO_POSITION).fillna(0.0)
    enriched["strategy_return"] = enriched["position"] * enriched["next_day_return"]
    enriched["buy_hold_return"] = enriched["next_day_return"]
    enriched["hit"] = (
        ((enriched["execution_action"] == "BUY") & (enriched["next_day_return"] > 0))
        | ((enriched["execution_action"] == "SELL") & (enriched["next_day_return"] < 0))
        | ((enriched["execution_action"] == "HOLD") & (enriched["next_day_return"].abs() <= 0.01))
    )
    enriched["month"] = enriched["date"].dt.to_period("M").astype(str)
    return enriched


def summarize_performance_metrics(results: pd.DataFrame) -> dict[str, float]:
    if "strategy_return" not in results:
        returns = pd.Series(dtype=float)
    else:
        sort_columns = [column for column in ["rerun_id", "date", "asset"] if column in results.columns]
        ordered = results.sort_values(sort_columns) if sort_columns else results
        returns = ordered["strategy_return"].dropna()
    if returns.empty:
        return {
            "cumulative_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "daily_volatility": 0.0,
            "annualized_volatility": 0.0,
        }

    equity_curve = pd.concat([pd.Series([1.0]), (1.0 + returns).cumprod()], ignore_index=True)
    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1.0
    daily_volatility = float(returns.std(ddof=0))
    sharpe_ratio = 0.0
    if daily_volatility > 0:
        sharpe_ratio = float((returns.mean() / daily_volatility) * (TRADING_DAYS_PER_YEAR**0.5))

    return {
        "cumulative_return": float(equity_curve.iloc[-1] - 1.0),
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": float(drawdown.min()),
        "daily_volatility": daily_volatility,
        "annualized_volatility": float(daily_volatility * (TRADING_DAYS_PER_YEAR**0.5)),
    }


def summarize_results(results: pd.DataFrame) -> dict[str, Any]:
    if results.empty:
        return {
            "rows": 0,
            "parse_success_rate": 0.0,
            "mean_latency_seconds": 0.0,
            "mean_strategy_return": 0.0,
            "mean_buy_hold_return": 0.0,
            "hit_rate": 0.0,
            "cumulative_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "daily_volatility": 0.0,
            "annualized_volatility": 0.0,
        }

    if "strategy_return" not in results.columns or "buy_hold_return" not in results.columns:
        results = add_outcome_columns(results)

    valid_returns = results["strategy_return"].dropna()
    valid_baseline = results["buy_hold_return"].dropna()
    performance = summarize_performance_metrics(results)
    return {
        "rows": int(len(results)),
        "parse_success_rate": float(results["json_parse_success"].mean()),
        "mean_latency_seconds": float(results["latency_seconds"].mean()),
        "mean_strategy_return": float(valid_returns.mean()) if not valid_returns.empty else 0.0,
        "mean_buy_hold_return": float(valid_baseline.mean()) if not valid_baseline.empty else 0.0,
        "hit_rate": float(results["hit"].dropna().mean()) if results["hit"].notna().any() else 0.0,
        **performance,
    }


def save_experiment_results(
    *,
    experiment_name: str,
    results: pd.DataFrame,
    metadata: dict[str, Any],
) -> Path:
    run_slug = _slugify(experiment_name)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ensure_results_dir() / f"{timestamp}_{run_slug}"
    run_dir.mkdir(parents=True, exist_ok=True)

    enriched_results = add_outcome_columns(results)

    predictions_csv = run_dir / "predictions.csv"
    predictions_jsonl = run_dir / "predictions.jsonl"
    summary_json = run_dir / "summary.json"

    enriched_results.to_csv(predictions_csv, index=False)
    with predictions_jsonl.open("w", encoding="utf-8") as handle:
        for row in enriched_results.to_dict(orient="records"):
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    payload = {
        "experiment_name": experiment_name,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
        "summary": summarize_results(enriched_results),
        "files": {
            "predictions_csv": predictions_csv.name,
            "predictions_jsonl": predictions_jsonl.name,
        },
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir


def list_saved_runs() -> pd.DataFrame:
    results_dir = ensure_results_dir()
    rows: list[dict[str, Any]] = []
    for run_dir in sorted([path for path in results_dir.iterdir() if path.is_dir()], reverse=True):
        summary_path = run_dir / "summary.json"
        predictions_path = run_dir / "predictions.csv"
        summary: dict[str, Any] = {}
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run_name": run_dir.name,
                "experiment_name": summary.get("experiment_name", run_dir.name),
                "saved_at_utc": summary.get("saved_at_utc", ""),
                "rows": summary.get("summary", {}).get("rows"),
                "parse_success_rate": summary.get("summary", {}).get("parse_success_rate"),
                "mean_strategy_return": summary.get("summary", {}).get("mean_strategy_return"),
                "cumulative_return": summary.get("summary", {}).get("cumulative_return"),
                "sharpe_ratio": summary.get("summary", {}).get("sharpe_ratio"),
                "path": str(run_dir),
                "predictions_csv": str(predictions_path),
            }
        )
    return pd.DataFrame(rows)


def load_saved_results(run_name: str) -> pd.DataFrame:
    run_dir = ensure_results_dir() / run_name
    predictions_path = run_dir / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"No predictions.csv found for run {run_name}")
    frame = pd.read_csv(predictions_path)
    return add_outcome_columns(frame)


def build_dataset_analysis_frame() -> pd.DataFrame:
    dataset = build_normalized_dataset().copy()
    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset["month"] = dataset["date"].dt.to_period("M").astype(str)
    return dataset
