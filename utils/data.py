from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "data" / "clef-task3-trading"
NOTES_PATH = ROOT_DIR / "notes.md"
RESULTS_DIR = ROOT_DIR / "results"

CORE_PAYLOAD_FIELDS = ("date", "price", "symbol", "momentum", "news")
ENRICHMENT_PAYLOAD_FIELDS = ("history_price", "10k", "10q")


@dataclass(frozen=True)
class DatasetAsset:
    asset: str
    file_path: Path
    file_type: str


def _humanize_heading(raw_heading: str) -> str:
    text = raw_heading.replace("\\section{", "").replace("\\subsection{", "").rstrip("}")
    return text.strip() or "Untitled Section"


@st.cache_data(show_spinner=False)
def list_dataset_assets() -> list[DatasetAsset]:
    assets: list[DatasetAsset] = []
    for file_path in sorted(DATASET_DIR.glob("*")):
        if file_path.suffix not in {".csv", ".parquet"}:
            continue
        assets.append(
            DatasetAsset(
                asset=file_path.stem,
                file_path=file_path,
                file_type=file_path.suffix.lstrip("."),
            )
        )
    return assets


def _preferred_assets() -> list[DatasetAsset]:
    preferred: dict[str, DatasetAsset] = {}
    for asset in list_dataset_assets():
        current = preferred.get(asset.asset)
        if current is None or asset.file_type == "parquet":
            preferred[asset.asset] = asset
    return [preferred[key] for key in sorted(preferred)]


@st.cache_data(show_spinner=False)
def load_asset_frame(asset: str) -> pd.DataFrame:
    for dataset_asset in _preferred_assets():
        if dataset_asset.asset != asset:
            continue
        if dataset_asset.file_type == "parquet":
            frame = pd.read_parquet(dataset_asset.file_path)
        else:
            frame = pd.read_csv(dataset_asset.file_path)
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        return frame.sort_values("date").reset_index(drop=True)
    raise FileNotFoundError(f"Asset file not found for {asset}")


def _parse_news(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass

    return [text]


def _format_history_price(history: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {
            "date": str(pd.to_datetime(row["date"]).date()),
            "price": float(row["prices"]),
        }
        for _, row in history.iterrows()
    ]


def _calculate_momentum(history: pd.DataFrame, lookback: int = 5) -> float | None:
    if len(history) <= lookback:
        return None
    current_price = float(history.iloc[-1]["prices"])
    prior_price = float(history.iloc[-lookback - 1]["prices"])
    if prior_price == 0:
        return None
    return (current_price - prior_price) / prior_price


def build_task3_request_payload(
    *,
    record: dict[str, object],
    history_price: list[dict[str, object]] | None = None,
    momentum: float | None = None,
) -> dict[str, object]:
    """Build the Agent Market Arena-compatible daily request payload."""
    symbol = str(record.get("asset", ""))
    has_equity_filings = symbol.upper() not in {"BTC", "ETH"}
    return {
        "date": record.get("date"),
        "price": record.get("prices"),
        "symbol": symbol,
        "momentum": momentum,
        "news": record.get("news_text", ""),
        "history_price": history_price or [],
        "10k": None if not has_equity_filings else record.get("10k"),
        "10q": None if not has_equity_filings else record.get("10q"),
    }


def normalize_trading_row(
    record: pd.Series,
    *,
    history_price: list[dict[str, object]] | None = None,
    momentum: float | None = None,
) -> dict[str, object]:
    row = record.to_dict()
    news_items = _parse_news(row.get("news"))
    news_text = "\n\n".join(news_items) if news_items else ""
    prompt_context = (
        f"Date: {pd.to_datetime(row.get('date')).date()}\n"
        f"Symbol: {row.get('asset', '')}\n"
        f"Price: {float(row.get('prices', 0.0)):.6f}\n"
        f"Momentum: {'N/A' if momentum is None else f'{momentum:.6f}'}\n"
        f"News summary:\n{news_text}"
    ).strip()

    row["date"] = str(pd.to_datetime(row["date"]).date())
    request_payload = build_task3_request_payload(
        record={
            "date": row["date"],
            "asset": row.get("asset", ""),
            "prices": float(row.get("prices", 0.0)),
            "news_text": news_text,
            "10k": row.get("10k"),
            "10q": row.get("10q"),
        },
        history_price=history_price,
        momentum=momentum,
    )

    return {
        "date": row["date"],
        "asset": str(row.get("asset", "")),
        "prices": float(row.get("prices", 0.0)),
        "symbol": str(row.get("asset", "")),
        "momentum": momentum,
        "news_items": news_items,
        "news_count": len(news_items),
        "news_length": len(news_text),
        "news_text": news_text,
        "history_price": history_price or [],
        "request_payload": request_payload,
        "prompt_context": prompt_context,
        "raw": row,
    }


@st.cache_data(show_spinner=False)
def build_normalized_dataset() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_asset in _preferred_assets():
        frame = load_asset_frame(dataset_asset.asset)
        for row_index, (_, row) in enumerate(frame.iterrows()):
            history = frame.iloc[max(0, row_index - 10) : row_index + 1][["date", "prices"]]
            history_price = _format_history_price(history)
            momentum = _calculate_momentum(history)
            normalized = normalize_trading_row(row, history_price=history_price, momentum=momentum)
            rows.append(
                {
                    "dataset_row_id": f"{dataset_asset.asset}-{row_index:04d}",
                    "row_index": row_index,
                    "date": normalized["date"],
                    "asset": normalized["asset"],
                    "symbol": normalized["symbol"],
                    "prices": normalized["prices"],
                    "momentum": normalized["momentum"],
                    "news_count": normalized["news_count"],
                    "news_length": normalized["news_length"],
                    "news_text": normalized["news_text"],
                    "history_price": normalized["history_price"],
                    "request_payload": normalized["request_payload"],
                    "prompt_context": normalized["prompt_context"],
                }
            )
    return pd.DataFrame(rows)


def load_normalized_asset_record(asset: str, row_index: int) -> dict[str, object]:
    frame = load_asset_frame(asset)
    row_index = max(0, min(int(row_index), len(frame) - 1))
    history = frame.iloc[max(0, row_index - 10) : row_index + 1][["date", "prices"]]
    return normalize_trading_row(
        frame.iloc[row_index],
        history_price=_format_history_price(history),
        momentum=_calculate_momentum(history),
    )


@st.cache_data(show_spinner=False)
def build_asset_catalog() -> pd.DataFrame:
    rows = []
    for dataset_asset in _preferred_assets():
        frame = load_asset_frame(dataset_asset.asset)
        available_formats = sorted(
            asset.file_type for asset in list_dataset_assets() if asset.asset == dataset_asset.asset
        )
        rows.append(
            {
                "asset": dataset_asset.asset,
                "rows": len(frame),
                "columns": len(frame.columns),
                "start_date": str(frame["date"].min().date()),
                "end_date": str(frame["date"].max().date()),
                "formats": ", ".join(available_formats),
                "path": str(dataset_asset.file_path.relative_to(ROOT_DIR)),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def build_price_history() -> pd.DataFrame:
    frames = [load_asset_frame(dataset_asset.asset) for dataset_asset in _preferred_assets()]
    if not frames:
        return pd.DataFrame(columns=["date", "asset", "prices"])
    combined = pd.concat(frames, ignore_index=True)
    return combined[["date", "asset", "prices"]].sort_values(["date", "asset"]).reset_index(drop=True)


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    return RESULTS_DIR


@st.cache_data(show_spinner=False)
def read_notes() -> str:
    return NOTES_PATH.read_text(encoding="utf-8") if NOTES_PATH.exists() else ""


@st.cache_data(show_spinner=False)
def load_notes_sections() -> list[dict[str, str]]:
    notes = read_notes()
    if not notes:
        return []

    matches = list(re.finditer(r"^(\\section\{.*?\}|\\subsection\{.*?\})\s*$", notes, flags=re.MULTILINE))
    if not matches:
        return [{"title": "Notes", "body": notes}]

    sections: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(notes)
        title = _humanize_heading(match.group(1))
        body = notes[start:end].strip()
        sections.append({"title": title, "body": body})
    return sections
