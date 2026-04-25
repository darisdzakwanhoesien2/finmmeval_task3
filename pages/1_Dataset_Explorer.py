from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from utils.data import (
    build_asset_catalog,
    build_normalized_dataset,
    build_price_history,
    load_asset_frame,
    load_normalized_asset_record,
)


st.set_page_config(page_title="Dataset Explorer", page_icon=":mag:")
st.title("Dataset Explorer")
st.caption("Inspect Task 3 daily trading bundles for BTC and TSLA")

catalog = build_asset_catalog()
assets = catalog["asset"].tolist()

selected_asset = st.sidebar.selectbox("Asset", assets)
frame = load_asset_frame(selected_asset)
normalized = build_normalized_dataset().loc[lambda data: data["asset"] == selected_asset].reset_index(drop=True)

stat_col_1, stat_col_2, stat_col_3, stat_col_4 = st.columns(4)
stat_col_1.metric("Rows", len(frame))
stat_col_2.metric("Columns", len(frame.columns))
stat_col_3.metric("First Date", str(pd.to_datetime(frame["date"]).min().date()))
stat_col_4.metric("Last Date", str(pd.to_datetime(frame["date"]).max().date()))

st.subheader("Schema")
schema_frame = pd.DataFrame(
    {
        "column": frame.columns,
        "dtype": [str(dtype) for dtype in frame.dtypes],
        "non_null": [int(frame[column].notna().sum()) for column in frame.columns],
        "sample_value": [str(frame.iloc[0][column])[:120] for column in frame.columns],
    }
)
st.dataframe(schema_frame, use_container_width=True, hide_index=True)

history = build_price_history()
asset_history = history.loc[history["asset"] == selected_asset].set_index("date")[["prices"]]
st.subheader(f"{selected_asset} Price Trend")
st.line_chart(asset_history.rename(columns={"prices": f"{selected_asset} price"}))

st.subheader("Bundle Browser")
row_index = st.slider("Row index", min_value=0, max_value=max(len(frame) - 1, 0), value=0)
record = load_normalized_asset_record(selected_asset, row_index)

left, right = st.columns([0.95, 1.05])
with left:
    st.markdown("**Bundle Metadata**")
    st.dataframe(
        pd.DataFrame(
            [
                {"field": "date", "value": record["date"]},
                {"field": "asset", "value": record["asset"]},
                {"field": "price", "value": record["prices"]},
                {"field": "momentum", "value": record["momentum"]},
                {"field": "news_items", "value": record["news_count"]},
                {"field": "news_characters", "value": record["news_length"]},
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("**Prompt-Ready Context**")
    st.text_area(
        "Trading bundle summary",
        value=record["prompt_context"],
        height=260,
        disabled=True,
        label_visibility="collapsed",
    )

with right:
    st.markdown("**News Narrative**")
    st.text_area(
        "News narrative",
        value=record["news_text"],
        height=420,
        disabled=True,
        label_visibility="collapsed",
    )

st.subheader("Agent Market Arena Request Payload")
st.code(json.dumps(record["request_payload"], ensure_ascii=False, indent=2, default=str), language="json")

st.subheader("Row Timeline")
timeline = normalized[["date", "prices", "news_count", "news_length"]].rename(
    columns={
        "date": "Date",
        "prices": "Price",
        "news_count": "News Items",
        "news_length": "News Characters",
    }
)
st.dataframe(timeline, use_container_width=True, hide_index=True)

st.subheader("Raw Row")
st.code(json.dumps(record["raw"], ensure_ascii=False, indent=2, default=str), language="json")
