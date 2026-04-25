Your current codebase is already a good exploration layer, but research-paper quality analysis needs one more layer on top: a disciplined evaluation protocol.

**What your code already gives you**
[utils/data.py](/Users/darisdzakwanhoesien/Documents/project_documentation/codebase/codex_based/finmmeval_task3/utils/data.py:54) is the foundation. It loads the canonical asset files, normalizes dates, parses the long `news` field, and builds a prompt-ready bundle in `normalize_trading_row()` at [utils/data.py](/Users/darisdzakwanhoesien/Documents/project_documentation/codebase/codex_based/finmmeval_task3/utils/data.py:89). That means each row is already a clean “unit of analysis”: one day, one asset, one price, one synthesized news context.

The app pages sit on top of that:
- [app.py](/Users/darisdzakwanhoesien/Documents/project_documentation/codebase/codex_based/finmmeval_task3/app.py:18) gives you corpus-level framing: row counts, asset coverage, date span, and price history.
- [pages/1_Dataset_Explorer.py](/Users/darisdzakwanhoesien/Documents/project_documentation/codebase/codex_based/finmmeval_task3/pages/1_Dataset_Explorer.py:49) lets you inspect individual bundles and compare them over time.
- [pages/2_Trading_Playground.py](/Users/darisdzakwanhoesien/Documents/project_documentation/codebase/codex_based/finmmeval_task3/pages/2_Trading_Playground.py:82) gives you a manual experiment harness for prompting a hosted HF model.
- [pages/3_Notes_Review.py](/Users/darisdzakwanhoesien/Documents/project_documentation/codebase/codex_based/finmmeval_task3/pages/3_Notes_Review.py:18) anchors the implementation to the benchmark narrative in `notes.md`.

That is enough to support a paper, but only if you analyze it in a structured way.

**How to turn this into research-paper quality**
Use three analysis levels.

1. Dataset analysis  
Start from [app.py](/Users/darisdzakwanhoesien/Documents/project_documentation/codebase/codex_based/finmmeval_task3/app.py:67) and [pages/1_Dataset_Explorer.py](/Users/darisdzakwanhoesien/Documents/project_documentation/codebase/codex_based/finmmeval_task3/pages/1_Dataset_Explorer.py:33). In the paper, describe:
- Temporal coverage: exact start and end dates per asset.
- Cross-asset balance: row counts for `BTC` vs `TSLA`.
- Context density: use `news_count` and `news_length` from [utils/data.py](/Users/darisdzakwanhoesien/Documents/project_documentation/codebase/codex_based/finmmeval_task3/utils/data.py:102) to report how much textual evidence each bundle contains.
- Price dynamics: use [build_price_history()](/Users/darisdzakwanhoesien/Documents/project_documentation/codebase/codex_based/finmmeval_task3/utils/data.py:137) to show volatility windows, regime shifts, and whether some periods are much noisier than others.

This becomes your “Dataset” section: not just “we have 545 rows per asset,” but “the benchmark spans August 1, 2024 to January 27, 2026, includes two assets with equal row counts, and provides long-form narrative context per day.”

2. Model evaluation analysis  
Right now [pages/2_Trading_Playground.py](/Users/darisdzakwanhoesien/Documents/project_documentation/codebase/codex_based/finmmeval_task3/pages/2_Trading_Playground.py:93) is interactive, which is great for probing behavior but not enough for a paper by itself. For paper-quality analysis, keep the same prompt structure and hosted inference path from [utils/hf_inference.py](/Users/darisdzakwanhoesien/Documents/project_documentation/codebase/codex_based/finmmeval_task3/utils/hf_inference.py:30), then run it systematically across all rows.

For each row, log:
- date
- asset
- input bundle
- model name
- full prompt
- model response
- parsed action: `BUY`, `HOLD`, or `SELL`
- confidence
- rationale length
- latency
- any JSON parse failure

Then analyze:
- Action distribution by asset and by month
- Stability: does the same prompt produce the same action across reruns?
- Calibration: when confidence is high, are outcomes actually better?
- Robustness: how often does the model fail to follow the required JSON/action schema?
- Sensitivity: how much does the action change when you vary temperature, provider, or prompt wording?

That gives you a real “Experiments” section rather than just screenshots of the app.

3. Outcome and error analysis  
This is the part that makes it feel like a paper instead of a demo. Use the action outputs to compute ex-post trading-style results:
- next-day return after `BUY`
- next-day return after `SELL`
- return of model policy vs buy-and-hold baseline
- hit rate by asset
- average return conditional on confidence bucket
- confusion-style analysis between market movement and action type

Then do qualitative error analysis:
- Cases where news was strongly positive but the model chose `SELL`
- Cases where there was little signal but the model was overconfident
- Cases where BTC and TSLA behave differently under similar prompt conditions
- Cases where the model cites irrelevant parts of the news blob

Your notes already point toward provenance, governance, and reproducibility. Tie those to the model outputs: if the rationale is not grounded in the supplied bundle, call that out as a failure mode.

**How I’d frame the paper using your current app**
A strong paper structure for this codebase would be:

- Task definition  
  “Daily end-of-day action recommendation from structured market bundle plus synthesized narrative context.”

- Dataset section  
  Derived from [build_asset_catalog()](/Users/darisdzakwanhoesien/Documents/project_documentation/codebase/codex_based/finmmeval_task3/utils/data.py:115), [build_price_history()](/Users/darisdzakwanhoesien/Documents/project_documentation/codebase/codex_based/finmmeval_task3/utils/data.py:137), and the row-normalization logic in [normalize_trading_row()](/Users/darisdzakwanhoesien/Documents/project_documentation/codebase/codex_based/finmmeval_task3/utils/data.py:89).

- Experimental setup  
  Use the exact hosted inference pattern in [stream_chat_completion()](/Users/darisdzakwanhoesien/Documents/project_documentation/codebase/codex_based/finmmeval_task3/utils/hf_inference.py:30), with fixed prompts and declared decoding settings from [pages/2_Trading_Playground.py](/Users/darisdzakwanhoesien/Documents/project_documentation/codebase/codex_based/finmmeval_task3/pages/2_Trading_Playground.py:66).

- Quantitative results  
  Accuracy is not enough here. Focus on action validity, decision consistency, return-based metrics, and robustness.

- Qualitative analysis  
  Pull exemplar rows from [pages/1_Dataset_Explorer.py](/Users/darisdzakwanhoesien/Documents/project_documentation/codebase/codex_based/finmmeval_task3/pages/1_Dataset_Explorer.py:49) and discuss why the model succeeded or failed.

- Limitations  
  Your current dataset has only two assets and uses synthesized `news`, so note that this is a benchmark for grounded decision behavior, not proof of deployable trading alpha.

The main gap in your existing code is that it is an analysis interface, not yet a batch experiment runner. If you want, I can build the next step for you: a reproducible evaluation script that runs the HF model over every row, stores outputs to CSV/JSONL, and computes paper-ready tables and charts.