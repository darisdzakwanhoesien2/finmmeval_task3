from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.data import build_normalized_dataset
from utils.evaluation import (
    evaluate_record,
    generate_results_discussion,
    load_saved_results,
    load_run_summary,
    save_experiment_results,
    save_run_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible FinMMEval Task 3 hosted-model evaluations.")
    parser.add_argument("--experiment-name", required=True, help="Human-readable label for the saved run.")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--provider", default="auto")
    parser.add_argument("--hf-provider-hint", default="auto")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9, dest="top_p")
    parser.add_argument("--max-tokens", type=int, default=700, dest="max_tokens")
    parser.add_argument("--asset", action="append", dest="assets", help="Optional asset filter; repeatable.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit after filtering.")
    parser.add_argument("--reruns", type=int, default=1, help="Number of repeated runs per row for stability analysis.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("OPENROUTER_API_KEY is required for provider=openrouter.", file=sys.stderr)
            return 1
    elif args.provider == "huggingface":
        api_key = os.getenv("HF_TOKEN")
        if not api_key:
            print("HF_TOKEN is required for provider=huggingface.", file=sys.stderr)
            return 1
    elif args.provider == "lmstudio":
        api_key = ""
    else:
        print(f"Unsupported provider: {args.provider}", file=sys.stderr)
        return 1

    dataset = build_normalized_dataset().copy()
    if args.assets:
        dataset = dataset.loc[dataset["asset"].isin(args.assets)].reset_index(drop=True)
    if args.limit is not None:
        dataset = dataset.head(args.limit).reset_index(drop=True)

    if dataset.empty:
        print("No rows matched the requested filters.", file=sys.stderr)
        return 1

    records: list[dict[str, object]] = []
    total = len(dataset) * max(args.reruns, 1)
    current = 0

    for rerun_id in range(max(args.reruns, 1)):
        for record in dataset.to_dict(orient="records"):
            current += 1
            print(
                f"[{current}/{total}] evaluating asset={record['asset']} date={record['date']} rerun={rerun_id}",
                flush=True,
            )
            result = evaluate_record(
                api_key=api_key,
                model=args.model,
                provider=args.provider,
                hf_provider_hint=args.hf_provider_hint,
                record=record,
                rerun_id=rerun_id,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
            records.append(result)

    results = pd.DataFrame(records)
    run_dir = save_experiment_results(
        experiment_name=args.experiment_name,
        results=results,
        metadata={
            "model": args.model,
            "provider": args.provider,
            "hf_provider_hint": args.hf_provider_hint,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "assets": args.assets or "ALL",
            "limit": args.limit,
            "reruns": args.reruns,
            "launched_from": "scripts/run_evaluation.py",
        },
    )
    run_name = run_dir.name
    saved_results = load_saved_results(run_name)
    report = generate_results_discussion(
        run_name=run_name,
        results=saved_results,
        run_summary=load_run_summary(run_name),
    )
    report_path = save_run_report(run_name, report)
    print(f"Saved results to {run_dir}")
    print(f"Saved results and discussion to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
