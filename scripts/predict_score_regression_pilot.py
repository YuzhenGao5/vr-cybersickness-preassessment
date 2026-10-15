"""Apply an exploratory saved model to numeric history-feature rows."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=root / "models/score_regression_pilot_2026-10-10/scene_candidates_lightgbm.pkl")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--row", type=int, default=0, help="Zero-based CSV row for one demonstration")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    table = pd.read_csv(args.input, encoding="utf-8-sig")
    if not 0 <= args.row < len(table):
        raise IndexError("row outside input table")
    missing = sorted(set(bundle["required_input_features"]) - set(table.columns))
    if missing:
        raise ValueError(f"missing required feature columns: {missing}")
    for field in bundle["feature_names"]:
        if field not in table:
            table[field] = np.nan  # Only features dropped by the fitted variance filter.
    example = table.iloc[[args.row]][bundle["feature_names"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if (example < 0).any().any():
        raise ValueError("nonnegative input contract violated")
    inputs = {field: None if pd.isna(example[field].iloc[0]) else float(example[field].iloc[0])
              for field in bundle["required_input_features"]}
    result = dict(model=str(args.model.resolve()), input_features=inputs,
                  predicted_ms_rating=float(bundle["pipeline"].predict(example)[0]),
                  observed_training_score_range=bundle["observed_target_range"],
                  warning=bundle["warning"],
                  demo_only_not_independent_validation="If input is the training matrix, this prediction is fitted-data illustration; use OOF results for evaluation.")
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
