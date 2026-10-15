#!/usr/bin/env python3
"""Inspect the locally downloaded EyeNavGS dataset and write an audit report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw" / "eyenavgS"
DEFAULT_REPORT = REPO_ROOT / "results" / "eyenavgs_check.json"
CANONICAL_SCENES = {
    "alameda", "berlin", "bicycle", "drjohnson", "london", "nyc",
    "playroom", "room", "stump", "train", "treehill", "truck",
}
POSE_COLUMNS = {
    "ViewIndex", "PositionX", "PositionY", "PositionZ",
    "QuaternionX", "QuaternionY", "QuaternionZ", "QuaternionW",
}
TIME_COLUMNS = ("Timestamp", "timestep")
SOURCES = [
    {
        "site": "Rutgers",
        "url": "https://github.com/symmru/EyeNavGS_Rutgers_Dataset",
        "commit": "7f264fda0223cbc74e191f115ef9fd3b580d8cf4",
        "license": "Apache-2.0",
    },
    {
        "site": "NTHU",
        "url": "https://github.com/sawalee0811/EyeNavGS_NTHU_Dataset",
        "commit": "608790e7d932534f18a2a124887605d1f6c85384",
        "license": "Apache-2.0",
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def site_for(path: Path) -> str:
    lowered = {part.casefold() for part in path.parts}
    if "rutgers" in lowered:
        return "Rutgers"
    if "nthu" in lowered:
        return "NTHU"
    return "unknown"


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as stream:
        return next(csv.reader(stream))


def is_trace(path: Path, columns: list[str]) -> bool:
    return path.stem.casefold().startswith("user") and POSE_COLUMNS.issubset(columns)


def inspect_deep(path: Path, time_column: str) -> dict[str, Any]:
    columns = [*sorted(POSE_COLUMNS), time_column]
    frame = pd.read_csv(path, usecols=columns)
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    finite_rows = np.isfinite(values).all(axis=1)
    view = numeric["ViewIndex"].to_numpy(dtype=np.float64)
    pair_rows = (len(frame) // 2) * 2
    alternating_pairs = 0
    quaternion_pair_max_difference = None
    if pair_rows:
        first_views = view[:pair_rows:2]
        second_views = view[1:pair_rows:2]
        valid_pairs = (
            np.isfinite(first_views)
            & np.isfinite(second_views)
            & (first_views != second_views)
            & np.isin(first_views, (0.0, 1.0))
            & np.isin(second_views, (0.0, 1.0))
        )
        alternating_pairs = int(valid_pairs.sum())
        q_cols = ["QuaternionX", "QuaternionY", "QuaternionZ", "QuaternionW"]
        first_q = numeric.loc[: pair_rows - 1, q_cols].to_numpy(dtype=np.float64)[::2]
        second_q = numeric.loc[: pair_rows - 1, q_cols].to_numpy(dtype=np.float64)[1::2]
        pair_finite = np.isfinite(first_q).all(axis=1) & np.isfinite(second_q).all(axis=1)
        if pair_finite.any():
            direct = np.linalg.norm(first_q[pair_finite] - second_q[pair_finite], axis=1)
            negated = np.linalg.norm(first_q[pair_finite] + second_q[pair_finite], axis=1)
            quaternion_pair_max_difference = float(np.minimum(direct, negated).max())

    left = numeric[numeric["ViewIndex"] == 0.0]
    timestamps = left[time_column].to_numpy(dtype=np.float64)
    finite_time = timestamps[np.isfinite(timestamps)]
    differences = np.diff(finite_time)
    return {
        "rows": int(len(frame)),
        "finite_rows": int(finite_rows.sum()),
        "candidate_stereo_pairs": int(pair_rows // 2),
        "alternating_stereo_pairs": alternating_pairs,
        "left_eye_rows": int(len(left)),
        "timestamp_resets": int(np.sum(differences < 0.0)),
        "nonpositive_timestamp_steps": int(np.sum(differences <= 0.0)),
        "positive_dt_median_ms": (
            float(np.median(differences[differences > 0.0]))
            if np.any(differences > 0.0)
            else None
        ),
        "quaternion_pair_max_sign_invariant_difference": quaternion_pair_max_difference,
    }


def inspect_dataset(data_dir: Path, deep: bool, list_files: bool) -> dict[str, Any]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"EyeNavGS directory does not exist: {data_dir}")

    files = sorted(path for path in data_dir.rglob("*") if path.is_file())
    csv_files = [path for path in files if path.suffix.casefold() == ".csv"]
    extensions = Counter(path.suffix.casefold() or "<none>" for path in files)
    schemas: Counter[tuple[str, ...]] = Counter()
    schema_examples: dict[tuple[str, ...], str] = {}
    valid_traces: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    invalid_csv: list[dict[str, str]] = []

    for path in csv_files:
        relative = path.relative_to(data_dir).as_posix()
        try:
            columns = read_header(path)
        except Exception as exc:
            invalid_csv.append({"path": relative, "reason": f"header read failed: {exc}"})
            continue
        schema = tuple(columns)
        schemas[schema] += 1
        schema_examples.setdefault(schema, relative)
        if not is_trace(path, columns):
            excluded.append({"path": relative, "reason": "metadata or non-trace CSV"})
            continue
        scene = path.parent.name.casefold()
        if scene not in CANONICAL_SCENES:
            excluded.append({"path": relative, "reason": f"non-canonical scene '{scene}'"})
            continue
        time_matches = [name for name in TIME_COLUMNS if name in columns]
        if len(time_matches) != 1:
            invalid_csv.append({"path": relative, "reason": "missing or ambiguous time column"})
            continue
        match = re.match(r"^(user[^_]+)_", path.stem, flags=re.IGNORECASE)
        if not match:
            invalid_csv.append({"path": relative, "reason": "cannot parse participant ID"})
            continue
        valid_traces.append(
            {
                "path": relative,
                "site": site_for(path.relative_to(data_dir)),
                "scene": scene,
                "participant": match.group(1).casefold(),
                "time_column": time_matches[0],
                "bytes": path.stat().st_size,
            }
        )

    deep_totals: dict[str, Any] | None = None
    deep_warnings: list[dict[str, Any]] = []
    if deep and valid_traces:
        aggregates: dict[str, float] = defaultdict(float)
        dt_by_site: dict[str, list[float]] = defaultdict(list)
        for index, item in enumerate(valid_traces, start=1):
            details = inspect_deep(data_dir / item["path"], item["time_column"])
            for field in (
                "rows", "finite_rows", "candidate_stereo_pairs", "alternating_stereo_pairs",
                "left_eye_rows", "timestamp_resets", "nonpositive_timestamp_steps",
            ):
                aggregates[field] += details[field]
            if details["positive_dt_median_ms"] is not None:
                dt_by_site[item["site"]].append(details["positive_dt_median_ms"])
            if (
                details["rows"] % 2
                or details["alternating_stereo_pairs"] != details["candidate_stereo_pairs"]
                or details["finite_rows"] != details["rows"]
                or details["timestamp_resets"]
            ):
                deep_warnings.append({"path": item["path"], **details})
            if index % 100 == 0:
                print(f"  deep check: {index}/{len(valid_traces)} traces", flush=True)
        deep_totals = {
            key: int(value) for key, value in aggregates.items()
        }
        deep_totals["median_trace_dt_ms_by_site"] = {
            site: float(np.median(values)) for site, values in sorted(dt_by_site.items())
        }

    site_scene_counts: dict[str, Counter[str]] = defaultdict(Counter)
    participants: dict[str, set[str]] = defaultdict(set)
    for item in valid_traces:
        site_scene_counts[item["site"]][item["scene"]] += 1
        participants[item["site"]].add(item["participant"])

    gate_passed = bool(valid_traces) and not invalid_csv
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "generated_at_utc": utc_now(),
        "data_root": str(data_dir.resolve()),
        "source_type": "EyeNavGS",
        "synthetic": False,
        "official_project_url": "https://symmru.github.io/EyeNavGS/",
        "sources": SOURCES,
        "gate": {
            "passed": gate_passed,
            "requirement": "at least one canonical trace with time, position and head quaternion",
            "reason": (
                f"{len(valid_traces)} canonical traces matched the required schema"
                if gate_passed
                else "no usable traces or one or more canonical trace CSVs had invalid schema"
            ),
        },
        "inventory": {
            "file_count": len(files),
            "total_bytes": sum(path.stat().st_size for path in files),
            "extension_counts": dict(sorted(extensions.items())),
            "csv_count": len(csv_files),
            "valid_trace_count": len(valid_traces),
            "valid_trace_bytes": sum(item["bytes"] for item in valid_traces),
            "participants_by_site": {
                site: len(values) for site, values in sorted(participants.items())
            },
            "scene_trace_counts_by_site": {
                site: dict(sorted(counts.items()))
                for site, counts in sorted(site_scene_counts.items())
            },
        },
        "schemas": [
            {"count": count, "columns": list(schema), "example": schema_examples[schema]}
            for schema, count in schemas.most_common()
        ],
        "known_column_aliases": {"time_ms": list(TIME_COLUMNS)},
        "stereo_semantics": (
            "Each rendered frame has left/right eye rows. Position is eye position; preprocessing "
            "must pair rows before estimating head-centre motion. Quaternion is head orientation."
        ),
        "excluded_files": excluded,
        "invalid_csv": invalid_csv,
        "deep_check": deep_totals,
        "deep_warnings": deep_warnings,
    }
    if list_files:
        report["files"] = [
            {"path": path.relative_to(data_dir).as_posix(), "bytes": path.stat().st_size}
            for path in files
        ]
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--deep", action="store_true", help="Read all canonical traces and validate stereo/time structure")
    parser.add_argument("--list-files", action="store_true", help="Include every local file in the JSON report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = inspect_dataset(args.data_dir, args.deep, args.list_files)
        atomic_json(args.report, report)
    except Exception as exc:
        print(f"EyeNavGS check failed: {exc}", file=sys.stderr)
        return 1

    inventory = report["inventory"]
    print("=" * 60)
    print("EyeNavGS dataset audit")
    print("=" * 60)
    print(f"Canonical traces : {inventory['valid_trace_count']}")
    print(f"Participants     : {inventory['participants_by_site']}")
    print(f"Scenes           : {len(CANONICAL_SCENES)}")
    print(f"Gate passed      : {report['gate']['passed']}")
    print(f"Report           : {args.report.resolve()}")
    return 0 if report["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

