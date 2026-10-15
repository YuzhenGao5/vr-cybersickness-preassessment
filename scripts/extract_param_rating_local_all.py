#!/usr/bin/env python3
"""Extract all unique local modern VR.net CSV sources, including ZIP members."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
import logging
from pathlib import Path
import platform
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd

from extract_param_rating_windows import process_session, sha256


class LocalCsvSource:
    """A canonical locator with explicit identity and a streaming CSV reader."""

    def __init__(self, record):
        self.record = record
        self.parent = Path(record["game_name"]) / record["session_id"]

    def resolve(self):
        return self.record["source_locator"]

    def stat(self):
        return SimpleNamespace(st_size=int(self.record["size_bytes"]))

    @contextmanager
    def open(self):
        if self.record["source_kind"] == "zip_member":
            with zipfile.ZipFile(self.record["archive_path"]) as archive:
                member = archive.getinfo(self.record["member_path"])
                if member.file_size != self.stat().st_size:
                    raise ValueError("ZIP member size changed since source inventory")
                with archive.open(member) as stream:
                    yield stream
        else:
            path = Path(self.record["source_locator"])
            if path.stat().st_size != self.stat().st_size:
                raise ValueError("CSV size changed since source inventory")
            with path.open("rb") as stream:
                yield stream


def read_source_csv(source, **kwargs):
    if kwargs.get("chunksize"):
        def chunks():
            with source.open() as stream:
                with pd.read_csv(stream, **kwargs) as reader:
                    yield from reader
        return chunks()
    with source.open() as stream:
        return pd.read_csv(stream, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=Path("results/param_rating_local_all_2026-10-10/source_inventory.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/param_rating_local_all_2026-10-10.json"))
    parser.add_argument("--output", type=Path, default=Path("features/param_rating_local_all_2026-10-10.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("features/param_rating_local_all_2026-10-10_preparation.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("features/param_rating_local_all_by_session_2026-10-10"))
    parser.add_argument("--log", type=Path, default=Path("results/param_rating_local_all_2026-10-10/extraction_log.txt"))
    parser.add_argument("--partition-count", type=int, default=1, help="Disjoint local parsing partitions; does not change feature/label rules")
    parser.add_argument("--partition-index", type=int, default=0)
    args = parser.parse_args()
    if not Path("PROTOCOL.md").is_file():
        raise SystemExit("PROTOCOL.md missing: experiments forbidden")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source_inventory = json.loads(args.sources.read_text(encoding="utf-8"))
    sources = source_inventory["sources"]
    if len({r["session_id"] for r in sources}) != len(sources):
        raise SystemExit("Canonical source inventory contains duplicate sessions")
    if args.partition_count < 1 or not 0 <= args.partition_index < args.partition_count:
        raise SystemExit("Invalid parsing partition")
    sources = [r for i, r in enumerate(sources) if i % args.partition_count == args.partition_index]
    for path in (args.output.parent, args.manifest.parent, args.cache_dir, args.log.parent):
        path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(args.log, encoding="utf-8"), logging.StreamHandler()])
    logger = logging.getLogger("param_rating_local_all")
    cache_definition = {"config_sha256": sha256(args.config),
                        "extraction_core_sha256": sha256(Path(__file__).with_name("extract_param_rating_windows.py")),
                        "extraction_wrapper_sha256": sha256(Path(__file__))}
    parts, inventory = [], []
    for index, source in enumerate(sources, 1):
        session = source["session_id"]
        record = dict(source)
        if not source["matrix_eligible"]:
            record.update(status="skipped_missing_columns", missing_columns=[
                c for c in ("timestamp", "camera_name", "p_matrix", "v_matrix", "m_matrix", "MS_rating")
                if c not in source["columns"]])
            inventory.append(record)
            logger.info("SKIP %s/%s %s matrix unavailable", index, len(sources), session)
            continue
        csv_path, json_path = args.cache_dir / f"{session}.csv", args.cache_dir / f"{session}.json"
        identity = dict(cache_definition, source_locator=source["source_locator"],
                        known_sha256=source["known_sha256"], size_bytes=source["size_bytes"],
                        game_name=source["game_name"], participant_id=source["participant_id"])
        logger.info("START %s/%s %s kind=%s bytes=%s", index, len(sources), session,
                    source["source_kind"], source["size_bytes"])
        try:
            cached = json.loads(json_path.read_text(encoding="utf-8")) if json_path.is_file() else {}
            if cached.get("cache_identity") == identity and csv_path.is_file() and cached.get("status") == "processed":
                part = pd.read_csv(csv_path, encoding="utf-8-sig")
                if len(part) != cached["output_windows"] or sha256(csv_path) != cached["cache_csv_sha256"]:
                    raise ValueError("Session cache content does not match completion record")
                record = cached
                logger.info("RESUME verified session cache %s windows=%s", session, len(part))
            else:
                rows, extracted = process_session(LocalCsvSource(source), config, logger, csv_reader=read_source_csv)
                record.update(extracted)
                if record["status"] != "processed":
                    raise ValueError(f"Unexpected extractor status {record['status']}")
                if record["raw_rows"] != source["raw_rows_expected"]:
                    raise ValueError(f"Raw row count mismatch: {record['raw_rows']} != {source['raw_rows_expected']}")
                if record["real_rating_events"] != source["rating_events_expected"]:
                    raise ValueError(f"Rating count mismatch: {record['real_rating_events']} != {source['rating_events_expected']}")
                part = pd.DataFrame(rows)
                if not part.empty:
                    part["participant_id"] = source["participant_id"]
                    part.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.17g")
                    record.update(cache_identity=identity, cache_csv_sha256=sha256(csv_path))
                json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            if not part.empty:
                parts.append(part)
            inventory.append(record)
        except Exception as exc:
            logger.exception("FAILED %s", session)
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            inventory.append(record)
        args.manifest.write_text(json.dumps(dict(status="running", session_inventory=inventory), ensure_ascii=False, indent=2), encoding="utf-8")
    if not parts:
        raise SystemExit("No extractable local matrix windows; see extraction log")
    frame = pd.concat(parts, ignore_index=True)
    if not frame["window_id"].is_unique:
        raise ValueError("Output has duplicate window identifiers")
    frame.to_csv(args.output, index=False, encoding="utf-8-sig", float_format="%.17g")
    usable = frame[frame["window_has_any_feature"] & frame["ms_rating"].notna()]
    distance = usable["rating_abs_distance_sec"]
    features = {}
    for column in config["feature_columns"]:
        values = pd.to_numeric(usable[column], errors="coerce")
        features[column] = dict(nonempty=int(values.notna().sum()), missing_fraction=float(values.isna().mean()),
                                min=float(values.min()) if values.notna().any() else None,
                                max=float(values.max()) if values.notna().any() else None,
                                unique_nonempty=int(values.nunique()))
    failed = sum(r["status"] == "failed" for r in inventory)
    summary = dict(status="complete" if not failed else "partial_some_sources_failed",
                   scope="all unique local modern CSV files and existing ZIP CSV members; not all remote VR.net",
                   config=config, config_sha256=sha256(args.config), source_inventory_path=str(args.sources.resolve()),
                   source_inventory_sha256=sha256(args.sources), output_csv=str(args.output.resolve()),
                   output_sha256=sha256(args.output), python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                   source_csv_count=len(sources), source_game_count=len({r["game_name"] for r in sources}),
                   processed_session_count=sum(r["status"] == "processed" for r in inventory),
                   skipped_session_count=sum(r["status"].startswith("skipped") for r in inventory), failed_session_count=failed,
                   output_windows=len(frame), usable_windows=len(usable), usable_sessions=int(usable.session_id.nunique()),
                   usable_games=int(usable.game_name.nunique()), real_rating_events_in_processed_sources=sum(r.get("real_rating_events", 0) for r in inventory),
                   paired_unique_rating_events=int(usable.rating_event_id.nunique()),
                   label_distance_seconds=dict(median=float(distance.median()), p90=float(distance.quantile(.9)), max=float(distance.max()),
                                               over_30s=int((distance > 30).sum()), over_60s=int((distance > 60).sum()),
                                               future_windows=int(usable.label_is_future.sum())),
                   feature_summary=features, unavailable_optional_features=["motion_flow", "depth"],
                   optional_feature_evidence="All 155 local CSV headers checked; no raw flow/depth fields.",
                   skip_games=dict(Counter(r["game_name"] for r in inventory if r["status"].startswith("skipped"))),
                   source_summary=source_inventory["summary"], session_inventory=inventory)
    args.manifest.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("COMPLETE status=%s source_sessions=%s processed=%s windows=%s usable=%s ratings=%s games=%s",
                summary["status"], len(sources), summary["processed_session_count"], len(frame), len(usable),
                summary["paired_unique_rating_events"], summary["usable_games"])
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
