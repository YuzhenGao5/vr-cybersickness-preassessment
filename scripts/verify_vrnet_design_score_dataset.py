"""Check export consistency and selected raw-source values; no model or downloads."""
import csv
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

csv.field_size_limit(2_147_483_647)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/vrnet_design_score_2026-10-10"


def load_csv(name):
    with (OUT / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    scores = load_csv("score_events.csv")
    ids = [r["event_id"] for r in scores]
    assert len(ids) == len(set(ids)) == summary["recorded_rating_rows"]
    coverage = Counter()
    for filename in ["scene_graphics_at_reports.csv", "interaction_at_reports.csv", "vrnet_score_parameter_dataset.csv"]:
        with (OUT / filename).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if filename.startswith("scene_"):
                assert not any(c.startswith("interaction__") for c in reader.fieldnames)
            actual_ids = []
            for row in reader:
                actual_ids.append(row["event_id"])
                assert row["training_ready"] == "False"
                assert row["strict_design_numeric_fields_verified"] == "0"
                provenance = json.loads(row["field_alignment_provenance_json"])
                for observation in provenance.values():
                    assert -1e-8 <= observation["age_seconds"] <= 1.0 + 1e-8
                if filename.startswith("vrnet_score_"):
                    for flag in ["scene_numeric_observation_present", "interaction_value_observation_present",
                                 "object_group_same_source_record", "camera_group_same_source_record", "gaze_group_same_source_record"]:
                        coverage[(row["dataset_family"], flag)] += row[flag] == "True"
            assert actual_ids == ids
    for family, flags in summary["numerical_and_coherent_coverage"].items():
        for flag, count in flags.items():
            assert coverage[(family, flag)] == count
    index = load_csv("global_index_inventory.csv")
    assert len(index) == summary["global_index_sessions"]
    dictionary = load_csv("field_dictionary.csv")
    assert all(r["strict_design_input_allowed"] == "no" for r in dictionary)
    assert all(r["is_interaction_parameter"] == "yes" for r in dictionary if r["category"].startswith("interaction_"))
    assert json.loads((OUT / "feature_groups.json").read_text(encoding="utf-8"))["strict_design_features"] == []
    for family, details in summary["families"].items():
        rows = [r for r in scores if r["dataset_family"] == family]
        assert len(rows) == details["recorded_rating_rows"]
        assert dict(Counter(str(float(r["score_value"])) for r in rows)) == details["rating_distribution"]
    candidates = [
        next(r for r in scores if r["dataset_family"] == "modern" and "::" not in r["source_file"]),
        next(r for r in scores if r["dataset_family"] == "modern" and "::" in r["source_file"]),
        next(r for r in scores if r["dataset_family"] == "native" and r["rating_source"] == "manual"),
        next(r for r in scores if r["dataset_family"] == "native" and r["rating_source"] == "auto"),
        next(r for r in scores if r["timestamp_quality"] == "rounded_scientific_not_aligned"),
    ]
    checks = []
    for event in candidates:
        locator = event["source_file"]
        archive = None
        if "::" in locator:
            archive_path, member = locator.split("::", 1)
            archive = zipfile.ZipFile(archive_path)
            stream = io.TextIOWrapper(archive.open(member), encoding="utf-8-sig", newline="")
        else:
            stream = Path(locator).open(encoding="utf-8-sig", newline="")
        try:
            modern = event["dataset_family"] == "modern"
            reader = csv.DictReader(stream) if modern else csv.reader(stream)
            for record, row in enumerate(reader, 1):
                if record == int(event["source_csv_record"]):
                    assert (row["MS_rating"] if modern else row[1].strip()) == event["score_raw"]
                    assert (row["timestamp"] if modern else row[0].strip()) == event["report_timestamp_raw"]
                    if not modern:
                        assert row[2].strip() == event["rating_source"]
                    checks.append(dict(event_id=event["event_id"], source_file=locator,
                                       source_csv_record=record, result="raw_score_timestamp_channel_match"))
                    break
            else:
                raise AssertionError("source record was not found")
        finally:
            stream.close()
            if archive:
                archive.close()
    result = dict(status="passed", scope="export consistency and five raw-record spot checks; not model validation",
                  exported_records=len(scores), full_index_records=len(index), dictionary_entries=len(dictionary),
                  matched_raw_records=checks, source_scan_failures=summary["source_scan_failures"])
    (OUT / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "matched_raw_records"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
