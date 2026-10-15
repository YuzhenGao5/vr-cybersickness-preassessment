"""Acquire verified original VR.net logs sequentially with size/disk guards."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import sys
import traceback
import zipfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

from prepare_vrnet_game import download, sha256, within, write_log

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/raw/vrnet/rollercoaster_raw"
LOG = ROOT / "results/rollercoaster_download_log.txt"
MANIFEST = ROOT / "results/rollercoaster_download_manifest.json"
LIMITS = {"max_zip_bytes": 5_000_000_000, "min_free_bytes": 10_000_000_000}


def unpack(item, archive, target, handle):
    target.mkdir(parents=True, exist_ok=True)
    record = {"name": item["name"], "source_url": item["webViewLink"],
              "drive_file_id": item["id"], "archive_bytes": archive.stat().st_size,
              "sha256": sha256(archive), "destination": str(target)}
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.infolist() if not m.is_dir()]
        total = sum(m.file_size for m in members)
        if shutil.disk_usage(target).free - total < LIMITS["min_free_bytes"]:
            raise OSError(f"Extraction disk guard: uncompressed={total}, free={shutil.disk_usage(target).free}")
        write_log(LOG, f"EXTRACTION PLAN {item['name']}: files={len(members)}, uncompressed_bytes={total}")
        types = {}
        extracted = []
        for member in members:
            raw = member.filename.replace("\\", "/")
            p = PurePosixPath(raw)
            if p.is_absolute() or ".." in p.parts or any(":" in v for v in p.parts):
                raise ValueError(f"Unsafe archive member {raw!r}")
            suffix = p.suffix.lower()
            types[suffix] = types.get(suffix, 0) + 1
            if suffix in {".exe", ".dll", ".bat", ".cmd", ".ps1", ".msi"}:
                raise ValueError(f"Executable content is not expected in a data log; stopped: {raw}")
            output = within(target.joinpath(*p.parts), target)
            output.parent.mkdir(parents=True, exist_ok=True)
            # All archive entries are read to EOF: zipfile verifies CRC.
            if output.exists():
                import hashlib
                h = hashlib.sha256()
                with z.open(member) as reader:
                    for block in iter(lambda: reader.read(1024 * 1024), b""):
                        h.update(block)
                if output.stat().st_size != member.file_size or sha256(output) != h.hexdigest():
                    raise ValueError(f"Existing extracted data differs; preserved and stopped: {output}")
            else:
                partial = output.with_name(output.name + ".extract.part")
                with z.open(member) as reader, partial.open("wb") as writer:
                    shutil.copyfileobj(reader, writer, length=1024 * 1024)
                if partial.stat().st_size != member.file_size:
                    raise ValueError(f"Extracted size mismatch: {raw}")
                partial.replace(output)
            handle.writerow({"archive": item["name"], "member": raw, "file_bytes": member.file_size,
                             "compressed_bytes": member.compress_size, "crc32": f"{member.CRC:08x}"})
            extracted.append(raw)
        record.update({"status": "success", "file_count": len(members), "uncompressed_bytes": total,
                       "extension_counts": types, "csv_files": [f for f in extracted if f.lower().endswith(".csv")],
                       "all_entries_read_and_crc_checked": True})
    archive = within(archive, DEST / ".download_work")
    archive.unlink()
    record["zip_deleted_after_extraction"] = True
    write_log(LOG, f"EXTRACTION SUCCESS {item['name']}: types={types}; ZIP deleted; free={shutil.disk_usage(target).free}")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="All publicly listed data ZIPs, not the game installer 7z")
    args = parser.parse_args()
    source = json.loads((ROOT / "results/rollercoaster_sources/author_site_history_check.json").read_text(encoding="utf-8"))
    raw = source["original_raw_dataset_drive"]
    scores = raw["selfreport_candidates"][0]
    scene_files = sorted(raw["epic_children"][0]["files"], key=lambda v: int(v.get("size", 0)))
    scene_files = [f for f in scene_files if f["name"].lower().endswith(".zip")]
    selected = [scores] + (scene_files if args.all else scene_files[:1])
    write_log(LOG, "COMMAND " + json.dumps(sys.argv, ensure_ascii=False))
    write_log(LOG, "CLIENT_DATE 2026-10-10; source=official access-form public download link; no form submitted or agreement signed")
    records = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else []
    DEST.mkdir(parents=True, exist_ok=True)
    with gzip.open(ROOT / "results/rollercoaster_zip_members.csv.gz", "at", encoding="utf-8", newline="") as handle:
        fields = ["archive", "member", "file_bytes", "compressed_bytes", "crc32"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not records:
            writer.writeheader()
        for item in selected:
            if any(r.get("drive_file_id") == item["id"] and r.get("status") == "success" for r in records):
                write_log(LOG, f"ALREADY EXTRACTED {item['name']}; use verified manifest, no new transfer")
                continue
            try:
                game = {"game_name": "Epic_Roller_Coasters" if item != scores else "SelfReports",
                        "archive_name": item["name"], "drive_file_id": item["id"], "archive_size_bytes": int(item["size"])}
                archive = download(game, SimpleNamespace(work_dir=DEST / ".download_work", log=LOG), LIMITS)
                target = DEST / ("self_reports" if item == scores else Path(item["name"]).stem)
                records.append(unpack(item, archive, target, writer))
            except Exception:
                error = traceback.format_exc()
                records.append({"name": item["name"], "source_url": item["webViewLink"],
                                "drive_file_id": item["id"], "status": "failed", "error": error})
                write_log(LOG, "FAILED " + error)
                MANIFEST.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
                raise
            MANIFEST.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
