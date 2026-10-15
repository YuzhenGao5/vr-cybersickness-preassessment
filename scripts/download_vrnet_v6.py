#!/usr/bin/env python3
"""Sequentially download VR.net game archives and retain selected data_file.csv files."""

from __future__ import annotations

import argparse
import shutil
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import pandas as pd
import requests

from download_vrnet_game_csvs_v3 import list_drive_archives, request_download


MAX_ZIP_BYTES = 5_000_000_000
MIN_FREE_BYTES = 10_000_000_000
MAX_ATTEMPTS = 12


def write_log(handle, message: str) -> None:
    print(message, flush=True)
    handle.write(message + "\n")
    handle.flush()


def download_archive(
    http: requests.Session,
    file_id: str,
    expected_size: int,
    part_path: Path,
    handle,
    game: str,
) -> None:
    if part_path.exists() and part_path.stat().st_size > expected_size:
        part_path.unlink()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        current = part_path.stat().st_size if part_path.exists() else 0
        if current == expected_size:
            break
        response = None
        try:
            response = request_download(http, file_id, range_start=current)
            append = current > 0 and response.status_code == 206
            if current and not append:
                write_log(handle, f"[{game}] resume not honored; restarting at byte 0")
                current = 0
            mode = "ab" if append else "wb"
            last_report = current
            with part_path.open(mode) as destination:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if not block:
                        continue
                    current += len(block)
                    if current > MAX_ZIP_BYTES:
                        raise ValueError(f"download crossed the 5 GB cap at {current} bytes")
                    destination.write(block)
                    if current - last_report >= 256 * 1024 * 1024:
                        write_log(handle, f"[{game}] progress={current}/{expected_size} bytes")
                        last_report = current
        except requests.RequestException as exc:
            current = part_path.stat().st_size if part_path.exists() else 0
            write_log(
                handle,
                f"[{game}] transfer attempt {attempt}/{MAX_ATTEMPTS} failed: "
                f"{type(exc).__name__}: {exc}; partial={current}",
            )
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(min(2 * attempt, 20))
        finally:
            if response is not None:
                response.close()
    actual = part_path.stat().st_size if part_path.exists() else 0
    if actual != expected_size:
        raise IOError(f"downloaded {actual} bytes; Drive metadata reports {expected_size}")


def find_member(names: list[str], session_name: str) -> str:
    matches: list[str] = []
    for member in names:
        parts = PurePosixPath(member.replace("\\", "/")).parts
        if parts and parts[-1].casefold() == "data_file.csv" and session_name in parts:
            matches.append(member)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one data_file.csv for {session_name}; found {len(matches)}: {matches[:10]}"
        )
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=Path("results/session_expansion_selection_v6.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/vrnet/v6_sessions"))
    parser.add_argument("--log", type=Path, default=Path("results/download_log_v6.txt"))
    args = parser.parse_args()
    selection = pd.read_csv(args.selection, encoding="utf-8-sig")
    required = {"game_name", "session_name", "scene_obj", "scene_cam", "MS_rating"}
    missing = sorted(required.difference(selection.columns))
    if missing:
        raise ValueError(f"selection is missing columns: {missing}")
    if not ((selection["scene_obj"] == 1) & (selection["scene_cam"] == 1) & (selection["MS_rating"] == 1)).all():
        raise ValueError("all selected sessions must have scene_obj=1, scene_cam=1, MS_rating=1")
    for column in (
        "download_status", "archive_name", "drive_file_id", "archive_size_bytes",
        "csv_size_bytes", "finished_utc", "failure_reason",
    ):
        if column not in selection.columns:
            selection[column] = pd.NA

    args.output_dir.mkdir(parents=True, exist_ok=True)
    work = args.output_dir / ".work"
    work.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    http = requests.Session()
    http.headers.update({"User-Agent": "Mozilla/5.0 (compatible; VRnet-v6-research/1.0)"})

    with args.log.open("a", encoding="utf-8", newline="\n") as handle:
        write_log(handle, "")
        write_log(handle, "=" * 80)
        write_log(handle, "VR.net v6 expansion download attempt")
        write_log(handle, f"Start UTC: {datetime.now(timezone.utc).isoformat()}")
        write_log(handle, "Command: .venv\\Scripts\\python.exe scripts\\download_vrnet_v6.py")
        write_log(handle, "Policy: one game ZIP at a time; cap 5,000,000,000 bytes; selected CSVs only; delete ZIP immediately.")
        try:
            archives = list_drive_archives(http)
            write_log(handle, f"Drive listing succeeded: {len(archives)} archives")
        except Exception:
            write_log(handle, "Drive listing failed with full traceback:")
            write_log(handle, traceback.format_exc())
            raise

        for game, rows in selection.groupby("game_name", sort=False):
            game = str(game)
            indices = list(rows.index)
            archive = archives.get(game)
            zip_path = work / f"{game}.zip"
            part_path = work / f"{game}.zip.part"
            destinations = {
                idx: args.output_dir / game / str(selection.at[idx, "session_name"]) / "data_file.csv"
                for idx in indices
            }
            missing_indices = [idx for idx in indices if not destinations[idx].is_file()]
            if not missing_indices:
                for idx in indices:
                    selection.at[idx, "download_status"] = "success_csv_already_present"
                    selection.at[idx, "csv_size_bytes"] = destinations[idx].stat().st_size
                write_log(handle, f"[{game}] SKIP: all {len(indices)} selected CSVs already exist")
                continue
            started = time.perf_counter()
            try:
                if archive is None:
                    raise FileNotFoundError(f"no public Drive archive metadata for {game}")
                archive_size = int(archive.get("size", 0))
                archive_name = str(archive.get("name", f"{game}.zip"))
                file_id = str(archive["id"])
                for idx in indices:
                    selection.at[idx, "archive_name"] = archive_name
                    selection.at[idx, "drive_file_id"] = file_id
                    selection.at[idx, "archive_size_bytes"] = archive_size
                if archive_size <= 0:
                    raise ValueError("Drive metadata has no positive archive size")
                if archive_size > MAX_ZIP_BYTES:
                    for idx in missing_indices:
                        selection.at[idx, "download_status"] = "skipped_over_5gb"
                    write_log(handle, f"[{game}] SKIP: ZIP size {archive_size} exceeds 5 GB")
                    continue
                free = shutil.disk_usage(args.output_dir).free
                if free < MIN_FREE_BYTES:
                    for idx in missing_indices:
                        selection.at[idx, "download_status"] = "skipped_free_space_under_10gb"
                    write_log(handle, f"[{game}] SKIP: free disk {free} is below 10 GB")
                    continue
                write_log(
                    handle,
                    f"[{game}] START archive={archive_name} size={archive_size} free={free}; "
                    f"extracting {len(missing_indices)} selected sessions",
                )
                download_archive(http, file_id, archive_size, part_path, handle, game)
                part_path.replace(zip_path)
                if not zipfile.is_zipfile(zip_path):
                    raise zipfile.BadZipFile("downloaded content is not a ZIP")
                with zipfile.ZipFile(zip_path) as bundle:
                    names = bundle.namelist()
                    for idx in missing_indices:
                        session_name = str(selection.at[idx, "session_name"])
                        member = find_member(names, session_name)
                        info = bundle.getinfo(member)
                        destination = destinations[idx]
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        partial = destination.with_suffix(".csv.part")
                        with bundle.open(info) as source, partial.open("wb") as target:
                            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                        if partial.stat().st_size != info.file_size:
                            raise IOError(
                                f"{session_name}: extracted {partial.stat().st_size}; ZIP reports {info.file_size}"
                            )
                        partial.replace(destination)
                        selection.at[idx, "download_status"] = "success_csv_extracted_zip_deleted"
                        selection.at[idx, "csv_size_bytes"] = destination.stat().st_size
                        write_log(handle, f"[{game}] extracted {session_name}: {destination.stat().st_size} bytes")
                write_log(handle, f"[{game}] SUCCESS elapsed={time.perf_counter()-started:.1f}s")
            except Exception as exc:
                for idx in missing_indices:
                    if pd.isna(selection.at[idx, "download_status"]):
                        selection.at[idx, "download_status"] = "failed"
                        selection.at[idx, "failure_reason"] = f"{type(exc).__name__}: {exc}"
                write_log(handle, f"[{game}] FAILED: {type(exc).__name__}: {exc}")
                write_log(handle, traceback.format_exc())
            finally:
                for disposable in (zip_path, part_path):
                    if disposable.exists():
                        disposable.unlink()
                        write_log(handle, f"[{game}] deleted temporary archive: {disposable}")
                now = datetime.now(timezone.utc).isoformat()
                for idx in indices:
                    selection.at[idx, "finished_utc"] = now
                selection.to_csv(args.selection, index=False, encoding="utf-8-sig")

        successes = int(selection["download_status"].astype(str).str.startswith("success").sum())
        failures = int(selection["download_status"].astype(str).eq("failed").sum())
        remaining = list(work.glob("*"))
        write_log(handle, f"Finished UTC: {datetime.now(timezone.utc).isoformat()}")
        write_log(handle, f"Successful selected CSVs: {successes}/{len(selection)}; failed={failures}")
        write_log(handle, f"Temporary ZIP/partial files remaining: {len(remaining)}")
    return 0 if successes == len(selection) else 2


if __name__ == "__main__":
    raise SystemExit(main())
