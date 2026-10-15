#!/usr/bin/env python3
"""Sequentially download selected VR.net game ZIPs and retain one CSV per game."""

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


def log(handle, message: str) -> None:
    handle.write(message + "\n")
    handle.flush()
    print(message, flush=True)


def selected_member(names: list[str], session_name: str) -> str:
    matches = []
    for name in names:
        path = PurePosixPath(name.replace("\\", "/"))
        if path.name.casefold() == "data_file.csv" and session_name in path.parts:
            matches.append(name)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one data_file.csv for {session_name}, found {len(matches)}: {matches[:10]}"
        )
    return matches[0]


def download_archive(
    http: requests.Session,
    file_id: str,
    expected_size: int,
    part_path: Path,
    log_handle,
    game: str,
) -> None:
    downloaded = part_path.stat().st_size if part_path.exists() else 0
    if downloaded > expected_size:
        part_path.unlink()
        downloaded = 0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if downloaded == expected_size:
            break
        response = None
        try:
            resume_from = part_path.stat().st_size if part_path.exists() else 0
            response = request_download(http, file_id, range_start=resume_from)
            append = resume_from > 0 and response.status_code == 206
            if resume_from and not append:
                log(log_handle, f"[{game}] server did not honor resume; restarting from byte 0")
                resume_from = 0
            mode = "ab" if append else "wb"
            downloaded = resume_from
            last_report = downloaded
            with part_path.open(mode) as output:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if not block:
                        continue
                    downloaded += len(block)
                    if downloaded > MAX_ZIP_BYTES:
                        raise ValueError(f"download crossed 5 GB limit at {downloaded} bytes")
                    output.write(block)
                    if downloaded - last_report >= 128 * 1024 * 1024:
                        log(log_handle, f"[{game}] progress={downloaded}/{expected_size} bytes")
                        last_report = downloaded
        except requests.RequestException as exc:
            downloaded = part_path.stat().st_size if part_path.exists() else 0
            log(
                log_handle,
                f"[{game}] transfer attempt {attempt}/{MAX_ATTEMPTS} failed: "
                f"{type(exc).__name__}: {exc}; partial={downloaded}",
            )
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(min(2 * attempt, 20))
        finally:
            if response is not None:
                response.close()
    final_size = part_path.stat().st_size if part_path.exists() else 0
    if final_size != expected_size:
        raise IOError(f"downloaded {final_size} bytes, Drive metadata reports {expected_size}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=Path("results/selected_sessions_v5.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/vrnet/v5_sessions"))
    parser.add_argument("--log", type=Path, default=Path("results/download_log_v5.txt"))
    args = parser.parse_args()
    if not args.selection.is_file():
        raise FileNotFoundError(f"Selection file not found: {args.selection}")

    selection = pd.read_csv(args.selection, encoding="utf-8-sig")
    required = {"game_name", "session_name", "scene_obj", "scene_cam", "MS_rating"}
    missing = sorted(required.difference(selection.columns))
    if missing:
        raise ValueError(f"Selection is missing columns: {missing}")
    if len(selection) != 10 or selection["game_name"].nunique() != 10:
        raise ValueError("Selection must contain exactly 10 sessions from 10 games")
    if not ((selection["scene_obj"] == 1) & (selection["scene_cam"] == 1) & (selection["MS_rating"] == 1)).all():
        raise ValueError("Every selected session must have scene_obj=1, scene_cam=1, MS_rating=1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = args.output_dir / ".work"
    work_dir.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    result_columns = [
        "download_status",
        "archive_name",
        "drive_file_id",
        "archive_size_bytes",
        "csv_size_bytes",
        "finished_utc",
        "failure_reason",
    ]
    for column in result_columns:
        if column not in selection.columns:
            selection[column] = pd.NA

    http = requests.Session()
    http.headers.update({"User-Agent": "Mozilla/5.0 (compatible; VRnet-v5-research/1.0)"})
    with args.log.open("w", encoding="utf-8", newline="\n") as handle:
        log(handle, "VR.net v5 sequential selected-session download log")
        log(handle, f"Start UTC: {datetime.now(timezone.utc).isoformat()}")
        log(handle, "Command: python scripts/download_vrnet_v5.py")
        log(handle, "Policy: one game ZIP at a time; 5,000,000,000-byte cap; extract one selected data_file.csv; delete ZIP/partial after each attempt.")
        log(handle, "Game types in selected_sessions_v5.csv are title-based heuristics, not verified VR.net metadata.")
        log(handle, "No model training or Git operation is performed by this script.")
        try:
            archives = list_drive_archives(http)
            log(handle, f"Drive listing succeeded: {len(archives)} ZIP archives")
        except Exception:
            log(handle, "Drive listing failed with full traceback:")
            log(handle, traceback.format_exc())
            raise

        for row_index, row in selection.iterrows():
            game = str(row["game_name"])
            session_name = str(row["session_name"])
            archive = archives.get(game)
            zip_path = work_dir / f"{game}.zip"
            part_path = work_dir / f"{game}.zip.part"
            output_path = args.output_dir / game / session_name / "data_file.csv"
            started = time.perf_counter()
            try:
                if output_path.is_file() and output_path.stat().st_size > 0:
                    selection.at[row_index, "download_status"] = "success_csv_already_present"
                    selection.at[row_index, "csv_size_bytes"] = output_path.stat().st_size
                    log(handle, f"[{game}] SKIP: {output_path} already exists ({output_path.stat().st_size} bytes)")
                    continue
                if archive is None:
                    raise FileNotFoundError(f"No public Drive ZIP metadata found for game {game}")
                archive_size = int(archive.get("size", 0))
                archive_name = str(archive.get("name", f"{game}.zip"))
                file_id = str(archive["id"])
                selection.at[row_index, "archive_name"] = archive_name
                selection.at[row_index, "drive_file_id"] = file_id
                selection.at[row_index, "archive_size_bytes"] = archive_size
                if archive_size <= 0:
                    raise ValueError("Drive metadata has no positive archive size")
                if archive_size > MAX_ZIP_BYTES:
                    selection.at[row_index, "download_status"] = "skipped_over_5gb"
                    log(handle, f"[{game}] SKIP: {archive_size} bytes exceeds 5 GB")
                    continue
                free = shutil.disk_usage(args.output_dir).free
                if free < MIN_FREE_BYTES:
                    selection.at[row_index, "download_status"] = "skipped_free_space_under_10gb"
                    log(handle, f"[{game}] SKIP: free disk {free} bytes is below 10 GB")
                    continue
                log(
                    handle,
                    f"[{game}] START session={session_name} archive={archive_name} size={archive_size} bytes free={free}",
                )
                download_archive(http, file_id, archive_size, part_path, handle, game)
                part_path.replace(zip_path)
                if not zipfile.is_zipfile(zip_path):
                    raise zipfile.BadZipFile("downloaded response is not a valid ZIP")
                with zipfile.ZipFile(zip_path) as archive_file:
                    member = selected_member(archive_file.namelist(), session_name)
                    info = archive_file.getinfo(member)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    temp_output = output_path.with_suffix(".csv.part")
                    with archive_file.open(info) as source, temp_output.open("wb") as destination:
                        shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
                    if temp_output.stat().st_size != info.file_size:
                        raise IOError(
                            f"extracted {temp_output.stat().st_size} bytes but ZIP member reports {info.file_size}"
                        )
                    temp_output.replace(output_path)
                selection.at[row_index, "download_status"] = "success_csv_extracted_zip_deleted"
                selection.at[row_index, "csv_size_bytes"] = output_path.stat().st_size
                log(
                    handle,
                    f"[{game}] SUCCESS csv={output_path} csv_size={output_path.stat().st_size} "
                    f"elapsed={time.perf_counter()-started:.1f}s",
                )
            except Exception as exc:
                selection.at[row_index, "download_status"] = "failed"
                selection.at[row_index, "failure_reason"] = f"{type(exc).__name__}: {exc}"
                log(handle, f"[{game}] FAILED: {type(exc).__name__}: {exc}")
                log(handle, traceback.format_exc())
            finally:
                for disposable in (zip_path, part_path):
                    if disposable.exists():
                        disposable.unlink()
                        log(handle, f"[{game}] deleted temporary archive: {disposable}")
                selection.at[row_index, "finished_utc"] = datetime.now(timezone.utc).isoformat()
                selection.to_csv(args.selection, index=False, encoding="utf-8-sig")
        successes = int(selection["download_status"].astype(str).str.startswith("success").sum())
        log(handle, f"Finished UTC: {datetime.now(timezone.utc).isoformat()}")
        log(handle, f"Successful selected CSVs: {successes}/{len(selection)}")
        log(handle, f"Temporary ZIP/partial files remaining: {len(list(work_dir.glob('*')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
