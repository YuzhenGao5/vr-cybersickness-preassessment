#!/usr/bin/env python3
"""Reproducibly sample VR.net sessions by game and download only ZIPs <=500 MiB."""

from __future__ import annotations

import csv
import html
import re
import shutil
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import pandas as pd
import requests


ROOT_FOLDER_ID = "1LLo4OE83Zeso8MkEXSvW9VrdjiqqlFRR"
MAX_ZIP_BYTES = 500 * 1024 * 1024
SEED = 42
FIELDS = [
    "session_name", "game_name", "main_log_ms_rating_flag", "video_flag", "game_session_count",
    "archive_name", "drive_file_id", "archive_size_bytes", "archive_size_mib",
    "sample_seed", "download_status", "csv_path",
]


def extract_drive_key(page: str) -> list[str]:
    return sorted(set(re.findall(r"AIza[0-9A-Za-z_-]{30,}", page)))


def find_folder_id(page: str, folder_name: str) -> str:
    labels = re.findall(r'aria-label="([^"]+)"', page)
    for label in labels:
        if label == f"{folder_name} Shared folder":
            start = page.find(f'aria-label="{label}"')
            match = re.search(r'data-id="([0-9A-Za-z_-]+)"', page[start:start + 5000])
            if match:
                return match.group(1)
    raise RuntimeError(f"Google Drive root page has no {folder_name!r} folder ID")


def list_drive_files(folder_id: str, session: requests.Session) -> list[dict]:
    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
    page_response = session.get(folder_url, timeout=45)
    page_response.raise_for_status()
    keys = extract_drive_key(page_response.text)
    if not keys:
        raise RuntimeError(f"No public Drive API key found in listing page {folder_url}")
    endpoint = "https://www.googleapis.com/drive/v3/files"
    params = {
        "q": f"'{folder_id}' in parents and trashed=false",
        "fields": "files(id,name,size,mimeType),nextPageToken",
        "pageSize": 1000,
    }
    last_error = None
    for key in keys:
        files: list[dict] = []
        token = None
        while True:
            query = {**params, "key": key}
            if token:
                query["pageToken"] = token
            response = None
            for attempt in range(1, 4):
                try:
                    response = session.get(
                        endpoint, params=query,
                        headers={"Referer": "https://drive.google.com/", "Connection": "close"},
                        timeout=60,
                    )
                    break
                except requests.RequestException as exc:
                    last_error = f"attempt {attempt}/3: {type(exc).__name__}: {exc}"
                    if attempt < 3:
                        time.sleep(attempt)
            if response is None:
                continue
            if not response.ok:
                last_error = f"HTTP {response.status_code}: {response.text[:1000]}"
                break
            try:
                payload = response.json()
            except Exception as exc:
                last_error = f"JSON decode failed: {type(exc).__name__}: {exc}; body={response.text[:1000]}"
                break
            files.extend(payload.get("files", []))
            token = payload.get("nextPageToken")
            if not token:
                return files
        # Try another key only when this key did not give a usable successful page.
    raise RuntimeError(f"Could not list Drive folder {folder_id}: {last_error or 'no response'}")


def get_csv_member(archive: zipfile.ZipFile, session_name: str) -> str | None:
    candidates = []
    for member in archive.namelist():
        path = PurePosixPath(member.replace("\\", "/"))
        if path.name.casefold() == "data_file.csv" and session_name in path.parts:
            candidates.append(member)
    return sorted(candidates, key=lambda value: (value.count("/"), value))[0] if candidates else None


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data/raw/vrnet"
    sample_dir = raw_dir / "sampled_csv"
    results = root / "results"
    sample_dir.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    selection_path = results / "vrnet_sampled_sessions_v2.csv"
    log_path = results / "sampled_csv_download_log_v2.txt"
    started = datetime.now(timezone.utc).isoformat()
    log_lines = [
        "VR.net v2 session sample and CSV download log",
        f"Started UTC: {started}",
        f"Local index: {raw_dir / 'main_log.csv'}",
        f"Drive root: https://drive.google.com/drive/folders/{ROOT_FOLDER_ID}",
        f"Maximum ZIP size: 500 MiB ({MAX_ZIP_BYTES} bytes)",
        f"Sampling: seed={SEED}; one eligible session with MS_rating flag per parsed game name.",
        "Video payloads are never extracted. Only a selected session's data_file.csv is extracted from eligible ZIPs.",
        "",
    ]
    if log_path.is_file():
        previous_log = log_path.read_text(encoding="utf-8")
        if "FATAL:" in previous_log:
            log_lines.extend(["Previous attempt (retained for audit):", previous_log.rstrip(), ""])
    try:
        main_log_path = raw_dir / "main_log.csv"
        main_file_path = raw_dir / "main_file.csv"
        if not main_log_path.is_file() or not main_file_path.is_file():
            missing = [str(path) for path in (main_log_path, main_file_path) if not path.is_file()]
            raise FileNotFoundError("Required local VR.net index missing: " + ", ".join(missing))
        main_log = pd.read_csv(main_log_path, encoding="utf-8-sig")
        main_file = pd.read_csv(main_file_path, encoding="utf-8-sig")
        required = {"name", "MS_rating", "video"}
        if not required.issubset(main_log.columns):
            raise ValueError(f"main_log.csv lacks columns: {sorted(required - set(main_log.columns))}")
        main_log["game_name"] = main_log["name"].astype(str).str.split("_", n=2).str[2]
        main_log["MS_rating"] = pd.to_numeric(main_log["MS_rating"], errors="coerce").fillna(0).astype(int)
        main_log["video"] = pd.to_numeric(main_log["video"], errors="coerce").fillna(0).astype(int)
        eligible = main_log.loc[main_log["MS_rating"].eq(1)].copy()
        rng = __import__("numpy").random.default_rng(SEED)
        chosen = []
        for game_name, group in sorted(eligible.groupby("game_name", sort=True)):
            if not game_name or game_name == "nan":
                continue
            selected_index = int(rng.integers(0, len(group)))
            chosen.append(group.iloc[selected_index])
        if not chosen:
            raise ValueError("No sessions with main_log MS_rating flag=1 available for sampling")
        selected = pd.DataFrame(chosen).reset_index(drop=True)
        if not 80 <= len(selected) <= 100:
            raise ValueError(f"Stratified one-per-game sample size {len(selected)} is outside requested 80–100")
        game_counts = main_file.set_index("game_name")["no_of_sessions"].to_dict()
        pending_rows = []
        for _, sample in selected.iterrows():
            game_name = str(sample["game_name"])
            pending_rows.append({
                "session_name": str(sample["name"]), "game_name": game_name,
                "main_log_ms_rating_flag": int(sample["MS_rating"]), "video_flag": int(sample["video"]),
                "game_session_count": int(game_counts.get(game_name, 0)),
                "archive_name": f"{game_name}.zip", "drive_file_id": "",
                "archive_size_bytes": "", "archive_size_mib": "", "sample_seed": SEED,
                "download_status": "pending_size_check", "csv_path": "",
            })
        pd.DataFrame(pending_rows, columns=FIELDS).to_csv(selection_path, index=False, encoding="utf-8-sig")

        drive = requests.Session()
        drive.headers.update({"User-Agent": "Mozilla/5.0 (compatible; dataset-metadata-check/1.0)"})
        root_page_url = f"https://drive.google.com/drive/folders/{ROOT_FOLDER_ID}"
        root_response = drive.get(root_page_url, timeout=45)
        root_response.raise_for_status()
        game_folder_id = find_folder_id(root_response.text, "game_sessions")
        files = list_drive_files(game_folder_id, drive)
        archives = {Path(file["name"]).stem: file for file in files if str(file.get("name", "")).lower().endswith(".zip")}
        log_lines.extend([
            f"Drive metadata listing: success; game_sessions folder ID={game_folder_id}; ZIP archives listed={len(archives)}.",
            f"main_log rows={len(main_log)}; unique sessions={main_log['name'].nunique()}; parsed games={main_log['game_name'].nunique()}; sessions with MS_rating flag=1={len(eligible)}.",
            "The Drive listing contains game-level ZIP archives, not session-level ZIPs; main_file.csv lists session counts per game.",
            "",
        ])

        rows = []
        for _, sample in selected.iterrows():
            session_name = str(sample["name"])
            game_name = str(sample["game_name"])
            archive_name = f"{game_name}.zip"
            archive = archives.get(game_name)
            record = {
                "session_name": session_name, "game_name": game_name,
                "main_log_ms_rating_flag": int(sample["MS_rating"]), "video_flag": int(sample["video"]),
                "game_session_count": int(game_counts.get(game_name, 0)),
                "archive_name": archive_name,
                "drive_file_id": archive.get("id", "") if archive else "",
                "archive_size_bytes": int(archive["size"]) if archive and archive.get("size") else "",
                "archive_size_mib": round(int(archive["size"]) / (1024 ** 2), 2) if archive and archive.get("size") else "",
                "sample_seed": SEED, "download_status": "pending", "csv_path": "",
            }
            if not archive:
                record["download_status"] = "skipped_archive_not_listed"
                log_lines.append(f"{session_name} | {game_name} | {archive_name} | SKIPPED: matching archive not listed in Drive metadata.")
                rows.append(record)
                continue
            size = int(archive.get("size", 0) or 0)
            if size <= 0:
                record["download_status"] = "skipped_size_unknown"
                log_lines.append(f"{session_name} | {game_name} | {archive_name} | SKIPPED: ZIP size missing; no download attempted.")
                rows.append(record)
                continue
            if size > MAX_ZIP_BYTES:
                record["download_status"] = "skipped_archive_over_500_mib"
                log_lines.append(
                    f"{session_name} | {game_name} | {archive_name} | SKIPPED: archive is {size} bytes ({size / (1024 ** 2):.2f} MiB), greater than 500 MiB; no ZIP GET issued."
                )
                rows.append(record)
                continue

            zip_path = sample_dir / f"{game_name}.zip"
            partial = zip_path.with_suffix(".zip.part")
            destination = sample_dir / session_name / "data_file.csv"
            try:
                url = f"https://drive.google.com/uc?export=download&id={archive['id']}"
                response = drive.get(url, stream=True, timeout=(30, 180))
                response.raise_for_status()
                confirm = next((v for k, v in response.cookies.items() if k.startswith("download_warning")), None)
                if confirm:
                    response.close()
                    response = drive.get(url + f"&confirm={confirm}", stream=True, timeout=(30, 180))
                    response.raise_for_status()
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                response.close()
                actual_size = partial.stat().st_size
                if actual_size > MAX_ZIP_BYTES:
                    raise ValueError(f"Downloaded response size {actual_size} bytes exceeds 500 MiB cap")
                if not zipfile.is_zipfile(partial):
                    raise ValueError("Downloaded response is not a valid ZIP archive")
                partial.replace(zip_path)
                with zipfile.ZipFile(zip_path) as zf:
                    member = get_csv_member(zf, session_name)
                    if member is None:
                        raise FileNotFoundError(f"ZIP contains no data_file.csv under session path {session_name!r}")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as source, destination.open("wb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                record["download_status"] = "downloaded_csv_zip_deleted"
                record["csv_path"] = str(destination.relative_to(root))
                log_lines.append(f"{session_name} | {game_name} | {archive_name} | SUCCESS: downloaded ZIP {actual_size} bytes; extracted only {member}; deleted ZIP.")
            except Exception as exc:
                record["download_status"] = "download_failed_or_csv_missing"
                error = f"{type(exc).__name__}: {exc}"
                log_lines.append(f"{session_name} | {game_name} | {archive_name} | FAILED: {error}")
                log_lines.append("Full traceback:")
                log_lines.extend(traceback.format_exc().rstrip().splitlines())
                if destination.exists():
                    destination.unlink()
                if zip_path.exists():
                    zip_path.unlink()
            finally:
                if partial.exists():
                    partial.unlink()
                if zip_path.exists():
                    zip_path.unlink()
            rows.append(record)
            # Preserve progress in case a later eligible ZIP fails.
            pd.DataFrame(rows, columns=FIELDS).to_csv(selection_path, index=False, encoding="utf-8-sig")
            log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

        selection = pd.DataFrame(rows, columns=FIELDS)
        selection.to_csv(selection_path, index=False, encoding="utf-8-sig")
        statuses = selection["download_status"].value_counts().to_dict()
        sizes = pd.to_numeric(selection["archive_size_bytes"], errors="coerce").dropna()
        log_lines.extend([
            "",
            f"Selected session count: {len(selection)}; successful CSV extractions: {int(selection['download_status'].eq('downloaded_csv_zip_deleted').sum())}.",
            f"Status counts: {statuses}",
            f"Archive size range: {sizes.min() / (1024**2):.2f}–{sizes.max() / (1024**2):.2f} MiB" if len(sizes) else "Archive size range: unknown.",
            "No video files were requested or extracted.",
            "",
        ])
        log_path.write_text("\n".join(log_lines), encoding="utf-8")
        print(f"selected={len(selection)} games={selection['game_name'].nunique()} downloaded_csvs={int(selection['download_status'].eq('downloaded_csv_zip_deleted').sum())} statuses={statuses}")
        return 0
    except Exception as exc:
        log_lines.extend([f"FATAL: {type(exc).__name__}: {exc}", "Full traceback:", traceback.format_exc().rstrip()])
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        print(f"ERROR {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
