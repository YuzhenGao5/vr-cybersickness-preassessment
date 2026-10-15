#!/usr/bin/env python3
"""Download up to three VR.net game archives sequentially and retain only data_file.csv."""

from __future__ import annotations

import html
import random
import re
import shutil
import time
import traceback
import zipfile
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath

import pandas as pd
import requests


DRIVE_ROOT_ID = "1LLo4OE83Zeso8MkEXSvW9VrdjiqqlFRR"
MAX_ZIP_BYTES = 5_000_000_000  # decimal 5 GB, as requested
MIN_FREE_BYTES = 10_000_000_000  # decimal 10 GB, as requested
SEED = 42
N_GAMES = 3
MAX_TRANSFER_ATTEMPTS = 12


class DownloadFormParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.action = ""
        self.inputs: dict[str, str] = {}
        self.in_download_form = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "form" and (attrs_dict.get("id") == "download-form" or "download" in attrs_dict.get("action", "")):
            self.in_download_form = True
            self.action = attrs_dict.get("action", "") or ""
        elif tag == "input" and self.in_download_form:
            name = attrs_dict.get("name")
            if name:
                self.inputs[name] = attrs_dict.get("value", "") or ""

    def handle_endtag(self, tag: str) -> None:
        if tag == "form":
            self.in_download_form = False


def extract_api_keys(page: str) -> list[str]:
    return sorted(set(re.findall(r"AIza[0-9A-Za-z_-]{30,}", page)))


def find_game_folder(page: str) -> str:
    for match in re.finditer(r'aria-label="game_sessions Shared folder"', page):
        snippet = page[match.start():match.start() + 5000]
        folder = re.search(r'data-id="([0-9A-Za-z_-]+)"', snippet)
        if folder:
            return folder.group(1)
    raise RuntimeError("Could not find game_sessions folder ID on the public VR.net Drive page")


def list_drive_archives(session: requests.Session) -> dict[str, dict]:
    root_url = f"https://drive.google.com/drive/folders/{DRIVE_ROOT_ID}"
    root_response = session.get(root_url, timeout=45)
    root_response.raise_for_status()
    folder_id = find_game_folder(root_response.text)
    keys = extract_api_keys(root_response.text)
    if not keys:
        raise RuntimeError("No public Google Drive API key found in VR.net Drive listing page")
    endpoint = "https://www.googleapis.com/drive/v3/files"
    last_error = "no response"
    for key in keys:
        page_token = None
        archives: dict[str, dict] = {}
        while True:
            params = {
                "q": f"'{folder_id}' in parents and trashed=false",
                "fields": "files(id,name,size,mimeType),nextPageToken",
                "pageSize": 1000,
                "key": key,
            }
            if page_token:
                params["pageToken"] = page_token
            response = None
            for attempt in range(1, 4):
                try:
                    response = session.get(
                        endpoint, params=params,
                        headers={"Referer": "https://drive.google.com/", "Connection": "close"},
                        timeout=60,
                    )
                    break
                except requests.RequestException as exc:
                    last_error = f"attempt {attempt}/3: {type(exc).__name__}: {exc}"
                    if attempt < 3:
                        time.sleep(attempt)
            if response is None:
                break
            if not response.ok:
                last_error = f"HTTP {response.status_code}: {response.text[:1000]}"
                break
            payload = response.json()
            for item in payload.get("files", []):
                name = str(item.get("name", ""))
                if name.lower().endswith(".zip"):
                    archives[Path(name).stem] = item
            page_token = payload.get("nextPageToken")
            if not page_token:
                return archives
    raise RuntimeError(f"Could not list Google Drive archives: {last_error}")


def request_download(session: requests.Session, file_id: str, range_start: int = 0) -> requests.Response:
    base_url = "https://drive.google.com/uc"
    params = {"export": "download", "id": file_id}
    headers = {"Range": f"bytes={range_start}-"} if range_start else {}
    response = session.get(base_url, params=params, headers=headers, stream=True, timeout=(30, 240))
    response.raise_for_status()
    token = next((value for name, value in response.cookies.items() if name.startswith("download_warning")), None)
    content_type = response.headers.get("Content-Type", "").lower()
    if token:
        response.close()
        params["confirm"] = token
        response = session.get(base_url, params=params, headers=headers, stream=True, timeout=(30, 240))
        response.raise_for_status()
    elif "text/html" in content_type:
        page = response.content.decode("utf-8", errors="replace")
        response.close()
        parser = DownloadFormParser()
        parser.feed(page)
        if parser.action:
            action = html.unescape(parser.action)
            response = session.get(action, params=parser.inputs, headers=headers, stream=True, timeout=(30, 240))
        else:
            match = re.search(r"[?&]confirm=([0-9A-Za-z_-]+)", page)
            if not match:
                raise RuntimeError("Google Drive returned an HTML confirmation page without a recognized download token")
            params["confirm"] = match.group(1)
            response = session.get(base_url, params=params, headers=headers, stream=True, timeout=(30, 240))
        response.raise_for_status()
    return response


def match_session(member_name: str, known_sessions: set[str]) -> str | None:
    path = PurePosixPath(member_name.replace("\\", "/"))
    parts = set(path.parts)
    matches = [name for name in known_sessions if name in parts]
    if len(matches) == 1:
        return matches[0]
    return None


def log_write(handle, message: str) -> None:
    handle.write(message + "\n")
    handle.flush()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data/raw/vrnet"
    sample_dir = raw_dir / "sampled_csv"
    results_dir = root / "results"
    sample_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    work_dir = sample_dir / ".work"
    work_dir.mkdir(parents=True, exist_ok=True)

    main_file_path = raw_dir / "main_file.csv"
    main_log_path = raw_dir / "main_log.csv"
    previous_selection_path = results_dir / "vrnet_sampled_sessions_v2.csv"
    selection_md = results_dir / "zip_download_selection.md"
    log_path = results_dir / "zip_download_log.txt"
    if not main_file_path.is_file() or not main_log_path.is_file():
        raise FileNotFoundError("Required local VR.net indexes main_file.csv and main_log.csv must exist")

    games = pd.read_csv(main_file_path, encoding="utf-8-sig")
    previous_samples = pd.read_csv(previous_selection_path, encoding="utf-8-sig")
    main_log = pd.read_csv(main_log_path, encoding="utf-8-sig")
    required = {"game_name", "no_of_sessions"}
    if not required.issubset(games.columns):
        raise ValueError(f"main_file.csv lacks columns: {sorted(required - set(games.columns))}")
    max_sessions = int(pd.to_numeric(games["no_of_sessions"], errors="raise").max())
    top_games = games.loc[pd.to_numeric(games["no_of_sessions"], errors="raise").eq(max_sessions)].copy()
    # main_file.csv order is retained so sampling is reproducible against the frozen local index.
    candidates = top_games["game_name"].astype(str).tolist()
    chosen_games = random.Random(SEED).sample(candidates, min(N_GAMES, len(candidates)))
    sample_meta = previous_samples.drop_duplicates("game_name").set_index("game_name")
    archive_meta = {}
    for game in chosen_games:
        if game not in sample_meta.index:
            raise ValueError(f"No previously verified Drive metadata for selected game {game!r}")
        row = sample_meta.loc[game]
        archive_meta[game] = {
            "id": str(row["drive_file_id"]),
            "size": int(row["archive_size_bytes"]),
            "name": str(row["archive_name"]),
        }

    session_column = "name" if "name" in main_log.columns else None
    if not session_column:
        raise ValueError("main_log.csv has no name column needed to verify extracted session paths")
    session_names = set(main_log[session_column].astype(str))
    selected_records = []
    for game in chosen_games:
        selected_records.append({
            "game_name": game,
            "no_of_sessions": int(games.loc[games["game_name"].eq(game), "no_of_sessions"].iloc[0]),
            "archive_name": archive_meta[game]["name"],
            "drive_file_id": archive_meta[game]["id"],
            "archive_size_bytes": archive_meta[game]["size"],
            "archive_size_gb": round(archive_meta[game]["size"] / 1_000_000_000, 3),
            "selection_seed": SEED,
            "selection_reason": "top session-count stratum; game genre not verified in local index, selected randomly with seed 42",
            "download_status": "pending",
            "csv_extracted": 0,
        })

    selection_csv = results_dir / "zip_download_selection.csv"
    pd.DataFrame(selected_records).to_csv(selection_csv, index=False, encoding="utf-8-sig")
    max_session_by_game = main_log.copy()
    max_session_by_game["parsed_game"] = max_session_by_game["name"].astype(str).str.split("_", n=2).str[2]
    rationale = [
        "# VR.net ZIP 试下载选择",
        "",
        f"抽样规则：在 `main_file.csv` 中选取 session 数最多的游戏层（{max_sessions} sessions/game），用固定种子 {SEED} 随机抽 3 个。",
        "本地索引不含经过核实的游戏类型字段；没有把名称推断为赛车、飞行或过山车类型。",
        "压缩包大小来自先前成功读取的公开 Drive 元数据，下载前脚本会再次核对磁盘空间、并以 5,000,000,000 bytes 为上限。",
        "",
        "| 游戏 | session 数 | ZIP | ZIP 大小 (GB) | 选择理由 |",
        "|---|---:|---|---:|---|",
    ]
    for item in selected_records:
        rationale.append(f"| {item['game_name']} | {item['no_of_sessions']} | {item['archive_name']} | {item['archive_size_gb']:.3f} | {item['selection_reason']} |")
    rationale.extend(["", f"机器可读选择清单：`{selection_csv.relative_to(root)}`。", ""])
    selection_md.write_text("\n".join(rationale), encoding="utf-8")

    start = datetime.now(timezone.utc).isoformat()
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        if log_path.stat().st_size:
            log_write(log, "")
            log_write(log, "--- New/resume attempt ---")
        log_write(log, "VR.net 5 GB game ZIP sequential download log")
        log_write(log, f"Start UTC: {start}")
        log_write(log, f"Drive root: https://drive.google.com/drive/folders/{DRIVE_ROOT_ID}")
        log_write(log, f"Selected games: {', '.join(chosen_games)}")
        log_write(log, f"ZIP size cap: {MAX_ZIP_BYTES} bytes (5 decimal GB)")
        log_write(log, f"Required free disk before each download: {MIN_FREE_BYTES} bytes (10 decimal GB)")
        log_write(log, "Processing is strictly sequential: download one ZIP, extract only verified data_file.csv members, delete ZIP, then continue.")
        log_write(log, "No video files are extracted; no model training or Git commit is performed.")
        log_write(log, "")

        drive = requests.Session()
        drive.headers.update({"User-Agent": "Mozilla/5.0 (compatible; VRnet-research-download/1.0)"})
        for item in selected_records:
            game = item["game_name"]
            archive = archive_meta[game]
            t0 = datetime.now(timezone.utc)
            log_write(log, f"[{game}] START UTC {t0.isoformat()} | expected_size={archive['size']} bytes ({item['archive_size_gb']:.3f} GB) | file_id={archive['id']}")
            zip_path = work_dir / f"{game}.zip"
            part_path = work_dir / f"{game}.zip.part"
            extracted_count = 0
            try:
                existing_csvs = sorted((sample_dir / game).glob("*/data_file.csv"))
                if len(existing_csvs) >= int(item["no_of_sessions"]):
                    extracted_count = len(existing_csvs)
                    item["download_status"] = "success_csvs_already_present_zip_absent"
                    log_write(
                        log,
                        f"[{game}] SKIPPED DOWNLOAD: {extracted_count} session CSVs already exist "
                        f"for {item['no_of_sessions']} indexed sessions; no ZIP GET issued.",
                    )
                    continue
                free_bytes = shutil.disk_usage(sample_dir).free
                log_write(log, f"[{game}] free_disk_before={free_bytes} bytes")
                if archive["size"] > MAX_ZIP_BYTES:
                    item["download_status"] = "skipped_over_5gb"
                    log_write(log, f"[{game}] SKIPPED: metadata size exceeds 5 GB; no ZIP GET issued.")
                    continue
                if free_bytes < MIN_FREE_BYTES:
                    item["download_status"] = "skipped_free_space_under_10gb"
                    log_write(log, f"[{game}] SKIPPED: free disk is below 10 GB; no ZIP GET issued.")
                    continue

                downloaded = part_path.stat().st_size if part_path.exists() else 0
                if downloaded > archive["size"]:
                    raise ValueError(f"partial archive is {downloaded} bytes, larger than Drive metadata size {archive['size']}")
                for transfer_attempt in range(1, MAX_TRANSFER_ATTEMPTS + 1):
                    if downloaded == archive["size"]:
                        break
                    response = None
                    try:
                        resume_from = part_path.stat().st_size if part_path.exists() else 0
                        response = request_download(drive, archive["id"], range_start=resume_from)
                        append_mode = resume_from > 0 and response.status_code == 206
                        if append_mode:
                            content_range = response.headers.get("Content-Range", "")
                            if not content_range.startswith(f"bytes {resume_from}-"):
                                raise IOError(f"server returned an unexpected Content-Range for resume: {content_range!r}")
                            log_write(
                                log,
                                f"[{game}] TRANSFER attempt {transfer_attempt}/{MAX_TRANSFER_ATTEMPTS}: "
                                f"resume accepted at byte {resume_from}; Content-Range={content_range}",
                            )
                        elif resume_from:
                            log_write(
                                log,
                                f"[{game}] TRANSFER attempt {transfer_attempt}/{MAX_TRANSFER_ATTEMPTS}: "
                                f"server did not honor Range (HTTP {response.status_code}); restarting at byte 0.",
                            )
                            resume_from = 0
                        else:
                            log_write(log, f"[{game}] TRANSFER attempt {transfer_attempt}/{MAX_TRANSFER_ATTEMPTS}: starting at byte 0.")
                        downloaded = resume_from
                        last_report = resume_from
                        with part_path.open("ab" if append_mode else "wb") as output:
                            for chunk in response.iter_content(chunk_size=1024 * 1024):
                                if not chunk:
                                    continue
                                downloaded += len(chunk)
                                if downloaded > MAX_ZIP_BYTES:
                                    raise ValueError(f"download crossed 5 GB cap at {downloaded} bytes; stopping")
                                output.write(chunk)
                                if downloaded - last_report >= 32 * 1024 * 1024:
                                    log_write(log, f"[{game}] download_progress={downloaded} bytes")
                                    last_report = downloaded
                    except requests.RequestException as exc:
                        downloaded = part_path.stat().st_size if part_path.exists() else 0
                        log_write(
                            log,
                            f"[{game}] TRANSFER attempt {transfer_attempt}/{MAX_TRANSFER_ATTEMPTS} interrupted: "
                            f"{type(exc).__name__}: {exc}; partial_bytes={downloaded}",
                        )
                        if transfer_attempt == MAX_TRANSFER_ATTEMPTS:
                            raise
                        time.sleep(min(transfer_attempt * 2, 20))
                    finally:
                        if response is not None:
                            response.close()
                if downloaded != archive["size"]:
                    raise IOError(f"downloaded {downloaded} bytes but Drive metadata reports {archive['size']} bytes")
                if not zipfile.is_zipfile(part_path):
                    raise zipfile.BadZipFile("download response is not a valid ZIP archive")
                part_path.replace(zip_path)
                item["download_status"] = "downloaded"
                log_write(log, f"[{game}] ZIP download complete: {downloaded} bytes ({downloaded / 1_000_000_000:.3f} GB); now extracting only data_file.csv members.")

                with zipfile.ZipFile(zip_path) as archive_file:
                    csv_members = [
                        name for name in archive_file.namelist()
                        if PurePosixPath(name.replace("\\", "/")).name.casefold() == "data_file.csv"
                    ]
                    for member in csv_members:
                        session_name = match_session(member, session_names)
                        if not session_name:
                            log_write(log, f"[{game}] CSV MEMBER SKIPPED: could not match archive member to an exact main_log session: {member}")
                            continue
                        if session_name.split("_", 2)[-1] != game:
                            log_write(log, f"[{game}] CSV MEMBER SKIPPED: matched session {session_name!r} has a different parsed game name: {member}")
                            continue
                        destination = sample_dir / game / session_name / "data_file.csv"
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with archive_file.open(member) as source, destination.open("wb") as output:
                            shutil.copyfileobj(source, output, length=1024 * 1024)
                        extracted_count += 1
                        log_write(log, f"[{game}] EXTRACTED: {member} -> {destination.relative_to(root)} ({destination.stat().st_size} bytes)")

                item["csv_extracted"] = extracted_count
                if extracted_count:
                    item["download_status"] = "success_csvs_extracted_zip_deleted"
                else:
                    item["download_status"] = "zip_downloaded_no_verified_csv"
                    log_write(log, f"[{game}] No CSV was extracted from the {len(csv_members)} data_file.csv member(s); no paths were guessed.")
            except Exception as exc:
                item["download_status"] = "failed"
                log_write(log, f"[{game}] FAILED: {type(exc).__name__}: {exc}")
                log_write(log, "Full traceback:")
                for line in traceback.format_exc().rstrip().splitlines():
                    log_write(log, line)
            finally:
                for candidate in (part_path, zip_path):
                    try:
                        if candidate.exists():
                            candidate.unlink()
                            log_write(log, f"[{game}] DELETED temporary archive: {candidate.name}")
                    except Exception as cleanup_exc:
                        log_write(log, f"[{game}] CLEANUP FAILED for {candidate}: {type(cleanup_exc).__name__}: {cleanup_exc}")
                item["csv_extracted"] = extracted_count
                item["finished_utc"] = datetime.now(timezone.utc).isoformat()
                pd.DataFrame(selected_records).to_csv(selection_csv, index=False, encoding="utf-8-sig")
                log_write(log, f"[{game}] END UTC {item['finished_utc']} | status={item['download_status']} | extracted_csvs={extracted_count}")
                log_write(log, "")

        pd.DataFrame(selected_records).to_csv(selection_csv, index=False, encoding="utf-8-sig")
        successful = sum(int(item["csv_extracted"]) for item in selected_records)
        log_write(log, f"SUMMARY: games_selected={len(selected_records)}; total_session_CSVs_extracted={successful}; statuses={ {item['game_name']: item['download_status'] for item in selected_records} }")
        log_write(log, f"End UTC: {datetime.now(timezone.utc).isoformat()}")
        log_write(log, f"Temporary archive directory empty: {not any(work_dir.iterdir())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
