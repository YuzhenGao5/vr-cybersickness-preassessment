"""Refresh public VR.net metadata to locate the paper's roller coaster release.

Only small public documents/metadata are fetched. No game archives or training.
"""
from __future__ import annotations

import hashlib
import json
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from audit_vrnet_graphics_labels import extract_api_keys, list_drive_children
from download_vrnet_game_csvs_v3 import request_download


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "rollercoaster_sources"
ROOT_ID = "1LLo4OE83Zeso8MkEXSvW9VrdjiqqlFRR"
# Verified via the PDF link on https://ahlab.org/projects/vrhook/.
PAPER_ID = "1eTlLX4o3eOzI7aPUcRHMiWoCrRe1aDqF"
LIMIT_BYTES = 32_000_000


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "results" / "rollercoaster_source_check_log.txt"
    records: list[dict] = []
    with log_path.open("a", encoding="utf-8") as log, requests.Session() as session:
        def note(message: str) -> None:
            line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
            log.write(line + "\n")
            log.flush()
            print(line, flush=True)

        def fetch(url: str, name: str, drive_id: str | None = None) -> None:
            note(f"GET {url}; maximum={LIMIT_BYTES} bytes; destination={OUT / name}")
            try:
                response = (request_download(session, drive_id) if drive_id else
                            session.get(url, stream=True, timeout=(20, 60)))
                with response:
                    response.raise_for_status()
                    count = 0
                    chunks = []
                    for chunk in response.iter_content(65536):
                        count += len(chunk)
                        if count > LIMIT_BYTES:
                            raise RuntimeError("Small-document download cap exceeded; stopped")
                        chunks.append(chunk)
                    blob = b"".join(chunks)
                    if name.endswith(".pdf") and not blob.startswith(b"%PDF"):
                        raise RuntimeError("Response is not a PDF; not saved as a successful PDF")
                    (OUT / name).write_bytes(blob)
                    records.append({"url": url, "name": name, "status": "success",
                                    "http_status": response.status_code, "bytes": count,
                                    "sha256": hashlib.sha256(blob).hexdigest()})
                    note(f"SUCCESS HTTP={response.status_code}; bytes={count}; name={name}")
            except Exception:
                error = traceback.format_exc()
                # Frontend public API keys are not useful diagnostic output.
                error = re.sub(r"AIza[0-9A-Za-z_-]{30,}", "[REDACTED_API_KEY]", error)
                note("FAILED\n" + error)
                records.append({"url": url, "name": name, "status": "failed", "error": error})

        note("Source audit only. Client date=2026-10-10; host UTC recorded separately. No game archive download.")
        root_url = f"https://drive.google.com/drive/folders/{ROOT_ID}"
        try:
            response = session.get(root_url, timeout=(20, 45))
            response.raise_for_status()
            keys = extract_api_keys(response.text)
            items = None
            for key in keys:
                try:
                    items = list_drive_children(session, key, ROOT_ID)
                    break
                except Exception as exc:
                    note("Public metadata attempt failed: " + re.sub(
                        r"AIza[0-9A-Za-z_-]{30,}", "[REDACTED_API_KEY]", str(exc)))
            if items is None:
                raise RuntimeError("Unable to obtain public root metadata")
            inventory = {"root_url": root_url, "root_items": items, "folders": {}}
            note(f"SUCCESS root metadata: {len(items)} items")
            for item in items:
                if item["mimeType"] == "application/vnd.google-apps.folder":
                    children = list_drive_children(session, key, item["id"])
                    inventory["folders"][item["name"]] = children
                    note(f"SUCCESS directory {item['name']}: {len(children)} children")
                    (OUT / f"{item['name']}_live_inventory.csv").write_text(
                        pd.DataFrame(children).to_csv(index=False), encoding="utf-8-sig")
            (OUT / "drive_inventory.json").write_text(
                json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
            for item in items:
                name = item["name"]
                if name in ("ReadMe.md", "main_file.csv", "main_log.csv"):
                    fetch(f"https://drive.google.com/file/d/{item['id']}/view", name,
                          drive_id=item["id"])
                elif name == "Updates_ReadMe":
                    fetch(f"https://docs.google.com/document/d/{item['id']}/export?format=txt",
                          "Updates_ReadMe.txt")
        except Exception:
            error = re.sub(r"AIza[0-9A-Za-z_-]{30,}", "[REDACTED_API_KEY]", traceback.format_exc())
            note("METADATA FAILED\n" + error)
            records.append({"url": root_url, "status": "failed", "error": error})

        fetch(f"https://drive.google.com/file/d/{PAPER_ID}/view?usp=drive_link",
              "vrnet_final_paper.pdf", drive_id=PAPER_ID)
        (OUT / "request_results.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        note("Source audit completed; no training or game archive download was performed")


if __name__ == "__main__":
    main()
