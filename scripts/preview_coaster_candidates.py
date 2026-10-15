"""Inspect a few real JPGs from public candidate archives via bounded Range reads.

No full game download, no CSV labels, no training, and no claim that these are
the paper's roller coaster. Originals are never overwritten.
"""
from __future__ import annotations

import io
import json
import re
import traceback
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import requests

from download_vrnet_game_csvs_v3 import request_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "rollercoaster_sources"
CAP = 32_000_000  # Directory plus images, per game.


class PublicRangeFile(io.RawIOBase):
    def __init__(self, session, item, note):
        self.session, self.size, self.note = session, int(item["size"]), note
        self.pos, self.transferred = 0, 0
        self.tail_start = max(0, self.size - 65536)
        response = request_download(session, item["id"], range_start=self.tail_start)
        self.url = response.url
        self.tail = self.consume(response, self.tail_start, self.size - 1)

    def consume(self, response, start, end):
        with response:
            cr = response.headers.get("Content-Range", "")
            self.note(f"HTTP={response.status_code}; Range={cr}")
            if response.status_code != 206 or cr != f"bytes {start}-{end}/{self.size}":
                raise RuntimeError("Exact Range not honored; response closed without reading payload")
            if self.transferred + end - start + 1 > CAP:
                raise RuntimeError("Candidate preview cap exceeded")
            blob = response.content
            self.transferred += len(blob)
            if len(blob) != end - start + 1:
                raise RuntimeError("Incomplete Range response")
            return blob

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if n + self.transferred > CAP:
            raise RuntimeError("Candidate preview cap exceeded before request")
        end = self.pos + n - 1
        if self.pos >= self.tail_start:
            result = self.tail[self.pos - self.tail_start:self.pos - self.tail_start + n]
        else:
            response = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"},
                                        stream=True, timeout=(20, 60))
            result = self.consume(response, self.pos, end)
        self.pos += len(result)
        return result

    def seek(self, offset, whence=0):
        self.pos = (offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset)
        if not 0 <= self.pos <= self.size:
            raise ValueError("Out of bounds seek")
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def readable(self):
        return True


def main():
    inventory = json.loads((OUT / "drive_inventory.json").read_text(encoding="utf-8"))
    games = {Path(i["name"]).stem: i for i in inventory["folders"]["game_sessions"]}
    results = []
    log_path = ROOT / "results" / "rollercoaster_candidate_preview_log.txt"
    with log_path.open("a", encoding="utf-8") as log:
        def note(text):
            log.write(text + "\n")
            log.flush()
            print(text, flush=True)
        for game in ["Puddles_Theme_Park", "Fun_House"]:
            item = games[game]
            record = {"game": game, "source": item["webViewLink"], "candidate_only": True,
                      "archive_bytes": int(item["size"]), "previews": []}
            rangefile = None
            try:
                if int(item["size"]) > 5_000_000_000:
                    raise RuntimeError("Game ZIP exceeds existing 5GB cap; skip")
                destination = OUT / "candidate_previews" / game
                destination.mkdir(parents=True, exist_ok=True)
                note(f"CANDIDATE {game}; bounded metadata/JPG only; not identified as paper data")
                with requests.Session() as session:
                    rangefile = PublicRangeFile(session, item, note)
                    with zipfile.ZipFile(rangefile) as archive:
                        sessions = defaultdict(list)
                        types = defaultdict(int)
                        for member in archive.infolist():
                            if member.is_dir():
                                continue
                            types[Path(member.filename).suffix.lower()] += 1
                            if member.filename.lower().endswith((".jpg", ".jpeg")):
                                parts = member.filename.replace("\\", "/").split("/")
                                matches = [p for p in parts if re.match(r"^\d+_\d+_", p)]
                                if len(matches) != 1:
                                    raise RuntimeError("Cannot identify JPG session name")
                                sessions[matches[0]].append(member)
                        record["archive_entry_count"] = len(archive.infolist())
                        record["extension_counts"] = dict(types)
                        record["session_names"] = sorted(sessions)
                        note(f"Directory found {len(sessions)} JPG sessions; types={dict(types)}")
                        # Three sessions at spread indices; four temporal checkpoints each.
                        names = sorted(sessions)
                        chosen = sorted({0, len(names) // 2, len(names) - 1})
                        panels = []
                        for index in chosen:
                            name = names[index]
                            entries = sorted(sessions[name], key=lambda m: int(Path(m.filename).stem))
                            for fraction in (0.1, 0.35, 0.65, 0.9):
                                member = entries[min(len(entries) - 1, int((len(entries) - 1) * fraction))]
                                blob = archive.read(member)  # Full selected JPG; CRC checked.
                                frame = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
                                if frame is None:
                                    raise RuntimeError(f"JPEG decode failed: {member.filename}")
                                filename = f"{name}__{Path(member.filename).name}"
                                (destination / filename).write_bytes(blob)
                                record["previews"].append({"session": name, "member": member.filename,
                                                           "bytes": len(blob), "image": str(destination / filename)})
                                thumbnail = cv2.resize(frame, (420, 220))
                                panel = np.full((260, 420, 3), 255, np.uint8)
                                panel[:220] = thumbnail
                                cv2.putText(panel, name, (5, 236), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1)
                                cv2.putText(panel, Path(member.filename).name, (5, 252), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1)
                                panels.append(panel)
                        montage = np.vstack([np.hstack(panels[i:i + 4]) for i in range(0, len(panels), 4)])
                        montage_path = destination / "preview_montage.jpg"
                        if not cv2.imencode(".jpg", montage)[0]:
                            raise RuntimeError("Montage encoding failed")
                        montage_path.write_bytes(cv2.imencode(".jpg", montage)[1].tobytes())
                        record.update({"status": "success", "montage": str(montage_path)})
            except Exception:
                record.update({"status": "failed", "error": traceback.format_exc()})
                note(record["error"])
            record["transferred_bytes"] = rangefile.transferred if rangefile else None
            results.append(record)
            (OUT / "candidate_preview_results.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            note(f"FINISH {game}: {record['status']}; bytes={record['transferred_bytes']}")


if __name__ == "__main__":
    main()
