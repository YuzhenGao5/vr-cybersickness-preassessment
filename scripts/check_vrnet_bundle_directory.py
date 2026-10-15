"""Read a bounded HTTP Range ZIP directory; never fetch the full VR.net bundle."""
from __future__ import annotations

import json
import struct
import traceback
from datetime import datetime, timezone
from pathlib import Path

import requests

from download_vrnet_game_csvs_v3 import request_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "rollercoaster_sources"
CAP = 32_000_000


def main() -> None:
    metadata = json.loads((OUT / "drive_inventory.json").read_text(encoding="utf-8"))
    item = next(i for i in metadata["root_items"] if i["name"] == "game_sessions.zip")
    size = int(item["size"])
    total = 0
    record = {"source": item["webViewLink"], "archive_bytes": size,
              "method": "HTTP Range directory metadata only", "transfer_cap_bytes": CAP}
    with (ROOT / "results" / "rollercoaster_bundle_check_log.txt").open("a", encoding="utf-8") as log:
        def note(message):
            log.write(f"[{datetime.now(timezone.utc).isoformat()}] {message}\n")
            log.flush()
            print(message, flush=True)

        try:
            with requests.Session() as session:
                start = max(0, size - 65536)
                note(f"Request public bundle suffix Range bytes={start}-{size - 1}; require HTTP 206")
                response = request_download(session, item["id"], range_start=start)
                def consume(r, expected_start, expected_end):
                    nonlocal total
                    with r:
                        cr = r.headers.get("Content-Range", "")
                        note(f"HTTP={r.status_code}; Content-Range={cr}")
                        if r.status_code != 206 or cr != f"bytes {expected_start}-{expected_end}/{size}":
                            raise RuntimeError("Server did not honor exact Range. Closed response without reading payload.")
                        expected = expected_end - expected_start + 1
                        if total + expected > CAP:
                            raise RuntimeError("Directory metadata transfer exceeds cap; stopped")
                        payload = r.content
                        total += len(payload)
                        if len(payload) != expected:
                            raise RuntimeError("Incomplete Range response")
                        return payload

                final_url = response.url
                suffix = consume(response, start, size - 1)
                eocd = suffix.rfind(b"PK\x05\x06")
                if eocd < 0:
                    raise RuntimeError("ZIP end-of-central-directory signature not found")
                fields = struct.unpack_from("<4s4H2LH", suffix, eocd)
                entries, cd_size, cd_offset = fields[4], fields[5], fields[6]

                def get_range(a, b):
                    note(f"Request bounded directory Range bytes={a}-{b}")
                    return consume(session.get(final_url, headers={"Range": f"bytes={a}-{b}"},
                                               timeout=(20, 60), stream=True), a, b)

                if entries == 65535 or cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
                    locator = suffix.rfind(b"PK\x06\x07", 0, eocd)
                    if locator < 0:
                        raise RuntimeError("ZIP64 locator not found")
                    _, _, zip64_offset, _ = struct.unpack_from("<4sLQL", suffix, locator)
                    if start <= zip64_offset and zip64_offset + 56 <= size:
                        z64 = suffix[zip64_offset - start:zip64_offset - start + 56]
                    else:
                        z64 = get_range(zip64_offset, zip64_offset + 55)
                    fields64 = struct.unpack("<4sQ2H2L4Q", z64)
                    if fields64[0] != b"PK\x06\x06":
                        raise RuntimeError("ZIP64 header invalid")
                    entries, cd_size, cd_offset = fields64[7], fields64[8], fields64[9]
                record.update({"entry_count": entries, "central_directory_bytes": cd_size,
                               "central_directory_offset": cd_offset})
                note(f"Directory entries={entries}; directory size={cd_size} bytes")
                if total + cd_size > CAP:
                    raise RuntimeError("Complete central directory exceeds small metadata cap; not downloaded")
                if start <= cd_offset and cd_offset + cd_size <= size:
                    directory = suffix[cd_offset - start:cd_offset - start + cd_size]
                else:
                    directory = get_range(cd_offset, cd_offset + cd_size - 1)
                offset = 0
                names = []
                while offset < len(directory):
                    if directory[offset:offset + 4] != b"PK\x01\x02":
                        raise RuntimeError(f"Invalid central directory at byte {offset}")
                    values = struct.unpack_from("<4s6H3L5H2L", directory, offset)
                    flags, n, extra, comment = values[3], values[10], values[11], values[12]
                    names.append(directory[offset + 46:offset + 46 + n].decode(
                        "utf-8" if flags & 0x800 else "cp437", errors="strict"))
                    offset += 46 + n + extra + comment
                if len(names) != entries:
                    raise RuntimeError(f"Directory count mismatch {len(names)} != {entries}")
                record.update({"status": "success", "names": names})
                note(f"SUCCESS complete directory read; count={len(names)}; payload bytes={total}")
        except Exception:
            record.update({"status": "failed", "error": traceback.format_exc()})
            note(record["error"])
        record["actual_payload_bytes"] = total
        (OUT / "bundle_directory_check.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
