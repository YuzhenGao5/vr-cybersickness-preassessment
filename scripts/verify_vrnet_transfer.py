#!/usr/bin/env python3
"""Create/resume a portable SHA256 inventory, or verify a transferred tree.

Python standard library only. Never modifies source data and never follows links.
The manifest stores an initial size/mtime snapshot and incrementally appended
hashes, so interrupted creation can resume. Verification compares content hashes
and sizes, not remote mtimes. Exit 0 means an exact, complete verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import time
from datetime import datetime, timezone


SCHEMA = "vrnet-transfer-sha256-v1"
CHUNK_BYTES = 8 * 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_link_or_reparse(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def canonical_relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value or "\x00" in value:
        raise ValueError(f"Invalid relative manifest path: {value!r}")
    parts = value.split("/")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"Unsafe relative manifest path: {value!r}")
    return value


def checked_root(value: str) -> Path:
    path = Path(value).absolute()
    if is_link_or_reparse(path):
        raise ValueError(f"Root must not be a symlink or reparse point: {path}")
    path = path.resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"Root is not a directory: {path}")
    return path


def output_outside_root(value: str, root: Path) -> Path:
    result = Path(value).absolute().resolve(strict=False)
    try:
        result.relative_to(root)
    except ValueError:
        return result
    raise ValueError(f"Manifest/report must be outside the source data tree: {result}")


def safe_file(root: Path, relative: str) -> Path:
    relative = canonical_relative(relative)
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if is_link_or_reparse(current):
            raise ValueError(f"Refusing symlink/reparse point: {relative}")
    resolved = current.resolve(strict=True)
    resolved.relative_to(root)
    if not resolved.is_file():
        raise ValueError(f"Not a regular file: {relative}")
    return resolved


def scan(root: Path) -> tuple[dict[str, dict], list[str]]:
    inventory: dict[str, dict] = {}
    skipped: list[str] = []
    stack = [root]
    while stack:
        directory = stack.pop()
        if is_link_or_reparse(directory):
            raise ValueError(f"Directory changed to a link while scanning: {directory}")
        with os.scandir(directory) as children:
            for child in children:
                path = Path(child.path)
                relative = path.relative_to(root).as_posix()
                if is_link_or_reparse(path):
                    skipped.append(relative)
                elif child.is_dir(follow_symlinks=False):
                    stack.append(path)
                elif child.is_file(follow_symlinks=False):
                    canonical_relative(relative)
                    info = child.stat(follow_symlinks=False)
                    inventory[relative] = {"size": info.st_size, "mtime_ns": info.st_mtime_ns}
                else:
                    skipped.append(relative)
    return inventory, sorted(skipped)


def hash_stable_file(root: Path, relative: str, expected: dict | None = None) -> tuple[str, int]:
    path = safe_file(root, relative)
    before = path.stat()
    if expected is not None and (before.st_size != expected["size"] or before.st_mtime_ns != expected["mtime_ns"]):
        raise ValueError(f"Source modified since manifest snapshot: {relative}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(CHUNK_BYTES):
            digest.update(data)
    after_path = safe_file(root, relative)
    after = after_path.stat()
    before_key = (before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev)
    after_key = (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev)
    if before_key != after_key:
        raise ValueError(f"File modified while hashing: {relative}")
    return digest.hexdigest(), before.st_size


def write_line(handle, record: dict) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_manifest(path: Path, resume: bool = False) -> tuple[dict, dict[str, dict], dict[str, str], bool]:
    # Only creation may discard a final partially written JSON line after an
    # interrupted append. No source data file is ever truncated or modified.
    if resume:
        with path.open("rb+") as handle:
            handle.seek(0, os.SEEK_END)
            length = handle.tell()
            if length:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    position = length
                    last_newline = -1
                    while position > 0 and last_newline < 0:
                        count = min(position, 65536)
                        position -= count
                        handle.seek(position)
                        block = handle.read(count)
                        found = block.rfind(b"\n")
                        if found >= 0:
                            last_newline = position + found
                    if last_newline < 0:
                        raise ValueError("Manifest has no complete header; create a new manifest path.")
                    handle.truncate(last_newline + 1)
                    print("Discarded only an interrupted final manifest line; resuming.", flush=True)
    header: dict = {}
    inventory: dict[str, dict] = {}
    hashes: dict[str, str] = {}
    inventory_complete = False
    complete = False
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            kind = record.get("type")
            if complete:
                raise ValueError(f"Records after manifest completion at line {line_number}")
            if line_number == 1:
                if kind != "header" or record.get("schema") != SCHEMA:
                    raise ValueError("Unsupported manifest header/schema")
                header = record
            elif kind == "inventory" and not inventory_complete:
                relative = canonical_relative(record["path"])
                if relative in inventory or not isinstance(record.get("size"), int) or record["size"] < 0:
                    raise ValueError(f"Invalid/duplicate inventory at line {line_number}")
                if not isinstance(record.get("mtime_ns"), int):
                    raise ValueError(f"Invalid source timestamp at line {line_number}")
                inventory[relative] = {"size": record["size"], "mtime_ns": record["mtime_ns"]}
            elif kind == "inventory_complete" and not inventory_complete:
                if record.get("file_count") != len(inventory) or record.get("total_bytes") != sum(x["size"] for x in inventory.values()):
                    raise ValueError("Inventory footer does not match inventory")
                inventory_complete = True
            elif kind == "hash" and inventory_complete:
                relative = canonical_relative(record["path"])
                sha = record.get("sha256", "")
                if relative not in inventory or relative in hashes or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                    raise ValueError(f"Invalid/duplicate hash at line {line_number}")
                hashes[relative] = sha
            elif kind == "complete" and inventory_complete:
                if record.get("file_count") != len(inventory) or len(hashes) != len(inventory):
                    raise ValueError("Manifest completion does not have every file hash")
                complete = True
            else:
                raise ValueError(f"Unexpected record at line {line_number}: {kind!r}")
    if not header or not inventory_complete:
        raise ValueError("Manifest inventory incomplete; use a new manifest path")
    return header, inventory, hashes, complete


def create(root: Path, manifest: Path) -> int:
    started = time.monotonic()
    current, skipped = scan(root)
    print(f"Scanned {len(current):,} regular files, {sum(x['size'] for x in current.values()):,} bytes; skipped links/special files: {len(skipped)}", flush=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        temporary = manifest.with_name(manifest.name + ".creating")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            write_line(handle, {"type": "header", "schema": SCHEMA, "created_utc": utc_now(), "source_root": str(root), "skipped_links_or_special": skipped})
            for relative in sorted(current):
                write_line(handle, {"type": "inventory", "path": relative, **current[relative]})
            write_line(handle, {"type": "inventory_complete", "file_count": len(current), "total_bytes": sum(x["size"] for x in current.values())})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, manifest)
    header, inventory, hashes, complete = load_manifest(manifest, resume=True)
    if header.get("source_root") != str(root):
        raise ValueError("Create/resume must use the original source root; use verify for a different tree")
    if current != inventory or skipped != header.get("skipped_links_or_special", []):
        added = sorted(current.keys() - inventory.keys())
        missing = sorted(inventory.keys() - current.keys())
        changed = sorted(p for p in current.keys() & inventory.keys() if current[p] != inventory[p])
        raise ValueError(f"Source changed since inventory: added={added[:10]}, missing={missing[:10]}, size/mtime_changed={changed[:10]}; do not resume this frozen manifest")
    if complete:
        print(f"Manifest already complete: {manifest}; source size/mtime snapshot unchanged.", flush=True)
        return 0
    print(f"Resuming with {len(hashes):,}/{len(inventory):,} SHA256 hashes already recorded.", flush=True)
    last_progress = 0.0
    with manifest.open("a", encoding="utf-8", newline="\n") as handle:
        for relative in sorted(inventory):
            if relative in hashes:
                continue
            sha, _ = hash_stable_file(root, relative, inventory[relative])
            write_line(handle, {"type": "hash", "path": relative, "sha256": sha})
            handle.flush()
            hashes[relative] = sha
            now = time.monotonic()
            if now - last_progress >= 5:
                print(f"Hashed {len(hashes):,}/{len(inventory):,} files; elapsed {now-started:.1f}s; {relative}", flush=True)
                last_progress = now
            if len(hashes) % 100 == 0:
                os.fsync(handle.fileno())
        final, final_skipped = scan(root)
        if final != inventory or final_skipped != skipped:
            raise ValueError("Source tree changed during hashing; manifest is not marked complete")
        write_line(handle, {"type": "complete", "file_count": len(inventory), "total_bytes": sum(x["size"] for x in inventory.values()), "completed_utc": utc_now()})
        handle.flush()
        os.fsync(handle.fileno())
    print(f"COMPLETE: {manifest}; elapsed {time.monotonic()-started:.1f}s", flush=True)
    return 0


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify(root: Path, manifest: Path, report: Path) -> int:
    started = time.monotonic()
    result = {"schema": SCHEMA, "verified_utc": utc_now(), "root": str(root), "manifest": str(manifest), "ok": False}
    try:
        _, inventory, hashes, complete = load_manifest(manifest)
        if not complete:
            raise ValueError("Source manifest is not complete; resume create before verifying transfer")
        current, skipped = scan(root)
        missing = sorted(inventory.keys() - current.keys())
        extra = sorted(current.keys() - inventory.keys())
        mismatches: list[dict] = []
        read_errors: list[dict] = []
        matched = 0
        last_progress = 0.0
        for relative in sorted(inventory.keys() & current.keys()):
            if current[relative]["size"] != inventory[relative]["size"]:
                mismatches.append({"path": relative, "reason": "size_mismatch", "expected": inventory[relative]["size"], "actual": current[relative]["size"]})
                continue
            try:
                sha, _ = hash_stable_file(root, relative)
                if sha != hashes[relative]:
                    mismatches.append({"path": relative, "reason": "sha256_mismatch", "expected": hashes[relative], "actual": sha})
                else:
                    matched += 1
            except (OSError, ValueError) as exc:
                read_errors.append({"path": relative, "error": str(exc)})
            now = time.monotonic()
            if now - last_progress >= 5:
                print(f"Verified {matched:,}/{len(inventory):,}; mismatches={len(mismatches)}; read_errors={len(read_errors)}", flush=True)
                last_progress = now
        final, final_skipped = scan(root)
        tree_changed = final != current or final_skipped != skipped
        result.update({"expected_file_count": len(inventory), "actual_file_count": len(current), "expected_total_bytes": sum(x["size"] for x in inventory.values()), "actual_total_bytes": sum(x["size"] for x in current.values()), "matched_file_count": matched, "missing_files": missing, "extra_files": extra, "mismatches": mismatches, "read_errors": read_errors, "skipped_links_or_special": skipped, "tree_changed_during_verify": tree_changed})
        result["ok"] = not (missing or extra or mismatches or read_errors or skipped or tree_changed)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        result["error"] = str(exc)
    result["elapsed_seconds"] = time.monotonic() - started
    save_json(report, result)
    print(f"{'PASS' if result['ok'] else 'FAIL'}: report {report}", flush=True)
    return 0 if result["ok"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser("create", help="create or resume a local SHA256 JSONL manifest")
    create_parser.add_argument("--root", required=True)
    create_parser.add_argument("--manifest", required=True)
    verify_parser = subparsers.add_parser("verify", help="verify all transferred files against a complete source manifest")
    verify_parser.add_argument("--root", required=True)
    verify_parser.add_argument("--manifest", required=True)
    verify_parser.add_argument("--report", required=True)
    args = parser.parse_args()
    try:
        root = checked_root(args.root)
        manifest = output_outside_root(args.manifest, root)
        if args.command == "create":
            return create(root, manifest)
        report = output_outside_root(args.report, root)
        return verify(root, manifest, report)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
