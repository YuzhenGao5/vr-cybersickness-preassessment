#!/usr/bin/env python3
"""Resume VRNET using one large-file SFTP batch and bounded, uncompressed tars.

Run locally: python scripts/resume_vrnet_packed_upload.py
Python 3.10+, native OpenSSH, and the two sibling helpers are required.
Only the current small-file pack occupies local disk. Failed packs are retained.
Equal sizes only select transfer work; exit 0 requires final full SHA256 PASS.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tarfile
import time
import uuid

import resume_vrnet_upload as old

PROJECT = old.PROJECT
RESULTS = "vrnet_packed_resume_2026-10-14"
SCHEMA = "vrnet-packed-block-v1"
BUFFER = 8 * 1024 * 1024


def digest_file(handle) -> tuple[str, int]:
    digest, count = hashlib.sha256(), 0
    while data := handle.read(BUFFER):
        digest.update(data)
        count += len(data)
    return digest.hexdigest(), count


def directory_fd(path: Path, create: bool = False) -> int:
    """Linux extractor: walk absolute directories with no-follow directory FDs."""
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:]:
            child = child_directory_fd(fd, part, create)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def child_directory_fd(parent: int, name: str, create: bool) -> int:
    if create:
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent)
        except FileExistsError:
            pass
    return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)


def regular_destination(fd: int, name: str) -> None:
    try:
        info = os.stat(name, dir_fd=fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Destination is a link or non-regular file: {name}")


def save_report(fd: int, name: str, value: dict) -> None:
    regular_destination(fd, name)
    temporary = name + ".writing-" + uuid.uuid4().hex
    output = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
    with os.fdopen(output, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, name, src_dir_fd=fd, dst_dir_fd=fd)


def extract(args: argparse.Namespace) -> int:
    """Install only catalogued regular members after archive and member SHA checks."""
    project, root, staging, archive, report_path = [Path(old.remote_path(value)) for value in
        (args.project, args.root, args.staging, args.archive, args.report)]
    base = project / "results" / RESULTS
    if (root == project or root in project.parents or root == staging or root in staging.parents or staging.parent.parent != base
            or staging.name != "staging" or not re.fullmatch(r"\d{8}T\d{6}Z_[0-9a-f]{8}", staging.parent.name)):
        raise ValueError("Staging must be the explicit run directory outside raw data")
    if (archive.parent != staging or not re.fullmatch(r"block-\d{6}\.tar", archive.name)
            or report_path != staging / (archive.stem + ".report.json")):
        raise ValueError("Archive/report must be direct, generated children of this run staging directory")
    if args.size <= 0 or not re.fullmatch(r"[0-9a-f]{64}", args.sha256):
        raise ValueError("Invalid expected archive size/SHA256")
    stage_fd, root_fd = directory_fd(staging), None
    result = {"schema": SCHEMA, "archive": str(archive), "root": str(root),
              "archive_sha256": args.sha256, "archive_size": args.size,
              "ok": False, "copied_file_count": 0, "copied_bytes": 0,
              "archive_cleanup": "Exact archive is deleted only after the successful report is saved."}
    started = time.monotonic()
    try:
        root_fd = directory_fd(root)
        if os.fstat(root_fd).st_dev != os.fstat(stage_fd).st_dev:
            raise ValueError("Raw root and staging must share a filesystem for atomic replacement")
        archive_fd = os.open(archive.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=stage_fd)
        with os.fdopen(archive_fd, "rb") as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size != args.size:
                raise ValueError("Archive is not a regular file of the expected size")
            sha, count = digest_file(source)
            if sha != args.sha256 or count != args.size:
                raise ValueError("Whole archive SHA256/size mismatch")
            source.seek(0)
            with tarfile.open(fileobj=source, mode="r:") as pack:
                first = pack.next()
                if first is None or first.name != "catalog.json" or not first.isreg() or first.linkname or first.size > 64 * 1024 * 1024:
                    raise ValueError("First archive member must be a bounded regular catalog.json")
                with pack.extractfile(first) as handle:
                    catalog = json.load(handle)
                if not isinstance(catalog, dict) or catalog.get("schema") != SCHEMA or not isinstance(catalog.get("files"), list):
                    raise ValueError("Invalid block catalog")
                expected = {}
                for entry in catalog["files"]:
                    relative = old.relative_path(entry["path"])
                    size, sha = entry["size"], entry["sha256"]
                    if relative in expected or type(size) is not int or size < 0 or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
                        raise ValueError("Invalid/duplicate catalog file")
                    expected[relative] = (size, sha)
                if not expected:
                    raise ValueError("Empty block catalog")
                result.update({"expected_file_count": len(expected), "expected_total_bytes": sum(value[0] for value in expected.values())})
                members = pack.getmembers()
                names = set()
                for member in members[1:]:
                    if not member.isreg() or member.linkname or not member.name.startswith("raw/"):
                        raise ValueError(f"Unexpected non-regular member: {member.name}")
                    relative = old.relative_path(member.name[4:])
                    if relative in names or relative not in expected or member.size != expected[relative][0]:
                        raise ValueError(f"Duplicate/unexpected member or wrong size: {relative}")
                    names.add(relative)
                if len(members) != len(expected) + 1 or names != expected.keys():
                    raise ValueError("Archive member set differs from its catalog")
                for member in members[1:]:
                    relative = member.name[4:]
                    parent_fd = os.dup(root_fd)
                    temporary = ".extract-" + uuid.uuid4().hex
                    try:
                        parts = PurePosixPath(relative).parts
                        for part in parts[:-1]:
                            child = child_directory_fd(parent_fd, part, True)
                            os.close(parent_fd)
                            parent_fd = child
                        regular_destination(parent_fd, parts[-1])
                        output = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=stage_fd)
                        digest, copied = hashlib.sha256(), 0
                        with pack.extractfile(member) as handle, os.fdopen(output, "wb") as target:
                            while data := handle.read(BUFFER):
                                target.write(data)
                                digest.update(data)
                                copied += len(data)
                            target.flush()
                        if (copied, digest.hexdigest()) != expected[relative]:
                            raise ValueError(f"Member SHA256/size mismatch: {relative}")
                        regular_destination(parent_fd, parts[-1])
                        os.replace(temporary, parts[-1], src_dir_fd=stage_fd, dst_dir_fd=parent_fd)
                        result["copied_file_count"] += 1
                        result["copied_bytes"] += copied
                    finally:
                        os.close(parent_fd)
                        # Only this exact generated temporary file can be removed.
                        try:
                            os.unlink(temporary, dir_fd=stage_fd)
                        except FileNotFoundError:
                            pass
            after = os.fstat(source.fileno())
            identity = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            if identity(before) != identity(after) or identity(before) != identity(os.stat(archive.name, dir_fd=stage_fd, follow_symlinks=False)):
                raise ValueError("Archive changed during extraction")
        result.update({"ok": True, "completed_utc": old.utc_now(), "elapsed_seconds": time.monotonic() - started})
        save_report(stage_fd, report_path.name, result)
        # The directory FD was opened without links; unlink only this archive.
        os.unlink(archive.name, dir_fd=stage_fd)
        print(f"Block installed: {result['copied_file_count']:,} files / {result['copied_bytes']:,} bytes; report {report_path}", flush=True)
        return 0
    except BaseException as exc:
        result.update({"ok": False, "error": str(exc), "elapsed_seconds": time.monotonic() - started})
        save_report(stage_fd, report_path.name, result)
        raise
    finally:
        if root_fd is not None:
            os.close(root_fd)
        os.close(stage_fd)


def groups(paths: list[str], inventory: dict, limit: int):
    block, estimate = [], 10240
    for relative in paths:
        # Include padded content, tar/PAX headers and catalog overhead.
        cost = ((inventory[relative]["size"] + 511) // 512) * 512 + 2048 + 2 * len(relative.encode("utf-8")) + 256
        if block and estimate + cost > limit:
            yield block
            block, estimate = [], 10240
        block.append(relative)
        estimate += cost
    if block:
        yield block


def make_pack(path: Path, paths: list[str], root: Path, inventory: dict, hashes: dict, verifier) -> tuple[str, int]:
    catalog = json.dumps({"schema": SCHEMA, "files": [{"path": p, "size": inventory[p]["size"], "sha256": hashes[p]} for p in paths]},
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    old.safe_output(path)
    with tarfile.open(path, "x", format=tarfile.PAX_FORMAT) as pack:
        entry = tarfile.TarInfo("catalog.json")
        entry.size = len(catalog)
        pack.addfile(entry, io.BytesIO(catalog))
        for relative in paths:
            local = verifier.safe_file(root, relative)
            before = local.stat()
            if (before.st_size, before.st_mtime_ns) != (inventory[relative]["size"], inventory[relative]["mtime_ns"]):
                raise ValueError(f"Frozen source changed before packing: {relative}")
            entry = tarfile.TarInfo("raw/" + relative)
            entry.size, entry.mode = before.st_size, 0o600
            with local.open("rb") as handle:
                pack.addfile(entry, handle)
            after = verifier.safe_file(root, relative).stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev) != (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev):
                raise ValueError(f"Frozen source changed while packing: {relative}")
    with path.open("rb") as handle:
        return digest_file(handle)


def upload(args: argparse.Namespace) -> int:
    import verify_vrnet_transfer as verifier

    if not re.fullmatch(r"(?:[A-Za-z0-9_][A-Za-z0-9_.-]*@)?[A-Za-z0-9_][A-Za-z0-9_.-]*", args.host):
        raise ValueError("Host must be a plain hostname or user@hostname")
    remote_root, remote_project = old.remote_path(args.remote_root), old.remote_path(args.remote_project)
    if remote_project == remote_root or remote_project.startswith(remote_root + "/"):
        raise ValueError("Remote project/results must be outside the raw data root")
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", args.remote_python) or args.remote_python.startswith("-"):
        raise ValueError("Remote Python must be an executable name or plain absolute path")
    if args.small_file_mib < 1 or args.block_mib <= args.small_file_mib:
        raise ValueError("Require 1 <= --small-file-mib < --block-mib")
    root = verifier.checked_root(args.root)
    old.safe_directory(root)
    manifest = verifier.output_outside_root(args.manifest, root)
    header, inventory, hashes, complete = verifier.load_manifest(manifest)
    if not complete or Path(header.get("source_root", "")).resolve(strict=True) != root:
        raise ValueError("Need the complete frozen SHA256 manifest for this source root")
    for relative in inventory:
        old.relative_path(relative)
    old.check_frozen_source(verifier, root, header, inventory)
    native = {name: old.native_tool(name) for name in ("ssh", "scp", "sftp")}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run = PROJECT / "results" / RESULTS / run_id
    if run == root or root in run.parents:
        raise ValueError("Local run results must be outside the raw data root")
    run = old.safe_directory(run, create=True)
    runner = old.Runner(run)
    remote_run = remote_project + "/results/" + RESULTS + "/" + run_id
    staging = remote_run + "/staging"
    if remote_run == remote_root or remote_run.startswith(remote_root + "/"):
        raise ValueError("Remote run/staging must be outside the raw data root")
    scripts = remote_project + "/scripts/"
    helper, legacy, checker = [scripts + name for name in ("resume_vrnet_packed_upload.py", "resume_vrnet_upload.py", "verify_vrnet_transfer.py")]
    remote_manifest = remote_project + "/results/vrnet_transfer_manifest_2026-10-12.jsonl"
    remote_inventory, remote_report = remote_run + "/remote_inventory.json", remote_run + "/remotecheck.json"
    summary = {"started_utc": old.utc_now(), "ok": False, "verification": "not_run", "source_root": str(root),
               "host": args.host, "remote_root": remote_root, "local_results": str(run), "remote_results": remote_run,
               "expected_file_count": len(inventory), "expected_total_bytes": sum(x["size"] for x in inventory.values()),
               "small_file_mib": args.small_file_mib, "block_mib": args.block_mib,
               "size_skip_notice": "Equal sizes only skip transfer; full final SHA256 verification is required.",
               "large_file_notice": "Ordinary put restarts a failed large file from byte zero; no append/resume mode is used."}
    summary_path = run / "summary.json"
    old.save_json(summary_path, summary)
    print(f"Run records: {run}\n{summary['size_skip_notice']}\n{summary['large_file_notice']}", flush=True)
    try:
        runner.run("Prepare remote directories and helper destinations", old.ssh_command(native["ssh"], args.host,
                   [args.remote_python, "-c", old.BOOTSTRAP, remote_root, scripts.rstrip("/"), remote_run, staging,
                    helper, legacy, checker, remote_manifest]))
        runner.run("Upload the three small standard-library helpers", [native["scp"], "-oBatchMode=no",
                   str(PROJECT / "scripts/resume_vrnet_upload.py"), str(Path(__file__).resolve()),
                   str(PROJECT / "scripts/verify_vrnet_transfer.py"), args.host + ":" + scripts])
        runner.run("Inventory current remote sizes", old.ssh_command(native["ssh"], args.host,
                   [args.remote_python, legacy, "inventory", "--root", remote_root, "--output", remote_inventory]))
        runner.run("Download current size inventory", [native["scp"], "-oBatchMode=no", args.host + ":" + remote_inventory, str(run / "remote_inventory.json")])
        remote_files, remote_dirs = old.load_remote_inventory(run / "remote_inventory.json", remote_root)
        forced = old.repair_paths(Path(args.repair_report) if args.repair_report else None, remote_root, inventory)
        pending, required = [], set()
        for relative in sorted(inventory):
            if relative in remote_dirs:
                raise ValueError(f"Remote directory conflicts with source file: {relative}")
            parts = PurePosixPath(relative).parts
            for length in range(1, len(parts)):
                parent = PurePosixPath(*parts[:length]).as_posix()
                if parent in remote_files:
                    raise ValueError(f"Remote file conflicts with parent directory: {parent}")
            if relative in forced or remote_files.get(relative) != inventory[relative]["size"]:
                pending.append(relative)
        large = [p for p in pending if inventory[p]["size"] > args.small_file_mib * 1024 * 1024]
        small = [p for p in pending if inventory[p]["size"] <= args.small_file_mib * 1024 * 1024]
        for relative in large:
            parts = PurePosixPath(relative).parts
            required.update(PurePosixPath(*parts[:n]).as_posix() for n in range(1, len(parts)))
        plan = {"large_files": large, "small_files": small, "force_reupload_files": sorted(forced),
                "upload_file_count": len(pending), "upload_bytes": sum(inventory[p]["size"] for p in pending),
                "skipped_equal_size_file_count": len(inventory) - len(pending),
                "skipped_equal_size_bytes": sum(inventory[p]["size"] for p in inventory.keys() - set(pending)),
                "extra_remote_files": sorted(remote_files.keys() - inventory.keys())}
        old.save_json(run / "plan.json", plan)
        summary.update({key: value for key, value in plan.items() if key.endswith("count") or key.endswith("bytes")})
        print(f"Plan: {len(large):,} large files; {len(small):,} packed small files; {plan['upload_bytes']:,} pending bytes.", flush=True)
        if large:
            batch = run / "large-files.sftp"
            with batch.open("x", encoding="utf-8", newline="\n") as handle:
                old.batch_line(handle, "progress")
                for parent in sorted(required - remote_dirs, key=lambda p: (p.count("/"), p)):
                    old.batch_line(handle, "mkdir " + old.sftp_quote(remote_root + "/" + parent))
                for relative in large:
                    local = verifier.safe_file(root, relative)
                    info = local.stat()
                    if (info.st_size, info.st_mtime_ns) != (inventory[relative]["size"], inventory[relative]["mtime_ns"]):
                        raise ValueError(f"Frozen source changed: {relative}")
                    old.batch_line(handle, "put " + old.sftp_quote(local.as_posix()) + " " + old.sftp_quote(remote_root + "/" + relative))
            started = time.monotonic()
            amount = sum(inventory[p]["size"] for p in large)
            runner.run(f"Upload {len(large):,} large files / {amount:,} bytes in one native SFTP connection",
                       [native["sftp"], "-oBatchMode=no", "-N", "-b", str(batch), args.host])
            summary.update({"large_completed_files": len(large), "large_completed_bytes": amount, "large_elapsed_seconds": time.monotonic() - started})
            old.save_json(summary_path, summary)
        summary["small_completed_files"] = summary["small_completed_bytes"] = summary["small_completed_blocks"] = 0
        blocks = list(groups(small, inventory, args.block_mib * 1024 * 1024))
        for index, paths in enumerate(blocks, 1):
            name = f"block-{index:06d}.tar"
            pack, remote_pack = run / name, staging + "/" + name
            started = time.monotonic()
            print(f"Packing block {index}/{len(blocks)}: {len(paths):,} files...", flush=True)
            if shutil.disk_usage(run).free < args.block_mib * 1024 * 1024 + 64 * 1024 * 1024:
                raise ValueError("Insufficient local free space for one block plus a 64 MiB reserve")
            sha, size = make_pack(pack, paths, root, inventory, hashes, verifier)
            if size > args.block_mib * 1024 * 1024:
                raise ValueError(f"Pack exceeds configured block limit; retained: {pack}")
            batch = run / "current-pack.sftp"
            with batch.open("w", encoding="utf-8", newline="\n") as handle:
                old.batch_line(handle, "progress")
                old.batch_line(handle, "put " + old.sftp_quote(pack.as_posix()) + " " + old.sftp_quote(remote_pack))
            runner.run(f"Upload block {index}/{len(blocks)} / {size:,} archive bytes", [native["sftp"], "-oBatchMode=no", "-N", "-b", str(batch), args.host])
            runner.run(f"Verify and install block {index}/{len(blocks)}", old.ssh_command(native["ssh"], args.host,
                       [args.remote_python, helper, "extract", "--project", remote_project, "--root", remote_root,
                        "--staging", staging, "--archive", remote_pack, "--size", str(size), "--sha256", sha,
                        "--report", staging + f"/block-{index:06d}.report.json"]))
            amount = sum(inventory[p]["size"] for p in paths)
            runner.event({"event": "block_completed", "index": index, "files": len(paths), "raw_bytes": amount,
                          "archive_bytes": size, "archive_sha256": sha, "elapsed_seconds": time.monotonic() - started})
            summary["small_completed_files"] += len(paths)
            summary["small_completed_bytes"] += amount
            summary["small_completed_blocks"] += 1
            old.save_json(summary_path, summary)
            if pack.parent != run or old.is_link(pack.lstat()) or not stat.S_ISREG(pack.lstat().st_mode):
                raise ValueError("Refusing unsafe local temporary-pack deletion")
            pack.unlink()
            print(f"Small files installed: {summary['small_completed_files']:,}/{len(small):,}; {summary['small_completed_bytes']:,} bytes.", flush=True)
        summary["transfer_completed"] = True
        old.check_frozen_source(verifier, root, header, inventory)
        runner.run("Upload frozen SHA256 manifest (verification metadata)", [native["scp"], "-oBatchMode=no", str(manifest), args.host + ":" + remote_manifest])
        summary["verification"] = "running"
        old.save_json(summary_path, summary)
        code = runner.run("Verify ALL remote raw files against frozen SHA256 hashes", old.ssh_command(native["ssh"], args.host,
                          [args.remote_python, checker, "verify", "--root", remote_root, "--manifest", remote_manifest,
                           "--report", remote_report]), allow_failure=True)
        runner.run("Download final SHA256 report", [native["scp"], "-oBatchMode=no", args.host + ":" + remote_report, str(run / "remotecheck.json")])
        with (run / "remotecheck.json").open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        summary["ok"] = (code == 0 and report.get("ok") is True and report.get("schema") == "vrnet-transfer-sha256-v1"
                         and report.get("root") == remote_root and report.get("manifest") == remote_manifest
                         and report.get("expected_file_count") == len(inventory)
                         and report.get("expected_total_bytes") == summary["expected_total_bytes"])
        summary.update({"verification": "passed" if summary["ok"] else "failed", "verification_returncode": code,
                        "verification_report": str(run / "remotecheck.json"), "finished_utc": old.utc_now()})
        old.save_json(summary_path, summary)
        print(f"{'PASS' if summary['ok'] else 'FAIL'}: full remote SHA256 verification; records {run}", flush=True)
        return 0 if summary["ok"] else 1
    except BaseException as exc:
        summary.update({"error": str(exc), "finished_utc": old.utc_now()})
        old.save_json(summary_path, summary)
        retained_packs = sum(1 for path in run.glob("block-*.tar") if path.is_file())
        print(f"Stopped; run records retained: {run}; retained local packs: {retained_packs}\nRerun to inventory again and skip equal-size files.", file=sys.stderr, flush=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    local = sub.add_parser("upload", help="resume with bounded packs, then require full SHA256 PASS")
    for flag, default in (("root", str(PROJECT / "data/raw/vrnet")), ("manifest", str(PROJECT / "results/vrnet_transfer_manifest_2026-10-12.jsonl")),
                          ("host", "ygao930@irl-ai"), ("remote-root", old.DEFAULT_REMOTE_ROOT),
                          ("remote-project", old.DEFAULT_REMOTE_PROJECT), ("remote-python", "python3")):
        local.add_argument("--" + flag, default=default)
    local.add_argument("--repair-report", help="force paths listed by a previous remote SHA256 report")
    local.add_argument("--small-file-mib", type=int, default=1)
    local.add_argument("--block-mib", type=int, default=256)
    remote = sub.add_parser("extract", help="internal Linux block verifier/extractor; refuses links/special files")
    for flag in ("project", "root", "staging", "archive", "sha256", "report"):
        remote.add_argument("--" + flag, required=True)
    remote.add_argument("--size", type=int, required=True)
    argv = sys.argv[1:]
    args = parser.parse_args(argv if argv and argv[0] in ("upload", "extract") else ["upload", *argv])
    try:
        return extract(args) if args.command == "extract" else upload(args)
    except KeyboardInterrupt:
        print("Interrupted; raw source is unchanged. Rerun to resume.", file=sys.stderr, flush=True)
        return 130
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, RuntimeError, tarfile.TarError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return exc.returncode if isinstance(exc, old.CommandError) and exc.returncode > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
