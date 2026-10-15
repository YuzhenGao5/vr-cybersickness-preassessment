#!/usr/bin/env python3
"""Resume the frozen VRNET transfer with native OpenSSH and finish with SHA256.

Run locally: python scripts/resume_vrnet_upload.py
Python 3.10+ and ssh/scp/sftp are required. Password prompts and transfer progress
use the caller's terminal. Passwords are never requested or saved by this script.
Equal remote sizes are only a transfer shortcut; only the final verifier can PASS.
The inventory subcommand is a standalone Linux helper using only stdlib.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import stat
import subprocess
import sys
import uuid


INVENTORY_SCHEMA = "vrnet-remote-size-inventory-v1"
PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_REMOTE_ROOT = "/data/scratch/ygao930/vrnet_migration_2026-10-12/vrnet"
DEFAULT_REMOTE_PROJECT = "/data/scratch/ygao930/vr-cybersickness-preassessment"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError(f"Invalid relative path: {value!r}")
    if "\\" in value or ":" in value or value.startswith("/") or any(p in ("", ".", "..") for p in value.split("/")):
        raise ValueError(f"Unsafe relative path: {value!r}")
    return value


def remote_path(value: str) -> str:
    # Plain ASCII bootstrap paths also keep scp safe with older SCP transports.
    if not value.startswith("/") or value == "/" or not re.fullmatch(r"/[A-Za-z0-9_./-]+", value):
        raise ValueError(f"Remote root/project must be an absolute plain ASCII path: {value!r}")
    if any(p in ("", ".", "..") for p in value[1:].split("/")):
        raise ValueError(f"Unsafe remote path: {value!r}")
    return value


def is_link(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def safe_directory(path: Path, create: bool = False) -> Path:
    path = path.absolute()
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            if not create:
                raise
            current.mkdir()
            info = current.lstat()
        if is_link(info) or not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"Directory is a symlink, reparse point or non-directory: {current}")
    return path


def safe_output(path: Path) -> Path:
    path = path.absolute()
    safe_directory(path.parent, create=True)
    if path.exists() or path.is_symlink():
        info = path.lstat()
        if is_link(info) or not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Output is a symlink or non-regular file: {path}")
    return path


def save_json(path: Path, value: dict) -> None:
    path = safe_output(path)
    temporary = path.with_name(path.name + ".writing-" + uuid.uuid4().hex)
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def inventory(root_value: str, output_value: str) -> int:
    root = safe_directory(Path(remote_path(root_value)))
    output = Path(remote_path(output_value))
    if output == root or root in output.parents:
        raise ValueError("Inventory output must be outside the raw data tree")
    files: dict[str, int] = {}
    directories: list[str] = []
    stack = [root]
    while stack:
        directory = stack.pop()
        safe_directory(directory)
        with os.scandir(directory) as children:
            for child in children:
                path = Path(child.path)
                relative = relative_path(path.relative_to(root).as_posix())
                info = child.stat(follow_symlinks=False)
                if is_link(info):
                    raise ValueError(f"Refusing remote symlink/reparse point: {relative}")
                if stat.S_ISDIR(info.st_mode):
                    directories.append(relative)
                    stack.append(path)
                elif stat.S_ISREG(info.st_mode):
                    files[relative] = info.st_size
                else:
                    raise ValueError(f"Refusing remote special file: {relative}")
    save_json(output, {
        "schema": INVENTORY_SCHEMA, "root": str(root), "created_utc": utc_now(),
        "files": files, "directories": sorted(directories),
        "file_count": len(files), "total_bytes": sum(files.values()),
    })
    print(f"Remote size inventory: {len(files):,} files, {sum(files.values()):,} bytes; {output}", flush=True)
    return 0


# The first SSH call creates/checks the target root before uploading the helper.
# Arguments are passed separately through a shlex-quoted remote command.
BOOTSTRAP = r'''
import os, pathlib, stat, sys
def directory(value):
    p = pathlib.Path(value)
    q = pathlib.Path('/')
    for part in p.parts[1:]:
        q = q / part
        try:
            info = q.lstat()
        except FileNotFoundError:
            q.mkdir()
            info = q.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError('Unsafe directory: ' + str(q))
for value in sys.argv[1:5]:
    directory(value)
for value in sys.argv[5:]:
    p = pathlib.Path(value)
    try:
        info = p.lstat()
    except FileNotFoundError:
        continue
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError('Unsafe upload/report destination: ' + str(p))
print('Remote directories and upload destinations checked.', flush=True)
'''


def native_tool(name: str) -> str:
    if os.name == "nt":
        candidate = Path(os.environ.get("WINDIR", "C:/Windows")) / "System32" / "OpenSSH" / (name + ".exe")
        if candidate.is_file():
            return str(candidate)
    found = shutil.which(name)
    if not found:
        raise ValueError(f"Required OpenSSH tool is missing: {name}")
    return found


class CommandError(RuntimeError):
    def __init__(self, stage: str, returncode: int):
        super().__init__(f"{stage} failed with exit code {returncode}; see terminal and saved commands.log")
        self.returncode = returncode


class Runner:
    def __init__(self, directory: Path):
        self.log = directory / "commands.log"

    def event(self, value: dict) -> None:
        with self.log.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"time_utc": utc_now(), **value}, ensure_ascii=False) + "\n")

    def run(self, stage: str, argv: list[str], *, allow_failure: bool = False) -> int:
        argv = [argv[0], "-oServerAliveInterval=30", "-oServerAliveCountMax=6", *argv[1:]]
        print(f"\n{stage}", flush=True)
        self.event({"event": "command_started", "stage": stage, "argv": argv})
        try:
            # Inherit all stdio and the console: native OpenSSH handles passwords
            # and progress. commands.log intentionally contains no terminal dump.
            result = subprocess.run(argv, shell=False)
        except BaseException as exc:
            self.event({"event": "command_exception", "stage": stage, "error": str(exc)})
            raise
        self.event({"event": "command_finished", "stage": stage, "returncode": result.returncode})
        if result.returncode and not allow_failure:
            raise CommandError(stage, result.returncode)
        return result.returncode


def ssh_command(ssh: str, host: str, argv: list[str]) -> list[str]:
    return [ssh, "-oBatchMode=no", host, shlex.join(argv)]


def sftp_quote(value: str) -> str:
    if any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError(f"Path cannot be represented in an SFTP batch: {value!r}")
    # OpenSSH's quoted-argument parser itself protects glob metacharacters.
    # Only literal backslashes and quotes need explicit batch escaping.
    escaped = "".join("\\" + c if c in '\\"' else c for c in value)
    return '"' + escaped + '"'


def batch_line(handle, command: str) -> None:
    value = command + "\n"
    if len(value.encode("utf-8")) > 2047:
        raise ValueError("SFTP command exceeds the native 2047-byte batch line limit")
    handle.write(value)


def load_remote_inventory(path: Path, expected_root: str) -> tuple[dict[str, int], set[str]]:
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if result.get("schema") != INVENTORY_SCHEMA or result.get("root") != expected_root:
        raise ValueError("Downloaded inventory has an unexpected schema/root")
    files = result.get("files")
    directories = result.get("directories")
    if not isinstance(files, dict) or not isinstance(directories, list):
        raise ValueError("Malformed remote size inventory")
    for path_value, size in files.items():
        relative_path(path_value)
        if type(size) is not int or size < 0:
            raise ValueError(f"Invalid remote file size: {path_value!r}")
    dirs = {relative_path(value) for value in directories}
    if len(dirs) != len(directories) or dirs & files.keys():
        raise ValueError("Duplicate/conflicting remote inventory paths")
    if result.get("file_count") != len(files) or result.get("total_bytes") != sum(files.values()):
        raise ValueError("Remote size inventory footer does not match")
    for value in set(files) | dirs:
        parts = PurePosixPath(value).parts
        for length in range(1, len(parts)):
            if PurePosixPath(*parts[:length]).as_posix() not in dirs:
                raise ValueError(f"Remote inventory is missing a parent directory: {value}")
    return files, dirs


def repair_paths(path: Path | None, expected_root: str, local_inventory: dict) -> set[str]:
    if path is None:
        return set()
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("schema") != "vrnet-transfer-sha256-v1" or report.get("root") != expected_root:
        raise ValueError("Repair report must be a verifier report for this remote root")
    values = list(report.get("missing_files", []))
    values.extend(item["path"] for key in ("mismatches", "read_errors") for item in report.get(key, []))
    paths = {relative_path(value) for value in values}
    if paths - local_inventory.keys():
        raise ValueError("Repair report names paths absent from the frozen manifest")
    return paths


def check_frozen_source(verifier, root: Path, header: dict, expected: dict) -> None:
    print("Checking that local source sizes and timestamps still match the frozen manifest...", flush=True)
    current, skipped = verifier.scan(root)
    if current != expected or skipped != header.get("skipped_links_or_special", []):
        added = sorted(current.keys() - expected.keys())[:10]
        missing = sorted(expected.keys() - current.keys())[:10]
        changed = sorted(p for p in current.keys() & expected.keys() if current[p] != expected[p])[:10]
        raise ValueError(f"Frozen source changed: added={added}, missing={missing}, size/mtime_changed={changed}")
    print(f"Frozen local source unchanged: {len(expected):,} files, {sum(x['size'] for x in expected.values()):,} bytes.", flush=True)


def upload(args: argparse.Namespace) -> int:
    # The copied helper's inventory mode never needs this sibling module.
    import verify_vrnet_transfer as verifier

    if not re.fullmatch(r"(?:[A-Za-z0-9_][A-Za-z0-9_.-]*@)?[A-Za-z0-9_][A-Za-z0-9_.-]*", args.host):
        raise ValueError("Host must be a plain hostname or user@hostname")
    remote_root = remote_path(args.remote_root)
    remote_project = remote_path(args.remote_project)
    if remote_project == remote_root or remote_project.startswith(remote_root + "/"):
        raise ValueError("Remote project/results must be outside the raw data root")
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", args.remote_python) or args.remote_python.startswith("-"):
        raise ValueError("Remote Python must be an executable name or plain absolute path")
    root = verifier.checked_root(args.root)
    manifest = verifier.output_outside_root(args.manifest, root)
    header, local_inventory, _hashes, complete = verifier.load_manifest(manifest)
    if not complete:
        raise ValueError("Local SHA256 manifest is incomplete")
    if Path(header.get("source_root", "")).resolve(strict=True) != root:
        raise ValueError("Local root differs from the manifest's frozen source root")
    for relative in local_inventory:
        relative_path(relative)
    check_frozen_source(verifier, root, header, local_inventory)
    tools = {name: native_tool(name) for name in ("ssh", "scp", "sftp")}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_directory = safe_directory(PROJECT / "results" / "vrnet_resume_2026-10-13" / run_id, create=True)
    runner = Runner(run_directory)
    remote_run = remote_project + "/results/vrnet_resume_2026-10-13/" + run_id
    remote_helper = remote_project + "/scripts/resume_vrnet_upload.py"
    remote_verifier = remote_project + "/scripts/verify_vrnet_transfer.py"
    remote_manifest = remote_project + "/results/vrnet_transfer_manifest_2026-10-12.jsonl"
    remote_inventory = remote_run + "/remote_inventory.json"
    remote_report = remote_run + "/remotecheck.json"
    summary = {
        "started_utc": utc_now(), "source_root": str(root), "manifest": str(manifest),
        "host": args.host, "remote_root": remote_root, "remote_project": remote_project,
        "local_results": str(run_directory), "remote_results": remote_run,
        "expected_file_count": len(local_inventory),
        "expected_total_bytes": sum(x["size"] for x in local_inventory.values()),
        "ok": False, "verification": "not_run",
        "size_skip_notice": "Equal sizes are skipped only to resume transfer; size equality does not verify SHA256.",
        "log_notice": "commands.log records stages, argv, errors and return codes; native terminal output/passwords are not captured.",
    }
    summary_path = run_directory / "summary.json"
    save_json(summary_path, summary)
    print(f"Run records: {run_directory}\n{summary['size_skip_notice']}", flush=True)
    try:
        destinations = [remote_helper, remote_verifier, remote_manifest, remote_inventory, remote_report]
        bootstrap_args = [args.remote_python, "-c", BOOTSTRAP, remote_root,
                          remote_project + "/scripts", remote_project + "/results", remote_run, *destinations]
        runner.run("1/6 Prepare remote directories and reject links/conflicts", ssh_command(tools["ssh"], args.host, bootstrap_args))
        runner.run("2/6 Upload the small inventory helper (this is not raw data progress)",
                   [tools["scp"], "-oBatchMode=no", str(Path(__file__).resolve()), args.host + ":" + remote_helper])
        runner.run("3/6 Inventory existing remote files by size", ssh_command(tools["ssh"], args.host,
                   [args.remote_python, remote_helper, "inventory", "--root", remote_root, "--output", remote_inventory]))
        inventory_file = run_directory / "remote_inventory.json"
        runner.run("Download the small remote inventory", [tools["scp"], "-oBatchMode=no", args.host + ":" + remote_inventory, str(inventory_file)])
        remote_files, remote_dirs = load_remote_inventory(inventory_file, remote_root)
        forced = repair_paths(Path(args.repair_report) if args.repair_report else None, remote_root, local_inventory)
        upload_paths: list[str] = []
        required_dirs: set[str] = set()
        skipped_count = skipped_bytes = 0
        for relative, expected in local_inventory.items():
            parts = PurePosixPath(relative).parts
            if relative in remote_dirs:
                raise ValueError(f"Remote directory conflicts with source file: {relative}")
            for length in range(1, len(parts)):
                parent = PurePosixPath(*parts[:length]).as_posix()
                if parent in remote_files:
                    raise ValueError(f"Remote file conflicts with required directory: {parent}")
                required_dirs.add(parent)
            if relative not in forced and remote_files.get(relative) == expected["size"]:
                skipped_count += 1
                skipped_bytes += expected["size"]
            else:
                upload_paths.append(relative)
        missing_dirs = sorted(required_dirs - remote_dirs, key=lambda value: (value.count("/"), value))
        upload_paths.sort()
        plan = {
            "upload_files": upload_paths, "create_directories": missing_dirs,
            "force_reupload_files": sorted(forced),
            "upload_file_count": len(upload_paths),
            "upload_bytes": sum(local_inventory[p]["size"] for p in upload_paths),
            "skipped_equal_size_file_count": skipped_count, "skipped_equal_size_bytes": skipped_bytes,
            "extra_remote_files": sorted(remote_files.keys() - local_inventory.keys()),
            "size_skip_notice": summary["size_skip_notice"],
        }
        save_json(run_directory / "plan.json", plan)
        summary.update({key: plan[key] for key in ("upload_file_count", "upload_bytes", "skipped_equal_size_file_count", "skipped_equal_size_bytes")})
        batch_file = run_directory / "upload.sftp"
        print(f"Generating the upload command list for {len(upload_paths):,} selected files...", flush=True)
        with batch_file.open("x", encoding="utf-8", newline="\n") as handle:
            batch_line(handle, "progress")
            for relative in missing_dirs:
                batch_line(handle, "mkdir " + sftp_quote(remote_root + "/" + relative))
            for index, relative in enumerate(upload_paths, start=1):
                local_file = verifier.safe_file(root, relative)
                info = local_file.stat()
                if (info.st_size, info.st_mtime_ns) != (local_inventory[relative]["size"], local_inventory[relative]["mtime_ns"]):
                    raise ValueError(f"Local file changed before batch generation: {relative}")
                batch_line(handle, "put " + sftp_quote(local_file.as_posix()) + " " + sftp_quote(remote_root + "/" + relative))
                if index % 5000 == 0:
                    print(f"Prepared upload commands: {index:,}/{len(upload_paths):,}", flush=True)
        save_json(summary_path, summary)
        print(f"Plan: upload {len(upload_paths):,} files / {plan['upload_bytes']:,} bytes; skip {skipped_count:,} equal-size files.\nEach selected file is uploaded in full using ordinary put. {summary['size_skip_notice']}", flush=True)
        if missing_dirs or upload_paths:
            runner.run("4/6 Upload missing/incomplete raw files (native SFTP progress)",
                       [tools["sftp"], "-oBatchMode=no", "-N", "-b", str(batch_file), args.host])
        else:
            print("4/6 No raw files need size-based upload; SHA256 verification is still required.", flush=True)
        summary["transfer_batch_completed"] = True
        check_frozen_source(verifier, root, header, local_inventory)
        runner.run("5/6 Upload the complete SHA256 manifest (about 63 MB; this is verification metadata)",
                   [tools["scp"], "-oBatchMode=no", str(manifest), args.host + ":" + remote_manifest])
        runner.run("Upload the standard-library SHA256 verifier", [tools["scp"], "-oBatchMode=no",
                   str(PROJECT / "scripts" / "verify_vrnet_transfer.py"), args.host + ":" + remote_verifier])
        if args.skip_final_verification:
            summary["verification"] = "skipped_by_option"
            summary["finished_utc"] = utc_now()
            save_json(summary_path, summary)
            print(f"TRANSFER FINISHED; SHA256 NOT VERIFIED (--skip-final-verification).\nRecords: {run_directory}", flush=True)
            return 0
        summary["verification"] = "running"
        save_json(summary_path, summary)
        verification_code = runner.run("6/6 Verify ALL remote raw files against frozen SHA256 hashes; this may take time", ssh_command(tools["ssh"], args.host,
                            [args.remote_python, remote_verifier, "verify", "--root", remote_root,
                             "--manifest", remote_manifest, "--report", remote_report]), allow_failure=True)
        local_report = run_directory / "remotecheck.json"
        runner.run("Download final verification report (also retained remotely)",
                   [tools["scp"], "-oBatchMode=no", args.host + ":" + remote_report, str(local_report)])
        with local_report.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        valid_report = (report.get("schema") == "vrnet-transfer-sha256-v1" and report.get("root") == remote_root
                        and report.get("manifest") == remote_manifest
                        and report.get("expected_file_count") == len(local_inventory)
                        and report.get("expected_total_bytes") == summary["expected_total_bytes"])
        summary["ok"] = verification_code == 0 and valid_report and report.get("ok") is True
        summary["verification"] = "passed" if summary["ok"] else "failed"
        summary["verification_returncode"] = verification_code
        summary["verification_report"] = str(local_report)
        summary["finished_utc"] = utc_now()
        save_json(summary_path, summary)
        print(f"{'PASS: complete remote SHA256 verification' if summary['ok'] else 'FAIL: remote SHA256 verification did not pass'}.\nReport: {local_report}\nRecords: {run_directory}", flush=True)
        if not summary["ok"]:
            print("If the report lists missing/mismatched/read-error files, repeat this command with --repair-report pointing to remotecheck.json. Extra remote files are reported and never deleted.", flush=True)
        return 0 if summary["ok"] else 1
    except BaseException as exc:
        summary.update({"error": str(exc), "finished_utc": utc_now()})
        save_json(summary_path, summary)
        print(f"Stopped; partial uploads and run records are preserved: {run_directory}\nRerun the same command to inventory again and skip complete-size files.", file=sys.stderr, flush=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    upload_parser = subparsers.add_parser("upload", help="resume the frozen upload and verify remote SHA256")
    upload_parser.add_argument("--root", default=str(PROJECT / "data" / "raw" / "vrnet"))
    upload_parser.add_argument("--manifest", default=str(PROJECT / "results" / "vrnet_transfer_manifest_2026-10-12.jsonl"))
    upload_parser.add_argument("--host", default="ygao930@irl-ai")
    upload_parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    upload_parser.add_argument("--remote-project", default=DEFAULT_REMOTE_PROJECT)
    upload_parser.add_argument("--remote-python", default="python3")
    upload_parser.add_argument("--repair-report", help="force full uploads for missing/mismatched/read-error paths in a previous remotecheck.json")
    upload_parser.add_argument("--skip-final-verification", action="store_true", help="finish transfer without a SHA256 PASS; summary remains unverified")
    inventory_parser = subparsers.add_parser("inventory", help="internal standalone remote size inventory; refuses links/special files")
    inventory_parser.add_argument("--root", required=True)
    inventory_parser.add_argument("--output", required=True)
    argv = sys.argv[1:]
    if not argv or argv[0] not in ("upload", "inventory"):
        argv = ["upload", *argv]
    args = parser.parse_args(argv)
    try:
        return inventory(args.root, args.output) if args.command == "inventory" else upload(args)
    except KeyboardInterrupt:
        print("Interrupted by user; no data was deleted. Rerun to resume.", file=sys.stderr, flush=True)
        return 130
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return exc.returncode if isinstance(exc, CommandError) and exc.returncode > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
