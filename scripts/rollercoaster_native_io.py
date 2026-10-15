"""Read the original VR.net Epic bundle without inventing a frame clock.

Observed in VRLOG-6051819: gamedata.log is gzip containing records with a
24-byte little-endian header (type, payload length, frame, source, Unix ms).
Type 5 supplies the saved-image frame IDs. Other payload semantics are not
decoded here. Scores in voice.csv use integer Unix microseconds.

Motion containers have a gzip envelope and two uint64 dimension candidates,
followed by an opaque stream. Numerical motion decoding requires documented
producer parameters; installing a codec alone cannot supply those parameters.
"""

from __future__ import annotations

import csv
import gzip
import math
import os
import re
import struct
from collections import Counter
from pathlib import Path
from typing import Iterator, Mapping

import pandas as pd


RECORD_HEADER = struct.Struct("<iiIIq")
MOTION_PREFIX = struct.Struct("<QQ")
FRAME_RECORD_TYPE = 5
FRAME_COLUMNS = ["frame_id", "timestamp_ms", "rgb_path", "depth_path", "motion_path"]
SCORE_COLUMNS = ["timestamp_us", "fms_raw", "rating_source"]


class NativeFormatError(ValueError):
    """A raw record cannot be aligned safely with the observed format."""


class MotionEncodingUnknown(RuntimeError):
    """The producer's motion array/codec parameters are not established."""


def resolve_session_dir(session_dir: str | Path) -> Path:
    """Accept either the session itself or its single extraction wrapper."""
    directory = Path(session_dir)
    if (directory / "gamedata.log").is_file():
        return directory
    candidates = [
        child for child in directory.iterdir()
        if child.is_dir() and child.name != "__MACOSX"
        and (child / "gamedata.log").is_file()
    ]
    if len(candidates) != 1:
        raise NativeFormatError(f"Expected one native session under {directory}; found {len(candidates)}")
    return candidates[0]


def iter_record_headers(log_path: str | Path) -> Iterator[dict[str, int]]:
    """Stream every header, checking lengths without loading the whole log."""
    path = Path(log_path)
    with path.open("rb") as raw:
        if raw.read(2) != b"\x1f\x8b":
            raise NativeFormatError(f"Expected gzip native log: {path}")
    with gzip.open(path, "rb") as stream:
        offset = 0
        while True:
            packed = stream.read(RECORD_HEADER.size)
            if not packed:
                break
            if len(packed) != RECORD_HEADER.size:
                raise NativeFormatError(f"Truncated record header at uncompressed offset {offset}")
            tag, size, frame, source, timestamp = RECORD_HEADER.unpack(packed)
            if tag < 0 or size < 0 or size > 256 * 1024 * 1024:
                raise NativeFormatError(f"Invalid type/payload length at offset {offset}: {tag}, {size}")
            if not 0 < timestamp < 100_000_000_000_000:
                raise NativeFormatError(f"Unsupported Unix-ms timestamp at offset {offset}: {timestamp}")
            remaining = size
            while remaining:
                part = stream.read(min(remaining, 65536))
                if not part:
                    raise NativeFormatError(f"Truncated payload at uncompressed offset {offset}")
                remaining -= len(part)
            yield {
                "record_type": tag, "payload_bytes": size, "frame_id": frame,
                "source_id": source, "timestamp_ms": timestamp,
                "uncompressed_offset": offset,
            }
            offset += RECORD_HEADER.size + size


def read_frame_index(session_dir: str | Path) -> pd.DataFrame:
    """Return type-5 frames in recorded order, with missing assets kept as None.

    Duplicate frame IDs, non-increasing times, nonzero type-5 source IDs, and
    orphan image IDs raise; they are never sorted away, merged, or repaired.
    Every timestamp_ms is the recorded integer. Compare timestamp_ms * 1000
    with score timestamp_us for strict score-relative windows.
    """
    directory = resolve_session_dir(session_dir).resolve()
    with os.scandir(directory) as entries:
        asset_names = {entry.name for entry in entries if entry.is_file(follow_symlinks=False)}
    rows: list[dict] = []
    seen: set[int] = set()
    counts: Counter = Counter()
    previous_ms: int | None = None
    for header in iter_record_headers(directory / "gamedata.log"):
        counts[header["record_type"]] += 1
        if header["record_type"] != FRAME_RECORD_TYPE:
            continue
        frame, timestamp = header["frame_id"], header["timestamp_ms"]
        if header["source_id"] != 0:
            raise NativeFormatError(f"Unknown type-5 source ID for frame {frame}: {header['source_id']}")
        if frame in seen:
            raise NativeFormatError(f"Duplicate type-5 frame ID: {frame}")
        if previous_ms is not None and timestamp <= previous_ms:
            raise NativeFormatError(f"Non-increasing type-5 timestamp at frame {frame}: {timestamp}")
        seen.add(frame)
        previous_ms = timestamp
        paths = {}
        for column, prefix, suffix in (
            ("rgb_path", "s", ".jpg"), ("depth_path", "d", ".jpg"),
            ("motion_path", "m", ".zfp"),
        ):
            file = directory / f"{prefix}{frame}{suffix}"
            paths[column] = str(file) if file.name in asset_names else None
        rows.append({"frame_id": frame, "timestamp_ms": timestamp, **paths})
    if not rows:
        raise NativeFormatError(f"No observed type-5 frame records in {directory}")
    asset_ids = {
        int(match.group(1))
        for name in asset_names
        if (match := re.fullmatch(r"[sdm](\d+)\.(?:jpg|zfp)", name))
    }
    orphans = asset_ids - seen
    if orphans:
        raise NativeFormatError(f"Assets lack a type-5 timestamp: {sorted(orphans)[:20]}")
    table = pd.DataFrame(rows, columns=FRAME_COLUMNS)
    table["frame_id"] = table["frame_id"].astype("int64")
    table["timestamp_ms"] = table["timestamp_ms"].astype("int64")
    table.attrs.update({
        "session_dir": str(directory), "frame_record_type": FRAME_RECORD_TYPE,
        "timestamp_unit": "Unix milliseconds", "log_record_counts": dict(counts),
        "alignment": "Recorded frame time; multiply integer milliseconds by 1000 to compare with score microseconds",
        "frame_type_provenance": "Empirical full filename/frame correspondence in VRLOG-6051819; not an author data dictionary",
    })
    return table


def load_scores(path: str | Path) -> pd.DataFrame:
    """Read observed headerless timestamp_us, FMS, source CSV without dedup.

    Auto/manual rows and repeated ratings/timestamps are retained. This is a
    parser, not a choice of the study's independent event-label protocol.
    """
    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, fields in enumerate(csv.reader(stream), 1):
            if not fields or all(not item.strip() for item in fields):
                continue
            if len(fields) != 3:
                raise NativeFormatError(f"Expected three score fields at {path}:{line_number}")
            try:
                timestamp = int(fields[0].strip())
                rating = float(fields[1].strip())
            except ValueError as exc:
                raise NativeFormatError(f"Invalid score timestamp/rating at {path}:{line_number}") from exc
            if not 100_000_000_000_000 <= timestamp < 100_000_000_000_000_000:
                raise NativeFormatError(f"Expected Unix microseconds at {path}:{line_number}; got {timestamp}")
            if not math.isfinite(rating):
                raise NativeFormatError(f"Nonfinite FMS at {path}:{line_number}")
            rows.append({"timestamp_us": timestamp, "fms_raw": rating, "rating_source": fields[2].strip()})
    table = pd.DataFrame(rows, columns=SCORE_COLUMNS)
    if len(table):
        table["timestamp_us"] = table["timestamp_us"].astype("int64")
    table.attrs.update({"timestamp_unit": "Unix microseconds", "rows_preserved": True})
    return table


def read_motion_container(path: str | Path) -> tuple[bytes, dict]:
    """Unwrap gzip and the observed 16-byte prefix; do not infer encoding."""
    file = Path(path)
    with file.open("rb") as stream:
        if stream.read(2) != b"\x1f\x8b":
            raise NativeFormatError(f"Expected gzip motion container: {file}")
    with gzip.open(file, "rb") as stream:
        prefix = stream.read(MOTION_PREFIX.size)
        if len(prefix) != MOTION_PREFIX.size:
            raise NativeFormatError(f"Truncated motion prefix: {file}")
        width, height = MOTION_PREFIX.unpack(prefix)
        payload = stream.read()
    if not 0 < width <= 65536 or not 0 < height <= 65536 or not payload:
        raise NativeFormatError(f"Unsupported motion container: {file}")
    return payload, {
        "path": str(file.resolve()), "outer_encoding": "gzip",
        "prefix_encoding": "two little-endian uint64 fields",
        "dimension_candidates": [width, height], "prefix_bytes": MOTION_PREFIX.size,
        "payload_bytes": len(payload), "full_zfp_magic_present": payload.startswith(b"zfp"),
        "array_shape": "unknown", "component_layout": "unknown",
        "scalar_type": "unknown", "compression_mode_and_parameters": "unknown",
        "numeric_motion_units": "unknown",
    }


def decode_motion(path: str | Path, *, encoding: Mapping | None = None):
    """Decode only with producer-verified parameters; default is explicit unknown.

    encoding must state producer_verified=True, source, shape in zfPy C order,
    dtype, component_layout, mode, codec_version and bitstream_word_bits.
    Modes: tolerance/rate/precision (with matching value), or reversible.
    No shape, vector layout, compression setting, or physical unit is guessed.
    """
    payload, metadata = read_motion_container(path)
    required = {"producer_verified", "source", "shape", "dtype", "component_layout", "mode", "codec_version", "bitstream_word_bits"}
    if encoding is None or not required.issubset(encoding) or encoding["producer_verified"] is not True:
        raise MotionEncodingUnknown(f"Producer dtype/shape/layout/codec parameters required for {path}; observed prefix {metadata['dimension_candidates']}")
    if not encoding["source"] or encoding["component_layout"] == "unknown":
        raise MotionEncodingUnknown("A producer source and component layout are required")
    shape = tuple(int(size) for size in encoding["shape"])
    if not 1 <= len(shape) <= 4 or any(size <= 0 for size in shape) or math.prod(shape) > 64_000_000:
        raise NativeFormatError(f"Unsupported producer array shape: {shape}")
    if encoding["dtype"] not in ("float32", "float64", "int32", "int64"):
        raise NativeFormatError(f"Unsupported producer scalar type: {encoding['dtype']}")
    mode = encoding["mode"]
    if mode not in ("tolerance", "rate", "precision", "reversible"):
        raise NativeFormatError(f"Unsupported producer compression mode: {mode}")
    arguments = {}
    if mode != "reversible":
        if mode not in encoding or not float(encoding[mode]) > 0:
            raise MotionEncodingUnknown(f"Producer {mode} value is required")
        arguments[mode] = encoding[mode]
    try:
        import numpy as np
        import zfpy
    except ImportError as exc:
        raise MotionEncodingUnknown("A compatible zfpy codec is not installed; producer parameters are still required") from exc
    array = zfpy._decompress(payload, zfpy.dtype_to_ztype(np.dtype(encoding["dtype"])), shape, **arguments)
    metadata.update({"verified_encoding": dict(encoding), "decoded_shape": list(array.shape), "decoded_dtype": str(array.dtype)})
    return array, metadata


decoder_motion = decode_motion
