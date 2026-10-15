"""Compute frame-to-frame Farneback optical-flow summaries for VR.net JPGs.

Example (from repository root)::

    python scripts/verify_optical_flow.py \
        --samples-dir data/raw/vrnet/samples_full \
        --output features/optical_flow_verification.csv \
        --resize-width 320

The script does not train a model or infer sickness labels. It processes only
JPG files already present under the supplied samples directory and preserves
the frame number gaps in its output.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np


FRAME_RE = re.compile(r"^(?P<frame>\d+)\.(?:jpe?g)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resize-width", type=int, default=320)
    return parser.parse_args()


def image_paths(session_dir: Path) -> list[tuple[int, Path]]:
    video_dir = session_dir / "video"
    if not video_dir.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    for path in video_dir.iterdir():
        match = FRAME_RE.match(path.name)
        if path.is_file() and match:
            found.append((int(match.group("frame")), path))
    return sorted(found, key=lambda item: (item[0], item[1].name.casefold()))


def load_gray(path: Path, resize_width: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    if resize_width > 0 and image.shape[1] > resize_width:
        height = max(1, round(image.shape[0] * resize_width / image.shape[1]))
        image = cv2.resize(image, (resize_width, height), interpolation=cv2.INTER_AREA)
    return image


def main() -> int:
    args = parse_args()
    if args.resize_width < 0:
        raise SystemExit("--resize-width must be 0 (no resize) or a positive integer")
    sessions_dir = args.samples_dir
    if not sessions_dir.is_dir():
        raise SystemExit(f"samples directory does not exist: {sessions_dir}")

    sessions = sorted(p for p in sessions_dir.iterdir() if p.is_dir())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "session",
        "frame_from",
        "frame_to",
        "frame_gap",
        "mean_magnitude_px",
        "p95_magnitude_px",
        "max_magnitude_px",
        "mean_direction_rad",
        "valid_pixels",
        "image_width",
        "image_height",
        "status",
        "error",
    ]
    t0 = time.time()
    session_counts: dict[str, dict[str, int]] = {}
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for session in sessions:
            frames = image_paths(session)
            counts = {"jpg": len(frames), "pairs": max(0, len(frames) - 1), "ok": 0, "failed": 0}
            session_counts[session.name] = counts
            if len(frames) < 2:
                continue
            previous_id, previous_path = frames[0]
            try:
                previous = load_gray(previous_path, args.resize_width)
            except Exception as exc:  # record decode failures rather than inventing a value
                counts["failed"] += 1
                writer.writerow({"session": session.name, "frame_from": previous_id,
                                 "frame_to": "", "frame_gap": "", "status": "failed",
                                 "error": str(exc)})
                continue
            for current_id, current_path in frames[1:]:
                try:
                    current = load_gray(current_path, args.resize_width)
                    if current.shape != previous.shape:
                        current = cv2.resize(current, (previous.shape[1], previous.shape[0]),
                                             interpolation=cv2.INTER_AREA)
                    flow = cv2.calcOpticalFlowFarneback(
                        previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1],
                                                       angleInDegrees=False)
                    finite = np.isfinite(magnitude) & np.isfinite(angle)
                    values = magnitude[finite]
                    directions = angle[finite]
                    writer.writerow({
                        "session": session.name,
                        "frame_from": previous_id,
                        "frame_to": current_id,
                        "frame_gap": current_id - previous_id,
                        "mean_magnitude_px": float(np.mean(values)) if values.size else "",
                        "p95_magnitude_px": float(np.percentile(values, 95)) if values.size else "",
                        "max_magnitude_px": float(np.max(values)) if values.size else "",
                        "mean_direction_rad": float(np.angle(np.mean(np.exp(1j * directions))))
                        if directions.size else "",
                        "valid_pixels": int(values.size),
                        "image_width": int(previous.shape[1]),
                        "image_height": int(previous.shape[0]),
                        "status": "ok",
                        "error": "",
                    })
                    counts["ok"] += 1
                    previous = current
                    previous_id = current_id
                except Exception as exc:  # keep processing later adjacent pairs
                    counts["failed"] += 1
                    writer.writerow({"session": session.name, "frame_from": previous_id,
                                     "frame_to": current_id,
                                     "frame_gap": current_id - previous_id,
                                     "status": "failed", "error": str(exc)})
                    try:
                        previous = load_gray(current_path, args.resize_width)
                        previous_id = current_id
                    except Exception:
                        pass
    elapsed = time.time() - t0
    total = {key: sum(value[key] for value in session_counts.values())
             for key in ("jpg", "pairs", "ok", "failed")}
    print(f"samples_dir={sessions_dir}")
    print(f"output={args.output}")
    print(f"resize_width={args.resize_width}")
    print(f"sessions={len(session_counts)} jpg={total['jpg']} pairs={total['pairs']} "
          f"flow_ok={total['ok']} failed={total['failed']} elapsed_seconds={elapsed:.1f}")
    for name, counts in session_counts.items():
        print(f"{name}: jpg={counts['jpg']} pairs={counts['pairs']} "
              f"flow_ok={counts['ok']} failed={counts['failed']}")
    return 0 if total["failed"] == 0 and total["ok"] == total["pairs"] else 1


if __name__ == "__main__":
    sys.exit(main())
