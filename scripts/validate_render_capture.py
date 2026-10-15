#!/usr/bin/env python3
"""Reject empty, near-black, near-flat or overwhelmingly magenta captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageStat


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    image_path = arguments.image.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    with Image.open(image_path) as opened:
        rgb = opened.convert("RGB")
        sample = rgb.resize((min(320, rgb.width), min(180, rgb.height)))
        stat = ImageStat.Stat(sample)
        mean = [float(value) for value in stat.mean]
        stddev = [float(value) for value in stat.stddev]
        pixels = list(sample.get_flattened_data())
        magenta = sum(1 for red, green, blue in pixels if red > 180 and blue > 180 and green < 90)
        magenta_ratio = magenta / max(1, len(pixels))
        luminance = 0.2126 * mean[0] + 0.7152 * mean[1] + 0.0722 * mean[2]
        passed = (
            rgb.width >= 320
            and rgb.height >= 180
            and 4.0 < luminance < 248.0
            and max(stddev) >= 4.0
            and magenta_ratio < 0.35
        )
        result = {
            "status": "passed" if passed else "failed",
            "image": str(image_path),
            "bytes": image_path.stat().st_size,
            "width": rgb.width,
            "height": rgb.height,
            "rgb_mean": mean,
            "rgb_stddev": stddev,
            "mean_luminance": luminance,
            "magenta_ratio": magenta_ratio,
            "checks": {
                "non_empty": image_path.stat().st_size > 0,
                "minimum_dimensions": rgb.width >= 320 and rgb.height >= 180,
                "not_black_or_white": 4.0 < luminance < 248.0,
                "has_colour_or_luminance_variance": max(stddev) >= 4.0,
                "not_overwhelmingly_magenta": magenta_ratio < 0.35,
            },
        }
    output = arguments.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
