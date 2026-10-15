#!/usr/bin/env bash
set -euo pipefail
# For ONE already-downloaded archive described by the manifest; verifies JPG
# references and deletes the ZIP only after successful extraction.
manifest=${1:?Supply verified manifest}
data_root=${2:?Supply data directory}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
"${PYTHON:-python3}" "$script_dir/prepare_vrnet_game.py" \
  --mode extract --manifest "$manifest" --work-dir "$data_root/.download_work" \
  --frames-root "$data_root/video_frames" --csv-root "$data_root/sampled_csv" \
  --log "$data_root/results/download_log.md" --summary "$data_root/results/downloaded_sessions.json"
