#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/download_vrnet_games.sh VERIFIED_MANIFEST.json DATA_ROOT
# Sequential download -> retained JPG/CSV -> ZIP deletion; never all ZIPs first.
manifest=${1:?Supply a verified manifest with real Drive IDs and sizes}
data_root=${2:?Supply a data directory, e.g. /data/scratch/ygao930/vrnet}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p -- "$data_root/results" "$data_root/.download_work"
"${PYTHON:-python3}" "$script_dir/prepare_vrnet_game.py" \
  --mode both --manifest "$manifest" --work-dir "$data_root/.download_work" \
  --frames-root "$data_root/video_frames" --csv-root "$data_root/sampled_csv" \
  --log "$data_root/results/download_log.md" --summary "$data_root/results/downloaded_sessions.json"
