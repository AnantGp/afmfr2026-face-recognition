#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DOWNLOAD_PID="${2:-}"
PASSWORD="${AFMFR_ZIP_PASSWORD:?Set AFMFR_ZIP_PASSWORD to the organizer-provided archive password}"

cd "$ROOT"
mkdir -p logs data/raw/archives/1st_track data/raw/track1 data/manifests runs/track1_rank1_a100

if [[ -n "$DOWNLOAD_PID" ]]; then
  echo "Waiting for Track 1 download PID $DOWNLOAD_PID" | tee -a logs/track1_pipeline.log
  while kill -0 "$DOWNLOAD_PID" 2>/dev/null; do
    du -sh data/raw/archives/1st_track 2>/dev/null | tee -a logs/track1_pipeline.log || true
    sleep 60
  done
fi

echo "Track 1 archives present:" | tee -a logs/track1_pipeline.log
ls -lh data/raw/archives/1st_track | tee -a logs/track1_pipeline.log

echo "Extracting official Track 1 multipart dataset" | tee -a logs/track1_pipeline.log
7z x -y -p"$PASSWORD" data/raw/archives/1st_track/IJCB_competition_1st_track.zip -odata/raw/track1 \
  | tee -a logs/track1_extract_part1.log

echo "Extracting official Track 1 second dataset" | tee -a logs/track1_pipeline.log
7z x -y -p"$PASSWORD" data/raw/archives/1st_track/IJCB_1st_track_data_2.zip -odata/raw/track1 \
  | tee -a logs/track1_extract_part2.log

echo "Track 1 extracted layout:" | tee -a logs/track1_pipeline.log
set +o pipefail
find data/raw/track1 -maxdepth 3 -type d | sort | head -80 | tee -a logs/track1_pipeline.log
set -o pipefail

echo "Building Track 1 manifest and splits" | tee -a logs/track1_pipeline.log
source .venv/bin/activate
python -m afmfr.build_manifest \
  --data-root data/raw/track1 \
  --track track1 \
  --output data/manifests/track1_all.csv \
  --identity-depth 1 \
  --prefix-depth 0 | tee -a logs/track1_pipeline.log

python -m afmfr.make_splits \
  --manifest data/manifests/track1_all.csv \
  --output data/manifests/track1_splits.csv \
  --identity-val-fraction 0.04 \
  --same-id-val-images 2 \
  --min-images-for-same-id-val 6 | tee -a logs/track1_pipeline.log

echo "Launching historical Track 1 A100 experiment (see docs/RESULTS.md)" | tee -a logs/track1_pipeline.log
nohup bash -lc "source .venv/bin/activate && python -u -m afmfr.train --config configs/track1_rank1_a100.yaml" \
  > runs/track1_rank1_a100/train.log 2>&1 &
echo $! > runs/track1_rank1_a100/train.pid
cat runs/track1_rank1_a100/train.pid | tee -a logs/track1_pipeline.log
