#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if [[ -z "${AFMFR_ZIP_PASSWORD:-}" ]]; then
  echo "Set AFMFR_ZIP_PASSWORD before running extraction." >&2
  exit 1
fi

if ! command -v 7z >/dev/null 2>&1; then
  echo "7z is required. Install with: sudo apt-get update && sudo apt-get install -y p7zip-full" >&2
  exit 1
fi

mkdir -p "$ROOT/data/raw/track1" "$ROOT/data/raw/track2"

TRACK1_ARCH="$ROOT/data/raw/archives/1st_track"
TRACK2_ARCH="$ROOT/data/raw/archives/2nd_track"

7z x -y -p"$AFMFR_ZIP_PASSWORD" "$TRACK1_ARCH/IJCB_competition_1st_track.zip" -o"$ROOT/data/raw/track1"
7z x -y -p"$AFMFR_ZIP_PASSWORD" "$TRACK1_ARCH/IJCB_1st_track_data_2.zip" -o"$ROOT/data/raw/track1"
7z x -y -p"$AFMFR_ZIP_PASSWORD" "$TRACK2_ARCH/IJCB_compet_2nd_track.zip" -o"$ROOT/data/raw/track2"

echo "Extraction complete:"
find "$ROOT/data/raw/track1" -maxdepth 2 -type d | sed -n '1,20p'
find "$ROOT/data/raw/track2" -maxdepth 2 -type d | sed -n '1,20p'
