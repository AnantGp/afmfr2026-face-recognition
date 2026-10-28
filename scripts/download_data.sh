#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "$ROOT/data/raw/archives"

gcloud storage cp -r gs://ijcb-competition-2026/1st_track "$ROOT/data/raw/archives/"
gcloud storage cp -r gs://ijcb-competition-2026/2nd_track "$ROOT/data/raw/archives/"

echo "Downloaded archives to $ROOT/data/raw/archives"
echo "Extract manually with the competition password, then build manifests."
