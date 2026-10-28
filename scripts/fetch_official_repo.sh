#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "$ROOT/official"

if [[ -d "$ROOT/official/IJCB-AFMFR-2026/.git" ]]; then
  git -C "$ROOT/official/IJCB-AFMFR-2026" pull --ff-only
else
  git clone https://github.com/TaharChettaoui/IJCB-AFMFR-2026.git "$ROOT/official/IJCB-AFMFR-2026"
fi
