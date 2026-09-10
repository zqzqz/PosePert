#!/usr/bin/env bash
#
# Fetch and unpack the two release archives.
#
#   posepert_data.zip     attack cases, ray-cast caches, occupancy maps,
#                         scenario features, CARLA meshes        (2.2 GB)
#   posepert_models.zip   every checkpoint the experiments load  (368 MB)
#
# The OPV2V and V2X-Real datasets themselves are NOT here: they are third-party
# releases under their own licences. Download them from the original sources and
# extract the splits under data/OPV2V/ and data/V2X-Real/ -- see docs/INSTALL.md.
#
# Usage:
#   bash scripts/download.sh              # both archives
#   bash scripts/download.sh models       # just one
#
# Already-downloaded archives are reused; pass --force to re-fetch.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# TODO(release): replace with the published Google Drive file IDs.
DATA_FILE_ID="PLACEHOLDER_DATA_FILE_ID"
MODELS_FILE_ID="PLACEHOLDER_MODELS_FILE_ID"

FORCE=0
TARGETS=()
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        data|models) TARGETS+=("$arg") ;;
        *) echo "usage: $0 [data|models] [--force]" >&2; exit 2 ;;
    esac
done
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(data models)

command -v gdown >/dev/null || {
    echo "gdown not found. Install it with:  pip install gdown" >&2
    exit 1
}

fetch() {
    # Separate statements on purpose: `local a=$1 b=$a` declares both names before
    # assigning, so $a is an unset local when b is expanded, which trips set -u.
    local name="$1"
    local file_id="$2"
    local zip="$name.zip"

    if [ "${file_id#PLACEHOLDER}" != "$file_id" ]; then
        cat >&2 <<MSG

  The Google Drive ID for $zip has not been filled in yet.

  Download it manually from the link in docs/INSTALL.md, place it in
  $ROOT, and unpack from the repository root:

      unzip -o $zip

MSG
        return 1
    fi

    if [ -f "$zip" ] && [ "$FORCE" = "0" ]; then
        echo "== $zip already present, skipping download (--force to re-fetch)"
    else
        echo "== downloading $zip"
        gdown -O "$zip" "$file_id"
    fi

    echo "== unpacking $zip"
    unzip -o -q "$zip"
}

status=0
for t in "${TARGETS[@]}"; do
    case "$t" in
        data)   fetch posepert_data   "$DATA_FILE_ID"   || status=1 ;;
        models) fetch posepert_models "$MODELS_FILE_ID" || status=1 ;;
    esac
done

if [ "$status" = "0" ]; then
    echo
    echo "Done. Verify the layout with:  python scripts/check_artifact.py"
fi
exit $status
