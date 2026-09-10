#!/usr/bin/env bash
#
# Build the two release archives that accompany the artifact.
#
#   release/posepert_models.zip   every checkpoint the experiments load  (368 MB)
#   release/posepert_data.zip     every non-public input file           (2.2 GB)
#
# OPV2V and V2X-Real themselves are NOT included: they are third-party datasets
# with their own distribution terms and are downloaded from the original sources
# (see docs/INSTALL.md). Everything the artifact adds on top of them is here.
#
# Symlinks are dereferenced, so the archives are self-contained and can be built
# from a tree populated by scripts/link_local_data.sh.
#
# Usage:
#   bash scripts/make_release.sh [OUT_DIR]        # default: release/

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/release}"
cd "$ROOT"

command -v zip >/dev/null || { echo "ERROR: zip is not installed" >&2; exit 1; }
mkdir -p "$OUT"

# --- models -----------------------------------------------------------------
# Small enough (~480 MB) to ship whole. link_local_data.sh already stages only
# the checkpoint each model directory actually loads, not every training epoch.
echo "== models =="
rm -f "$OUT/posepert_models.zip"
zip -r -q -FS "$OUT/posepert_models.zip" models -x '*/__pycache__/*' '*.DS_Store'
echo "   $(du -h "$OUT/posepert_models.zip" | cut -f1)  posepert_models.zip"

# --- data -------------------------------------------------------------------
# Explicit include list: everything here is required by a documented experiment.
# Deliberately excluded, and why:
#   data/OPV2V/{train,test,validate}       public dataset, ~110 GB
#   data/V2X-Real/{train,test,val,validate} public dataset
#   data/perturbation_train*_paper         only needed to retrain PertNet (3.5 GB)
#   data/prediction                        only needed to retrain GRIP++/Trajectron++ (5.7 GB)
#   scenario/normal/*__ab3dmot*.pkl        legacy feature combinations no script reads (~5 GB)
#   scenario/normal/*/occupancy.png        visualisation, not an input
echo "== data =="
LIST="$(mktemp)"
trap 'rm -f "$LIST"' EXIT

add() { [ -e "$1" ] && find -L "$1" -type f -print >> "$LIST" || echo "   MISSING $1" >&2; }

add data/carla
add data/model_3d
for f in data/OPV2V/*.pkl;    do [ -e "$f" ] && echo "$f" >> "$LIST"; done
for f in data/V2X-Real/*.pkl; do [ -e "$f" ] && echo "$f" >> "$LIST"; done
add data/OPV2V/attack/lidar_shift.pkl
add data/OPV2V/attack_cache_paper
add data/OPV2V/normal
add data/OPV2V/multi_frame/attack/lidar_shift_early_Sampled_dense1
add data/V2X-Real/attack/lidar_shift.pkl
add data/V2X-Real/attack_cache_paper
add data/V2X-Real/normal

# Scenario features: only the ones run_scenario_all.py and run_scenario_defense.py load.
for f in pointpillar_intermediate ab3dmot grip occupancy v2vnet_intermediate cobevt_intermediate; do
    find -L data/OPV2V/scenario/normal -name "$f.pkl" -type f -print >> "$LIST" 2>/dev/null || true
done

sort -u "$LIST" -o "$LIST"
echo "   $(wc -l < "$LIST") files"
rm -f "$OUT/posepert_data.zip"
zip -q -FS "$OUT/posepert_data.zip" -@ < "$LIST"
echo "   $(du -h "$OUT/posepert_data.zip" | cut -f1)  posepert_data.zip"

echo
echo "Archives written to $OUT"
echo "Unpack both from the repository root: unzip -o posepert_<name>.zip"
