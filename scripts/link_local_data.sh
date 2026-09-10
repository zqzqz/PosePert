#!/usr/bin/env bash
#
# Populate ./data and ./models with the minimal set of files needed to
# reproduce the paper results, by symlinking them from a local storage root.
#
# This is the setup used on our lab machines, where the OPV2V / V2X-Real
# datasets and the trained checkpoints already live on a shared disk. Artifact
# evaluators who obtained the release archives instead should use
# scripts/download.sh; the resulting layout is identical.
#
# Usage:
#   bash scripts/link_local_data.sh [SRC_ROOT]
#
# SRC_ROOT defaults to the lab path below and must contain data/ and models/.

set -euo pipefail

SRC="${1:-/workspace/hdd/users/qzzhang/AdvCollaborativePerception}"
DST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -d "$SRC/data" ] || [ ! -d "$SRC/models" ]; then
    echo "ERROR: $SRC does not contain data/ and models/" >&2
    exit 1
fi

echo "Source : $SRC"
echo "Target : $DST"
echo

n_ok=0
n_miss=0

# link <relative-src> <relative-dst>
link() {
    local src="$SRC/$1" dst="$DST/$2"
    if [ ! -e "$src" ]; then
        echo "  MISSING  $1"
        n_miss=$((n_miss + 1))
        return
    fi
    mkdir -p "$(dirname "$dst")"
    ln -sfn "$src" "$dst"
    n_ok=$((n_ok + 1))
}

echo "== data: OPV2V =="
# Raw scenes (~110 GB total) stay on the shared disk.
for split in train test validate; do
    link "data/OPV2V/$split" "data/OPV2V/$split"
done
# Dataset indices. OPV2VDataset regenerates these if absent, which is slow.
for f in test.pkl test_cases.pkl test_attacks.pkl test_scenario_attacks.pkl \
         train.pkl train_cases.pkl train_attacks.pkl train_scenario_attacks.pkl \
         validate.pkl validate_cases.pkl validate_attacks.pkl; do
    link "data/OPV2V/$f" "data/OPV2V/$f"
done
link "data/OPV2V/attack/lidar_shift.pkl" "data/OPV2V/attack/lidar_shift.pkl"
# Pre-computed ray-tracing init for the PGD baseline (test/run_pgd_baseline.py).
# Without it every case hits the "no init data" skip and the run reports NaN.
link "data/OPV2V/multi_frame/attack/lidar_shift_early_Sampled_dense1" \
     "data/OPV2V/multi_frame/attack/lidar_shift_early_Sampled_dense1"
link "data/OPV2V/attack_cache_paper"     "data/OPV2V/attack_cache_paper"
# Precomputed detection / tracking / prediction per scenario case. The scenario
# attacker reads observed_trajectories from here; without it every case dies with
# KeyError: 'observed_trajectories'. 16 case dirs cover all 102 scenario cases.
link "data/OPV2V/scenario"               "data/OPV2V/scenario"
link "data/OPV2V/normal"                 "data/OPV2V/normal"

echo "== data: V2X-Real =="
for split in train test val validate; do
    link "data/V2X-Real/$split" "data/V2X-Real/$split"
done
for f in test.pkl test_cases.pkl test_attacks.pkl test_scenario_attacks.pkl \
         train.pkl validate.pkl; do
    link "data/V2X-Real/$f" "data/V2X-Real/$f"
done
link "data/V2X-Real/attack/lidar_shift.pkl" "data/V2X-Real/attack/lidar_shift.pkl"
link "data/V2X-Real/attack_cache_paper"     "data/V2X-Real/attack_cache_paper"
link "data/V2X-Real/normal"                 "data/V2X-Real/normal"

echo "== data: shared assets =="
link "data/carla"    "data/carla"        # CARLA map meshes / lane info for CAD
link "data/model_3d" "data/model_3d"     # 3D vehicle meshes for ray casting

echo "== data: PertNet training sets (only needed to retrain) =="
link "data/perturbation_train_paper"         "data/perturbation_train_paper"
# Training sets for the trajectory predictors. Only needed to retrain GRIP++ or
# Trajectron++; the shipped checkpoints under models/ cover every documented run.
link "data/prediction"                       "data/prediction"
link "data/perturbation_train_v2xreal_paper" "data/perturbation_train_v2xreal_paper"

echo "== models: collaborative perception backbones =="
# Only config.yaml plus the checkpoint OpenCOOD's load_saved_model() actually
# picks (latest.pth, else the highest net_epoch*.pth). The upstream directories
# hold every training epoch; linking them wholesale would add several GB.
link "models/OpenCOOD/pointpillar_attentive_fusion/config.yaml" \
     "models/OpenCOOD/pointpillar_attentive_fusion/config.yaml"
link "models/OpenCOOD/pointpillar_attentive_fusion/latest.pth" \
     "models/OpenCOOD/pointpillar_attentive_fusion/latest.pth"

link "models/OpenCOOD/v2vnet/config.yaml"     "models/OpenCOOD/v2vnet/config.yaml"
link "models/OpenCOOD/v2vnet/net_epoch83.pth" "models/OpenCOOD/v2vnet/net_epoch83.pth"

link "models/OpenCOOD/pointpillar_attentive_fusion_cobevt/config.yaml" \
     "models/OpenCOOD/pointpillar_attentive_fusion_cobevt/config.yaml"
link "models/OpenCOOD/pointpillar_attentive_fusion_cobevt/latest.pth" \
     "models/OpenCOOD/pointpillar_attentive_fusion_cobevt/latest.pth"

link "models/OpenCOOD/pointpillar_attentive_fusion_v2xreal/config.yaml" \
     "models/OpenCOOD/pointpillar_attentive_fusion_v2xreal/config.yaml"
link "models/OpenCOOD/pointpillar_attentive_fusion_v2xreal/latest.pth" \
     "models/OpenCOOD/pointpillar_attentive_fusion_v2xreal/latest.pth"

# Late fusion. The scenario attacker (results_paper/run_scenario_all.py) builds its
# detector with fusion_method="late", so Table 3 needs these on top of the
# intermediate-fusion models above. Neither directory has a latest.pth, so link the
# highest net_epoch*.pth, which is what load_saved_model() picks.
link "models/OpenCOOD/pointpillar_late_fusion/config.yaml" \
     "models/OpenCOOD/pointpillar_late_fusion/config.yaml"
link "models/OpenCOOD/pointpillar_late_fusion/net_epoch30.pth" \
     "models/OpenCOOD/pointpillar_late_fusion/net_epoch30.pth"
link "models/OpenCOOD/pointpillar_late_fusion_v2xreal/config.yaml" \
     "models/OpenCOOD/pointpillar_late_fusion_v2xreal/config.yaml"
link "models/OpenCOOD/pointpillar_late_fusion_v2xreal/net_epoch150.pth" \
     "models/OpenCOOD/pointpillar_late_fusion_v2xreal/net_epoch150.pth"

echo "== models: PertNet checkpoints (paper) =="
link "models/perturbation_net_paper_pointpillar/perturbation_net_ep35.pt" \
     "models/perturbation_net_paper_pointpillar/perturbation_net_ep35.pt"
# carla_demo/mvp_carla/config.py loads the "best" checkpoint rather than ep35.
link "models/perturbation_net_paper_pointpillar/perturbation_net_best.pt" \
     "models/perturbation_net_paper_pointpillar/perturbation_net_best.pt"
# The pipeline figure's loader needs a PertNet with explicit rank/hidden_dim, which the
# *_paper_* checkpoints do not carry. This older low-rank net is only for that figure.
link "models/perturbation_net/perturbation_net_best.pt" \
     "models/perturbation_net/perturbation_net_best.pt"
link "models/perturbation_net_paper_v2vnet/perturbation_net_best.pt" \
     "models/perturbation_net_paper_v2vnet/perturbation_net_best.pt"
link "models/perturbation_net_paper_cobevt/perturbation_net_best.pt" \
     "models/perturbation_net_paper_cobevt/perturbation_net_best.pt"
link "models/perturbation_net_paper_pointpillar_V2X-Real/perturbation_net_best.pt" \
     "models/perturbation_net_paper_pointpillar_V2X-Real/perturbation_net_best.pt"

echo "== models: defenses and downstream modules =="
link "models/MADE/residual_ae.pt"       "models/MADE/residual_ae.pt"       # MADE
link "models/SqueezeSegV3"              "models/SqueezeSegV3"              # CAD
link "models/GRIP/OPV2V/checkpoint.pt"  "models/GRIP/OPV2V/checkpoint.pt"  # scenario
link "models/GRIP/V2X-Real/checkpoint.pt" "models/GRIP/V2X-Real/checkpoint.pt"
link "models/Trajectron/OPV2V"          "models/Trajectron/OPV2V"          # transfer eval

echo "== third_party: large external repos =="
# OpenCOOD and SqueezeSegV3 are vendored in the artifact. These two are too big
# to vendor (1.3 GB / 323 MB), so link the local working copies. A fresh clone
# also works, but V2X-Real then needs third_party/patches applied first.
PARENT="$(dirname "$DST")"
for repo in V2X-Real AdvTrajectoryPrediction; do
    if [ -d "$PARENT/third_party/$repo" ]; then
        ln -sfn "$PARENT/third_party/$repo" "$DST/third_party/$repo"
        echo "  linked   third_party/$repo"
        n_ok=$((n_ok + 1))
    else
        echo "  MISSING  third_party/$repo (clone it, see README)"
        n_miss=$((n_miss + 1))
    fi
done

echo
echo "Linked $n_ok entries, $n_miss missing."
echo "Run 'python scripts/check_artifact.py' to verify the layout."
[ "$n_miss" -eq 0 ]
