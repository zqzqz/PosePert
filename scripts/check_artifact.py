#!/usr/bin/env python3
"""
Verify that the artifact has everything it needs before running experiments.

Checks the data/, models/ and third_party/ paths each experiment depends on and
reports what is present, what is missing, and which experiments are runnable.
Symlinks that dangle are reported as missing, not present.

Usage:
    python scripts/check_artifact.py            # check everything
    python scripts/check_artifact.py --group T2 # check one experiment group
"""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# group -> (description, [(path, required, note), ...])
CHECKS = {
    "core": ("Shared assets needed by every experiment", [
        ("third_party/OpenCOOD", True, "perception backbones"),
        ("data/carla", True, "map meshes / lane info (CAD defense)"),
        ("data/model_3d", True, "3D vehicle meshes (ray casting)"),
    ]),
    "T2": ("Table 2 - perception attack + defenses on OPV2V", [
        ("data/OPV2V/test", True, "raw test scenes"),
        ("data/OPV2V/test.pkl", True, "dataset index"),
        ("data/OPV2V/attack/lidar_shift.pkl", True, "300 perception test cases"),
        ("data/OPV2V/attack_cache_paper", True, "cached cases (regenerated if absent, slow)"),
        ("data/OPV2V/normal", True, "occupancy maps for CAD"),
        ("models/OpenCOOD/pointpillar_attentive_fusion/latest.pth", True, "AttFusion"),
        ("models/OpenCOOD/v2vnet/net_epoch83.pth", True, "V2VNet"),
        ("models/OpenCOOD/pointpillar_attentive_fusion_cobevt/latest.pth", True, "CoBEVT"),
        ("models/perturbation_net_paper_pointpillar/perturbation_net_ep35.pt", True, "PertNet AttFusion"),
        ("models/perturbation_net_paper_v2vnet/perturbation_net_best.pt", True, "PertNet V2VNet"),
        ("models/perturbation_net_paper_cobevt/perturbation_net_best.pt", True, "PertNet CoBEVT"),
        ("models/MADE/residual_ae.pt", True, "MADE defense"),
        ("models/SqueezeSegV3", True, "CAD defense segmentation"),
        ("data/OPV2V/multi_frame/attack/lidar_shift_early_Sampled_dense1", True,
         "ray-tracing init for the PGD baseline; without it the run reports NaN"),
    ]),
    "T3": ("Table 3 - scenario attack + defenses on OPV2V", [
        ("data/OPV2V/test_scenario_attacks.pkl", True, "102 scenario test cases"),
        ("data/OPV2V/scenario/normal", True,
         "precomputed detection/tracking/prediction per scenario case"),
        ("models/GRIP/OPV2V/checkpoint.pt", True, "GRIP++ trajectory prediction"),
        ("models/OpenCOOD/pointpillar_late_fusion/net_epoch30.pth", True,
         "the scenario attacker detects with late fusion"),
        # The GRIP++ model code itself lives inside AdvTrajectoryPrediction, so
        # scenario attacks need it too, not just the transfer evaluation.
        ("third_party/AdvTrajectoryPrediction", True, "GRIP++ model code"),
        ("models/Trajectron/OPV2V", False, "transfer eval only"),
    ]),
    "V2X": ("V2X-Real experiments", [
        ("data/V2X-Real/test", True, "raw test scenes"),
        ("data/V2X-Real/test.pkl", True, "dataset index"),
        ("data/V2X-Real/attack/lidar_shift.pkl", True, "perception test cases"),
        ("data/V2X-Real/attack_cache_paper", True, "cached cases"),
        ("data/V2X-Real/normal", True, "occupancy maps for CAD"),
        ("data/V2X-Real/test_scenario_attacks.pkl", True, "scenario test cases"),
        ("models/OpenCOOD/pointpillar_attentive_fusion_v2xreal/latest.pth", True, "V2X-Real AttFusion"),
        ("models/OpenCOOD/pointpillar_late_fusion_v2xreal/net_epoch150.pth", True,
         "V2X-Real scenario attacker (late fusion)"),
        ("models/perturbation_net_paper_pointpillar_V2X-Real/perturbation_net_best.pt", True, "PertNet V2X-Real"),
        ("models/GRIP/V2X-Real/checkpoint.pt", False, "scenario attack on V2X-Real"),
        ("third_party/V2X-Real", True, "V2X-Real dataset API"),
    ]),
    "carla": ("CARLA closed-loop braking cases (separate env, needs CARLA server)", [
        ("carla_demo/run_paper_cases.py", True, "phantom cut-in / braking driver"),
        ("carla_demo/run_collision_cases.py", True, "suppression / collision driver"),
        ("models/perturbation_net_paper_pointpillar/perturbation_net_best.pt", True,
         "PertNet checkpoint used by carla_demo/mvp_carla/config.py"),
        ("models/GRIP/OPV2V/checkpoint.pt", True, "GRIP++ trajectory prediction"),
        ("third_party/AdvTrajectoryPrediction", True, "GRIP++ model code"),
    ]),
    "train": ("Retraining PertNet from scratch (optional)", [
        ("data/OPV2V/train", True, "raw train scenes"),
        ("data/OPV2V/train.pkl", True, "dataset index"),
        ("data/perturbation_train_paper", False, "cached training set used for the paper"),
        ("data/perturbation_train_v2xreal_paper", False, "V2X-Real training set"),
        ("data/prediction", False, "GRIP++/Trajectron++ training sets; retraining only"),
    ]),
}


def human(path):
    """Size of a file, or entry count for a directory."""
    try:
        if os.path.isdir(path):
            return "%d entries" % len(os.listdir(path))
        n = os.path.getsize(path)
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024 or unit == "GB":
                return "%.0f %s" % (n, unit)
            n /= 1024.0
    except OSError:
        return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=sorted(CHECKS), help="check a single group")
    ap.add_argument("-v", "--verbose", action="store_true", help="list every path")
    args = ap.parse_args()

    groups = [args.group] if args.group else list(CHECKS)
    blocked, warned = [], []

    for g in groups:
        desc, entries = CHECKS[g]
        print("\n[%s] %s" % (g, desc))
        for rel, required, note in entries:
            path = os.path.join(ROOT, rel)
            ok = os.path.exists(path)           # False for dangling symlinks
            dangling = os.path.islink(path) and not ok
            if ok:
                if args.verbose:
                    print("  ok       %-62s %s" % (rel, human(path)))
            else:
                kind = "BROKEN " if dangling else "MISSING"
                tag = "required" if required else "optional"
                print("  %s  %-62s (%s: %s)" % (kind, rel, tag, note))
                (blocked if required else warned).append((g, rel))
        if not args.verbose:
            n_ok = sum(1 for rel, _, _ in entries
                       if os.path.exists(os.path.join(ROOT, rel)))
            print("  %d/%d present" % (n_ok, len(entries)))

    print("\n" + "=" * 64)
    if blocked:
        runnable = sorted({g for g in groups} - {g for g, _ in blocked})
        print("%d required path(s) missing." % len(blocked))
        print("Runnable groups: %s" % (", ".join(runnable) or "none"))
        print("Run 'bash scripts/link_local_data.sh' (lab machines) or "
              "'bash scripts/download.sh' (release archives).")
    else:
        print("All required paths present. Every experiment group is runnable.")
    if warned:
        print("%d optional path(s) missing: %s"
              % (len(warned), ", ".join(rel for _, rel in warned)))
    return 1 if blocked else 0


if __name__ == "__main__":
    sys.exit(main())
