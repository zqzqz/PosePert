# PosePert: Stealthy Pose Perturbation Attacks on Collaborative Perception

PosePert attacks LiDAR-based collaborative perception by perturbing the *spatial
pose* of shared intermediate features. A single malicious collaborator scales a
small voxel neighbourhood of its own bird's-eye-view feature map and adds a
learned residual correction, which shifts or fabricates an object in the victim's
fused detection while keeping the shared feature statistically close to benign.
The repository also implements the defenses the paper evaluates, including our
**PoseGuard** local-consistency detector.

This is the research artifact: the attack, the defenses, the evaluation harness,
and the closed-loop CARLA study.

## Documentation

| | |
|---|---|
| **[docs/INSTALL.md](docs/INSTALL.md)** | environment, datasets, release archives, verification |
| **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)** | what to run to reproduce each table and figure |
| [carla_demo/SETUP.md](carla_demo/SETUP.md) | CARLA server and client for the closed-loop study |

```bash
bash scripts/setup.sh          # or: docker build -t posepert .
bash scripts/download.sh       # data + model archives
python scripts/check_artifact.py
CUDA_VISIBLE_DEVICES=0 python test/run_eval.py --model pointpillar --beta 2.0 --dataset OPV2V --n_cases 5
```

## What the attack does

The victim fuses feature maps from its collaborators. The attacker owns one of
those maps, so it can edit the features that describe a chosen object before
sharing them. Two levels are evaluated:

* **Object level** — move one target's perceived pose in a single frame. A
  ray-cast baseline relocates the object's points; `beta` scaling amplifies the
  feature response at the destination; **PertNet**, a small learned network,
  supplies a per-voxel correction that roughly doubles the achieved IoU at one
  metre of shift while keeping the perturbation bounded.
* **Scenario level** — plan a multi-frame sequence of shifts, each within a
  0.5 m per-frame stealth bound, so that the victim's *predicted* trajectory for
  the target intrudes into its own lane. The consequence is a driving decision,
  not just a detection error: the victim brakes for a phantom cut-in, or fails to
  brake for a vehicle shifted out of its lane.

Everything is evaluated on four settings, AttFusion / V2VNet / CoBEVT on OPV2V and
AttFusion on V2X-Real, plus a closed-loop CARLA study of both scenario outcomes.

## Architecture

```
mvp/
├── attack/
│   ├── posepert_attacker.py                PosePert, the voxelwise feature attack
│   ├── perturbation_network.py             PertNet architecture
│   ├── perturbation_train.py               PertNet training, build_perception()
│   ├── pertnet_pipeline*.py                beta scan -> collect -> train -> eval
│   ├── shift_rotation.py                   spoofed-pose definition (apply_shift)
│   └── scenario_attacker.py                multi-frame scenario planner
├── defense/
│   ├── perception_defender.py   CAD, occupancy consistency
│   ├── lucia/                   LUCIA, global and local (PoseGuard's anomaly score)
│   ├── made/                    MADE residual autoencoder, global and local
│   └── poseguard_defender.py    PoseGuard, the combined defense pipeline
├── perception/                  OpenCOOD and HEAL wrappers, CUDA IoU operator
├── prediction/                  GRIP++ and Trajectron++ interfaces
├── tracking/                    AB3DMOT
└── data/                        dataset loaders, geometry utilities
```

The attack path is: **perception** produces BEV features, **attack** edits the
attacker's slice of them, **tracking** and **prediction** turn the corrupted
detections into a trajectory, and **defense** scores the shared features for
consistency. `mvp/config.py` resolves `data/`, `models/` and `third_party/`.

Around the library:

```
test/           experiment entry points and unit-style checks
results_paper/  scenario attack runner and all figure generation
carla_demo/     closed-loop CARLA study (separate environment)
scripts/        setup, download, link_local_data, check_artifact, make_release
third_party/    git submodules (OpenCOOD, SqueezeSegV3, AdvTrajectoryPrediction, V2X-Real)
docs/           installation and experiment guides
```
