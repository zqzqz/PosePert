# Installation

Two things to set up: the Python environment, and the data and checkpoints.
Budget roughly an hour, most of it downloading OPV2V.

Verified on Ubuntu 20.04/22.04, CUDA 11.6, one NVIDIA GPU with >= 8 GB
(RTX 2080 Ti and A100). The CARLA study needs a second, newer environment; see
[carla_demo/SETUP.md](../carla_demo/SETUP.md).

---

## 1. Environment

### Option A: Docker (recommended)

```bash
docker build -t posepert .
docker run --gpus all -it \
    -v $PWD/data:/workspace/PosePert/data \
    -v $PWD/models:/workspace/PosePert/models \
    posepert bash
```

The image builds OpenCOOD's Cython extensions and the CUDA IoU operator, which are
the two steps most likely to fail in a hand-rolled environment. Mount `data/` and
`models/` from the host rather than baking them in, so the image stays small and
the datasets are shared between runs.

### Option B: Conda

```bash
git clone --recursive <repo-url> PosePert && cd PosePert

conda env create --name advCP --file environment.yml
conda activate advCP

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 \
    pytorch-cuda=11.6 -c pytorch -c nvidia
pip install spconv-cu116

cd third_party/OpenCOOD
pip install -e .
python opencood/utils/setup.py build_ext --inplace
python opencood/pcdet_utils/setup.py build_ext --inplace
cd ../..

cd mvp/perception/cuda_op && python setup.py install && cd ../../..
```

`bash scripts/setup.sh` runs the same sequence.

### External repositories

`third_party/OpenCOOD` and `third_party/SqueezeSegV3` are vendored. Two more are
too large to ship:

```bash
# GRIP++ model code. Required for ALL scenario-level experiments, not just
# the transfer evaluation -- the GRIP++ implementation itself lives here.
git clone https://github.com/zqzqz/AdvTrajectoryPrediction.git \
    third_party/AdvTrajectoryPrediction

# V2X-Real dataset API. Required only for the V2X-Real setting.
git clone https://github.com/ucla-mobility/V2X-Real.git third_party/V2X-Real
cd third_party/V2X-Real
pip install -e .
git apply ../patches/v2x-real-ego-id.patch
cd ../..
cp third_party/patches/point_pillar_intermediate_fusion_*.yaml \
   third_party/V2X-Real/opencood/hypes_yaml/
```

The V2X-Real patch is **required**, not optional. Upstream uses `ego_id = -1` as a
"not found yet" sentinel and then asserts on it, but V2X-Real gives vehicle agents
negative IDs, so any scene whose ego is vehicle `-1` trips the assertion. See
[third_party/patches/README.md](../third_party/patches/README.md).

---

## 2. Datasets

OPV2V and V2X-Real are third-party datasets under their own licences and are not
redistributed here. Download them from the original sources:

| Dataset | Source | Size |
|---|---|---|
| OPV2V | https://mobility-lab.seas.ucla.edu/opv2v/ | ~110 GB (train/validate/test) |
| V2X-Real | https://mobility-lab.seas.ucla.edu/v2x-real/ | ~131 GB (Lidar-128) |

Extract them so the splits sit directly under each dataset directory:

```
data/OPV2V/{train,validate,test}/
data/V2X-Real/{train,val,test}/
```

V2X-Real is needed only for the V2X-Real experiments; OPV2V alone is enough for
most of the results.

---

## 3. Release archives

Everything the artifact adds on top of those datasets ships as two archives:
precomputed attack cases, ray-cast caches, occupancy maps, scenario features, and
all trained checkpoints.

```bash
pip install gdown
bash scripts/download.sh          # fetches and unpacks both archives
```

| Archive | Download |
|---|---|
| `posepert_data.zip` | *(link to be published)* |
| `posepert_models.zip` | *(link to be published)* |

Until those links are filled in, `scripts/download.sh` prints the manual
instructions instead of failing. Unpack both from the repository root:

| Archive | Contents | Download | On disk |
|---|---|---|---|
| `posepert_data.zip` | attack cases, ray-cast caches, occupancy maps, scenario features, CARLA meshes | 2.2 GB | 7.1 GB |
| `posepert_models.zip` | CP backbones, PertNet, MADE, SqueezeSegV3, GRIP++, Trajectron++ | 368 MB | 480 MB |

```bash
unzip -o posepert_data.zip        # -> data/
unzip -o posepert_models.zip      # -> models/
```

Two categories are deliberately left out to keep the archives manageable. Both are
needed only to retrain components whose trained weights are already included:
PertNet's training sets (`data/perturbation_train*_paper`, ~3.5 GB) and the
trajectory-predictor training sets (`data/prediction`, ~5.7 GB).

### Lab machines

If the shared storage is mounted, skip the archives:

```bash
bash scripts/link_local_data.sh [SRC_ROOT]
```

It symlinks the same layout out of the storage root, so nothing is duplicated.

---

The archives were verified by extracting them into an empty tree alongside the
source, linking the two public datasets, and running the layout check and a short
experiment: all required paths present, and the attack ladder reproduced.

## 4. Verify

```bash
python scripts/check_artifact.py
```

It reports, per experiment group, which required and optional paths are present,
and treats a dangling symlink as missing. Check this before running anything: it
names the missing file directly, where an experiment would fail hours in with a
`FileNotFoundError` or, worse, quietly skip every case and report `NaN`.

```
[T2] Table 2 - perception attack + defenses on OPV2V
  14/14 present
...
All required paths present. Every experiment group is runnable.
```

Use `--group T2` to check one group, `-v` to list every path.

A short end-to-end check, about a minute on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python test/run_eval.py \
    --model pointpillar --beta 2.0 --dataset OPV2V --n_cases 5
```

Unit-style checks that need no GPU or dataset:

```bash
python test/test_shift_rotation.py             # shift attack yaw perturbation
python carla_demo/tests/test_moveout_offline.py  # CARLA suppression geometry
```

---

## 5. Expected layout

```
PosePert/
├── data/
│   ├── OPV2V/
│   │   ├── train,validate,test/     raw scenes (downloaded separately)
│   │   ├── {split}.pkl, ...         dataset indices
│   │   ├── attack/lidar_shift.pkl   300 perception test cases
│   │   ├── attack_cache_paper/      ray-cast spoofed point clouds
│   │   ├── normal/                  occupancy maps for CAD
│   │   ├── multi_frame/attack/...   ray-tracing init for the PGD baseline
│   │   ├── scenario/normal/         per-case detection/tracking/prediction
│   │   └── test_scenario_attacks.pkl   102 scenario test cases
│   ├── V2X-Real/                    same structure; 200 / 74 cases
│   ├── carla/                       CARLA map meshes and lane info
│   └── model_3d/                    3D vehicle meshes for ray casting
├── models/
│   ├── OpenCOOD/                    CP backbones, intermediate and late fusion
│   ├── perturbation_net_paper_*/    trained PertNet per setting
│   ├── MADE/, SqueezeSegV3/         defense models
│   └── GRIP/, Trajectron/           trajectory predictors
└── third_party/                     OpenCOOD, SqueezeSegV3, patches (+ 2 cloned)
```

## Troubleshooting

**`ModuleNotFoundError: opencood`** — OpenCOOD is not installed. Run
`pip install -e .` inside `third_party/OpenCOOD`.

**`AssertionError` in `late_fusion_dataset.py` on V2X-Real** — the ego-id patch
was not applied. See section 1.

**An experiment reports `NaN` over 0 cases** — an input is missing and the script
skipped every case. Run `python scripts/check_artifact.py` first.

**`could not spawn near 0 m` in the CARLA study** — leftover actors from a crashed
run. Run `python carla_demo/cleanup_world.py`.
