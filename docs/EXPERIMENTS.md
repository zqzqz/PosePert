# Experiments

Every command below is run **from the repository root** with the `advCP`
environment active; scripts resolve `data/`, `models/` and `results_paper/`
relative to the working directory. Run `python scripts/check_artifact.py` first.

Device selection is not uniform across scripts, and the commands here use the
right form for each: `run_defense_eval_full.py`, `run_cad_normal.py`,
`run_scenario_defense{,_v2xreal}.py`, `run_transfer_eval.py` and
`run_scenario_all.py` take `--gpu N`; the rest read `CUDA_VISIBLE_DEVICES`.

## The four settings

| Setting | Backbone | Dataset | beta |
|---|---|---|---|
| 1 | AttFusion (PointPillars) | OPV2V | 2.0 |
| 2 | V2VNet | OPV2V | 3.0 |
| 3 | CoBEVT | OPV2V | 2.0 |
| 4 | AttFusion (PointPillars) | V2X-Real | 1.2 |

Scale: 300 perception and 102 scenario cases on OPV2V; 200 and 74 on V2X-Real.
Most scripts accept `--n_cases N` to truncate for a smoke run.

---

## 1. Object-level attack (perception table)

Reports the ablation ladder in one pass: clean, ray-cast baseline (`beta=1`),
`beta` scaling, and `beta` + PertNet.

```bash
CUDA_VISIBLE_DEVICES=0 python test/run_eval.py --model pointpillar --beta 2.0 --dataset OPV2V
CUDA_VISIBLE_DEVICES=0 python test/run_eval.py --model v2vnet     --beta 3.0 --dataset OPV2V
CUDA_VISIBLE_DEVICES=0 python test/run_eval.py --model cobevt     --beta 2.0 --dataset OPV2V
CUDA_VISIBLE_DEVICES=0 python test/run_eval.py --model pointpillar --beta 1.2 --dataset V2X-Real
```

Writes `results_paper/E{1,2,3}_*` and `E4_v2xreal`, each holding
`per_case_results.pkl`.

The PGD baseline for the same table:

```bash
CUDA_VISIBLE_DEVICES=0 python test/run_pgd_baseline.py --model {pointpillar,v2vnet,cobevt}
```

## 2. Defenses on the object-level attack

Scores CAD, LUCIA (global and local), MADE (global and local) on each case.

```bash
python test/run_defense_eval_full.py --model pointpillar --beta 2.0 --gpu 0
python test/run_defense_eval_full.py --model v2vnet      --beta 3.0 --gpu 0
python test/run_defense_eval_full.py --model cobevt      --beta 2.0 --gpu 0

# CAD on V2X-Real (needs data/V2X-Real/normal)
DATASET_NAME=V2X-Real CUDA_VISIBLE_DEVICES=0 python test/run_cad_v2xreal.py

# Clean CAD score distribution, for ROC calibration
python test/run_cad_normal.py --model {pointpillar,v2vnet,cobevt} --gpu 0
```

Writes `results_paper/D_{pp_attentive,v2vnet,cobevt}` and `D_v2xreal`.
Roughly 45 minutes per model for the full 300 cases on one A100.

## 3. Scenario-level attack (scenario table)

Two stages: the attack writes one `case_NNN.pkl` per scenario, then the defense
script scores those files. Running the defense stage first only warns that the
directory is missing.

```bash
# Stage 1 - attack
python results_paper/run_scenario_all.py --model pointpillar --attack_type blackbox --gpu 0
python results_paper/run_scenario_all.py --model pointpillar --attack_type whitebox --gpu 0
#   ... repeat for --model v2vnet and --model cobevt
python results_paper/S4_scenario_v2xreal/run_scenario_test.py --n_cases 74 --gpu 0

# Stage 2 - defenses
python test/run_scenario_defense.py --model {pointpillar,v2vnet,cobevt} --gpu 0
python test/run_scenario_defense_v2xreal.py --gpu 0

# Transfer row: re-scores the whitebox results with Trajectron++ instead of GRIP++
python test/run_transfer_eval.py --model {pointpillar,v2vnet,cobevt} --gpu 0
```

`--attack_type blackbox` is the paper's Query-Access row, `whitebox` the White-Box
row. The transfer evaluation needs the whitebox results to exist first.

The attack stage needs the late-fusion detector
(`models/OpenCOOD/pointpillar_late_fusion`) and the precomputed per-case features
under `data/OPV2V/scenario/normal`, both staged by the release archives.

## 4. CARLA closed-loop study

Separate environment and a running CARLA server; see
[../carla_demo/SETUP.md](../carla_demo/SETUP.md).

```bash
export CARLA_ROOT=$PWD/third_party/CARLA

python carla_demo/run_paper_cases.py       # phantom cut-in: 12 braking cases
python carla_demo/run_collision_cases.py   # suppression: 10 collision cases
```

Each case runs a screening pass and an attack pass at 2-3 minutes per episode, so
a full run is a couple of hours. Outcomes vary between runs; pool several with

```bash
python results_paper/agg_carla_collision.py
```

Do not run the two scripts concurrently against one server: both set global world
settings. If spawns start failing, run `python carla_demo/cleanup_world.py`.

---

## Figures

Figure scripts read the `results_paper/` directories the experiments produce, so
**run the experiments first**. No precomputed results ship with the artifact.
Output goes to `results_paper/figures/`; set `FIG_OUT_DIR` to redirect.

| Script | Needs |
|---|---|
| `gen_fig_iou_distributions.py` | section 1 (E1-E4) |
| `gen_fig_ablation.py` | sections 1 and 2 |
| `gen_fig_defense_roc.py` | section 2, all three OPV2V models + CAD normal |
| `gen_fig_defense_dist.py` | section 2, plus `test/run_small_shift_defense.py` |
| `gen_fig_scenario_analysis.py` | section 3 |
| `gen_fig_case_studies_batch.py` | section 3 (`--result_dir results_paper/S1_scenario_pp`) |
| `gen_fig_factors.py` | section 1, plus a `multi_attacker` sweep not included here |
| `gen_fig_params.py`, `gen_fig_beta_visualization.py`, `gen_fig_pipeline.py`, `gen_insight_figures.py` | caches and checkpoints only |

---

## Regenerating inputs

None of this is needed to reproduce the results above; the release archives ship
all of it precomputed.

```bash
# Test cases
python test/refine_test_cases.py                                    # OPV2V + V2X-Real
CUDA_VISIBLE_DEVICES=0 python test/generate_v2xreal_test_cases.py
python test/generate_v2xreal_scenario_cases.py --gpu 0

# Ray-cast attack cache. Rebuild after changing test cases or shift parameters:
# run_eval.py and run_defense_eval_full.py read the cached point clouds, so the
# cache -- not the consuming script -- fixes the spoofed target pose.
DATASET_NAME=OPV2V     python test/regen_attack_cache.py --gpu 0
DATASET_NAME=V2X-Real  python test/regen_attack_cache.py --gpu 0

# Occupancy maps for CAD, built on the frame the defense scores
CUDA_VISIBLE_DEVICES=0 python test/test_occupancy_map.py      # OPV2V, frame 9
CUDA_VISIBLE_DEVICES=0 python test/gen_v2xreal_occupancy.py   # V2X-Real, frame 0

# MADE autoencoder
python mvp/defense/made/train_residual_ae.py --root data/OPV2V --out models/MADE/residual_ae.pt

# PertNet: beta scan -> data collection -> training -> evaluation
python mvp/attack/pertnet_pipeline.py --model {pointpillar,v2vnet,cobevt} --gpu 0
python mvp/attack/pertnet_pipeline_v2xreal.py --gpu 0
```

Retraining writes to `data/perturbation_train_{model}/` and
`models/perturbation_net_{model}/`, **not** the `*_paper` directories, so the
shipped checkpoints are never overwritten. Point an evaluation at a retrained
network with `test/run_eval.py --checkpoint <path>`.

### Target rotation

The shift attack translates the target and rotates it by `attack_opts["rotation"]`.
Every case in the shipped `lidar_shift.pkl` stores `0.0`, so the published
perception numbers are translation-only. The generators now sample +/-10 degrees
(`mvp/attack/shift_rotation.py`), but the data is unchanged, so results stay
identical until the cases **and then** the cache are regenerated, in that order.
Regenerating cases without the cache leaves the two inconsistent: the boxes carry
a yaw the rendered points do not.

---

## Reproduction status

What has and has not been reproduced, including where measurements disagree with
the paper, is recorded in [../README.md](../README.md#reproduction-status).
