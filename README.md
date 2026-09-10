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
| [third_party/patches/README.md](third_party/patches/README.md) | the V2X-Real fix and why it is required |

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
│   ├── lidar_shift_voxelwise_attacker.py   the voxelwise feature attack
│   ├── perturbation_network.py             PertNet architecture
│   ├── perturbation_train.py               PertNet training, build_perception()
│   ├── pertnet_pipeline*.py                beta scan -> collect -> train -> eval
│   ├── shift_rotation.py                   spoofed-pose definition (apply_shift)
│   └── scenario_attacker.py                multi-frame scenario planner
├── defense/
│   ├── perception_defender.py   CAD, occupancy consistency
│   ├── lucia/                   LUCIA, global and local (PoseGuard)
│   ├── made/                    MADE residual autoencoder, global and local
│   └── integrated_defender.py   combined pipeline
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
third_party/    OpenCOOD, SqueezeSegV3, patches for V2X-Real
docs/           installation and experiment guides
```

## Quick checks

No GPU or dataset required:

```bash
python test/test_shift_rotation.py                # shift attack yaw perturbation
python carla_demo/tests/test_moveout_offline.py   # CARLA suppression geometry
```

With data staged, `test/test_posepert_attack.py`, `test/test_poseguard_defense.py`
and `test/test_integrated_defense.py` exercise one cached case each.

## Reproduction status

Per paper claim, against [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). "Reproduced"
means a documented command regenerates the result and has been executed here at
least once; except for the CARLA study, the runs were smoke tests of a few cases,
not full-scale sweeps, so the numbers themselves are not re-verified.

| Paper claim | Status | Command / reason |
|---|---|---|
| Table 2, perception attack effectiveness (4 settings x PGD / ray-cast / +beta / +PertNet) | Reproduced | EXPERIMENTS section 1 |
| Table 2, defense TPR@5%FPR (CAD, LUCIA, MADE, Ours) | Reproduced | EXPERIMENTS section 2 |
| Table 3, scenario attack (White-Box, Query-Access) | Reproduced | EXPERIMENTS section 3, stage 1 |
| Table 3, scenario Transfer row | Reproduced | EXPERIMENTS section 3, transfer |
| Table 3, scenario defense TPR | Reproduced | EXPERIMENTS section 3, stage 2 |
| Table 4, CARLA phantom cut-in (false braking, 12 roads) | Braking depth reproduced; rate lower | EXPERIMENTS section 4; see below |
| Table 4, CARLA suppression (collision, 10 roads) | Mechanism reproduced; rate consistent, collisions not | EXPERIMENTS section 4; see below |
| **Overhead table (per-frame latency)** | **Not reproduced** | no timing harness in the repo |
| Fig. IoU distributions | Reproduced | `gen_fig_iou_distributions.py` |
| Fig. ablation | Reproduced | `gen_fig_ablation.py` |
| Fig. defense ROC / defense distance | Reproduced | `gen_fig_defense_roc.py`, `gen_fig_defense_dist.py` |
| Fig. scenario analysis | Reproduced | `gen_fig_scenario_analysis.py` |
| Fig. case studies | Reproduced | `gen_fig_case_studies_batch.py` |
| Fig. parameter sensitivity | Reproduced | `gen_fig_params.py` |
| **Fig. factor-distance** | **Partial** | needs a `multi_attacker` sweep that is absent |
| Fig. CARLA | Reproducible | both panels from EXPERIMENTS section 4 |

### Details on the gaps

* **The published perception results used no target rotation.** The shift attack
  translates the target by `shift_distance` along `shift_direction` *and* rotates it
  by `attack_opts["rotation"]`, but every case in the shipped `lidar_shift.pkl`
  (300 OPV2V, 200 V2X-Real) stores `rotation = 0.0`, so all published numbers are
  translation-only. Two script-level causes, both now fixed:

  - The case generators (`test/refine_test_cases.py`, `test/generate_v2xreal_test_cases.py`)
    sampled +/-30 deg; they now draw +/-10 deg from `mvp.attack.shift_rotation`, which is
    the single definition of the range. `pertnet_pipeline_v2xreal.py` sampled its PertNet
    training yaw from the same wrong +/-30 deg and now uses it too; the OPV2V pipeline
    applies no training yaw at all.
  - `test/regen_attack_cache.py`, which renders the spoofed point clouds, applied
    only the two translation terms and dropped the yaw one. A non-zero rotation in
    a test case would therefore still have produced an unrotated point cloud.

  Both attackers and the cache builder now share `apply_shift()`, so the spoofed
  pose has one definition. The data is deliberately unchanged, so **results are
  identical until the cases and the cache are regenerated**, in that order:

  ```bash
  python test/refine_test_cases.py                       # rewrites lidar_shift.pkl
  DATASET_NAME=OPV2V python test/regen_attack_cache.py --gpu 0
  ```

  Regenerating the cases without the cache leaves the two inconsistent: the boxes
  carry a yaw the rendered points do not. `python test/test_shift_rotation.py`
  covers the sampler and `apply_shift` (7 checks, no GPU).

* **CARLA phantom cut-in: braking depth reproduces, rate comes in lower.**
  Measured on CARLA 0.9.15 / Town10HD_Opt, 12 screened cruise-clean roads:

  | Metric | Paper | Measured |
  |---|---|---|
  | Unwarranted braking (% roads) | 50% | 25% (3/12) |
  | Worst-case min-speed | 11.3 km/h | 14.6 km/h (23.2 -> 14.6, -8.6) |
  | Mean slowdown | -- | 1.8 km/h |

  Where the attack lands, it lands hard and at the paper's magnitude: the worst
  case brakes the ego from 23.2 to 14.6 km/h, against the `carla_demo/README.md`
  note of "braking 23->15 km/h (-7.9 km/h worst case)". The rate is half the
  paper's, on 12 screened roads rather than the 8 that note refers to; at n=12
  a 3/12 result is not distinguishable from 50%.

  The per-case data points at what separates a hit from a miss, and it is not the
  road: it is how much perceived shift the attack actually realizes. All three
  dangerous cases realized 0.68-0.77 m, while the four cases with no predicted
  intrusion at all (`intr=99`) realized only 0.35-0.51 m. `MAX_OFFSET` is 1.0 m and
  `config.py` describes the faithful attack as reliably realizing about that, so
  the attack is under-realizing on roughly a third of roads. Worth investigating
  before quoting a rate; the shift is reported per case in the run output.

* **CARLA suppression: unsafe rate reproduces, collision rate does not.**
  `MoveOutScene` spawns a stopped vehicle that marginally blocks the ego lane; the
  attack runs in `moveout` mode, where the planner maximizes rather than minimizes
  predicted lane clearance, so the ACC stops reporting a lead. `longitudinal_ttc()`
  and an ego collision sensor supply the paper's three suppression metrics from
  ground-truth actor state, never from the attacked perception.

  Measured on CARLA 0.9.15 / Town10HD_Opt, three runs of 10 screened cases
  (`python results_paper/agg_carla_collision.py` reproduces this table):

  | Run | K | Unsafe | Collision | mean min-TTC |
  |---|---|---|---|---|
  | K=40 run 1 | 40 | 40% | 0% | 3.32 -> 1.86 s |
  | K=40 run 2 | 40 | 30% | 0% | 3.20 -> 2.14 s |
  | K=400 sustained | 400 | 20% | 0% | 3.07 -> 2.22 s |
  | **Pooled (n=30)** | | **30%** | **0%** | **3.20 -> 2.07 s (-1.13 s, -35%)** |
  | *Paper* | | *40%* | *10%, 13.5 km/h* | *2.6 -> <1.5 s* |

  The mechanism reproduces clearly and stably: the ACC is left without an in-lane
  lead for 19-35 of the attacked steps, and the attack removes **1.13 s (35%)** of
  the ego's mean time-to-collision, with the worst case reaching 0.72 s. No attack
  step failed (`err=0`) in any of the 30 cases.

  Two caveats, neither of which tuning would fix honestly:

  1. **No case reached contact**, against the paper's one collision in ten. The ego
     brakes late but always stops short.
  2. **The unsafe *rate* is a noisy statistic here.** It swung 40% / 30% / 20%
     across three runs, and running the attack all the way to the blocker (K=400)
     *lowered* it, which is the opposite of the expected direction. The reason is
     visible in the pooled data: 12 of 30 cases land within +/-0.3 s of the 1.5 s
     cut, so the threshold slices through the middle of the TTC distribution and
     small perturbations flip cases across it. The pooled 30% is consistent with
     the paper's 40% at n=30, but does not confirm it.

  Quote the pooled rate together with the TTC drop, which barely moves between
  runs. `--attack_frames` exposes the window cap; per-frame stealth is unaffected
  by it. If a clean baseline collides or the attack never bites, the geometry knobs
  are `BLOCKER_AHEAD`, `BLOCKER_ENCROACH` and `MOVEOUT_STEPS` in
  `carla_demo/mvp_carla/config.py`.
* **Suppression case count.** A single map yields only a handful of straight
  stretches with an adjacent lane, fewer than the paper's 10 suppression cases.
  `build_moveout_scenarios()` expands them deterministically by perturbing blocker
  distance (+/-5 m) and lane encroachment (+/-0.15 m), varying roads fastest so the
  first cases span distinct geometry. Screening keeps only scenarios whose clean
  run reaches the blocker, does not collide, and holds TTC >= 1.5 s.
* **Overhead table.** The per-frame latency figures (ray casting 20 ms, encoding
  5 ms, PertNet <1 ms, PGD 70 ms) have no measurement script in this repository.
* **`gen_fig_factors.py`** additionally needs `results_paper/multi_attacker/results.pkl`,
  produced by a multi-attacker sweep that is not part of this artifact. The rest
  of the figure works from the section 1 results.
* **OPV2V occupancy maps** are reproducible via `test/test_occupancy_map.py`
  (Section 5). Regenerating cases 0 and 1 matched the shipped files exactly on
  vehicle set, field names and area counts, with 252 of 255 polygons geometrically
  identical; the remaining three differ by at most 1.5% of their area, which is
  segmentation nondeterminism rather than a different computation. The maps cover
  all 300 perception test cases: those share 136 distinct `case_id`s, and every one
  has a map.

### Environment caveats

* **CARLA runs need a server and the CARLA Python client.** Verified against a
  CARLA **0.9.15** server in Docker on `Town10HD_Opt`, with the `cp37` client wheel
  installed into the same `advCP` environment as everything else; the paper used
  0.9.16 and the code is unchanged between them. `agents.navigation.controller`
  ships with the CARLA PythonAPI rather than the client wheel, so `CARLA_ROOT` has
  to point at a directory containing `PythonAPI/carla`. Step-by-step instructions,
  including extracting both pieces out of the Docker image, are in
  `carla_demo/SETUP.md`. Episodes take 2-3 minutes each and outcomes vary between
  runs, so quote rates over the whole case set, not a single road.
* **HEAL heterogeneous experiments** (`mvp/attack/pertnet_pipeline_heal.py`,
  `mvp/perception/heal_perception.py`) are kept for completeness but are outside
  the four settings above; they need `third_party/HEAL` and HEAL checkpoints,
  which this artifact does not link.

## License

See [LICENSE](LICENSE).
