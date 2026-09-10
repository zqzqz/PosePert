# mvp_carla — closed-loop PosePert attack in V2Xverse/CARLA

A closed-loop testbed for the **PosePert** pose-perturbation attack against intermediate-fusion
collaborative perception. A custom AV stack (collaborative detection → tracking → prediction →
ACC) drives in CARLA 0.9.16; a malicious collaborator runs the *faithful* voxelwise feature attack
to make the perceived/predicted pose of a neighboring vehicle drift into the ego lane, inducing the
ego's adaptive cruise control to brake when the road was actually clear.

Everything here is **simulation-only**, for the security-research paper in
`../MultiVehiclePerceptionPaper` (defensive characterization of the attack).

## Layout

```
carla_demo/
  mvp_carla/                 # the package (the production stack)
    __init__.py              #   env bootstrap: sys.path for mvp/OpenCOOD/GRIP + pin CUDA device 0
    config.py                #   ALL tunable constants (sensor spec, scenario, attack, ACC, metric)
    sim.py                   #   CARLA scene: OPV2V-spec LiDAR, geometry conversions, MoveInScene
    stack.py                 #   AVStack (detect->track->predict), ACC law, online attack planner
    runner.py                #   one closed-loop episode + cruise-clean screening evaluation
  run_demo.py                # single episode, baseline vs attack (CLI)
  run_eval.py                # headline paired danger-rate eval over cruise-clean roads (CLI)
  sweep_variants.py          # variant sweep: K (attack depth) x target speed, same clean roads
  view_demo.py               # windowed visualizer (run the server WITHOUT -RenderOffScreen)
  perception_attack_bench.py # perception-only ablation (raycast / +beta / beta8 / beta2+PertNet)
  diagnose.py                # per-frame attack-chain logger (plan -> det -> track -> vel -> intrusion)
  sort_vertices.py           # pure-PyTorch drop-in for the CUDA oriented-IoU op (no nvcc needed)
  tests/
    test_offline.py          # GPU-only unit checks (no CARLA server)
    test_integration.py      # short closed-loop episodes (REQUIRES the CARLA server)
```

## The scenario (`MoveInScene`)

On a straight drivable stretch: the **ego** (with a collaborative-perception LiDAR) closes on a
slower **target** in the adjacent lane; a malicious **collaborator** (also LiDAR-equipped) drives
nearby and is the attack's injection point. Straight stretches are auto-discovered from the map
spawn points (`find_straight_roads`). Geometry/speeds are in `config.py`
(`TARGET_AHEAD`, `COLLAB_AHEAD`, `TARGET_SPEED`, `EGO_KMH`).

## The AV stack (`AVStack`)

Per perception step (every 2 sim ticks, 10 Hz):

1. **Detect** — OPV2V-trained PointPillar **AttFusion** (OpenCOOD intermediate fusion) fuses the
   ego + collaborator LiDAR. The LiDAR is configured to the OPV2V spec so the pretrained model
   transfers to CARLA (see the `mvp-perception-on-carla` memory for the domain-transfer check).
2. **Track** — AB3DMOT associates detections into tracks.
3. **Predict** — GRIP++ forecasts each track 20 frames ahead.
4. **Control** — a constant-time-gap **ACC** law treats the nearest predicted *in-lane* point of the
   target track as a lead obstacle and sets the target speed for CARLA's `VehiclePIDController`
   (`acc_target_speed`). Braking is therefore smooth and principled, not a binary emergency stop.

## The attack

**PosePert** = make the *perceived* target pose drift laterally toward the ego lane within a small
per-frame stealth bound, so GRIP++ extrapolates a cut-in and the ACC brakes.

- **Online planner** (`AVStack._plan_offset`): each frame, query GRIP++ over candidate offsets in
  `[prev_off, prev_off + STEALTH]` and pick the one minimizing predicted lane intrusion
  (observe-predict-plan, re-planned every frame, capped at `MAX_OFFSET`). This is a query-based
  approximation of mvp's `ScenarioAttacker`.
- **Faithful realization** (`build_attacker`): the planned offset is realized through the real
  `LidarShiftVoxelwiseAttacker.run_multi_vehicle` (ray-cast init + **beta=2** feature scaling +
  **PertNet** learned per-voxel correction), which perturbs the collaborator's *shared feature map*.
  This is the paper-faithful attack — `beta>3` corrupts features; PertNet ~doubles the IoU at a 1 m
  shift. `run_*.py --beta B` overrides beta for ablation.

**Mechanism (verified, see `config.py` comments + `diagnose.py`):**
1. the realized perceived shift **saturates at ~0.7–0.9 m** regardless of the offset cap;
2. GRIP-predicted lane intrusion is driven by the track's lateral **velocity** (extrapolation),
   not its static offset — so a short, sustained lane-ward drift matters more than magnitude;
3. a **slow lead is required**: the ACC only brakes hard for a slow in-lane lead; a faster target
   reads as a safe-gap lead and the ego does not slow.

## Danger metric & cruise-clean screen

`runner.evaluate` reports a **paired** danger rate (calibration-free):

1. **Screen** for *cruise-clean* roads — baseline (no-attack) min-speed `>= CLEAN_MIN_SPEED`
   (22 km/h). This excludes roads where the slow adjacent target is already mis-gated in-lane in the
   baseline (ego dips to ~19 km/h with no headroom), which would otherwise mask the attack.
2. On each clean road, run the attack on the *same* scenario. **Danger** = the attack drops the
   ego's own min-speed `>= SLOWDOWN_MARGIN` (4 km/h) below *its own* baseline.

## Locked-in result

Best variant (committed in `config.py`): **`ATTACK_FRAMES = 5` (K=5) + slow (9 km/h) target →
50% danger (4/8 cruise-clean roads)**, braking 23→15 km/h (−7.9 km/h worst case). K=5 yields the
same danger *rate* as the minimal K=3 but ~2× deeper braking. 50% sits at the top of the paper's
19–50% realization-gap range. The `faithful-v1` tag corresponds to K=3. Reproduce the comparison
with `python sweep_variants.py`.

## What is real vs approximated

- **Real:** AttFusion perception (OPV2V-trained, no retraining), AB3DMOT, GRIP++, and the voxelwise
  feature attack (`LidarShiftVoxelwiseAttacker`). The adversarial trajectory is online-optimized
  against GRIP within the per-frame stealth bound.
- **Approximated:** the OPV2V→CARLA domain transfer (sensor matched, ~0.6 recall); the attack
  realizes the planned shift only partially/noisily → the outcome is **probabilistic** (a danger
  *rate*, not a guaranteed brake), matching the paper's framing.

## What the numbers mean (scope)

The danger rate is **conditional efficacy**, not a real-world base rate. The scenarios are generated
from one procedural move-in template, so the rate characterizes how reliably the attack succeeds
*given* its precondition (an attacker-influenced vehicle in a lead/adjacent position while the ego
cruises in ACC) — it is **not** a probability that such situations occur in the wild. A credible
incidence rate is not obtainable from CARLA (and the V2Xverse evaluation routes are
CARLA-Leaderboard routes deliberately *seeded* with NHTSA pre-crash scenarios, so a probability
computed over them would condition on an already-adversarial distribution). The defensible framing
separates **occurrence** (argued qualitatively — car-following is the dominant driving state) from
**efficacy** (mapped here by sweeping the precondition parameters), rather than claiming a single
frequency.

## Running

**Prerequisites** (see the `v2xverse-carla-env` and `mvp-perception-on-carla` memories): the WSL2
conda env `mvp` (py3.10, torch 2.11+cu128 for Blackwell sm_120) and pretrained weights under
`models/` (AttFusion, GRIP++/OPV2V, PertNet).

```bash
# once per shell (in WSL): conda env mvp + PYTHONPATH
source third_party/V2Xverse/activate_mvp.sh
```

Start the CARLA 0.9.16 server on Windows (`CarlaUE4.exe -carla-rpc-port=2000 -RenderOffScreen`; omit
`-RenderOffScreen` for `view_demo.py`); WSL reaches it on `localhost:2000` via mirrored networking.
Run from `carla_demo/` — importing `mvp_carla` first bootstraps paths + CUDA.

```bash
python tests/test_offline.py          # GPU-only checks (no server)
python tests/test_integration.py      # short closed-loop episodes (needs server)
python run_demo.py                     # one illustrative baseline-vs-attack episode
python run_eval.py --n_clean 8 --max_screen 45   # headline danger rate
python sweep_variants.py               # K x target-speed comparison
python perception_attack_bench.py      # perception-only IoU/%-success ablation
python diagnose.py                     # per-frame attack-chain trace
```

All tunables live in `mvp_carla/config.py`.
