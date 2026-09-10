"""Shared configuration constants for the mvp_carla AV stack + attack."""

# LiDAR spec matched to OPV2V (so the OPV2V-trained perception transfers to CARLA).
OPV2V_LIDAR = dict(channels="64", range="120", points_per_second="1300000",
                   rotation_frequency="20", upper_fov="2.0", lower_fov="-25.0")
MOUNT_Z = 1.9          # LiDAR mount height (m)

# Tracking / prediction (GRIP++ trained at 10 Hz, 20-frame obs/pred).
OBS_LEN = 20
PRED_LEN = 20
DT = 0.1               # prediction time step (s) == perception cadence (every 2 sim ticks @ 20 Hz)
SIM_DT = 0.05          # CARLA fixed_delta_seconds (20 Hz)

# Scenario kinematics (move-in): ego closes on a slower target in the adjacent lane.
TARGET_SPEED = 9.0 / 3.6   # m/s for target + collaborator
EGO_KMH = 25.0             # ego target speed (km/h)
TARGET_AHEAD = 16.0        # target spawn distance ahead of ego (m), adjacent lane
COLLAB_AHEAD = 8.0         # collaborator (attacker) spawn distance ahead (m), adjacent lane

# Attack.
STEALTH = 0.5          # max per-frame perceived-pose perturbation (m) -- stealth bound
MAX_OFFSET = 1.0       # cap on cumulative perceived shift (m): the faithful attack reliably
                       # realizes ~1.0 m; asking for more (1.5 m) collapses the detection
ATTACK_START = 8       # perception-step at which the attack window opens
ATTACK_FRAMES = 5      # K: number of stealthy attack frames. K=5 (vs the minimal K=3) sustains the
                       # lane-ward perceived velocity ~1 extra frame -> same danger RATE on cruise-clean
                       # roads but ~2x deeper braking (23->15 vs 23->18 km/h). Per-frame stealth (0.5 m)
                       # is unchanged, so K=5 is still a short, bounded burst. (faithful-v1 tag = K=3.)
BETA = 2.0             # voxelwise feature-scaling factor (paper PP-AttFusion; beta>3 corrupts features)
PLAN_GRID = 6          # candidate offsets queried per frame by the online optimizer
PERTNET_CKPT = "models/perturbation_net_paper_pointpillar/perturbation_net_best.pt"  # learned correction

# Lane membership for treating a predicted point as a lead obstacle.
LANE_HALF = 1.7        # |lateral offset| < this (m) => predicted point lies in the ego lane
LOOKAHEAD = 30.0       # only consider predicted points within this longitudinal range (m)

# Adaptive Cruise Control (longitudinal). The nearest predicted-in-lane point of the tracked
# target is the lead obstacle; a constant-time-gap law sets the target speed fed to the speed PID.
ACC_S0 = 5.0           # minimum standoff distance (m)
ACC_TGAP = 1.5         # desired time gap (s)
ACC_KGAP = 0.4         # gap-error feedback gain (1/s)
DANGER_SLOWDOWN = 0.4  # (screening) ego slows > this fraction below cruise => not a clean baseline
CLEAN_MIN_SPEED = 22.0 # (screening) a road counts as clean ONLY if its baseline min-speed stays >= this
                       # km/h. Roads where the slow adjacent target is already mis-gated in-lane in
                       # baseline (ego dips to ~19) have no headroom and mask the attack -- exclude them.
SLOWDOWN_MARGIN = 4.0  # (paired danger) attack makes ego min-speed >= this km/h below its OWN baseline

# ---- Move-out / suppression scenario (collision cases) ---------------------
# A stopped vehicle marginally blocks the ego lane. Clean: the ego perceives it in
# lane and brakes to a safe stop. Attacked: the collaborator shifts its perceived
# pose OUT of the lane, the ACC sees no lead, and braking is delayed or suppressed.
BLOCKER_AHEAD = 45.0     # blocker spawn distance ahead of the ego, same lane (m)
BLOCKER_ENCROACH = 1.0   # blocker lateral offset from lane centre (m), toward the adjacent
                         # lane. < LANE_HALF, so it marginally blocks: the clean stack still
                         # gates it in-lane, and ~1 m of perceived shift is enough to clear it.
MOVEOUT_ATTACK_RANGE = 35.0  # m. The suppression window is triggered by proximity, not by a
                             # fixed step index: the attack has to be live while the ego would
                             # otherwise gate the blocker in lane (LOOKAHEAD = 30 m) and brake.
                             # A fixed start fires while the ego is still far away and expires
                             # before the braking decision, leaving TTC unchanged.
MOVEOUT_ATTACK_FRAMES = 40   # K for suppression: perception steps the window stays open once
                             # triggered. Per-frame stealth (STEALTH) is unchanged; the
                             # perturbation simply has to persist while the ego closes on the
                             # blocker, rather than fire as the 5-frame burst move-in uses.
MOVEOUT_STEPS = 160          # ticks per episode: enough to reach a blocker 45 m ahead at EGO_KMH.
TTC_UNSAFE = 1.5             # s. Unsafe outcome = a collision, or min TTC below this.
TTC_SAFE_BASELINE = 1.5      # screening: a clean episode must keep min TTC >= this and not collide.
