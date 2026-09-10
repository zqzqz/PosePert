"""Episode loop + scenario-screening evaluation."""
import math
import numpy as np
import carla
from agents.navigation.controller import VehiclePIDController

from .config import (EGO_KMH, ATTACK_START, ATTACK_FRAMES, DANGER_SLOWDOWN, CLEAN_MIN_SPEED,
                     SLOWDOWN_MARGIN, MOVEOUT_ATTACK_FRAMES, MOVEOUT_STEPS, TTC_UNSAFE,
                     TTC_SAFE_BASELINE, MOVEOUT_ATTACK_RANGE)
from .sim import MoveInScene, MoveOutScene, actor_world_bbox, longitudinal_ttc
from .stack import AVStack, acc_target_speed

_LAT = {'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0, 'dt': 0.05}
_LON = {'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.05, 'dt': 0.05}


def run_episode(client, world, road, perception, predictor, vox_attacker, attack, steps=56, warmup=40,
                attack_frames=None, target_ahead=None, collab_ahead=None, target_speed=None, max_offset=None):
    """Run one closed-loop episode on `road`. Returns
    dict(min_intrusion, min_speed, decelerated, realized_shift). Optional overrides let a sweep vary
    the attack window (attack_frames), scenario geometry/speed, and the offset cap."""
    K = ATTACK_FRAMES if attack_frames is None else attack_frames
    scene = MoveInScene(world, road, target_ahead, collab_ahead, target_speed).spawn()
    try:
        ctrl = VehiclePIDController(scene.ego, args_lateral=_LAT, args_longitudinal=_LON)
        av = AVStack(perception, predictor, vox_attacker if attack else None, max_offset=max_offset)
        route = scene.ego_route()
        # warmup: fill sensors AND accelerate the ego to cruise (free road, no perception/attack yet)
        idx = 0
        for _ in range(warmup):
            scene.drive_constant()
            world.tick()
            eloc = scene.ego.get_location()
            while idx < len(route) - 1 and eloc.distance(route[idx].transform.location) < 3.5:
                idx += 1
            scene.ego.apply_control(ctrl.run_step(EGO_KMH, route[idx]))
        min_intr = 99.0
        cur_lead = None           # latest predicted lead obstacle (held between perception steps)
        post_speeds = []          # ego speeds during/after the attack window (for the danger metric)
        for step in range(steps):
            scene.drive_constant()
            world.tick()
            eloc = scene.ego.get_location()
            spd = 3.6 * scene.ego.get_velocity().length()
            while idx < len(route) - 1 and eloc.distance(route[idx].transform.location) < 3.5:
                idx += 1
            twp = route[idx]
            if step % 2 == 0 and scene.sensors_ready():
                frame, ep, cp = scene.build_frame(spd)
                tw = actor_world_bbox(scene.target)
                ego_xy = np.array([eloc.x, eloc.y])
                ego_fwd = np.array([math.cos(math.radians(ep[4])), math.sin(math.radians(ep[4]))])
                in_window = attack and (ATTACK_START <= step // 2 < ATTACK_START + K)
                r = av.step(frame, ep, cp, tw, ego_xy, ego_fwd, attack=in_window)
                min_intr = min(min_intr, r["intrusion"])
                cur_lead = r["lead"]
                if step // 2 >= ATTACK_START:
                    post_speeds.append(spd)
            # ACC: the predicted in-lane lead obstacle sets the target speed; the SAME speed PID tracks it.
            v_target = acc_target_speed(EGO_KMH, spd, cur_lead)
            control = ctrl.run_step(v_target, twp)
            scene.ego.apply_control(control)
        min_speed = min(post_speeds) if post_speeds else EGO_KMH
        return dict(min_intrusion=min_intr, min_speed=min_speed,
                    decelerated=(min_speed < EGO_KMH * (1 - DANGER_SLOWDOWN)),
                    realized_shift=av.realized_shift)
    finally:
        scene.destroy(client)


def evaluate(client, world, roads, perception, predictor, vox_attacker,
             n_clean=6, max_screen=24, steps=56, log=print, **variant):
    """Screen for baseline-SAFE scenarios, then run the attack on each (same scenario).
    Danger = PAIRED: attack makes the ego's own min-speed drop >= SLOWDOWN_MARGIN km/h below its
    baseline (calibration-free; immune to the ego not holding exactly cruise). `variant` overrides
    are forwarded to run_episode (attack_frames, target_ahead, collab_ahead, target_speed, max_offset)."""
    scene_kw = {k: variant.get(k) for k in ("target_ahead", "collab_ahead", "target_speed")}
    clean = []
    for ri, road in enumerate(roads[:max_screen]):
        if len(clean) >= n_clean:
            break
        try:
            b = run_episode(client, world, road, perception, predictor, None, attack=False,
                            steps=steps, **scene_kw)
        except RuntimeError as e:
            # Occupied or otherwise unusable spawn slots: skip the road rather than losing
            # the whole evaluation to one bad stretch.
            log("screen road %d: SKIPPED (%s)" % (ri, e))
            continue
        is_clean = b["min_speed"] >= CLEAN_MIN_SPEED
        log("screen road %d: baseline min_speed=%.1f clean=%s" % (ri, b["min_speed"], is_clean))
        if is_clean:
            clean.append((road, b))

    results = []
    for i, (road, b) in enumerate(clean):
        try:
            a = run_episode(client, world, road, perception, predictor, vox_attacker,
                            attack=True, steps=steps, **variant)
        except RuntimeError as e:
            log("clean %d: SKIPPED on attack pass (%s)" % (i, e))
            continue
        slowdown = b["min_speed"] - a["min_speed"]
        danger = slowdown >= SLOWDOWN_MARGIN
        results.append(dict(baseline=b, attack=a, slowdown=slowdown, danger=danger))
        log("clean %d | baseline %.1f -> attack %.1f km/h (slow %.1f) intr=%.1f shift=%.2f danger=%s"
            % (i, b["min_speed"], a["min_speed"], slowdown, a["min_intrusion"], a["realized_shift"], danger))

    rate = 100.0 * sum(r["danger"] for r in results) / max(len(results), 1)
    mean_slow = float(np.mean([r["slowdown"] for r in results])) if results else 0.0
    return results, rate, mean_slow


# ---- move-out / suppression (collision cases) ------------------------------
def run_episode_moveout(client, world, road, perception, predictor, vox_attacker, attack,
                        steps=None, warmup=40, attack_frames=None, blocker_ahead=None,
                        collab_ahead=None, encroach=None, max_offset=None):
    """One closed-loop suppression episode: the ego approaches a stopped blocker that
    marginally occupies its lane.

    Returns dict(min_ttc, collided, impact_kmh, min_speed, braked, realized_shift, reached).
    TTC and the collision flag come from ground-truth actor state, so an attack that fools
    perception cannot flatter the safety metric. `reached` is False when the episode ended
    before the ego got near the blocker, which makes the run uninformative rather than safe.
    """
    K = MOVEOUT_ATTACK_FRAMES if attack_frames is None else attack_frames
    n_steps = MOVEOUT_STEPS if steps is None else steps
    scene = MoveOutScene(world, road, blocker_ahead=blocker_ahead, collab_ahead=collab_ahead,
                         encroach=encroach).spawn()
    try:
        ctrl = VehiclePIDController(scene.ego, args_lateral=_LAT, args_longitudinal=_LON)
        av = AVStack(perception, predictor, vox_attacker if attack else None,
                     max_offset=max_offset, mode="moveout")
        route = scene.ego_route(n=260, step=4)
        idx = 0
        for _ in range(warmup):
            scene.drive_constant()
            world.tick()
            eloc = scene.ego.get_location()
            while idx < len(route) - 1 and eloc.distance(route[idx].transform.location) < 3.5:
                idx += 1
            scene.ego.apply_control(ctrl.run_step(EGO_KMH, route[idx]))

        min_ttc, impact_kmh, min_speed, closest = float("inf"), None, EGO_KMH, 1e9
        cur_lead, n_attacked, ttc_at_trigger = None, 0, None
        for step in range(n_steps):
            scene.drive_constant()
            world.tick()
            eloc = scene.ego.get_location()
            spd = 3.6 * scene.ego.get_velocity().length()
            while idx < len(route) - 1 and eloc.distance(route[idx].transform.location) < 3.5:
                idx += 1
            twp = route[idx]

            if step % 2 == 0 and scene.sensors_ready():
                frame, ep, cp = scene.build_frame(spd)
                tw = actor_world_bbox(scene.target)
                ego_xy = np.array([eloc.x, eloc.y])
                ego_fwd = np.array([math.cos(math.radians(ep[4])), math.sin(math.radians(ep[4]))])
                # Proximity-triggered window: open it once the ego is close enough that a
                # clean stack would gate the blocker in lane, then hold it for K steps.
                gap = scene.gap_to_blocker()
                in_window = attack and gap < MOVEOUT_ATTACK_RANGE and n_attacked < K
                if in_window:
                    if n_attacked == 0:
                        ttc_at_trigger = longitudinal_ttc(scene.ego, scene.target, ego_fwd)
                    n_attacked += 1
                r = av.step(frame, ep, cp, tw, ego_xy, ego_fwd, attack=in_window)
                cur_lead = r["lead"]
                min_ttc = min(min_ttc, longitudinal_ttc(scene.ego, scene.target, ego_fwd))

            min_speed = min(min_speed, spd)
            closest = min(closest, eloc.distance(scene.target.get_location()))
            if scene.collided() and impact_kmh is None:
                impact_kmh = spd
                break
            v_target = acc_target_speed(EGO_KMH, spd, cur_lead)
            scene.ego.apply_control(ctrl.run_step(v_target, twp))

        return dict(min_ttc=min_ttc, collided=scene.collided(),
                    impact_kmh=impact_kmh, min_speed=min_speed,
                    braked=(min_speed < EGO_KMH * (1 - DANGER_SLOWDOWN)),
                    realized_shift=av.realized_shift, closest=closest,
                    n_attacked=n_attacked, ttc_at_trigger=ttc_at_trigger,
                    n_attack_ok=av.n_attack_ok, n_attack_err=av.n_attack_err,
                    attack_err=av.last_attack_err, n_lead_suppressed=av.n_lead_suppressed,
                    reached=(closest < 25.0 or scene.collided()))
    finally:
        scene.destroy(client)


def evaluate_moveout(client, world, scenarios, perception, predictor, vox_attacker,
                     n_clean=10, max_screen=40, attack_frames=None, log=print):
    """Screen for suppression scenarios the clean ego handles safely, then attack each.

    `scenarios` is a list of dicts with keys road, blocker_ahead, encroach, collab_ahead --
    see build_moveout_scenarios(). A scenario is kept only if the clean run reaches the
    blocker, stops without colliding, and keeps min TTC >= TTC_SAFE_BASELINE, so any unsafe
    outcome under attack is attributable to the attack.
    """
    clean = []
    for si, sc in enumerate(scenarios[:max_screen]):
        if len(clean) >= n_clean:
            break
        kw = {k: sc.get(k) for k in ("blocker_ahead", "collab_ahead", "encroach")}
        try:
            b = run_episode_moveout(client, world, sc["road"], perception, predictor, None,
                                    attack=False, **kw)
        except RuntimeError as e:
            log("screen scenario %d: SKIPPED (%s)" % (si, e))
            continue
        ok = b["reached"] and not b["collided"] and b["min_ttc"] >= TTC_SAFE_BASELINE
        log("screen scenario %d (ahead=%.0f encroach=%.2f): min_ttc=%.2f min_speed=%.1f "
            "closest=%.1f collided=%s reached=%s safe=%s"
            % (si, sc.get("blocker_ahead") or -1, sc.get("encroach") or -1,
               b["min_ttc"], b["min_speed"], b["closest"], b["collided"], b["reached"], ok))
        if ok:
            clean.append((sc, b))

    results = []
    for i, (sc, b) in enumerate(clean):
        kw = {k: sc.get(k) for k in ("blocker_ahead", "collab_ahead", "encroach")}
        try:
            a = run_episode_moveout(client, world, sc["road"], perception, predictor, vox_attacker,
                                    attack=True, attack_frames=attack_frames, **kw)
        except RuntimeError as e:
            log("case %d: SKIPPED on attack pass (%s)" % (i, e))
            continue
        unsafe = bool(a["collided"] or a["min_ttc"] < TTC_UNSAFE)
        results.append(dict(scenario=sc, baseline=b, attack=a, unsafe=unsafe,
                            collided=bool(a["collided"])))
        log("case %d | TTC %.2f -> %.2f s | minspd %.1f -> %.1f | collided=%s impact=%s | "
            "atk=%d ok=%d err=%d lead_suppressed=%d shift=%.2f | unsafe=%s"
            % (i, b["min_ttc"], a["min_ttc"], b["min_speed"], a["min_speed"], a["collided"],
               ("%.1f km/h" % a["impact_kmh"]) if a["impact_kmh"] is not None else "-",
               a["n_attacked"], a["n_attack_ok"], a["n_attack_err"], a["n_lead_suppressed"],
               a["realized_shift"], unsafe))
        if a["n_attack_err"]:
            log("    first attack error: %s" % a["attack_err"])

    n = max(len(results), 1)
    unsafe_rate = 100.0 * sum(r["unsafe"] for r in results) / n
    coll_rate = 100.0 * sum(r["collided"] for r in results) / n
    impacts = [r["attack"]["impact_kmh"] for r in results if r["attack"]["impact_kmh"] is not None]
    worst_impact = max(impacts) if impacts else 0.0
    return results, unsafe_rate, coll_rate, worst_impact

