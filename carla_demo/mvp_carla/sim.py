"""CARLA scene helpers: sensor config, geometry conversions, scenario setup."""
import math
import numpy as np
import carla

from .config import (OPV2V_LIDAR, MOUNT_Z, TARGET_SPEED, TARGET_AHEAD, COLLAB_AHEAD,
                     BLOCKER_AHEAD, BLOCKER_ENCROACH)


def lidar_blueprint(bp_lib):
    bp = bp_lib.find("sensor.lidar.ray_cast")
    for k, v in OPV2V_LIDAR.items():
        if bp.has_attribute(k):
            bp.set_attribute(k, v)
    return bp


def sensor_pose(sensor):
    """LiDAR world pose as mvp expects: [x, y, z, roll, yaw, pitch] (degrees)."""
    t = sensor.get_transform()
    return np.array([t.location.x, t.location.y, t.location.z,
                     t.rotation.roll, t.rotation.yaw, t.rotation.pitch], dtype=np.float64)


def actor_world_bbox(actor):
    """Vehicle GT box in world frame: [x, y, z_bottom, l, w, h, yaw_rad]."""
    t = actor.get_transform()
    bb = actor.bounding_box
    return np.array([t.location.x, t.location.y, t.location.z - bb.extent.z,
                     2 * bb.extent.x, 2 * bb.extent.y, 2 * bb.extent.z,
                     math.radians(t.rotation.yaw)], dtype=np.float64)


def lidar_points(measurement):
    """CARLA LiDAR -> (N, 4) [x, y, z, intensity], raw CARLA frame (do NOT y-flip; see P0)."""
    return np.frombuffer(measurement.raw_data, dtype=np.float32).reshape(-1, 4).copy()


def gt_vehicles_dict(world, exclude_ids):
    """OPV2V-format GT vehicles dict (one entry per nearby vehicle actor)."""
    gt = {}
    for a in world.get_actors().filter("vehicle.*"):
        if a.id in exclude_ids:
            continue
        bb = actor_world_bbox(a)
        gt[a.id] = dict(location=list(bb[:3]), angle=[0, math.degrees(bb[6]), 0],
                        extent=[bb[3] / 2, bb[4] / 2, bb[5] / 2], center=[0, 0, 0])
    return gt


def find_straight_roads(amap, min_len=40.0, max_yaw_change=6.0):
    """Return [(ego_waypoint, adjacent_lane_waypoint), ...] on straight drivable stretches."""
    out = []
    for sp in amap.get_spawn_points():
        wp = amap.get_waypoint(sp.location)
        adj = wp.get_left_lane() or wp.get_right_lane()
        if adj is None or adj.lane_type != carla.LaneType.Driving:
            continue
        nxt = wp.next(min_len)
        if not nxt:
            continue
        dyaw = abs((nxt[0].transform.rotation.yaw - wp.transform.rotation.yaw + 180) % 360 - 180)
        if dyaw < max_yaw_change:
            out.append((wp, adj))
    return out


class MoveInScene:
    """Spawns the move-in scenario: ego + parallel target (adjacent lane) + attacker collaborator,
    each with an OPV2V-spec LiDAR. Provides per-tick frame building and cleanup."""

    def __init__(self, world, road, target_ahead=None, collab_ahead=None, target_speed=None):
        self.world = world
        self.bp = world.get_blueprint_library()
        self.ego_wp, self.adj_wp = road
        self.fwd = self.ego_wp.transform.get_forward_vector()
        self.actors, self.sensors, self._latest = [], {}, {}
        self.ego = self.target = self.collab = None
        self.target_ahead = TARGET_AHEAD if target_ahead is None else target_ahead
        self.collab_ahead = COLLAB_AHEAD if collab_ahead is None else collab_ahead
        self.target_speed = TARGET_SPEED if target_speed is None else target_speed

    def _ahead(self, wp, dist):
        return wp.next(dist)[0] if dist > 0 else wp

    def _spawn_on(self, bp, base_wp, dist):
        """Spawn ~dist ahead on base_wp's lane, nudging forward if the slot is occupied."""
        z = carla.Location(z=0.3)
        for d in (dist, dist + 3, dist + 6, dist + 9, max(0.0, dist - 3)):
            wp = self._ahead(base_wp, d) if d > 0 else base_wp
            a = self.world.try_spawn_actor(bp, carla.Transform(wp.transform.location + z, base_wp.transform.rotation))
            if a:
                return a
        raise RuntimeError("could not spawn near %.0f m" % dist)

    def spawn(self):
        try:
            self.ego = self._spawn_on(self.bp.find("vehicle.lincoln.mkz_2017"), self.ego_wp, 0.0)
            self.actors.append(self.ego)
            self.target = self._spawn_on(self.bp.find("vehicle.tesla.model3"), self.adj_wp, self.target_ahead)
            self.actors.append(self.target)
            self.collab = self._spawn_on(self.bp.find("vehicle.audi.tt"), self.adj_wp, self.collab_ahead)
            self.actors.append(self.collab)
            for name, veh in [("ego", self.ego), ("collab", self.collab)]:
                s = self.world.spawn_actor(lidar_blueprint(self.bp),
                                           carla.Transform(carla.Location(z=MOUNT_Z)), attach_to=veh)
                self.sensors[name] = s
                self.actors.append(s)
                s.listen(lambda d, n=name: self._latest.__setitem__(n, d))
        except Exception:
            # A partial spawn must not leak: leftover vehicles occupy the spawn slots and
            # make every later episode on this road fail too.
            self.abort()
            raise
        return self

    def abort(self):
        """Destroy whatever was spawned before a failure. Safe to call repeatedly."""
        for s in self.sensors.values():
            try:
                s.stop()
            except Exception:
                pass
        for a in self.actors:
            try:
                a.destroy()
            except Exception:
                pass
        self.actors, self.sensors, self._latest = [], {}, {}

    def ego_route(self, n=140, step=4):
        return [self.ego_wp] + [self._ahead(self.ego_wp, d) for d in range(step, n, step)]

    def drive_constant(self):
        """Keep target + collaborator driving straight at the target speed."""
        v = carla.Vector3D(self.fwd.x * self.target_speed, self.fwd.y * self.target_speed, 0.0)
        for veh in (self.target, self.collab):
            veh.set_target_velocity(v)

    def sensors_ready(self):
        return "ego" in self._latest and "collab" in self._latest

    def build_frame(self, ego_speed_kmh):
        """Build the mvp perception frame {vehicle: {lidar, lidar_pose, params}} for this tick."""
        ep = sensor_pose(self.sensors["ego"])
        cp = sensor_pose(self.sensors["collab"])
        gt = gt_vehicles_dict(self.world, {self.ego.id, self.collab.id})
        frame = {}
        for name, pose in [("ego", ep), ("collab", cp)]:
            frame[name] = {
                "lidar": lidar_points(self._latest[name]),
                "lidar_pose": pose,
                "params": {"ego_speed": ego_speed_kmh, "vehicles": gt, "lidar_pose": pose},
            }
        return frame, ep, cp

    def destroy(self, client):
        for s in self.sensors.values():
            try:
                s.stop()
            except Exception:
                pass
        client.apply_batch([carla.command.DestroyActor(a) for a in self.actors])
        self.actors, self.sensors, self._latest = [], {}, {}


class MoveOutScene(MoveInScene):
    """Suppression scenario: a STOPPED vehicle marginally blocks the ego lane, with the
    attacker collaborator alongside in the adjacent lane.

    Clean behaviour: the ego's stack gates the blocker as an in-lane lead and the ACC brakes
    to a safe stop. Under attack the collaborator shifts the blocker's perceived pose out of
    the lane, so no lead is reported and braking is delayed or suppressed.

    Inherits sensor setup, frame building and teardown from MoveInScene; only the spawn
    geometry, the stationary target and the ego collision sensor differ.
    """

    def __init__(self, world, road, blocker_ahead=None, collab_ahead=None, encroach=None,
                 **_ignored):
        super().__init__(world, road, target_ahead=None, collab_ahead=collab_ahead,
                         target_speed=0.0)
        self.blocker_ahead = BLOCKER_AHEAD if blocker_ahead is None else blocker_ahead
        self.encroach = BLOCKER_ENCROACH if encroach is None else encroach
        self.collisions = []

    def _adjacent_sign(self):
        """+1 if the adjacent lane lies to the ego lane's right, -1 if to its left.

        The blocker leans toward that side, so the move-out direction (further toward the
        adjacent lane) is also the direction that clears the ego lane.
        """
        tf = self.ego_wp.transform
        right = tf.get_right_vector()
        d = self.adj_wp.transform.location - tf.location
        return 1.0 if (d.x * right.x + d.y * right.y) >= 0 else -1.0

    def _spawn_offset(self, bp, base_wp, dist, lateral):
        """Spawn `dist` ahead of base_wp, displaced `lateral` m along the lane's right vector."""
        for d in (dist, dist + 4, dist + 8, dist - 4):
            if d <= 0:
                continue
            nxt = base_wp.next(d)
            if not nxt:
                continue
            tf = nxt[0].transform
            r = tf.get_right_vector()
            loc = carla.Location(tf.location.x + r.x * lateral,
                                 tf.location.y + r.y * lateral,
                                 tf.location.z + 0.3)
            a = self.world.try_spawn_actor(bp, carla.Transform(loc, tf.rotation))
            if a:
                return a
        raise RuntimeError("could not spawn blocker near %.0f m" % dist)

    def spawn(self):
        lateral = self._adjacent_sign() * self.encroach
        try:
            self.ego = self._spawn_on(self.bp.find("vehicle.lincoln.mkz_2017"), self.ego_wp, 0.0)
            self.actors.append(self.ego)
            # Blocker sits in the EGO lane, nudged toward the adjacent lane so it only marginally
            # blocks. self.target is the attacked object, matching MoveInScene's naming.
            self.target = self._spawn_offset(self.bp.find("vehicle.tesla.model3"),
                                             self.ego_wp, self.blocker_ahead, lateral)
            self.actors.append(self.target)
            self.collab = self._spawn_on(self.bp.find("vehicle.audi.tt"), self.adj_wp, self.collab_ahead)
            self.actors.append(self.collab)
            for name, veh in [("ego", self.ego), ("collab", self.collab)]:
                s = self.world.spawn_actor(lidar_blueprint(self.bp),
                                           carla.Transform(carla.Location(z=MOUNT_Z)), attach_to=veh)
                self.sensors[name] = s
                self.actors.append(s)
                s.listen(lambda d, n=name: self._latest.__setitem__(n, d))
            cs = self.world.spawn_actor(self.bp.find("sensor.other.collision"),
                                        carla.Transform(), attach_to=self.ego)
            cs.listen(self.on_collision)
            self.sensors["collision"] = cs
            self.actors.append(cs)
            # Hold the blocker still: handbrake plus zero velocity survives the physics step.
            self.target.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
        except Exception:
            self.abort()
            raise
        return self

    def drive_constant(self):
        """The blocker is stationary and the collaborator holds station beside it."""
        self.target.set_target_velocity(carla.Vector3D(0, 0, 0))
        self.collab.set_target_velocity(carla.Vector3D(0, 0, 0))

    def sensors_ready(self):
        return "ego" in self._latest and "collab" in self._latest

    def on_collision(self, event):
        """Record only impacts with the blocker.

        The ego can also clip scenery or the collaborator, neither of which says anything
        about whether braking was suppressed, so those events are dropped.
        """
        if getattr(event.other_actor, "id", None) == self.target.id:
            self.collisions.append(event)

    def collided(self):
        """True only if the ego struck the blocker."""
        return len(self.collisions) > 0

    def gap_to_blocker(self):
        """Longitudinal ego->blocker distance (m) along the ego heading, box to box."""
        et, bt = self.ego.get_transform(), self.target.get_transform()
        yaw = math.radians(et.rotation.yaw)
        fwd = np.array([math.cos(yaw), math.sin(yaw)])
        rel = np.array([bt.location.x - et.location.x, bt.location.y - et.location.y])
        return float(rel @ fwd) - (self.ego.bounding_box.extent.x + self.target.bounding_box.extent.x)


def longitudinal_ttc(ego_actor, blocker_actor, ego_fwd):
    """Ground-truth time-to-collision (s) between the ego and a stationary blocker.

    Uses the true actor poses, never perception, so the metric stays valid under attack.
    Returns inf when the ego is not closing on the blocker or the paths do not overlap
    laterally. The gap is measured between bounding boxes, not centres.
    """
    et, bt = ego_actor.get_transform(), blocker_actor.get_transform()
    ego_xy = np.array([et.location.x, et.location.y])
    blk_xy = np.array([bt.location.x, bt.location.y])
    fwd = np.asarray(ego_fwd, dtype=np.float64)
    fwd = fwd / (np.linalg.norm(fwd) or 1.0)

    rel = blk_xy - ego_xy
    lon = float(rel @ fwd)
    lat = abs(float(rel[0] * fwd[1] - rel[1] * fwd[0]))
    if lon <= 0:
        return float("inf")                       # blocker is behind the ego

    eb, bb = ego_actor.bounding_box, blocker_actor.bounding_box
    half_w = eb.extent.y + bb.extent.y
    if lat > half_w:
        return float("inf")                       # lateral paths do not overlap

    gap = lon - (eb.extent.x + bb.extent.x)
    v_close = float(np.array([ego_actor.get_velocity().x,
                              ego_actor.get_velocity().y]) @ fwd)
    if v_close <= 0.1:
        return float("inf")                       # stopped or reversing: not closing
    return max(gap, 0.0) / v_close


def build_moveout_scenarios(roads, n_wanted=10, ahead_deltas=(0.0, -5.0, +5.0),
                            encroach_deltas=(0.0, -0.15, +0.15)):
    """Expand straight roads into distinct suppression scenarios.

    A map yields only a handful of straight stretches with an adjacent lane, which is fewer
    than the case count the paper reports. Each road is therefore perturbed along the two
    parameters that actually change the outcome: how far ahead the blocker sits (how much
    room the ego has to react) and how far it encroaches (how marginal the in-lane gating is).

    Roads vary fastest so the first N scenarios are spread across distinct geometry rather
    than being N perturbations of one road. Deterministic: no RNG, so a rerun screens the
    same scenarios in the same order.
    """
    variants = [(da, de) for de in encroach_deltas for da in ahead_deltas]
    out = []
    for da, de in variants:
        for road in roads:
            out.append(dict(road=road,
                            blocker_ahead=BLOCKER_AHEAD + da,
                            encroach=max(0.4, BLOCKER_ENCROACH + de),
                            collab_ahead=COLLAB_AHEAD))
            if len(out) >= n_wanted * 3:      # screening headroom: not every scenario is clean
                return out
    return out

