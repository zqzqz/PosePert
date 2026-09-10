"""
Clear leftover actors from the CARLA world and restore asynchronous mode.

An episode that dies mid-spawn can leave vehicles behind. They occupy the spawn
slots the scenarios use, so the next run fails with "could not spawn near 0 m",
and a crashed run also leaves the world in synchronous mode, where it only
advances when some client ticks it. Run this between sessions, or any time a run
starts failing to spawn.

    python carla_demo/cleanup_world.py
    python carla_demo/cleanup_world.py --keep-sync   # leave synchronous mode alone

Note: in synchronous mode the actor list is stale until the world is ticked, so
this ticks once before enumerating. Without that, a world full of leftovers
reports zero actors.
"""
import argparse

import carla


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--keep-sync", action="store_true",
                    help="do not switch the world back to asynchronous mode")
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    settings = world.get_settings()
    print("synchronous mode: %s" % settings.synchronous_mode)

    if settings.synchronous_mode:
        world.tick()          # refresh the actor snapshot before enumerating

    vehicles = list(world.get_actors().filter("vehicle.*"))
    sensors = list(world.get_actors().filter("sensor.*"))
    print("found %d vehicle(s), %d sensor(s)" % (len(vehicles), len(sensors)))

    for s in sensors:
        try:
            s.stop()
        except Exception:
            pass
    if vehicles or sensors:
        client.apply_batch_sync(
            [carla.command.DestroyActor(a) for a in sensors + vehicles], True)
        if world.get_settings().synchronous_mode:
            world.tick()
        print("destroyed; %d vehicle(s) remain"
              % len(list(world.get_actors().filter("vehicle.*"))))

    if not args.keep_sync and settings.synchronous_mode:
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        print("restored asynchronous mode")


if __name__ == "__main__":
    main()
