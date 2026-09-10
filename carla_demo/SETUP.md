# Running the CARLA closed-loop study

The closed-loop experiments need a CARLA server plus the matching Python client.
Everything else in the artifact runs without them.

Verified configuration: **CARLA 0.9.15** server in Docker, client installed into
the same `advCP` environment as the rest of the artifact (Python 3.7), map
`Town10HD_Opt`. The paper used 0.9.16; 0.9.15 works unchanged.

## 1. Start the server

```bash
docker run -d --name carla-server --gpus all --net=host \
    carlasim/carla:0.9.15 ./CarlaUE4.sh -RenderOffScreen

# check it is listening
python -c "import socket; socket.create_connection(('127.0.0.1', 2000), 5); print('CARLA up')"
```

## 2. Install the Python client

The image ships wheels for several interpreters. Pick the one matching `advCP`
(`cp37`) and install it into that environment:

```bash
docker cp carla-server:/home/carla/PythonAPI/carla/dist/carla-0.9.15-cp37-cp37m-manylinux_2_27_x86_64.whl /tmp/
conda run -n advCP pip install /tmp/carla-0.9.15-cp37-cp37m-manylinux_2_27_x86_64.whl
```

## 3. Provide the `agents` package

`mvp_carla.runner` drives the ego with `agents.navigation.controller.VehiclePIDController`,
which ships with the CARLA PythonAPI rather than the `carla` wheel. Extract it and
point `CARLA_ROOT` at the directory that contains `PythonAPI/carla`:

```bash
mkdir -p third_party/CARLA/PythonAPI/carla
docker cp carla-server:/home/carla/PythonAPI/carla/agents third_party/CARLA/PythonAPI/carla/
export CARLA_ROOT=$PWD/third_party/CARLA
```

`mvp_carla/__init__.py` appends `$CARLA_ROOT/PythonAPI/carla` to `sys.path`; if
`agents` is already importable, `CARLA_ROOT` can be left unset.

## 4. Run

```bash
export CARLA_ROOT=$PWD/third_party/CARLA

# offline checks first: geometry and metrics, no server needed
python carla_demo/tests/test_moveout_offline.py

python carla_demo/run_paper_cases.py                    # phantom cut-in / braking
python carla_demo/run_collision_cases.py                # suppression / collision
python carla_demo/run_collision_cases.py --n_cases 1 --max_screen 3   # quick check
```

Both scripts put the server in synchronous mode at 20 Hz and restore the previous
settings on exit, including after an exception. Do not run them concurrently
against one server: they share global world settings.

## Recovering a stuck world

A run that dies mid-spawn leaves vehicles behind, and they sit in the slots the
scenarios spawn into, so the next run fails with `could not spawn near 0 m`. A
crashed run also leaves the world in synchronous mode, where it advances only when
a client ticks it, which looks like a frozen simulator to anything else connected.

```bash
python carla_demo/cleanup_world.py
```

Both scenario classes now destroy their own partial spawns on failure, and the
evaluation loops skip a road they cannot spawn on rather than aborting, so this
should be rare. It is still the first thing to try when spawns start failing.

In synchronous mode the actor list is stale until the world is ticked; querying it
without ticking reports zero actors even when the map is full of leftovers.

## Notes

* Runtime is dominated by the closed loop, roughly 2-3 minutes per episode
  (collaborative perception plus GRIP++ at every second tick). A screening pass
  and an attack pass run per case, so a 10-case suppression run with 30 screened
  scenarios is a couple of hours.
* Outcomes vary between runs. Perception, tracking and the physics step are not
  seeded end to end, so TTC on a given road moves by a few tenths of a second and
  a marginal case can flip across the 1.5 s threshold. Report rates over the full
  case set rather than a single road.
* Roads where the ego fails to move (a blocked spawn) are rejected by screening
  as `reached=False`; they are not counted as safe baselines.
