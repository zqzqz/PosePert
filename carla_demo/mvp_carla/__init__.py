"""
mvp_carla: clean implementation of the custom collaborative-perception AV stack
and the PosePert scenario attack, running in V2Xverse/CARLA closed-loop.

Importing this package bootstraps the environment (sys.path + CUDA device) so the
mvp/, OpenCOOD, and GRIP++ third-party code resolves. Import it BEFORE importing
torch/opencood/GRIP in any entry script.
"""
import os
import sys

# Single GPU (index 0). GRIP's main.py hardcodes CUDA_VISIBLE_DEVICES='1'; we pin 0.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Repository root (the directory holding mvp/, data/, models/, third_party/).
# Defaults to the parent of carla_demo/; override with MVP_ROOT to point elsewhere.
MVP_ROOT = os.environ.get(
    "MVP_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
)

_PATHS = [
    MVP_ROOT,
    os.path.join(MVP_ROOT, "third_party/AdvTrajectoryPrediction"),              # prediction.model.GRIP
    os.path.join(MVP_ROOT, "third_party/AdvTrajectoryPrediction/prediction/model/GRIP/GRIP"),  # nested GRIP (layers/...)
    # `agents.navigation.controller` (the PID controller the runner drives the ego
    # with) ships with the CARLA PythonAPI. Add it here if it is not already on
    # PYTHONPATH, e.g. $CARLA_ROOT/PythonAPI/carla.
    os.path.join(os.environ.get("CARLA_ROOT", ""), "PythonAPI/carla")
    if os.environ.get("CARLA_ROOT") else None,
]
for _p in _PATHS:
    if _p and _p not in sys.path:
        sys.path.insert(0, _p) if _p == MVP_ROOT else sys.path.append(_p)

# mvp/perception/opencood_perception resolves data/models via paths relative to MVP_ROOT.
os.chdir(MVP_ROOT)

# Force CUDA + cuDNN init on device 0 NOW, before any GRIP import. GRIP's main.py sets
# CUDA_VISIBLE_DEVICES='1' on import; if cuDNN initializes after that (e.g. GRIP's GRU
# flatten_parameters), it sees zero devices and crashes. Initializing here pins device 0.
try:
    import torch as _torch
    if _torch.cuda.is_available():
        _t = _torch.zeros(1, device="cuda")
        _torch.backends.cudnn.is_acceptable(_t)
        del _t
except Exception:
    pass


def reassert_cuda_device():
    """Re-pin device 0 (call around GRIP imports, which set CUDA_VISIBLE_DEVICES='1')."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
