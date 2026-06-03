# reBot GraspNet Demo

This repository runs a reBot Arm B601-DM visual grasping demo on Jetson with an
Orbbec Gemini 2 RGB-D camera. The current project uses:

- `conda` environment: `graspnet`
- Camera: Orbbec Gemini 2 through `pyorbbecsdk`
- Detector: TensorRT YOLO segmentation engine, `models/yolo11n-seg.engine`
- Grasp planner: GraspNet baseline checkpoint, `sdk/graspnet-baseline/checkpoints/checkpoint-rs.tar`
- Calibration: Eye-in-hand hand-eye calibration saved as `config/calibration/orbbec_gemini2/hand_eye.npz`

The normal safe workflow is:

1. Verify the camera.
2. Verify the arm connection.
3. Verify or redo hand-eye calibration.
4. Verify the GraspNet Python/CUDA stack.
5. Run camera-only demos.
6. Run robot dry-run.
7. Run real grasp motion.

## Hardware Layout

- Mount the Orbbec Gemini 2 rigidly on the robot end effector.
- Keep the ArUco calibration marker fixed on the table during calibration.
- Keep the object workspace clear of the arm before any robot motion command.
- Connect the Gemini 2 by USB 3.0.
- Connect the reBot arm through the USB2CAN adapter.

The default calibration marker settings are in `config/default.yaml`:

```yaml
calibration:
  aruco:
    marker_length_m: 0.1
    dict_id: 0
    target_marker_id: 0
  hand_eye_method: TSAI
```

This means the printed marker must be ArUco dictionary `DICT_4X4_50`, marker ID
`0`, with a measured black-square edge length of `100 mm`. The repo includes
`aruco100x100.pdf`.

## Environment

Activate the existing environment from the repo root:

```bash
conda activate graspnet
cd /home/seeed/Downloads/rebot_grasp
```

On Jetson, make sure CUDA is visible before building GraspNet CUDA extensions:

```bash
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```

Install the Jetson PyTorch wheel that matches the JetPack/L4T version. Do not
install generic PyPI `torch` on Jetson for this project. After PyTorch is
installed, install the remaining GraspNet runtime packages:

```bash
pip install -r requirements-graspnet-jetson.txt
```

Build the GraspNet CUDA operators:

```bash
cd /home/seeed/Downloads/rebot_grasp/sdk/graspnet-baseline/pointnet2
MAX_JOBS=1 python setup.py install

cd /home/seeed/Downloads/rebot_grasp/sdk/graspnet-baseline/knn
FORCE_CUDA=1 MAX_JOBS=1 python setup.py install

cd /home/seeed/Downloads/rebot_grasp
```

## Configuration

Main settings live in `config/default.yaml`.

Important current values:

```yaml
camera:
  type: orbbec_gemini2
  color_width: 1280
  color_height: 720
  depth_width: 1280
  depth_height: 720
  fps: 30

yolo:
  model_name: "yolo11n-seg.engine"
  device: "auto"
  use_world: false
  custom_classes: []

graspnet:
  checkpoint: "checkpoint-rs.tar"
```

Use `device: "auto"` for TensorRT engine inference. Open-vocabulary class lists
are only useful for `.pt` world/YOLOE models, not for `.engine` files.

## Verification

Run these checks before attempting a real grasp.

### 1. Verify Orbbec RGB-D stream

Text-only check:

```bash
python scripts/verify_pyorbbec_stream.py
```

Preview windows:

```bash
python scripts/verify_pyorbbec_stream.py --preview --seconds 10
```

No preview window is shown unless `--preview` is passed.

### 2. Verify arm connection

Read-only state check:

```bash
python scripts/verify_rebot_arm_motion.py --read-only
```

Small joint-6 jog and return:

```bash
python scripts/verify_rebot_arm_motion.py --deg 5
```

Only run the jog test with a clear robot workspace.

### 3. Verify hand-eye calibration file

```bash
python scripts/verify_handeye_calibration.py
```

Expected result:

```text
[OK] hand-eye calibration looks usable
```

This checks that `hand_eye.npz` exists, is `eye_in_hand`, has at least 5 samples,
and contains a valid rotation matrix.

### 4. Verify GraspNet stack

```bash
python scripts/verify_graspnet_stack.py
```

Expected result:

```text
[OK] GraspNet stack is ready
```

If this reports missing `torch`, install a Jetson-matched PyTorch wheel. If it
reports missing `pointnet2._ext` or `knn_pytorch`, rebuild the CUDA operators in
`sdk/graspnet-baseline/pointnet2` and `sdk/graspnet-baseline/knn`.

## Hand-Eye Calibration

Redo calibration whenever the camera mount changes, the end-effector geometry
changes, the marker size changes, or grasp poses are consistently shifted.

### Automatic mode

Use this first. The arm traverses preset viewpoints and records a sample when
the marker is detected stably.

```bash
conda activate graspnet
cd /home/seeed/Downloads/rebot_grasp
python scripts/collect_handeye_eih.py
```

Minimum samples: `5`. Recommended samples: `15+`. The automatic sequence has 50
candidate poses. Press `c` or `q` to stop and compute from collected samples.

The result is saved to:

```text
config/calibration/orbbec_gemini2/hand_eye.npz
```

Verify it:

```bash
python scripts/verify_handeye_calibration.py
```

### Manual gravity-compensation mode

Use this if automatic poses do not see the marker reliably.

```bash
python scripts/collect_handeye_eih.py --manual
```

Controls:

- `Enter`: capture one sample at the current pose.
- `pos`: print current end-effector pose.
- `c` or `q`: finish and compute calibration.

In manual mode, push the arm to different viewpoints around the fixed marker.
Use varied roll, pitch, yaw, and distance. Avoid collecting many samples from
nearly identical poses.

## Running Demos

### YOLO detection only

```bash
python scripts/object_detection.py
```

This verifies the `yolo11n-seg.engine` detector and camera display path.

### Ordinary OBB grasp perception

```bash
python scripts/ordinary_grasp_pipeline.py
```

This uses YOLO masks and depth to estimate a simple grasp point. It does not
use GraspNet.

### GraspNet camera demo

Start with full-scene GraspNet, without YOLO target filtering:

```bash
python scripts/graspnet_camera_demo.py --camera-type orbbec_gemini2 --no-yolo --debug-frames
```

Then test target-aware mode with the TensorRT YOLO engine:

```bash
python scripts/graspnet_camera_demo.py --camera-type orbbec_gemini2 --yolo-model yolo11n-seg.engine
```

Useful options:

```bash
--no-visualizer       Run without the Open3D window.
--auto               Run inference periodically.
--infer-interval 2   Set automatic inference interval in seconds.
--target-class cup   Prefer a specific detected class.
--no-yolo            Run full-scene GraspNet.
```

Keyboard controls in the visual demo:

- `g` or `Space`: run GraspNet on the current RGB-D frame.
- `q` or `Esc`: quit.

### GraspNet web UI

`scripts/grasp_web.py` combines the local web frontend and backend in one file.
It streams the camera as MJPEG, lets you select a YOLO target class, and
automatically overlays the current best GraspNet grasp point on the RGB frame.

Preview only, no robot connection:

```bash
python scripts/grasp_web.py --host 0.0.0.0 --port 8000
```

Full-scene GraspNet without YOLO target filtering:

```bash
python scripts/grasp_web.py --host 0.0.0.0 --port 8000 --no-yolo
```

Allow real grasp execution from the web button:

```bash
python scripts/grasp_web.py --host 0.0.0.0 --port 8000 --enable-robot
```

Open the browser at:

```text
http://localhost:8000
```

Automatic updates only refresh the GraspNet grasp point. The robot only moves
when `--enable-robot` is used and the `真实抓取` button is clicked.

## Robot Grasp Execution

Always dry-run first.

Full-scene GraspNet dry-run:

```bash
python scripts/grasp.py --dry-run --camera-type orbbec_gemini2 --no-yolo
```

Target-aware dry-run:

```bash
python scripts/grasp.py --dry-run --camera-type orbbec_gemini2 --target-class cup
```

If the printed target pose and preview look correct, remove `--dry-run`:

```bash
python scripts/grasp.py --camera-type orbbec_gemini2 --target-class cup
```

The script initializes the arm and gripper, moves to the configured ready pose,
captures an RGB-D frame, estimates a grasp, transforms it through hand-eye
calibration, and executes pre-grasp, descend, close, lift, and return motions.

## Troubleshooting

### `pyorbbecsdk import failed`

Rebuild/install the local Orbbec SDK package:

```bash
cd /home/seeed/Downloads/rebot_grasp/sdk/pyorbbecsdk
cmake --install build
pip install --force-reinstall dist/pyorbbecsdk2-*.whl
```

Install udev rules:

```bash
cd /home/seeed/Downloads/rebot_grasp/sdk/pyorbbecsdk/scripts/env_setup
sudo ./install_udev_rules.sh
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### No camera window

`verify_pyorbbec_stream.py` is text-only by default. Use:

```bash
python scripts/verify_pyorbbec_stream.py --preview
```

For headless GraspNet tests, use `--no-visualizer`.

### `nvbufsurftransform: Could not get EGL display connection`

This warning can appear on Jetson when a display/EGL session is not available.
It is not fatal for text-only checks. For GUI preview, run from a local desktop
session or use a proper X11/Wayland forwarding setup.

### `torch.cuda.is_available()` is false

The installed PyTorch is wrong for Jetson or CUDA is not visible. Install the
NVIDIA Jetson PyTorch wheel matching the board's JetPack/L4T release, then
check:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### `No module named pointnet2._ext` or `knn_pytorch`

Rebuild the GraspNet CUDA ops:

```bash
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"

cd /home/seeed/Downloads/rebot_grasp/sdk/graspnet-baseline/pointnet2
MAX_JOBS=1 python setup.py install

cd /home/seeed/Downloads/rebot_grasp/sdk/graspnet-baseline/knn
FORCE_CUDA=1 MAX_JOBS=1 python setup.py install
```

### ArUco marker is not detected

Check:

- The marker is ID `0` from `DICT_4X4_50`.
- The measured marker edge length matches `marker_length_m`.
- The marker is flat, well lit, and fully visible.
- The camera intrinsics file exists at `config/calibration/orbbec_gemini2/intrinsics.npz`.

## Main Files

```text
config/default.yaml                         Main runtime configuration
models/yolo11n-seg.engine                   TensorRT YOLO segmentation engine
config/calibration/orbbec_gemini2/          Camera intrinsics and hand-eye files
scripts/verify_pyorbbec_stream.py           Orbbec RGB-D stream check
scripts/verify_rebot_arm_motion.py          Arm read/jog check
scripts/verify_handeye_calibration.py       Calibration file sanity check
scripts/verify_graspnet_stack.py            GraspNet dependency check
scripts/collect_handeye_eih.py              Eye-in-hand calibration collection
scripts/graspnet_camera_demo.py             Camera-only GraspNet demo
scripts/grasp_web.py                        Web UI for GraspNet point preview/execution
scripts/grasp.py                            Robot GraspNet execution
```

## References

- Seeed wiki: https://wiki.seeedstudio.com/rebot_arm_b601_dm_grasping_demo/
- reBot arm SDK: https://github.com/vectorBH6/reBotArm_control_py
- Orbbec SDK v2: https://github.com/orbbec/OrbbecSDK_v2
- pyorbbecsdk: https://github.com/orbbec/pyorbbecsdk
- GraspNet baseline: https://github.com/graspnet/graspnet-baseline
- GraspNet API: https://github.com/graspnet/graspnetAPI
- NVIDIA PyTorch for Jetson: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html
