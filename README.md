# 🦾 reBot Arm B601-DM Visual Grasping Demo (Jetson Edition)

<p align="center">
  <img src="https://media-cdn.seeedstudio.com/media/catalog/product/cache/bb49d3ec4ee05b6f018e93f896b8a25d/1/1/110110147.jpg" alt="reComputer" width="450">
  <img src="https://raw.githubusercontent.com/Seeed-Projects/reBot-DevArm/main/media/v1.0.png" alt="reBot Arm B601" width="450">
</p>

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python 3.10+">
    <img src="https://img.shields.io/badge/Platform-Jetson%20(ARM64)-orange.svg" alt="Platform: Jetson">
    <img src="https://img.shields.io/badge/Camera-Orbbec%20Gemini%202-green.svg" alt="Camera: Orbbec Gemini 2">
    <img src="https://img.shields.io/badge/Detection-GraspNet%2BYOLO-yellow.svg" alt="Detection: GraspNet + YOLO">
</p>

<p align="center">
  <strong>RGB-D Perception · Object Detection · Hand-Eye Calibration · GraspNet 6-DoF Pose · Robot Control</strong>
</p>

<p align="center">
  <a href="#readme-en"><strong>English Guide</strong></a>
  &nbsp;|&nbsp;
  <a href="./README_zh.md"><strong>中文文档</strong></a>
</p>

---

<!-- ═══════════════════════════════════════════════════════════ -->
<!-- ENGLISH SECTION                                             -->
<!-- ═══════════════════════════════════════════════════════════ -->
<div id="readme-en"></div>

## Table of Contents

> **Jump to:** [1. Overview](#1-overview--en) · [2. Hardware Requirements](#2-hardware-requirements--en) · [3. System Architecture](#3-system-architecture--en) · [4. Environment Setup](#4-environment-setup--en) · [5. Four-Step Environment Verification](#5-four-step-environment-verification--en) · [6. Hand-Eye Calibration](#6-hand-eye-calibration--en) · [7. YOLO / TensorRT Model Export](#7-yolo--tensorrt-model-export--en) · [8. Running the Web Demo](#8-running-the-web-demo--en) · [9. Web UI Guide](#9-web-ui-guide--en) · [10. Script Reference](#10-script-reference--en) · [11. Configuration Reference](#11-configuration-reference--en) · [12. CLI Grasping](#12-cli-grasping--en) · [13. Troubleshooting](#13-troubleshooting--en) · [14. References](#14-references--en)

---

<div id="1-overview--en"></div>

## 1. Overview <span style="font-size:0.6em">[<a href="./README_zh.md#1-overview--zh">中文</a>]</span>

This project implements a complete visual grasping pipeline for the **reBot Arm B601-DM** on **NVIDIA Jetson**, combining multi-modal perception, deep-learning-based grasp pose estimation, and real robot control.

### Pipeline Flow

```
Orbbec Gemini 2 RGB-D Camera
        ↓
  YOLO Instance Segmentation (target filtering)
        ↓
  GraspNet 6-DoF Grasp Pose Estimation
        ↓
  Hand-Eye Calibration (Eye-in-Hand, TSAI algorithm)
        ↓
  Coordinate Transform: Camera Frame → Robot Base Frame
        ↓
  Arm IK Trajectory + Gripper Force Control
        ↓
  Base Rotation + Object Placement
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Dual grasp estimation** | GraspNet (6-DoF, pretrained) + ordinary grasp (depth-based, high-frequency) |
| **Eye-in-Hand calibration** | TSAI algorithm, fully automatic collection |
| **YOLO target filtering** | Segment anything in the scene, or pick by class name |
| **Web UI** | Real-time MJPEG stream, target selection, grasp preview, extrinsic tuning |
| **Base motor jog** | Direct joint1 control via web panel |
| **Multi-camera support** | Orbbec Gemini 2, RealSense D435i, RealSense D405 |
| **Extrinsic compensation** | Per-frame XYZ + RPY offsets for gripper / camera / base |

### Default Configuration

| Item | Default |
|------|---------|
| Conda environment | `graspnet` |
| Camera | Orbbec Gemini 2 (`pyorbbecsdk`) |
| Object detection | YOLO `yolo11n-seg.engine` |
| Grasp estimation | GraspNet `checkpoint-rs.tar` |
| Hand-eye calibration | `config/calibration/orbbec_gemini2/hand_eye.npz` |
| Main entry point | `scripts/grasp_web.py` |

---

<div id="2-hardware-requirements--en"></div>

## 2. Hardware Requirements <span style="font-size:0.6em">[<a href="./README_zh.md#2-hardware-requirements--zh">中文</a>]</span>

| Component | Model / Notes |
|-----------|---------------|
| Robot arm | reBot Arm B601-DM (DAMIAO motor version) |
| Depth camera | Orbbec Gemini 2 |
| Connectivity | USB2CAN adapter (robot CAN bus); USB 3.0 (camera) |
| Host | Jetson (Ubuntu 22.04, Python 3.10, ARM64) |
| ArUco calibration board | 4x4, ID=0, 0.1 m edge length (for hand-eye calibration) |

### Wiring

```bash
# 1. Connect Gemini 2 to Jetson via USB 3.0
# 2. Connect USB2CAN adapter to robot CAN bus and insert into Jetson USB port
# 3. Set device permissions (required on first use)
sudo chmod a+rw /dev/bus/usb/*/*
sudo chmod 666 /dev/ttyUSB0   # Adjust port number as needed
```

---

<div id="3-system-architecture--en"></div>

## 3. System Architecture <span style="font-size:0.6em">[<a href="./README_zh.md#3-system-architecture--zh">中文</a>]</span>

```
┌──────────────────────────────────────────────────────────────┐
│                 grasp_web.py (Web UI)                        │
│  Live MJPEG Stream · Target Class Selector · Grasp Button    │
│              Base Motor Jog Panel (joint1)                   │
└───────────────────────┬──────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
┌──────────────────┐        ┌─────────────────────────┐
│  YOLO Seg.       │        │  GraspNet 6-DoF        │
│  Target Filter   │        │  Grasp Pose Estimation │
│  (TensorRT)      │        │  + Ordinary Grasp      │
└────────┬─────────┘        └────────────┬────────────┘
         │                               │
         └───────────────┬───────────────┘
                         ▼
              ┌──────────────────┐
              │  Hand-Eye Transform │ ← hand_eye.npz (calibration result)
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │  Arm IK Trajectory │
              │  + Gripper FSM     │
              └──────────────────┘
```

---

<div id="4-environment-setup--en"></div>

## 4. Environment Setup <span style="font-size:0.6em">[<a href="./README_zh.md#4-environment-setup--zh">中文</a>]</span>

> **Recommended order**: Follow each step in sequence. Do not skip the verification steps in Section 5.

### 4.1 Clone the Repository

```bash
git clone https://github.com/Seeed-Projects/reBot-DevArm-Grasp.git rebot_grasp
cd rebot_grasp
```

### 4.2 Create a conda Environment

```bash
conda create -n graspnet python=3.10 -y
conda activate graspnet
```

### 4.3 Install NVIDIA PyTorch for Jetson

> **Critical:** On Jetson, **do not** install the generic PyPI `torch`. You must install a JetPack-matched wheel from NVIDIA.

```bash
# Query JetPack version
dpkg -l | grep jetpack
# or
cat /etc/nv_tegra_release
```

Reference: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html

```bash
# Example (JetPack 6.0 + CUDA 12.6, Python 3.10 aarch64)
pip install torch-2.6.0 torchvision-0.21.0 -f https://developer.download.nvidia.com/compute/pytorchicrob/links.html
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 4.4 Install Non-PyTorch Dependencies

```bash
pip install -r requirements-graspnet-jetson.txt
```

### 4.5 Set CUDA Environment Variables

```bash
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```

> Add to `~/.bashrc` to avoid repeating this every session:
> ```bash
> echo 'export CUDA_HOME=/usr/local/cuda-12.6' >> ~/.bashrc
> echo 'export PATH="$CUDA_HOME/bin:$PATH"' >> ~/.bashrc
> echo 'export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
> source ~/.bashrc
> ```

### 4.6 Prepare SDK Submodules

#### 4.6.1 reBotArm SDK

```bash
git clone https://github.com/vectorBH6/reBotArm_control_py.git sdk/reBotArm_control_py
cd sdk/reBotArm_control_py
pip install -e .
cd ../..
```

#### 4.6.2 GraspNet Baseline

```bash
git clone https://github.com/graspnet/graspnet-baseline.git sdk/graspnet-baseline
```

Download the GraspNet pretrained checkpoint (required, ~1.6 GB):

```bash
mkdir -p sdk/graspnet-baseline/checkpoints
# Download checkpoint-rs.tar from https://graspnet.net/download
# Place it in sdk/graspnet-baseline/checkpoints/
```

#### 4.6.3 GraspNet API

```bash
git clone https://github.com/graspnet/graspnetAPI.git sdk/graspnetAPI
cd sdk/graspnetAPI
pip install -e .
cd ../..
```

#### 4.6.4 Build GraspNet CUDA Operators (Critical!)

> The `No module named pointnet2._ext` error is almost always caused by skipping this step.

```bash
cd sdk/graspnet-baseline/pointnet2
MAX_JOBS=1 python setup.py install

cd /home/seeed/Downloads/rebot_grasp/sdk/graspnet-baseline/knn
FORCE_CUDA=1 MAX_JOBS=1 python setup.py install

cd /home/seeed/Downloads/rebot_grasp
```

### 4.7 Install Orbbec Python SDK

```bash
# Install build dependencies
sudo apt-get install -y cmake build-essential libusb-1.0-0-dev

# Get pyorbbecsdk
cd sdk
git clone https://github.com/orbbec/pyorbbecsdk.git
cd pyorbbecsdk
pip install -e .

# Build from source if pre-built version is incompatible:
# bash ../../scripts/install_pyorbbecsdk.sh --from-source
```

### 4.8 Configure Orbbec udev Rules (Required)

```bash
cd /home/seeed/Downloads/rebot_grasp/sdk/pyorbbecsdk/scripts/env_setup
sudo ./install_udev_rules.sh
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### 4.9 Download YOLO Model Weights

```bash
# Create models directory
mkdir -p models

# Download YOLO11n-seg model
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt -O models/yolo11n-seg.pt
```

### 4.10 Export YOLO TensorRT Engine (Must Run on Target Jetson)

> TensorRT `.engine` files are tightly coupled to the Jetson device, GPU, CUDA, and TensorRT versions. **Do not** copy engines from other machines — export on the target Jetson:

```bash
yolo export model=models/yolo11n-seg.pt format=engine imgsz=640 half=True device=0 workspace=4
```

Output: `models/yolo11n-seg.engine`

If FP16 export fails, fall back to FP32:

```bash
yolo export model=models/yolo11n-seg.pt format=engine imgsz=640 device=0 workspace=4
```

---

<div id="5-four-step-environment-verification--en"></div>

## 5. Four-Step Environment Verification <span style="font-size:0.6em">[<a href="./README_zh.md#5-four-step-environment-verification--zh">中文</a>]</span>

> **Run after every deployment.** Complete all four steps in order before proceeding to calibration or demo.

### Step 1: Orbbec RGB-D Stream

```bash
conda activate graspnet
cd /home/seeed/Downloads/rebot_grasp

# Text-only check
python scripts/verify_pyorbbec_stream.py

# With preview window
python scripts/verify_pyorbbec_stream.py --preview --seconds 10
```

**Expected:** RGB and Depth resolution + frame rate reported, no errors.

### Step 2: Robot Arm Connection

```bash
# Read-only check (safe — does not move the arm)
python scripts/verify_rebot_arm_motion.py --read-only

# Small-angle jog test (make sure the arm path is clear!)
python scripts/verify_rebot_arm_motion.py --deg 5
```

**Expected:** `[OK]` or normal pose data returned.

### Step 3: Hand-Eye Calibration File

```bash
python scripts/verify_handeye_calibration.py
```

**Expected:**

```
[OK] hand-eye calibration looks usable
```

> If you get `No such file`, hand-eye calibration has not been done yet. Proceed to [Section 6](#6-hand-eye-calibration--en).

### Step 4: GraspNet Stack

```bash
python scripts/verify_graspnet_stack.py
```

**Expected:**

```
[OK] GraspNet stack is ready
```

> If you see `No module named pointnet2._ext`, re-run [Section 4.6.4](#464-build-graspnet-cuda-operators-critical).

---

<div id="6-hand-eye-calibration--en"></div>

## 6. Hand-Eye Calibration <span style="font-size:0.6em">[<a href="./README_zh.md#6-hand-eye-calibration--zh">中文</a>]</span>

This system uses **Eye-in-Hand** calibration: the camera is mounted on the robot end-effector and moves with it. The ArUco marker is fixed on the table.

### 6.1 When to Recalibrate

Recalibrate when any of the following occurs:

- Camera mount position changes (bracket moved or re-attached)
- Gripper geometry changes (different gripper or position moved)
- ArUco marker size changed
- Table layout significantly changed
- Grasp accuracy noticeably degraded

### 6.2 Calibration Board Setup

Use an **ArUco DICT_4X4_50, ID=0** board with edge length **0.1 m** (must match `aruco.marker_length_m` in `config/default.yaml`).

Print the board and affix it flat on the table, or use a physical calibration target. The marker must be clearly visible from the arm's viewing angles.

### 6.3 Automatic Collection (Recommended)

```bash
python scripts/collect_handeye_eih.py
```

The arm automatically traverses 50 predefined poses. When the ArUco marker is stably detected, it automatically samples. A minimum of **5 samples** is required; **15+** is recommended for stable results.

| Key | Action |
|-----|--------|
| `c` or `q` | End collection and compute calibration |
| `Ctrl+C` | Also triggers computation on collected data mid-collection |

### 6.4 Manual Collection Mode

```bash
python scripts/collect_handeye_eih.py --manual
```

The arm enters gravity-compensation mode. Manually push it to various viewing angles:

| Key | Action |
|-----|--------|
| `Enter` | Sample current pose |
| `pos` | Print current end-effector pose |
| `c` / `q` | Finish and compute |

### 6.5 Calibration Output

Results are auto-saved to:

```
config/calibration/orbbec_gemini2/hand_eye.npz
config/calibration/orbbec_gemini2/intrinsics.npz
```

### 6.6 Verify Calibration

```bash
python scripts/verify_handeye_calibration.py
```

---

<div id="7-yolo--tensorrt-model-export--en"></div>

## 7. YOLO / TensorRT Model Export <span style="font-size:0.6em">[<a href="./README_zh.md#7-yolo--tensorrt-model-export--zh">中文</a>]</span>

### 7.1 Export ONNX from .pt (Intermediate Format)

```bash
yolo export model=models/yolo11n-seg.pt format=onnx imgsz=640 opset=12 simplify=True
```

### 7.2 Export TensorRT Engine Directly from .pt (Recommended)

```bash
yolo export model=models/yolo11n-seg.pt format=engine imgsz=640 half=True device=0 workspace=4
```

### 7.3 Convert ONNX to TensorRT Engine

```bash
trtexec \
  --onnx=models/yolo11n-seg.onnx \
  --saveEngine=models/yolo11n-seg.engine \
  --fp16 \
  --workspace=4096
```

### 7.4 Verify the Exported Engine

```bash
python scripts/verify_graspnet_stack.py --engine models/yolo11n-seg.engine
python scripts/object_detection.py
```

### 7.5 Switching Models

Edit `config/default.yaml`:

```yaml
yolo:
  model_name: "yolo11n-seg.engine"  # Replace with your engine filename
  device: "auto"
```

Or override from the command line:

```bash
python scripts/grasp_web.py --yolo-model yolo11n-seg.engine
```

> **Note:** `.engine` files are device-specific and cannot be shared across machines. If JetPack / TensorRT / CUDA is updated, re-export the engine on the target Jetson.

---

<div id="8-running-the-web-demo--en"></div>

## 8. Running the Web Demo <span style="font-size:0.6em">[<a href="./README_zh.md#8-running-the-web-demo--zh">中文</a>]</span>

### 8.1 Preview Mode (No Robot Connection)

```bash
conda activate graspnet
cd /home/seeed/Downloads/rebot_grasp
python scripts/grasp_web.py --host 0.0.0.0 --port 8000
```

Open in browser:

```
http://<jetson_ip>:8000
```

In this mode you can:
- View live RGB-D video stream
- See YOLO detection results overlaid
- Preview GraspNet grasp points
- Tune extrinsic compensation parameters (gripper, camera, base)
- Jog the base motor

### 8.2 Enable Real Robot Execution

```bash
python scripts/grasp_web.py --host 0.0.0.0 --port 8000 --enable-robot
```

Click **Real Grasp** in the web interface to trigger the full grasp sequence.

### 8.3 Disable Post-Grasp Actions for Debugging

Base rotation and placement after each grasp interfere with continuous grasp debugging. Disable them:

```bash
python scripts/grasp_web.py --host 0.0.0.0 --port 8000 --enable-robot --no-place-after-grasp
```

### 8.4 Full-Scene GraspNet (No YOLO Filtering)

```bash
python scripts/grasp_web.py --host 0.0.0.0 --port 8000 --no-yolo
```

### 8.5 Common Command-Line Options

| Flag | Description |
|------|-------------|
| `--host 0.0.0.0 --port 8000` | Bind to all interfaces |
| `--enable-robot` | Allow real arm movement |
| `--no-yolo` | Skip YOLO detection, show full-scene GraspNet |
| `--camera-type orbbec_gemini2` | Force camera type |
| `--target-class cup` | Auto-select this class |
| `--no-place-after-grasp` | Skip base rotation and placement |
| `--no-auto-graspnet` | Disable automatic GraspNet updates |
| `--graspnet-interval 2.0` | GraspNet update interval (seconds) |

---

<div id="9-web-ui-guide--en"></div>

## 9. Web UI Guide <span style="font-size:0.6em">[<a href="./README_zh.md#9-web-ui-guide--zh">中文</a>]</span>

### UI Layout

```
┌─────────────────────────────────────────────────────────────┐
│ [reBot Grasp Web] [EN/中文] Target:[▼] [Set] [Refresh] [Real Grasp] │
│ mode hint...                                                │
├─────────────────────────────────┬───────────────────────────┤
│                                 │  ┌─ Compensation ───────┐ │
│                                 │  │ Gripper F/L/U (m)    │ │
│     Live MJPEG Camera Stream    │  │ [___] [___] [___]    │ │
│     + YOLO boxes                │  │ Gripper RPY (deg)    │ │
│     + GraspNet grasp point      │  │ [___] [___] [___]    │ │
│                                 │  │ Camera XYZ (m)       │ │
│                                 │  │ [___] [___] [___]    │ │
│                                 │  │ Camera RPY (deg)     │ │
│                                 │  │ [___] [___] [___]    │ │
│                                 │  │ Base XYZ (m)         │ │
│                                 │  │ [___] [___] [___]    │ │
│                                 │  │ Base RPY (deg)       │ │
│                                 │  │ [___] [___] [___]    │ │
│                                 │  │ [Set Compensation]   │ │
│                                 │  └──────────────────────┘ │
│                                 │  ┌─ Base Motor ────────┐ │
│                                 │  │ joint1 jog: [___]   │ │
│                                 │  │ Duration: [___]     │ │
│                                 │  │ Margin:  [___]       │ │
│                                 │  │ [-30°] [Apply] [+30°]│ │
│                                 │  └──────────────────────┘ │
│                                 │  status line...            │
│                                 │  {state JSON}              │
└─────────────────────────────────┴───────────────────────────┘
```

### Extrinsic Compensation Fields

The page has three categories of compensation fields:

| Category | Fields | Effect |
|----------|--------|--------|
| **Gripper Fwd/Lat/Up** | forward, lateral, vertical (meters) | Corrects final grasp TCP pose |
| **Gripper RPY** | roll, pitch, yaw (degrees) | Local TCP rotation offset |
| **Camera XYZ/RPY** | 6-DOF (meters / degrees) | Corrects camera extrinsics |
| **Base XYZ/RPY** | 6-DOF (meters / degrees) | Corrects base-frame extrinsics (**does not rotate the base motor**) |

### Base Motor Jog Panel

The base motor debug panel **directly jogs `joint1`** and returns JSON with:

```json
{
  "before_deg": 12.5,
  "after_deg": -17.5,
  "limit_deg": -30.0,
  "safe_limit_deg": -25.0
}
```

Use this panel to:
- Test base motor response
- Find safe joint limits before running grasp sequences
- Tune `base_delta_deg` and `base_safety_margin_deg` in `config/default.yaml`

---

<div id="10-script-reference--en"></div>

## 10. Script Reference <span style="font-size:0.6em">[<a href="./README_zh.md#10-script-reference--zh">中文</a>]</span>

| Script | Purpose |
|--------|---------|
| `verify_pyorbbec_stream.py` | Orbbec RGB-D stream check (text / preview) |
| `verify_rebot_arm_motion.py` | Robot connection and jog check |
| `verify_handeye_calibration.py` | Hand-eye calibration file integrity check |
| `verify_graspnet_stack.py` | GraspNet / CUDA / YOLO engine sanity check |
| `collect_handeye_eih.py` | Eye-in-Hand hand-eye calibration (auto / manual modes) |
| `graspnet_camera_demo.py` | Camera-side GraspNet preview with Open3D window |
| `object_detection.py` | Pure YOLO detection demo |
| `grasp_web.py` | **Main entry:** Web UI, target selection, real grasp, base jog |
| `grasp.py` | CLI real grasp pipeline (non-Web, OpenCV window) |
| `install_pyorbbecsdk.sh` | pyorbbecsdk installation helper script |

---

<div id="11-configuration-reference--en"></div>

## 11. Configuration Reference <span style="font-size:0.6em">[<a href="./README_zh.md#11-configuration-reference--zh">中文</a>]</span>

Main config file: `config/default.yaml`

### 11.1 Camera Configuration

```yaml
camera:
  type: orbbec_gemini2
  color_width: 1280
  color_height: 720
  depth_width: 1280
  depth_height: 720
  fps: 30
```

### 11.2 YOLO Configuration

```yaml
yolo:
  model_name: "yolo11n-seg.engine"  # Replace with your engine filename
  device: "auto"                    # Auto-select GPU
  use_world: false                   # Set true for open-vocabulary detection (requires .pt model)
  custom_classes: []                # Open-vocabulary custom classes
```

### 11.3 Grasp Offset Configuration

```yaml
grasp_pipeline:
  grasp:
    pregrasp_offset_m: 0.080         # Pre-grasp height offset (meters)
    grasp_forward_offset_m: 0.000    # Move grasp point forward along approach axis
    camera_x_offset_m: 0.000         # Camera extrinsic compensation X
    camera_y_offset_m: 0.000         # Camera extrinsic compensation Y
    camera_z_offset_m: 0.000         # Camera extrinsic compensation Z
    camera_roll_offset_deg: 0.0      # Camera extrinsic compensation roll
    camera_pitch_offset_deg: 0.0     # Camera extrinsic compensation pitch
    camera_yaw_offset_deg: 0.0       # Camera extrinsic compensation yaw
    base_x_offset_m: 0.000           # Base extrinsic compensation X
    base_yaw_offset_deg: 0.0         # Base extrinsic compensation yaw
```

### 11.4 Placement Configuration

```yaml
grasp_pipeline:
  place:
    enabled: true
    base_joint: joint1
    base_delta_deg: 90.0             # Base rotation angle (negative = negative direction)
    base_direction: auto              # auto / positive / negative
    base_rotate_duration: 2.5        # Rotation duration (seconds)
    base_safety_margin_deg: 5.0      # Safety margin from joint limits
    return_home: true                # Whether to return to home after placing
```

### 11.5 ArUco Calibration Configuration

```yaml
calibration:
  aruco:
    marker_length_m: 0.1
    dict_id: 0
    target_marker_id: 0
  hand_eye_method: TSAI
```

---

<div id="12-cli-grasping--en"></div>

## 12. CLI Grasping <span style="font-size:0.6em">[<a href="./README_zh.md#12-cli-grasping--zh">中文</a>]</span>

Suitable for headless or automated invocation.

### 12.1 Dry-Run (No Arm Movement)

```bash
# Full-scene grasping
python scripts/grasp.py --dry-run --camera-type orbbec_gemini2 --no-yolo

# With target class
python scripts/grasp.py --dry-run --camera-type orbbec_gemini2 --target-class cup
```

### 12.2 Real Execution

```bash
python scripts/grasp.py --camera-type orbbec_gemini2 --target-class cup
```

### 12.3 Disable Post-Grasp Placement

```bash
python scripts/grasp.py --camera-type orbbec_gemini2 --target-class cup --no-place-after-grasp
```

### 12.4 Custom Base Rotation

```bash
python scripts/grasp.py \
  --camera-type orbbec_gemini2 \
  --target-class cup \
  --place-base-delta-deg -30 \
  --place-base-rotate-duration 2.5
```

---

<div id="13-troubleshooting--en"></div>

## 13. Troubleshooting <span style="font-size:0.6em">[<a href="./README_zh.md#13-troubleshooting--zh">中文</a>]</span>

### Q1: `pyorbbecsdk import failed`

```bash
bash scripts/install_pyorbbecsdk.sh
# If still failing, build from source:
bash scripts/install_pyorbbecsdk.sh --from-source
```

### Q2: Camera frame is all black

```bash
# Check udev rules
ls -la /dev/bus/usb/*/*
# Reload udev
sudo udevadm control --reload-rules && sudo udevadm trigger
# Check USB 3.0 connection
```

### Q3: `nvbufsurftransform: Could not get EGL display connection`

This is normal on Jetson without a desktop display. Text-only checks (`--preview` omitted) are unaffected.

### Q4: `torch.cuda.is_available()` returns False

```bash
# Check PyTorch version
python -c "import torch; print(torch.__version__)"
# Reinstall the wheel matching your JetPack version
```

### Q5: `No module named pointnet2._ext`

Rebuild the CUDA operators:

```bash
cd sdk/graspnet-baseline/pointnet2
MAX_JOBS=1 python setup.py install

cd sdk/graspnet-baseline/knn
FORCE_CUDA=1 MAX_JOBS=1 python setup.py install
```

### Q6: YOLO engine fails to load

Common causes:

- Engine was copied from another machine
- JetPack / TensorRT / CUDA versions changed
- Ultralytics version changed

Solution: Re-export on the target Jetson:

```bash
yolo export model=models/yolo11n-seg.pt format=engine imgsz=640 half=True device=0 workspace=4
python scripts/verify_graspnet_stack.py --engine models/yolo11n-seg.engine
```

### Q7: ArUco marker not detected

- Verify the marker is `DICT_4X4_50` with ID `0`
- Verify `marker_length_m` matches the actual black square edge length
- Marker must be flat, complete, and under stable lighting
- Verify `config/calibration/orbbec_gemini2/intrinsics.npz` exists

### Q8: Base jog does not change the angle

First check: joint1 motor enable state, encoder feedback, limit settings, and low-level control mode.

### Q9: Grasp offset tuning

If the gripper consistently misses the object:

| Symptom | Tune |
|---------|------|
| Gripper approaches from wrong depth | `grasp_forward_offset_m` |
| Gripper misses left/right | `grasp_lateral_offset_m` or `camera_x_offset_m` |
| Gripper misses height | `grasp_vertical_offset_m` or `camera_z_offset_m` |
| Rotation is off | `grasp_roll_offset_deg` / `grasp_pitch_offset_deg` / `grasp_yaw_offset_deg` |
| Camera mount is slightly off | `camera_x/y/z_offset_m` and `camera_r/p/y_offset_deg` |

Use the web UI sliders for live tuning, then copy values to `config/default.yaml` for persistence.

---

<div id="14-references--en"></div>

## 14. References <span style="font-size:0.6em">[<a href="./README_zh.md#14-references--zh">中文</a>]</span>

| Resource | Link |
|----------|------|
| Seeed wiki | https://wiki.seeedstudio.com/rebot_arm_b601_dm_grasping_demo/ |
| reBot arm SDK | https://github.com/vectorBH6/reBotArm_control_py |
| Orbbec SDK v2 | https://github.com/orbbec/OrbbecSDK_v2 |
| pyorbbecsdk | https://github.com/orbbec/pyorbbecsdk |
| GraspNet baseline | https://github.com/graspnet/graspnet-baseline |
| GraspNet API | https://github.com/graspnet/graspnetAPI |
| NVIDIA PyTorch for Jetson | https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html |

---

<!-- ═══════════════════════════════════════════════════════════ -->
<!-- END OF ENGLISH SECTION                                     -->
<!-- ═══════════════════════════════════════════════════════════ -->

<p align="center">
  <strong>If this project is helpful to you, feel free to star it!</strong>
</p>
