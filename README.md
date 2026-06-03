# reBot DevArm Grasp

这个仓库用于在 Jetson 上运行 reBot Arm B601-DM 视觉抓取流程。当前版本以
Orbbec Gemini 2 RGB-D 相机、YOLO 分割模型和 GraspNet baseline 为主，提供
相机检查、手眼标定、GraspNet 预览、Web 选目标、真实抓取、放置和底座 joint1
调试入口。

当前默认配置：

- Conda 环境：`graspnet`
- 相机：Orbbec Gemini 2，`pyorbbecsdk`
- YOLO：`models/yolo11n-seg.engine`
- GraspNet checkpoint：`sdk/graspnet-baseline/checkpoints/checkpoint-rs.tar`
- 手眼标定：`config/calibration/orbbec_gemini2/hand_eye.npz`
- Web 入口：`scripts/grasp_web.py`

## 快速启动

```bash
conda activate graspnet
cd /home/seeed/Downloads/rebot_grasp
```

先做四个检查：

```bash
python scripts/verify_pyorbbec_stream.py
python scripts/verify_rebot_arm_motion.py --read-only
python scripts/verify_handeye_calibration.py
python scripts/verify_graspnet_stack.py
```

只开 Web 预览，不连接机械臂：

```bash
python scripts/grasp_web.py --host 0.0.0.0 --port 8000
```

允许 Web 点击真实抓取：

```bash
python scripts/grasp_web.py --host 0.0.0.0 --port 8000 --enable-robot
```

打开：

```text
http://localhost:8000
```

调试连续抓取时，可以先禁用抓取后的底座旋转和放置流程：

```bash
python scripts/grasp_web.py --host 0.0.0.0 --port 8000 --enable-robot --no-place-after-grasp
```

## 项目结构

```text
config/default.yaml                         主配置
config/calibration/orbbec_gemini2/          Orbbec 内参和手眼标定结果
drivers/camera/orbbec_gemini2.py            Orbbec Gemini 2 相机驱动
drivers/robot/rebot_arm.py                  reBot Arm 封装、夹爪控制、joint1 底座旋转
utils/yolo_runtime.py                       YOLO `.pt` / `.onnx` / `.engine` 路径和推理参数
scripts/install_pyorbbecsdk.sh              pyorbbecsdk 安装辅助脚本
scripts/verify_pyorbbec_stream.py           Orbbec RGB-D 流检查
scripts/verify_rebot_arm_motion.py          机械臂连接和小角度 jog 检查
scripts/verify_handeye_calibration.py       手眼标定文件检查
scripts/verify_graspnet_stack.py            GraspNet、CUDA、YOLO engine 检查
scripts/collect_handeye_eih.py              eye-in-hand 手眼标定采集
scripts/graspnet_camera_demo.py             相机侧 GraspNet 预览
scripts/grasp_web.py                        Web 预览、目标选择、真实抓取、底座 jog
scripts/grasp.py                            CLI 真实抓取流程
```

`sdk/`、`models/` 和 `.venv/` 默认不提交到 Git。`sdk/` 需要在本机准备好
GraspNet baseline、GraspNet API、pyorbbecsdk 和 reBotArm SDK。

## 环境安装

Jetson 上不要安装普通 PyPI 版 `torch`。先安装与 JetPack/L4T 匹配的 NVIDIA
PyTorch wheel，再装本项目的非 PyTorch 依赖：

```bash
pip install -r requirements-graspnet-jetson.txt
```

设置 CUDA 环境：

```bash
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```

构建 GraspNet CUDA 算子：

```bash
cd /home/seeed/Downloads/rebot_grasp/sdk/graspnet-baseline/pointnet2
MAX_JOBS=1 python setup.py install

cd /home/seeed/Downloads/rebot_grasp/sdk/graspnet-baseline/knn
FORCE_CUDA=1 MAX_JOBS=1 python setup.py install

cd /home/seeed/Downloads/rebot_grasp
```

安装或重装 Orbbec Python SDK：

```bash
bash scripts/install_pyorbbecsdk.sh
```

如果需要从源码构建：

```bash
bash scripts/install_pyorbbecsdk.sh --from-source
```

安装 udev rules：

```bash
cd /home/seeed/Downloads/rebot_grasp/sdk/pyorbbecsdk/scripts/env_setup
sudo ./install_udev_rules.sh
sudo udevadm control --reload-rules
sudo udevadm trigger
```

## 配置

主要配置在 `config/default.yaml`。

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

grasp_pipeline:
  grasp:
    pregrasp_offset_m: 0.080
    grasp_forward_offset_m: 0.000
    camera_x_offset_m: 0.000
    base_yaw_offset_deg: 0.0
  place:
    enabled: true
    base_joint: joint1
    base_delta_deg: 90.0
    base_direction: auto
    base_rotate_duration: 2.5
    base_safety_margin_deg: 5.0
    return_home: true
```

注意：

- Web 页面里的“基座XYZ/RPY”是外参补偿，不是底座电机控制。
- 底座电机是 `joint1`，Web 里的“底座电机调试”会直接调用 joint1 相对 jog。
- `base_delta_deg: -30` 会按 30 度幅度加 `negative` 方向处理。
- `base_direction: auto` 会在正负方向都安全时选离限位更远的一侧。

## YOLO 模型和 TensorRT Engine

默认使用：

```text
models/yolo11n-seg.engine
```

`.engine` 文件和 Jetson 设备、GPU、CUDA、TensorRT、Ultralytics 版本强相关，不能稳定跨机器复用。如果本地 engine 不可用、报 TensorRT 反序列化失败、类别名异常，或者 JetPack/TensorRT 更新过，建议在目标 Jetson 上重新导出。

### 从 `.pt` 导出 ONNX

准备一个 PyTorch 模型，例如：

```text
models/yolo11n-seg.pt
```

导出 ONNX：

```bash
yolo export model=models/yolo11n-seg.pt format=onnx imgsz=640 opset=12 simplify=True
```

常见输出：

```text
models/yolo11n-seg.onnx
```

也可以用 Python：

```bash
python - <<'PY'
from ultralytics import YOLO
model = YOLO("models/yolo11n-seg.pt")
model.export(format="onnx", imgsz=640, opset=12, simplify=True)
PY
```

ONNX 适合作为中间格式，也方便在导出 TensorRT 前检查模型是否正常。

### 从 `.pt` 直接导出 TensorRT engine

推荐在最终运行的 Jetson 上导出：

```bash
yolo export model=models/yolo11n-seg.pt format=engine imgsz=640 half=True device=0 workspace=4
```

常见输出：

```text
models/yolo11n-seg.engine
```

如果 `half=True` 报错，可以先用 FP32：

```bash
yolo export model=models/yolo11n-seg.pt format=engine imgsz=640 device=0 workspace=4
```

### 从 ONNX 转 TensorRT engine

如果已经有 ONNX，也可以用 TensorRT 的 `trtexec`：

```bash
trtexec \
  --onnx=models/yolo11n-seg.onnx \
  --saveEngine=models/yolo11n-seg.engine \
  --fp16 \
  --workspace=4096
```

如果 FP16 不稳定，去掉 `--fp16`：

```bash
trtexec \
  --onnx=models/yolo11n-seg.onnx \
  --saveEngine=models/yolo11n-seg.engine \
  --workspace=4096
```

导出后检查：

```bash
python scripts/verify_graspnet_stack.py --engine models/yolo11n-seg.engine
python scripts/object_detection.py
```

### 切换模型

修改 `config/default.yaml`：

```yaml
yolo:
  model_name: "yolo11n-seg.engine"
  device: "auto"
```

也可以临时在命令行指定：

```bash
python scripts/grasp_web.py --yolo-model yolo11n-seg.engine
python scripts/graspnet_camera_demo.py --yolo-model yolo11n-seg.engine
```

`.engine` 推理时不要设置 `device: cpu`。如果要用 open-vocabulary / YOLOE，
请使用对应 `.pt` 模型，并设置 `use_world: true` 和 `custom_classes`；TensorRT
engine 不适合动态改 open-vocabulary 类别列表。

## 验证流程

### Orbbec RGB-D 流

文本检查：

```bash
python scripts/verify_pyorbbec_stream.py
```

带窗口预览：

```bash
python scripts/verify_pyorbbec_stream.py --preview --seconds 10
```

### 机械臂连接

只读检查：

```bash
python scripts/verify_rebot_arm_motion.py --read-only
```

小角度 joint6 jog：

```bash
python scripts/verify_rebot_arm_motion.py --deg 5
```

确认机械臂周围没有障碍物后再执行 jog。

### 手眼标定文件

```bash
python scripts/verify_handeye_calibration.py
```

期望输出：

```text
[OK] hand-eye calibration looks usable
```

### GraspNet 运行栈

```bash
python scripts/verify_graspnet_stack.py
```

期望输出：

```text
[OK] GraspNet stack is ready
```

如果缺 `pointnet2._ext` 或 `knn_pytorch`，重建 GraspNet CUDA 算子。

## 手眼标定

相机安装位置、夹爪几何、ArUco 尺寸或工作台布局变化后，都需要重新标定。

自动采样：

```bash
python scripts/collect_handeye_eih.py
```

最少样本数是 5，建议 15 个以上。自动路径会尝试多个视角，稳定看到 marker
时采样。按 `c` 或 `q` 结束并计算。

手动模式：

```bash
python scripts/collect_handeye_eih.py --manual
```

手动模式控制：

- `Enter`：采样当前位姿
- `pos`：打印当前末端位姿
- `c` 或 `q`：结束并计算

输出文件：

```text
config/calibration/orbbec_gemini2/hand_eye.npz
```

标定后运行：

```bash
python scripts/verify_handeye_calibration.py
```

## GraspNet 相机 Demo

全场景 GraspNet，不用 YOLO 目标过滤：

```bash
python scripts/graspnet_camera_demo.py --camera-type orbbec_gemini2 --no-yolo --debug-frames
```

YOLO 目标过滤：

```bash
python scripts/graspnet_camera_demo.py --camera-type orbbec_gemini2 --yolo-model yolo11n-seg.engine
```

常用参数：

```bash
--no-visualizer       不打开 Open3D 抓取窗口
--auto                周期性运行 GraspNet
--infer-interval 2    自动推理间隔，单位秒
--target-class cup    指定目标类别
--no-yolo             全场景 GraspNet
```

键盘：

- `g` 或 `Space`：对当前 RGB-D 帧运行 GraspNet
- `q` 或 `Esc`：退出

## Web UI

预览模式：

```bash
python scripts/grasp_web.py --host 0.0.0.0 --port 8000
```

禁用 YOLO，使用全场景 GraspNet：

```bash
python scripts/grasp_web.py --host 0.0.0.0 --port 8000 --no-yolo
```

启用真实抓取：

```bash
python scripts/grasp_web.py --host 0.0.0.0 --port 8000 --enable-robot
```

临时禁用抓取后的底座旋转/放置：

```bash
python scripts/grasp_web.py --host 0.0.0.0 --port 8000 --enable-robot --no-place-after-grasp
```

Web 自动更新只刷新 GraspNet 抓取点。机械臂只会在使用 `--enable-robot`
启动后，点击 `真实抓取` 时移动。

页面上的补偿分三类：

- `夹爪前/左/上` 和 `夹爪RPY`：修正最终抓取 TCP。
- `相机XYZ/RPY`：修正相机外参。
- `基座XYZ/RPY`：修正基座系外参，不会转底座电机。

底座电机调试面板会直接 jog `joint1`。返回 JSON 里会显示：

```text
before_deg
after_deg
limit_deg
safe_limit_deg
```

如果 `before_deg` 和 `after_deg` 没有变化，优先检查 joint1 电机使能、反馈、
限位和底层控制模式。

## CLI 真实抓取

先 dry-run：

```bash
python scripts/grasp.py --dry-run --camera-type orbbec_gemini2 --no-yolo
```

指定目标类别 dry-run：

```bash
python scripts/grasp.py --dry-run --camera-type orbbec_gemini2 --target-class cup
```

真实执行：

```bash
python scripts/grasp.py --camera-type orbbec_gemini2 --target-class cup
```

跳过抓取后的底座旋转/放置：

```bash
python scripts/grasp.py --camera-type orbbec_gemini2 --target-class cup --no-place-after-grasp
```

设置抓取后底座旋转：

```bash
python scripts/grasp.py \
  --camera-type orbbec_gemini2 \
  --target-class cup \
  --place-base-delta-deg -30 \
  --place-base-rotate-duration 2.5
```

`--place-base-delta-deg -30` 表示 joint1 负方向 30 度。

## 常见问题

### `pyorbbecsdk import failed`

```bash
bash scripts/install_pyorbbecsdk.sh
```

仍然失败时，从源码安装：

```bash
bash scripts/install_pyorbbecsdk.sh --from-source
```

### 没有相机窗口

`verify_pyorbbec_stream.py` 默认只做文本检查。需要窗口时加：

```bash
python scripts/verify_pyorbbec_stream.py --preview
```

Headless 环境下运行 GraspNet demo，使用 `--no-visualizer`。

### `nvbufsurftransform: Could not get EGL display connection`

这是 Jetson 没有可用 EGL/桌面显示时的常见提示。文本检查通常不受影响。
需要图形窗口时，使用本地桌面会话或正确配置 X11/Wayland。

### `torch.cuda.is_available()` 是 false

PyTorch 版本不匹配 Jetson，或 CUDA 环境不可见。检查：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### `No module named pointnet2._ext`

重新构建：

```bash
cd /home/seeed/Downloads/rebot_grasp/sdk/graspnet-baseline/pointnet2
MAX_JOBS=1 python setup.py install

cd /home/seeed/Downloads/rebot_grasp/sdk/graspnet-baseline/knn
FORCE_CUDA=1 MAX_JOBS=1 python setup.py install
```

### YOLO engine 不可用

典型表现：

- TensorRT 反序列化失败
- engine 加载后类别名变成泛型
- JetPack/TensorRT/CUDA 更新后推理失败
- 从另一台机器复制过来的 engine 不能运行

处理方式：

```bash
yolo export model=models/yolo11n-seg.pt format=engine imgsz=640 half=True device=0 workspace=4
python scripts/verify_graspnet_stack.py --engine models/yolo11n-seg.engine
```

如果没有 `.pt`，需要先取得对应的 PyTorch 模型权重。不要指望旧 `.engine`
在新环境里稳定可用。

### ArUco 检测不到

检查：

- marker 是 `DICT_4X4_50`，ID 为 `0`
- `marker_length_m` 和实际黑色方块边长一致
- marker 平整、完整、光照稳定
- `config/calibration/orbbec_gemini2/intrinsics.npz` 存在

## 参考

- Seeed wiki: https://wiki.seeedstudio.com/rebot_arm_b601_dm_grasping_demo/
- reBot arm SDK: https://github.com/vectorBH6/reBotArm_control_py
- Orbbec SDK v2: https://github.com/orbbec/OrbbecSDK_v2
- pyorbbecsdk: https://github.com/orbbec/pyorbbecsdk
- GraspNet baseline: https://github.com/graspnet/graspnet-baseline
- GraspNet API: https://github.com/graspnet/graspnetAPI
- NVIDIA PyTorch for Jetson: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html

---

# English Guide

This repository runs a reBot Arm B601-DM visual grasping stack on Jetson. The
current setup uses an Orbbec Gemini 2 RGB-D camera, YOLO segmentation, GraspNet
baseline, a local Web UI, real robot execution, post-grasp placement, and a
joint1 base-motor jog tool.

Default runtime:

- Conda environment: `graspnet`
- Camera: Orbbec Gemini 2 through `pyorbbecsdk`
- YOLO model: `models/yolo11n-seg.engine`
- GraspNet checkpoint: `sdk/graspnet-baseline/checkpoints/checkpoint-rs.tar`
- Hand-eye calibration: `config/calibration/orbbec_gemini2/hand_eye.npz`
- Web entry point: `scripts/grasp_web.py`

## Quick Start

```bash
conda activate graspnet
cd /home/seeed/Downloads/rebot_grasp
```

Run the basic checks first:

```bash
python scripts/verify_pyorbbec_stream.py
python scripts/verify_rebot_arm_motion.py --read-only
python scripts/verify_handeye_calibration.py
python scripts/verify_graspnet_stack.py
```

Start the Web UI in preview mode:

```bash
python scripts/grasp_web.py --host 0.0.0.0 --port 8000
```

Start the Web UI with real robot execution enabled:

```bash
python scripts/grasp_web.py --host 0.0.0.0 --port 8000 --enable-robot
```

Open:

```text
http://localhost:8000
```

For repeated grasp debugging, temporarily disable post-grasp base rotation and
placement:

```bash
python scripts/grasp_web.py --host 0.0.0.0 --port 8000 --enable-robot --no-place-after-grasp
```

## Key Files

```text
config/default.yaml                         Main runtime config
config/calibration/orbbec_gemini2/          Orbbec intrinsics and hand-eye data
drivers/camera/orbbec_gemini2.py            Orbbec Gemini 2 camera driver
drivers/robot/rebot_arm.py                  Robot wrapper, gripper, base joint1 control
utils/yolo_runtime.py                       YOLO `.pt` / `.onnx` / `.engine` helpers
scripts/install_pyorbbecsdk.sh              pyorbbecsdk install helper
scripts/verify_pyorbbec_stream.py           Orbbec RGB-D stream check
scripts/verify_rebot_arm_motion.py          Robot read-only and jog check
scripts/verify_handeye_calibration.py       Hand-eye calibration sanity check
scripts/verify_graspnet_stack.py            GraspNet/CUDA/YOLO engine check
scripts/collect_handeye_eih.py              Eye-in-hand calibration collection
scripts/graspnet_camera_demo.py             Camera-side GraspNet demo
scripts/grasp_web.py                        Web preview, target selection, real grasp, base jog
scripts/grasp.py                            CLI real grasp pipeline
```

The `sdk/`, `models/`, and `.venv/` directories are local runtime assets and are
not meant to be committed.

## Environment Setup

On Jetson, install the NVIDIA PyTorch wheel that matches the board's
JetPack/L4T version. Do not use generic PyPI `torch` for this project.

Install non-PyTorch dependencies:

```bash
pip install -r requirements-graspnet-jetson.txt
```

Set CUDA paths:

```bash
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```

Build the GraspNet CUDA operators:

```bash
cd /home/seeed/Downloads/rebot_grasp/sdk/graspnet-baseline/pointnet2
MAX_JOBS=1 python setup.py install

cd /home/seeed/Downloads/rebot_grasp/sdk/graspnet-baseline/knn
FORCE_CUDA=1 MAX_JOBS=1 python setup.py install

cd /home/seeed/Downloads/rebot_grasp
```

Install or reinstall the Orbbec Python SDK:

```bash
bash scripts/install_pyorbbecsdk.sh
```

Build from source if needed:

```bash
bash scripts/install_pyorbbecsdk.sh --from-source
```

Install Orbbec udev rules:

```bash
cd /home/seeed/Downloads/rebot_grasp/sdk/pyorbbecsdk/scripts/env_setup
sudo ./install_udev_rules.sh
sudo udevadm control --reload-rules
sudo udevadm trigger
```

## Configuration

Main settings live in `config/default.yaml`.

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

grasp_pipeline:
  place:
    enabled: true
    base_joint: joint1
    base_delta_deg: 90.0
    base_direction: auto
    base_rotate_duration: 2.5
    base_safety_margin_deg: 5.0
    return_home: true
```

Important notes:

- The Web UI `base XYZ/RPY` fields are extrinsic compensation fields. They do
  not move the base motor.
- The base motor is `joint1`.
- The Web UI base-motor debug panel directly jogs `joint1`.
- `base_delta_deg: -30` is interpreted as 30 degrees in the negative direction.
- `base_direction: auto` chooses the safer direction based on current angle and
  joint limits.

## YOLO ONNX and TensorRT Engine Export

The default model is:

```text
models/yolo11n-seg.engine
```

TensorRT `.engine` files are tied to the target Jetson, GPU, CUDA, TensorRT, and
Ultralytics versions. They are not portable across machines. If the local engine
does not load, deserialization fails, class names look wrong, or JetPack/TensorRT
was updated, rebuild the engine on the target Jetson.

### Export ONNX from `.pt`

Start with a PyTorch model:

```text
models/yolo11n-seg.pt
```

Export ONNX:

```bash
yolo export model=models/yolo11n-seg.pt format=onnx imgsz=640 opset=12 simplify=True
```

Expected output:

```text
models/yolo11n-seg.onnx
```

Python alternative:

```bash
python - <<'PY'
from ultralytics import YOLO
model = YOLO("models/yolo11n-seg.pt")
model.export(format="onnx", imgsz=640, opset=12, simplify=True)
PY
```

### Export TensorRT engine directly from `.pt`

Run this on the Jetson that will execute the model:

```bash
yolo export model=models/yolo11n-seg.pt format=engine imgsz=640 half=True device=0 workspace=4
```

Expected output:

```text
models/yolo11n-seg.engine
```

If FP16 export fails, use FP32:

```bash
yolo export model=models/yolo11n-seg.pt format=engine imgsz=640 device=0 workspace=4
```

### Convert ONNX to TensorRT engine

```bash
trtexec \
  --onnx=models/yolo11n-seg.onnx \
  --saveEngine=models/yolo11n-seg.engine \
  --fp16 \
  --workspace=4096
```

If FP16 is unstable:

```bash
trtexec \
  --onnx=models/yolo11n-seg.onnx \
  --saveEngine=models/yolo11n-seg.engine \
  --workspace=4096
```

Verify the exported engine:

```bash
python scripts/verify_graspnet_stack.py --engine models/yolo11n-seg.engine
python scripts/object_detection.py
```

Switch the model in `config/default.yaml`:

```yaml
yolo:
  model_name: "yolo11n-seg.engine"
  device: "auto"
```

Or override it from the command line:

```bash
python scripts/grasp_web.py --yolo-model yolo11n-seg.engine
python scripts/graspnet_camera_demo.py --yolo-model yolo11n-seg.engine
```

Do not set `device: cpu` for `.engine` inference. Use `.pt` models for
open-vocabulary / YOLOE workflows.

## Verification

Camera:

```bash
python scripts/verify_pyorbbec_stream.py
python scripts/verify_pyorbbec_stream.py --preview --seconds 10
```

Robot:

```bash
python scripts/verify_rebot_arm_motion.py --read-only
python scripts/verify_rebot_arm_motion.py --deg 5
```

Hand-eye calibration:

```bash
python scripts/verify_handeye_calibration.py
```

Expected:

```text
[OK] hand-eye calibration looks usable
```

GraspNet stack:

```bash
python scripts/verify_graspnet_stack.py
```

Expected:

```text
[OK] GraspNet stack is ready
```

## Hand-Eye Calibration

Redo calibration whenever the camera mount, gripper geometry, marker size, or
table layout changes.

Automatic mode:

```bash
python scripts/collect_handeye_eih.py
```

Manual mode:

```bash
python scripts/collect_handeye_eih.py --manual
```

Manual controls:

- `Enter`: capture the current pose
- `pos`: print the current end-effector pose
- `c` or `q`: finish and solve

Output:

```text
config/calibration/orbbec_gemini2/hand_eye.npz
```

Verify after calibration:

```bash
python scripts/verify_handeye_calibration.py
```

## GraspNet Camera Demo

Full-scene GraspNet:

```bash
python scripts/graspnet_camera_demo.py --camera-type orbbec_gemini2 --no-yolo --debug-frames
```

YOLO target filtering:

```bash
python scripts/graspnet_camera_demo.py --camera-type orbbec_gemini2 --yolo-model yolo11n-seg.engine
```

Common options:

```bash
--no-visualizer       Disable the Open3D grasp window
--auto                Run GraspNet periodically
--infer-interval 2    Automatic inference interval in seconds
--target-class cup    Prefer a target class
--no-yolo             Run full-scene GraspNet
```

Keyboard:

- `g` or `Space`: run GraspNet on the current RGB-D frame
- `q` or `Esc`: quit

## Web UI

Preview only:

```bash
python scripts/grasp_web.py --host 0.0.0.0 --port 8000
```

Full-scene GraspNet without YOLO:

```bash
python scripts/grasp_web.py --host 0.0.0.0 --port 8000 --no-yolo
```

Enable real robot execution:

```bash
python scripts/grasp_web.py --host 0.0.0.0 --port 8000 --enable-robot
```

Disable post-grasp placement while debugging:

```bash
python scripts/grasp_web.py --host 0.0.0.0 --port 8000 --enable-robot --no-place-after-grasp
```

Automatic updates only refresh the displayed GraspNet grasp point. The arm moves
only after the process is started with `--enable-robot` and the `真实抓取` button
is clicked.

The Web compensation fields are:

- Gripper offsets: correct the final TCP grasp pose.
- Camera offsets: correct camera extrinsics.
- Base offsets: correct base-frame extrinsics; these do not rotate the base
  motor.

The base motor debug panel jogs `joint1` and reports `before_deg`, `after_deg`,
`limit_deg`, and `safe_limit_deg`.

## CLI Real Grasp

Dry-run first:

```bash
python scripts/grasp.py --dry-run --camera-type orbbec_gemini2 --no-yolo
python scripts/grasp.py --dry-run --camera-type orbbec_gemini2 --target-class cup
```

Real execution:

```bash
python scripts/grasp.py --camera-type orbbec_gemini2 --target-class cup
```

Disable post-grasp placement:

```bash
python scripts/grasp.py --camera-type orbbec_gemini2 --target-class cup --no-place-after-grasp
```

Configure post-grasp base rotation:

```bash
python scripts/grasp.py \
  --camera-type orbbec_gemini2 \
  --target-class cup \
  --place-base-delta-deg -30 \
  --place-base-rotate-duration 2.5
```

`--place-base-delta-deg -30` means rotate joint1 30 degrees in the negative
direction.

## Troubleshooting

### `pyorbbecsdk import failed`

```bash
bash scripts/install_pyorbbecsdk.sh
```

Or build from source:

```bash
bash scripts/install_pyorbbecsdk.sh --from-source
```

### No camera window

`verify_pyorbbec_stream.py` is text-only by default. Use:

```bash
python scripts/verify_pyorbbec_stream.py --preview
```

For headless GraspNet runs, use `--no-visualizer`.

### `torch.cuda.is_available()` is false

Install the NVIDIA Jetson PyTorch wheel matching the board's JetPack/L4T
release, then check:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### `No module named pointnet2._ext`

Rebuild the CUDA ops:

```bash
cd /home/seeed/Downloads/rebot_grasp/sdk/graspnet-baseline/pointnet2
MAX_JOBS=1 python setup.py install

cd /home/seeed/Downloads/rebot_grasp/sdk/graspnet-baseline/knn
FORCE_CUDA=1 MAX_JOBS=1 python setup.py install
```

### YOLO engine does not work

Typical causes:

- TensorRT deserialization failure
- Wrong/generic class names after loading the engine
- JetPack/TensorRT/CUDA changed
- The engine was copied from a different machine

Rebuild on the target Jetson:

```bash
yolo export model=models/yolo11n-seg.pt format=engine imgsz=640 half=True device=0 workspace=4
python scripts/verify_graspnet_stack.py --engine models/yolo11n-seg.engine
```

If no `.pt` model is available, obtain the matching PyTorch weights first. Old
`.engine` files should not be treated as portable artifacts.
