"""
grasp.py - 基于 GraspNet 的机械臂视觉夹取主程序
================================================
流程：
  1. 初始化机械臂、夹爪和 RGB-D 相机，移动到预备位
  2. YOLO 选择目标，GraspNet 在当前 RGB-D 帧上估计 6D 夹取姿态
  3. G/SPACE 键：冻结当前帧并执行夹取（--dry-run 只打印坐标）
  4. R 键：恢复实时预览
  5. Q/ESC 键：退出，释放夹爪并回零位

用法：
  cd /home/chlorine/seeed/cameraws
  conda activate seeed
  python scripts/grasp.py --dry-run
  python scripts/grasp.py --target-class cup
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEED_ROOT = PROJECT_ROOT.parent
GRASPNET_ROOT = PROJECT_ROOT / "sdk" / "graspnet-baseline"


def _prepare_imports() -> None:
    for path in (SEEED_ROOT, PROJECT_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    for subdir in ("models", "dataset", "utils", "pointnet2", "graspnetAPI"):
        sys.path.insert(0, str(GRASPNET_ROOT / subdir))
    sys.path.insert(0, str(GRASPNET_ROOT))


_prepare_imports()

from cameraws.drivers.camera import make_camera  # noqa: E402
from cameraws.drivers.robot.rebot_arm import RebotArm  # noqa: E402
from cameraws.scripts.graspnet_camera_demo import (  # noqa: E402
    DirectRealSenseCamera,
    build_end_points,
    build_net,
    detect_targets,
    draw_target_overlay,
    infer_grasps,
    load_yolo,
    overlay_status,
    target_status_text,
)
from cameraws.utils.transforms import (  # noqa: E402
    canonicalize_parallel_gripper_tcp_rotation,
    mat4_to_pose6d,
    rotation_matrix_to_euler_zyx,
)
from graspnetAPI import Grasp, GraspGroup  # noqa: E402


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_hand_eye(project_root: Path, cam_type: str) -> tuple[Optional[np.ndarray], Optional[str]]:
    hand_eye_path = project_root / "config" / "calibration" / cam_type / "hand_eye.npz"
    if not hand_eye_path.exists():
        return None, None

    data = np.load(str(hand_eye_path), allow_pickle=False)
    T = data["T_result"].astype(np.float64)
    mode = str(data["mode"][0])
    return T, mode


def configure_camera(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cam_cfg = cfg.setdefault("camera", {})
    if args.camera_type is not None:
        cam_cfg["type"] = args.camera_type
    cam_type = str(cam_cfg.get("type", "")).lower()
    if not cam_type:
        raise ValueError("camera.type is missing in config; pass --camera-type or set it in YAML")

    if args.width is not None:
        cam_cfg["color_width"] = args.width
        cam_cfg["depth_width"] = args.width
    if args.height is not None:
        cam_cfg["color_height"] = args.height
        cam_cfg["depth_height"] = args.height
    if args.fps is not None:
        cam_cfg["fps"] = args.fps
    elif args.camera_type is not None and "realsense" in cam_type:
        cam_cfg["fps"] = 15
    return cfg


def _move_ready(robot: RebotArm, ready_cfg: dict[str, Any]) -> None:
    duration = float(ready_cfg.get("duration", 3.0))
    robot.move_to(
        float(ready_cfg.get("x", 0.25)),
        float(ready_cfg.get("y", 0.0)),
        float(ready_cfg.get("z", 0.35)),
        float(ready_cfg.get("roll", 0.0)),
        float(ready_cfg.get("pitch", 1.2)),
        float(ready_cfg.get("yaw", 0.0)),
        duration=duration,
    )
    robot.wait_motion(duration)


def _cam_to_base(T_hand_eye: np.ndarray, robot: RebotArm) -> np.ndarray:
    return robot.get_tcp_pose() @ T_hand_eye


def _graspnet_to_rebot_tcp_rotation(grasp_rotation: np.ndarray) -> np.ndarray:
    R = np.asarray(grasp_rotation, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"grasp_rotation must be (3, 3), got {R.shape}")

    tcp_x = R[:, 0]
    tcp_y = R[:, 1] - float(np.dot(R[:, 1], tcp_x)) * tcp_x
    tcp_x = tcp_x / max(np.linalg.norm(tcp_x), 1e-8)
    tcp_y = tcp_y / max(np.linalg.norm(tcp_y), 1e-8)
    tcp_z = np.cross(tcp_x, tcp_y)
    tcp_z = tcp_z / max(np.linalg.norm(tcp_z), 1e-8)

    tcp_y = np.cross(tcp_z, tcp_x)
    tcp_y = tcp_y / max(np.linalg.norm(tcp_y), 1e-8)
    R_tcp = np.column_stack([tcp_x, tcp_y, tcp_z]).astype(np.float64)
    if np.linalg.det(R_tcp) < 0.0:
        R_tcp[:, 2] *= -1.0
    return R_tcp


def _transform_grasp(
    grasp: Grasp,
    T_cam2base: np.ndarray,
    pregrasp_offset_m: float,
    retreat_offset_m: float,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    T_grasp_cam = np.eye(4, dtype=np.float64)
    T_grasp_cam[:3, :3] = _graspnet_to_rebot_tcp_rotation(grasp.rotation_matrix)
    T_grasp_cam[:3, 3] = np.asarray(grasp.translation, dtype=np.float64)

    T_grasp_base = T_cam2base @ T_grasp_cam
    grasp_pos_base = T_grasp_base[:3, 3].copy()
    grasp_rot_base = canonicalize_parallel_gripper_tcp_rotation(T_grasp_base[:3, :3])
    T_grasp_base[:3, :3] = grasp_rot_base

    T_pregrasp_base = T_grasp_base.copy()
    T_pregrasp_base[:3, 3] = grasp_pos_base - grasp_rot_base[:, 0] * float(pregrasp_offset_m)

    T_retreat_base = T_grasp_base.copy()
    T_retreat_base[:3, 3] = grasp_pos_base - grasp_rot_base[:, 0] * float(retreat_offset_m)

    return mat4_to_pose6d(T_grasp_base), mat4_to_pose6d(T_pregrasp_base), mat4_to_pose6d(T_retreat_base)


def _execute_grasp(
    robot: RebotArm,
    grasp6d: tuple[float, ...],
    pre6d: tuple[float, ...],
    retreat6d: tuple[float, ...],
    ready_cfg: dict[str, Any],
    dry_run: bool,
    gripper_width_m: float,
) -> bool:
    xg, yg, zg, rxg, ryg, rzg = grasp6d
    xp, yp, zp, rxp, ryp, rzp = pre6d
    xr, yr, zr, rxr, ryr, rzr = retreat6d

    print(f"[Grasp] pregrasp xyz=({xp:+.3f},{yp:+.3f},{zp:+.3f}) rpy=({rxp:+.3f},{ryp:+.3f},{rzp:+.3f})")
    print(f"[Grasp] grasp    xyz=({xg:+.3f},{yg:+.3f},{zg:+.3f}) rpy=({rxg:+.3f},{ryg:+.3f},{rzg:+.3f})")
    print(f"[Grasp] retreat  xyz=({xr:+.3f},{yr:+.3f},{zr:+.3f}) rpy=({rxr:+.3f},{ryr:+.3f},{rzr:+.3f})")

    if dry_run:
        print("[Grasp] --dry-run: 跳过机械臂执行")
        return False

    print(f"[Grasp] 打开夹爪 width={gripper_width_m:.3f}m...")
    robot.open_gripper(distance_m=gripper_width_m)

    print("[Grasp] 移动到预夹取位...")
    if not robot.move_to(xp, yp, zp, rxp, ryp, rzp, duration=2.0):
        print("[Grasp] 预夹取 IK 失败，中止")
        return False
    robot.wait_motion(2.0)

    print("[Grasp] 移动到夹取位...")
    if not robot.move_to(xg, yg, zg, rxg, ryg, rzg, duration=1.5):
        print("[Grasp] 夹取 IK 失败，中止")
        return False
    robot.wait_motion(1.5)

    print("[Grasp] 夹取中...")
    ok = robot.grasp()
    print("[Grasp] 夹取成功，力控保持中" if ok else "[Grasp] 空夹取")

    print("[Grasp] 退回预夹取位...")
    if robot.move_to(xr, yr, zr, rxr, ryr, rzr, duration=1.5):
        robot.wait_motion(1.5)

    print("[Grasp] 返回预备位...")
    _move_ready(robot, ready_cfg)
    return ok


def _select_best_grasp(gg: GraspGroup) -> Optional[Grasp]:
    if len(gg) == 0:
        return None
    ranked = GraspGroup(gg.grasp_group_array.copy())
    try:
        ranked = ranked.nms()
    except Exception as exc:
        print(f"[WARN] GraspNet NMS 不可用，直接按 score 排序: {exc}")
    ranked.sort_by_score()
    return ranked[0] if len(ranked) > 0 else None


def _print_grasp(grasp: Grasp) -> None:
    tcp_rotation = canonicalize_parallel_gripper_tcp_rotation(_graspnet_to_rebot_tcp_rotation(grasp.rotation_matrix))
    print("\n[G] GraspNet 最佳夹取:")
    print(f"  score={grasp.score:.4f} width={grasp.width:.4f} height={grasp.height:.4f} depth={grasp.depth:.4f}")
    print(f"  position_xyz={grasp.translation.tolist()}")
    print(f"  graspnet_rpy={rotation_matrix_to_euler_zyx(grasp.rotation_matrix).tolist()}")
    print(f"  tcp_rpy={rotation_matrix_to_euler_zyx(tcp_rotation).tolist()}")


def _draw_best_grasp_projection(display: np.ndarray, grasp: Optional[Grasp], K: np.ndarray) -> None:
    if grasp is None:
        return
    x, y, z = [float(v) for v in grasp.translation]
    if z <= 1e-6:
        return
    u = int(round(float(K[0, 0]) * x / z + float(K[0, 2])))
    v = int(round(float(K[1, 1]) * y / z + float(K[1, 2])))
    if 0 <= u < display.shape[1] and 0 <= v < display.shape[0]:
        cv2.drawMarker(display, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
        label = f"best score={grasp.score:.2f} width={grasp.width * 100:.1f}cm"
        cv2.putText(display, label, (u + 10, max(24, v - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)


def _infer_current_frame(
    net: Any,
    yolo_model: Optional[Any],
    yolo_opts: dict[str, Any],
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
    args: argparse.Namespace,
) -> tuple[GraspGroup, Optional[Grasp], str, str, list[Any], Optional[Any]]:
    target_mask = None
    selected_target = None
    targets = []
    target_label = "full scene"

    if yolo_model is not None:
        _, targets, selected_target = detect_targets(yolo_model, color_bgr, yolo_opts, args.target_class)
        if selected_target is None:
            target_status = target_status_text(selected_target, targets, args.target_class)
            return GraspGroup(), None, f"inference skipped: {target_status}", target_status, targets, selected_target
        target_mask = selected_target.mask
        target_label = f"{selected_target.class_name} {selected_target.conf:.2f}"

    tic = time.time()
    end_points, _, raw_cloud = build_end_points(
        color_bgr,
        depth_mm,
        K,
        args.num_point,
        args.min_depth,
        args.max_depth,
    )
    gg, decoded_count, target_count = infer_grasps(
        net,
        end_points,
        raw_cloud,
        args.collision_thresh,
        args.voxel_size,
        target_mask=target_mask,
        K=K,
        target_margin_px=args.target_margin_px,
    )
    best = _select_best_grasp(gg)
    elapsed = time.time() - tic

    if yolo_model is None:
        status = f"grasps={len(gg)} decoded={decoded_count} inference={elapsed:.2f}s"
        target_status = "YOLO disabled: full-scene GraspNet"
    else:
        status = f"{target_label} grasps={len(gg)} target={target_count}/{decoded_count} inference={elapsed:.2f}s"
        target_status = target_status_text(selected_target, targets, args.target_class)
    return gg, best, status, target_status, targets, selected_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于 GraspNet 的机械臂夹取主程序")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "default.yaml"))
    parser.add_argument("--checkpoint", default=str(GRASPNET_ROOT / "checkpoints" / "checkpoint-rs.tar"))
    parser.add_argument("--dry-run", action="store_true", help="只估计姿态，不移动机械臂")
    parser.add_argument("--camera-type", choices=("realsense_d435i", "realsense_d405", "orbbec_gemini2"), default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--num-point", type=int, default=20000)
    parser.add_argument("--num-view", type=int, default=300)
    parser.add_argument("--collision-thresh", type=float, default=0.01)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--min-depth", type=float, default=0.05, help="meters")
    parser.add_argument("--max-depth", type=float, default=2.0, help="meters")
    parser.add_argument("--target-class", default=None)
    parser.add_argument("--target-margin-px", type=int, default=12)
    parser.add_argument("--no-yolo", action="store_true", help="禁用 YOLO，全场景 GraspNet")
    parser.add_argument("--yolo-model", default=None)
    parser.add_argument("--yolo-device", default=None)
    parser.add_argument("--yolo-conf", type=float, default=None)
    parser.add_argument("--yolo-iou", type=float, default=None)
    parser.add_argument("--infer-every-live", type=int, default=None)
    parser.add_argument("--pregrasp-offset", type=float, default=None, help="meters")
    parser.add_argument("--retreat-offset", type=float, default=None, help="meters")
    parser.add_argument("--gripper-open-width", type=float, default=0.09, help="meters")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = configure_camera(load_config(Path(args.config)), args)

    robot_cfg = cfg.get("robot", {})
    ready_cfg = robot_cfg.get(
        "ready_pose",
        {"x": 0.25, "y": 0.0, "z": 0.35, "roll": 0.0, "pitch": 1.2, "yaw": 0.0, "duration": 3.0},
    )
    grasp_cfg = cfg.get("grasp_pipeline", {}).get("grasp", {})
    pregrasp_offset_m = float(args.pregrasp_offset if args.pregrasp_offset is not None else grasp_cfg.get("pregrasp_offset_m", 0.08))
    retreat_offset_m = float(args.retreat_offset if args.retreat_offset is not None else pregrasp_offset_m)

    print("=== 初始化机械臂 ===")
    robot = RebotArm(
        config_path=robot_cfg.get("config_path"),
        urdf_path=robot_cfg.get("urdf_path"),
        repo_root=robot_cfg.get("repo_root"),
    )
    robot.connect(enable=True)
    robot.init_gripper()

    print("[Robot] 移动到预备位置...")
    _move_ready(robot, ready_cfg)

    cam_type = str(cfg.get("camera", {}).get("type", "")).lower()
    T_hand_eye, hand_eye_mode = load_hand_eye(PROJECT_ROOT, cam_type)
    if T_hand_eye is None or hand_eye_mode != "eye_in_hand":
        print("[WARN] 手眼标定不可用或非 eye_in_hand，夹取执行将被禁用")
        T_hand_eye = None

    print("=== 加载模型 ===")
    yolo_model, yolo_opts = load_yolo(cfg, args)
    net = build_net(args.checkpoint, args.num_view)

    cam_cfg = cfg["camera"]
    print(f"=== 初始化相机: {cam_cfg['type']} {cam_cfg.get('color_width')}x{cam_cfg.get('color_height')}@{cam_cfg.get('fps')} ===")
    if "realsense" in cam_type:
        cam = DirectRealSenseCamera(
            cam_cfg.get("color_width", 1280),
            cam_cfg.get("color_height", 720),
            cam_cfg.get("fps", 15),
        )
    else:
        cam = make_camera(cfg)

    last_targets: list[Any] = []
    selected_target: Optional[Any] = None
    last_target_status = "YOLO disabled: full-scene GraspNet" if yolo_model is None else "target detector warming up..."
    status = "warming up camera..."
    frozen = False
    last_display: Optional[np.ndarray] = None
    frame_index = 0
    fps_counter = 0
    fps_timer = time.perf_counter()
    fps_value = 0.0
    window_name = "Main - GraspNet Grasp"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("\n[Keys] G/SPACE=GraspNet夹取  R=恢复  Q/ESC=退出\n")

    try:
        cam.open()
        cam.warm_up(args.warmup)
        K = cam.K.astype(np.float64)
        print("Camera intrinsics:")
        print(K)

        while True:
            color_bgr, depth_mm = cam.get_frame()
            if color_bgr is None or depth_mm is None:
                continue

            frame_index += 1
            fps_counter += 1
            now = time.perf_counter()
            if now - fps_timer >= 1.0:
                fps_value = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

            if not frozen and yolo_model is not None and (frame_index == 1 or frame_index % int(yolo_opts["infer_every"]) == 0):
                try:
                    _, last_targets, selected_target = detect_targets(yolo_model, color_bgr, yolo_opts, args.target_class)
                    last_target_status = target_status_text(selected_target, last_targets, args.target_class)
                except Exception as exc:
                    last_targets = []
                    selected_target = None
                    last_target_status = f"YOLO failed: {exc}"

            if frozen and last_display is not None:
                display = last_display.copy()
                cv2.putText(display, "[FROZEN]", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)
            else:
                display_base = color_bgr
                if yolo_model is not None:
                    display_base = draw_target_overlay(color_bgr, last_targets, selected_target, args.target_class)
                display = overlay_status(
                    display_base,
                    f"{'LIVE' if not frozen else 'FROZEN'} {fps_value:.1f}fps | {status}",
                    False,
                    last_target_status,
                )
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                frozen = False
                last_display = None
                status = "live preview"
                continue

            if key in (ord("g"), ord("G"), ord(" ")):
                print("\n[G] 采帧并运行 GraspNet...")
                snap_color, snap_depth = cam.get_frame()
                if snap_color is None or snap_depth is None:
                    print("[G] 采帧失败")
                    continue

                try:
                    gg, best, status, last_target_status, last_targets, selected_target = _infer_current_frame(
                        net,
                        yolo_model,
                        yolo_opts,
                        snap_color,
                        snap_depth,
                        K,
                        args,
                    )
                except Exception as exc:
                    status = f"inference failed: {exc}"
                    print(f"[G] {status}")
                    continue

                print(f"[G] {status}")
                if best is None:
                    print("[G] 未找到有效 GraspNet 夹取候选")
                    continue

                _print_grasp(best)
                frozen = True
                display_base = snap_color
                if yolo_model is not None:
                    display_base = draw_target_overlay(snap_color, last_targets, selected_target, args.target_class)
                snap_display = overlay_status(display_base, f"SNAPSHOT | {status}", False, last_target_status)
                _draw_best_grasp_projection(snap_display, best, K)
                last_display = snap_display

                if T_hand_eye is None:
                    print("[G] 手眼标定不可用，无法执行夹取")
                    continue

                T_cam2base = _cam_to_base(T_hand_eye, robot)
                grasp6d, pre6d, retreat6d = _transform_grasp(best, T_cam2base, pregrasp_offset_m, retreat_offset_m)
                _execute_grasp(
                    robot,
                    grasp6d,
                    pre6d,
                    retreat6d,
                    ready_cfg,
                    dry_run=args.dry_run,
                    gripper_width_m=max(float(args.gripper_open_width), float(best.width) + 0.02),
                )

    finally:
        print("\n[退出] 释放夹爪并回零...")
        try:
            robot.release_gripper()
            robot.safe_home()
        except Exception as exc:
            print(f"[退出] {exc}")
        robot.disconnect()
        try:
            cam.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("已退出。")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
