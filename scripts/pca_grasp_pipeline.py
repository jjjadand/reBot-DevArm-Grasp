"""
实时目标检测 + 快照夹取姿态估计管线 (Live Detection + Snapshot Grasp)
支持 D435i / Gemini 2 双相机接入。

工作流:
    实时预览：持续运行 YOLO 实例分割，叠加每个目标的 XYZ 坐标 (米)。
        ↓  按下 [S] 键
    快照：对当前帧生成局部点云 + PCA 夹取姿态估算，终端打印姿态参数。
    Open3D 3D 窗口显示点云与虚拟夹爪，关闭后继续实时检测。

用法:
    pip install open3d
    cd /home/chlorine/seeed/cameraws
    python scripts/pca_grasp_pipeline.py
    python scripts/pca_grasp_pipeline.py --infer-every 2
    python scripts/pca_grasp_pipeline.py --model yoloe-26s-seg.pt --device cpu
"""

import os
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

import sys
import time
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import cv2
import yaml
import numpy as np
from ultralytics import YOLO

try:
    import open3d as o3d
except ImportError:
    print("[ERROR] Missing open3d: pip install open3d")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────
# 鼠标测距回调
# ──────────────────────────────────────────────────────────────
clicked_point = {"u": -1, "v": -1}

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point["u"] = x
        clicked_point["v"] = y
        print(f"[Click] pixel ({x}, {y})")


# ──────────────────────────────────────────────────────────────
# 深度采样（ROI 中位数）
# ──────────────────────────────────────────────────────────────
def get_center_depth_rs(depth_frame, x, y, roi=5):
    w, h = depth_frame.get_width(), depth_frame.get_height()
    half = roi // 2
    depths = [
        depth_frame.get_distance(i, j)
        for i in range(max(0, x - half), min(w, x + half + 1))
        for j in range(max(0, y - half), min(h, y + half + 1))
        if depth_frame.get_distance(i, j) > 0
    ]
    return float(np.median(depths)) if depths else 0.0


def get_center_depth_ob(depth_map, x, y, roi=5):
    h, w = depth_map.shape
    half = roi // 2
    region = depth_map[max(0, y-half):min(h, y+half+1),
                       max(0, x-half):min(w, x+half+1)]
    valid = region[region > 0]
    return float(np.median(valid)) if len(valid) > 0 else 0.0


# ──────────────────────────────────────────────────────────────
# 1. PCA 夹取姿态估计器
# ──────────────────────────────────────────────────────────────
class GraspEstimator:
    """
    基于点云 PCA 的 6D 夹取姿态估计。
    正式环境可在 predict() 中替换为 AnyGrasp/Contact-GraspNet。
    """
    MAX_GRIPPER_WIDTH = 0.08   # 夹爪最大张开 8 cm

    def predict(self, pcd: "o3d.geometry.PointCloud"):
        """
        返回 (T_4x4, width, score, R_3x3, center, shape_hint)
        若拒绝抓取则第一项返回 None。
        """
        points = np.asarray(pcd.points)
        if len(points) < 50:
            print("  [Skip] Too few points.")
            return None, 0.0, 0.0, None, None, None

        center = np.mean(points, axis=0)
        cov = np.cov(points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)   # 升序

        idx = np.argsort(eigenvalues)[::-1]               # 降序
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # ── 轴分配（几何法）──
        # 长轴（PC1）= 物体最长方向；短轴（PC3）= 物体最薄方向
        long_axis  = eigenvectors[:, 0].copy()
        short_axis = eigenvectors[:, 2].copy()

        # open_axis = 短轴：夹爪开合方向跨过物体最薄处，所需开度最小
        open_axis = short_axis / np.linalg.norm(short_axis)

        # approach = long × open，天然垂直于两轴，定义夹爪下探平面
        # 此叉积有 ± 两个解，即两种相反的下探方向
        approach_raw = np.cross(long_axis, open_axis)
        norm = np.linalg.norm(approach_raw)
        if norm < 1e-8:                        # 极端退化情况（长轴≈短轴）
            approach_raw = eigenvectors[:, 1].copy()
            norm = np.linalg.norm(approach_raw)
        approach_raw /= norm

        # 两个候选解：approach_raw 和 -approach_raw
        # 保留与相机向下方向（+Y）夹角更小的，即 dot > 0 的那个
        # （丢弃"从下往上"或"从桌面穿出"的无效解）
        cam_down = np.array([0.0, 1.0, 0.0])   # 相机坐标系 +Y ≈ 物理向下
        approach = approach_raw if np.dot(approach_raw, cam_down) >= 0 \
                   else -approach_raw

        # 重新正交化 grip_axis（确保三轴严格正交）
        grip_axis = np.cross(open_axis, approach)
        norm = np.linalg.norm(grip_axis)
        if norm < 1e-8:
            grip_axis = long_axis.copy()
        else:
            grip_axis /= norm

        # R 列：[开合轴, 夹爪纵轴, 下探轴]
        R = np.column_stack([open_axis, grip_axis, approach])
        if np.linalg.det(R) < 0:
            R[:, 0] = -R[:, 0]

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = center

        # 沿夹取点截面估计夹持宽度
        # 只取重心附近 ±2cm 切片内的点，避免细长物体取到最宽端
        along_grip = np.dot(points - center, grip_axis)
        slice_half = 0.02   # 2 cm 截面半宽
        in_slice = np.abs(along_grip) < slice_half
        slice_pts = points[in_slice] if in_slice.sum() >= 10 else points
        proj_slice = np.dot(slice_pts - center, open_axis)
        # 去除 5% 极端值再取范围，抑制边缘噪点
        lo, hi = np.percentile(proj_slice, 5), np.percentile(proj_slice, 95)
        width = float(hi - lo) + 0.010   # +1 cm 余量

        if width > self.MAX_GRIPPER_WIDTH:
            print(f"  [Skip] Too wide: {width*100:.1f} cm")
            return None, 0.0, 0.0, None, None, None

        # 形状分类
        r01 = eigenvalues[0] / (eigenvalues[1] + 1e-8)
        r12 = eigenvalues[1] / (eigenvalues[2] + 1e-8)
        shape = "elongated" if r01 > 3.0 else ("flat" if r12 > 3.0 else "compact")

        score = min(1.0, len(points) / 3000.0)
        return T, width, score, R, center, shape


# ──────────────────────────────────────────────────────────────
# 2. Open3D 可视化工具
# ──────────────────────────────────────────────────────────────
def create_virtual_gripper(T, width=0.08, depth=0.06):
    """根据 4x4 变换矩阵生成红色线框夹爪。"""
    w2 = width / 2.0
    pts_local = [
        [0,   0,   -0.05],   # 基座
        [-w2, 0,    0   ],   # 左指根
        [ w2, 0,    0   ],   # 右指根
        [-w2, 0,    depth],  # 左指尖
        [ w2, 0,    depth],  # 右指尖
    ]
    lines = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4]]
    pts_world = [(T @ np.array([*p, 1.0]))[:3] for p in pts_local]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts_world)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(lines))
    return ls


# ──────────────────────────────────────────────────────────────
# 3. Mask → 点云
# ──────────────────────────────────────────────────────────────
def generate_masked_pointcloud(color_img, depth_mm, mask, intrinsics):
    fx, fy, cx, cy = intrinsics
    h, w = depth_mm.shape
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    valid = (mask > 0) & (depth_mm > 100) & (depth_mm < 2000)
    v_px, u_px = np.where(valid)
    z_m = depth_mm[valid] / 1000.0
    x_m = (u_px - cx) * z_m / fx
    y_m = (v_px - cy) * z_m / fy

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.stack([x_m, y_m, z_m], axis=-1))
    pcd.colors = o3d.utility.Vector3dVector(color_img[valid][:, ::-1] / 255.0)
    pcd = pcd.voxel_down_sample(voxel_size=0.003)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd


# ──────────────────────────────────────────────────────────────
# 4. 旋转矩阵 → 欧拉角（度）
# ──────────────────────────────────────────────────────────────
def rotation_to_euler_deg(R):
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0.0
    return np.degrees([rx, ry, rz])


# ──────────────────────────────────────────────────────────────
# 5. 快照抓取（S 键触发）
# ──────────────────────────────────────────────────────────────
def run_snapshot(color_image, depth_mm, results, model,
                 cam_intrinsics, grasp_estimator):
    """
    对当前帧（使用实时推理缓存的 results）生成点云并计算夹取姿态。
    在 Open3D 窗口中显示结果，关闭后返回继续实时循环。
    """
    print("\n[Snapshot] Computing grasp poses...")

    geometries = []
    # 翻转矩阵：相机坐标系 → Open3D 世界坐标系（Y/Z 取反，使 Z 朝上）
    flip = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    found = False

    if results is None:
        print("  No detection results available yet.")
        return

    for r in results:
        has_masks = r.masks is not None
        for i, box in enumerate(r.boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]

            # 构建掩码（优先用多边形轮廓，比 resize 更精准）
            target_mask = np.zeros(depth_mm.shape, dtype=np.uint8)
            if has_masks:
                pts = np.array(r.masks.xy[i], dtype=np.int32)
                cv2.fillPoly(target_mask, [pts], 255)
                # 形态学腐蚀：切断边缘飞点与桌面的连接
                kernel = np.ones((9, 9), np.uint8)
                target_mask = cv2.erode(target_mask, kernel, iterations=1)
            else:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cv2.rectangle(target_mask, (x1, y1), (x2, y2), 255, -1)

            pcd = generate_masked_pointcloud(color_image, depth_mm,
                                             target_mask, cam_intrinsics)
            n_pts = len(pcd.points)
            if n_pts < 100:
                print(f"  [{class_name}] Skipped (only {n_pts} valid points)")
                continue

            print(f"\n  [{class_name} {conf:.2f}]  pts={n_pts}  → PCA...")
            T_cam, width, score, R_mat, center, shape = grasp_estimator.predict(pcd)
            if T_cam is None:
                continue

            rpy = rotation_to_euler_deg(R_mat)
            print(f"    Position (camera frame):")
            print(f"      X={center[0]:+.4f}  Y={center[1]:+.4f}  Z={center[2]:+.4f}  (m)")
            print(f"    Euler RPY (deg):  R={rpy[0]:+.1f}  P={rpy[1]:+.1f}  Y={rpy[2]:+.1f}")
            print(f"    Gripper width:    {width*100:.1f} cm   Shape: {shape}   Score: {score:.2f}")
            print(f"    Rotation matrix:\n{np.array2string(R_mat, precision=4, suppress_small=True)}")

            # Open3D 可视化（翻转到世界坐标系）
            pcd.transform(flip)
            gripper = create_virtual_gripper(flip @ T_cam, width=width)
            geometries.extend([pcd, gripper])
            found = True

    if found:
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        geometries.append(coord)
        print("\n[Open3D] Close this window to resume live detection.")
        o3d.visualization.draw_geometries(geometries,
                                          window_name="Grasp Pose Estimation")
    else:
        print("  No valid grasp targets found in current frame.")


# ──────────────────────────────────────────────────────────────
# 6. 主流程
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Live Detection + Snapshot Grasp Pipeline")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--model", default=None, help="YOLO model name (overrides yaml)")
    parser.add_argument("--device", default=None, help="cpu | cuda:0 (overrides yaml)")
    parser.add_argument("--infer-every", type=int, default=3,
                        help="Run YOLO every N frames in live mode (default: 3)")
    args = parser.parse_args()

    config_path = project_root / args.config
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        sys.exit(1)
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cam_cfg  = cfg.get("camera", {})
    cam_type = cam_cfg.get("type", "orbbec_gemini2").lower()
    color_w  = cam_cfg.get("color_width", 640)
    color_h  = cam_cfg.get("color_height", 480)
    fps      = cam_cfg.get("fps", 30)

    yolo_cfg      = cfg.get("yolo", {})
    model_name    = args.model  or yolo_cfg.get("model_name", "yoloe-26s-seg.pt")
    device        = args.device or yolo_cfg.get("device", "cpu")
    custom_classes = yolo_cfg.get("custom_classes", ["bottle", "cup", "book"])
    use_world     = yolo_cfg.get("use_world", True)

    # ── YOLO ──
    model_path = project_root / "models" / model_name
    print(f"=== Loading YOLO: {model_name} ===")
    model = YOLO(str(model_path))
    if use_world and ("world" in model_name.lower() or "yoloe" in model_name.lower()):
        model.set_classes(custom_classes)
        print(f"  Classes: {custom_classes}")

    grasp_estimator = GraspEstimator()

    # ── 相机初始化 ──
    print(f"\n=== Camera: {cam_type} ===")
    cam_intrinsics   = (0.0, 0.0, 0.0, 0.0)
    rs_pipeline = rs_align = rs_intr = None
    ob_pipeline = None
    fx = fy = icx = icy = 0.0

    if "realsense" in cam_type:
        try:
            import pyrealsense2 as rs
        except ImportError:
            print("[ERROR] pyrealsense2 not installed.")
            sys.exit(1)
        rs_pipeline = rs.pipeline()
        rs_cfg = rs.config()
        rs_cfg.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, fps)
        rs_cfg.enable_stream(rs.stream.depth, color_w, color_h, rs.format.z16, fps)
        try:
            profile = rs_pipeline.start(rs_cfg)
        except RuntimeError as e:
            print(f"[ERROR] RealSense start failed: {e}")
            sys.exit(1)
        rs_align = rs.align(rs.stream.color)
        rs_intr  = (profile.get_stream(rs.stream.color)
                    .as_video_stream_profile().get_intrinsics())
        cam_intrinsics = (rs_intr.fx, rs_intr.fy, rs_intr.ppx, rs_intr.ppy)
        fx, fy, icx, icy = cam_intrinsics
        print(f"  D435i ready  fx={rs_intr.fx:.1f}")

    elif "orbbec" in cam_type:
        try:
            from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode
        except ImportError:
            print("[ERROR] pyorbbecsdk not installed.")
            sys.exit(1)
        try:
            from pyorbbecsdk import Context, OBLogSeverity
            Context().set_logger_severity(OBLogSeverity.FATAL)
        except Exception:
            pass
        try:
            ob_pipeline = Pipeline()
        except Exception as e:
            print(f"[ERROR] Orbbec device not found: {e}")
            print("  Try: sudo chmod a+rw /dev/bus/usb/*/*")
            sys.exit(1)
        ob_cfg = Config()
        pl = ob_pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            cp = pl.get_video_stream_profile(color_w, color_h, OBFormat.MJPG, fps)
        except Exception:
            try:
                cp = pl.get_video_stream_profile(color_w, color_h, OBFormat.RGB, fps)
            except Exception:
                cp = pl.get_default_video_stream_profile()
        ob_cfg.enable_stream(cp)
        dl = ob_pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        try:
            dp = dl.get_video_stream_profile(color_w, color_h, OBFormat.Y16, fps)
        except Exception:
            dp = dl.get_default_video_stream_profile()
        ob_cfg.enable_stream(dp)
        ob_cfg.set_align_mode(OBAlignMode.HW_MODE)
        try:
            ob_pipeline.start(ob_cfg)
        except Exception as e:
            print(f"[ERROR] Orbbec pipeline start failed: {e}")
            sys.exit(1)
        ri = ob_pipeline.get_camera_param().rgb_intrinsic
        cam_intrinsics = (ri.fx, ri.fy, ri.cx, ri.cy)
        fx, fy, icx, icy = cam_intrinsics
        print(f"  Gemini2 ready  fx={ri.fx:.1f}")

        # 丢弃启动帧，fd 重定向抑制 C++ timestamp 日志
        print("  Warmup...", end="", flush=True)
        try:
            _dn = os.open(os.devnull, os.O_WRONLY)
            _old = os.dup(2)
            os.dup2(_dn, 2)
            for _ in range(30):
                ob_pipeline.wait_for_frames(200)
            os.dup2(_old, 2)
            os.close(_dn); os.close(_old)
        except Exception:
            for _ in range(30):
                ob_pipeline.wait_for_frames(200)
        print(" done")
    else:
        print(f"[ERROR] Unknown camera type: {cam_type}")
        sys.exit(1)

    # ── UI 初始化 ──
    WIN = f"Live Detection + Grasp  [{cam_type}]"
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, mouse_callback)
    print(f"\n[Keys]  S=snapshot+grasp   Q/ESC=quit   Left-click=measure depth")
    print(f"        Live YOLO every {args.infer_every} frames\n")

    # ── 主循环状态 ──
    frame_count   = 0
    last_results  = None      # 缓存 YOLO 结果（用于实时叠加和快照）
    last_seg_vis  = None      # 缓存 masks.plot() 图层
    color_image   = None
    depth_mm      = None
    depth_frame_rs = None

    _fps_t   = time.perf_counter()
    _fps_cnt = 0
    fps_disp = 0.0

    try:
        while True:
            # ── 取帧 ──
            if "realsense" in cam_type:
                frames  = rs_pipeline.wait_for_frames()
                aligned = rs_align.process(frames)
                cf, df  = aligned.get_color_frame(), aligned.get_depth_frame()
                if not cf or not df:
                    continue
                color_image    = np.asanyarray(cf.get_data())
                depth_frame_rs = df
                # RealSense depth 已是 uint16 mm（×depth_scale 后才是米，这里保持 mm）
                depth_mm = (np.asanyarray(df.get_data()).astype(np.float32)).astype(np.uint16)
            else:
                frames = ob_pipeline.wait_for_frames(500)
                if frames is None:
                    continue
                cf, df = frames.get_color_frame(), frames.get_depth_frame()
                if not cf or not df:
                    continue
                w_ob, h_ob = cf.get_width(), cf.get_height()
                fmt_ob = cf.get_format()
                raw = np.ascontiguousarray(np.asanyarray(cf.get_data()), dtype=np.uint8)
                try:
                    if fmt_ob == OBFormat.MJPG:
                        color_image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                    elif fmt_ob == OBFormat.RGB:
                        color_image = cv2.cvtColor(raw.reshape(h_ob, w_ob, 3),
                                                   cv2.COLOR_RGB2BGR)
                    else:
                        color_image = raw.reshape(h_ob, w_ob, 3)
                except Exception:
                    continue
                if color_image is None:
                    continue
                depth_mm = np.frombuffer(df.get_data(),
                                         dtype=np.uint16).reshape(h_ob, w_ob)

            frame_count += 1

            # ── YOLO 推理（每 N 帧一次）──
            if frame_count % args.infer_every == 0:
                last_results = model.predict(
                    color_image, verbose=False, device=device, conf=0.2)
                # 缓存 mask 多边形（而非整张标注图，避免重影）
                last_seg_vis = None
                if last_results and last_results[0].masks is not None:
                    r0 = last_results[0]
                    h_img, w_img = color_image.shape[:2]
                    overlay = np.zeros((h_img, w_img, 3), dtype=np.uint8)
                    _palette = [
                        (255, 80,  80), (80, 200,  80), (80, 120, 255),
                        (255,200,   0), ( 0, 220, 220), (200,   0, 200),
                    ]
                    for i in range(len(r0.boxes)):
                        pts = np.array(r0.masks.xy[i], dtype=np.int32)
                        color = _palette[i % len(_palette)]
                        cv2.fillPoly(overlay, [pts], color)
                    last_seg_vis = overlay   # 纯 mask 色块，背景全黑

            # ── 构建显示帧：mask 色块叠加到当前实时帧（无重影）──
            display = color_image.copy()
            if last_seg_vis is not None:
                # 只在 mask 非黑处混合，背景保持原样
                mask_region = last_seg_vis.any(axis=2)
                blended = cv2.addWeighted(color_image, 0.55, last_seg_vis, 0.45, 0)
                display[mask_region] = blended[mask_region]

            # ── 叠加检测框与 XYZ 坐标 ──
            if last_results:
                for r in last_results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        cls_id = int(box.cls[0])
                        conf   = float(box.conf[0])
                        cname  = model.names[cls_id]
                        u, v   = (x1 + x2) // 2, (y1 + y2) // 2

                        xm = ym = zm = 0.0
                        valid = False
                        if "realsense" in cam_type and depth_frame_rs:
                            zm = get_center_depth_rs(depth_frame_rs, u, v)
                            if zm > 0:
                                import pyrealsense2 as rs
                                pt = rs.rs2_deproject_pixel_to_point(
                                    rs_intr, [u, v], zm)
                                xm, ym, zm = pt
                                valid = True
                        elif depth_mm is not None:
                            zmm = get_center_depth_ob(depth_mm, u, v)
                            if zmm > 0:
                                zm = zmm / 1000.0
                                xm = (u - icx) * zm / fx
                                ym = (v - icy) * zm / fy
                                valid = True

                        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        cv2.circle(display, (u, v), 4, (0, 0, 255), -1)
                        if valid:
                            lb1 = f"{cname} {conf:.2f}"
                            lb2 = f"X:{xm:.2f} Y:{ym:.2f} Z:{zm:.2f}m"
                            bw  = max(len(lb1), len(lb2)) * 8 + 4
                            by  = max(0, y1 - 38)
                            cv2.rectangle(display, (x1, by), (x1 + bw, y1),
                                          (0, 0, 0), -1)
                            cv2.putText(display, lb1, (x1 + 3, by + 14),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                                        (0, 255, 0), 1, cv2.LINE_AA)
                            cv2.putText(display, lb2, (x1 + 3, by + 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                                        (0, 255, 255), 1, cv2.LINE_AA)
                        else:
                            cv2.putText(display, f"{cname} (no depth)",
                                        (x1, max(15, y1 - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                                        (0, 165, 255), 1, cv2.LINE_AA)

            # ── 鼠标测距 ──
            cu, cv_y = clicked_point["u"], clicked_point["v"]
            if cu != -1 and cv_y != -1:
                pz = px = py = 0.0
                valid_click = False
                if "realsense" in cam_type and depth_frame_rs:
                    pz = get_center_depth_rs(depth_frame_rs, cu, cv_y)
                    if pz > 0:
                        import pyrealsense2 as rs
                        pt = rs.rs2_deproject_pixel_to_point(rs_intr, [cu, cv_y], pz)
                        px, py, pz = pt
                        valid_click = True
                elif depth_mm is not None:
                    zmm = get_center_depth_ob(depth_mm, cu, cv_y)
                    if zmm > 0:
                        pz = zmm / 1000.0
                        px = (cu - icx) * pz / fx
                        py = (cv_y - icy) * pz / fy
                        valid_click = True
                if valid_click:
                    cv2.drawMarker(display, (cu, cv_y),
                                   (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
                    tl = f"X:{px:.3f} Y:{py:.3f} Z:{pz:.3f}m"
                    cv2.rectangle(display, (cu + 5, cv_y - 22),
                                  (cu + 280, cv_y + 4), (0, 0, 0), -1)
                    cv2.putText(display, tl, (cu + 8, cv_y - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 255), 1, cv2.LINE_AA)

            # ── FPS 与提示 ──
            _fps_cnt += 1
            now = time.perf_counter()
            if now - _fps_t >= 1.0:
                fps_disp = _fps_cnt / (now - _fps_t)
                _fps_cnt = 0
                _fps_t   = now
            cv2.putText(display,
                        f"{fps_disp:.1f}fps | S=Grasp Snapshot  Q=Quit",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(WIN, display)

            # ── 按键 ──
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in [ord('q'), ord('Q'), 27]:
                break
            if key in [ord('s'), ord('S')]:
                if color_image is not None and depth_mm is not None:
                    run_snapshot(color_image, depth_mm, last_results,
                                 model, cam_intrinsics, grasp_estimator)
                else:
                    print("[S] No frame available yet.")

    except KeyboardInterrupt:
        print("\n[Ctrl+C] Exiting.")
    finally:
        if rs_pipeline:
            try: rs_pipeline.stop()
            except Exception: pass
        if ob_pipeline:
            try: ob_pipeline.stop()
            except Exception: pass
        cv2.destroyAllWindows()

    print("Done.")


if __name__ == "__main__":
    main()
    os._exit(0)
