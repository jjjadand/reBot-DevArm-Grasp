"""
Intel RealSense D435i 专属 3D 视觉目标检测管线 (已修复 YOLOE-26 识别与沙盒路径)
特性：未检测到设备优雅退出、自动读取统一配置文件、支持 YOLOE-26 开放词汇

用法：
    cd /your/path/to/cameraws
    python unused/yolo_d435i_3d.py
"""

import os
import sys
import cv2
import yaml
import numpy as np
from pathlib import Path
from ultralytics import YOLO

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

try:
    import pyrealsense2 as rs
except ImportError:
    print("[错误] 未安装 pyrealsense2，请执行: pip install pyrealsense2")
    sys.exit(1)

clicked_point = {"u": -1, "v": -1}

def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point["u"] = x; clicked_point["v"] = y

def get_center_depth(depth_frame, x, y, roi_size=5):
    w, h = depth_frame.get_width(), depth_frame.get_height()
    half_roi = roi_size // 2
    x_min, x_max = max(0, x - half_roi), min(w, x + half_roi + 1)
    y_min, y_max = max(0, y - half_roi), min(h, y + half_roi + 1)
    depths = [depth_frame.get_distance(i, j) for i in range(x_min, x_max) for j in range(y_min, y_max) if depth_frame.get_distance(i, j) > 0]
    return np.median(depths) if len(depths) > 0 else 0.0

def main():
    # --- 加载统一架构配置 ---
    # 因为当前脚本在 unused 文件夹，所以 project_root 依然是 parent.parent
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config" / "default.yaml"
    
    if not os.path.exists(config_path):
        print(f"[错误] 找不到配置文件: {config_path}")
        sys.exit(1)
        
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    cam_cfg = cfg.get("camera", {})
    color_w, color_h = cam_cfg.get("color_width", 640), cam_cfg.get("color_height", 480)
    fps = cam_cfg.get("fps", 30)

    yolo_cfg = cfg.get("yolo", {})
    model_name = yolo_cfg.get("model_name", "yoloe-26s-seg.pt")
    device = yolo_cfg.get("device", "cpu")
    use_world = yolo_cfg.get("use_world", False)
    custom_classes = yolo_cfg.get("custom_classes", ["person", "cup"])
    
    # 【修复1】：动态读取模型目录，默认回到 models 文件夹
    custom_model_dir = yolo_cfg.get("model_dir", "models")
    if os.path.isabs(custom_model_dir):
        models_dir = Path(custom_model_dir)
    else:
        models_dir = project_root / custom_model_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    # --- 初始化 YOLO ---
    print("=== 初始化 YOLO 模型 ===")
    
    # 【修复2】：应用沙盒目录切换技术，确保 mobileclip2_b.ts 不乱跑
    original_cwd = os.getcwd()
    try:
        os.chdir(models_dir)
        model = YOLO(model_name) 
        
        # 【修复3】：加入对 yoloe 关键词的识别，修复无法加载自定义词汇的 Bug
        is_open_vocab = use_world and ("world" in model_name.lower() or "yoloe" in model_name.lower())
        if is_open_vocab:
            print(f"启用开放词汇模式，正在注入 {len(custom_classes)} 种概念...")
            model.set_classes(custom_classes)
    finally:
        os.chdir(original_cwd)
        
    print(f"YOLO 模型就绪 (Device: {device})")

    # --- 初始化 D435i ---
    print("\n=== 初始化 RealSense D435i ===")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, color_w, color_h, rs.format.z16, fps)

    # 优雅捕获设备未连接错误
    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"\n[致命错误] 无法启动 RealSense 相机！")
        print(f"可能原因：1. 相机未连接；2. USB 接口松动；3. 被其他进程占用。")
        print(f"底层报错: {e}")
        sys.exit(1)

    align = rs.align(rs.stream.color)
    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
    
    window_name = "D435i Object Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)

    try:
        # 跳过预热帧
        for _ in range(15): pipeline.wait_for_frames()

        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame, depth_frame = aligned_frames.get_color_frame(), aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame: continue
            
            color_image = np.asanyarray(color_frame.get_data())

            results = model.predict(color_image, verbose=False, device=device)
            
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cls_id, conf = int(box.cls[0]), float(box.conf[0])
                    class_name = model.names[cls_id]
                    
                    u, v = (x1 + x2) // 2, (y1 + y2) // 2
                    z_m = get_center_depth(depth_frame, u, v, 5)
                    
                    if z_m > 0:
                        pt = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], z_m)
                        x_m, y_m, z_m = pt[0], pt[1], pt[2]
                        
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(color_image, (u, v), 5, (0, 0, 255), -1)
                        cv2.putText(color_image, f"{class_name} {conf:.2f}", (x1+5, y1-22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(color_image, f"X:{x_m:.2f} Y:{y_m:.2f} Z:{z_m:.2f} (m)", (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 165, 255), 2)

            cu, cv = clicked_point["u"], clicked_point["v"]
            if cu != -1 and cv != -1 and 0 <= cu < intrinsics.width and 0 <= cv < intrinsics.height:
                cz_m = get_center_depth(depth_frame, cu, cv)
                if cz_m > 0:
                    pt = rs.rs2_deproject_pixel_to_point(intrinsics, [cu, cv], cz_m)
                    cv2.drawMarker(color_image, (cu, cv), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
                    cv2.putText(color_image, f"TEST-> X:{pt[0]:.3f} Y:{pt[1]:.3f} Z:{pt[2]:.3f}m", (cu+10, cv-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            cv2.putText(color_image, f"D435i | {model_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow(window_name, color_image)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27] or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()