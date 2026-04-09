"""
Orbbec Gemini 2 专属 3D 视觉目标检测管线
特性：未检测到设备优雅退出、自动读取统一配置文件、支持开放词汇

用法：
    cd /your/path/to/cameraws
    python unused/yolo_gemini2_3d.py
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
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode
except ImportError:
    print("[错误] 未安装 pyorbbecsdk，请确保环境正确。")
    sys.exit(1)

clicked_point = {"u": -1, "v": -1}

def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point["u"] = x; clicked_point["v"] = y

def get_center_depth(depth_map, x, y, roi_size=5):
    h, w = depth_map.shape
    half_roi = roi_size // 2
    x_min, x_max = max(0, x - half_roi), min(w, x + half_roi + 1)
    y_min, y_max = max(0, y - half_roi), min(h, y + half_roi + 1)
    roi = depth_map[y_min:y_max, x_min:x_max]
    valid_depths = roi[roi > 0]
    return np.median(valid_depths) if len(valid_depths) > 0 else 0.0

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
    color_w, color_h = cam_cfg.get("color_width", 1280), cam_cfg.get("color_height", 720)
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

    # --- 初始化 Gemini 2 ---
    print("\n=== 初始化 Orbbec Gemini 2 ===")
    pipeline = Pipeline()
    config = Config()
    
    profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    try:
        color_profile = profile_list.get_video_stream_profile(color_w, color_h, OBFormat.MJPG, fps)
    except Exception:
        try: color_profile = profile_list.get_video_stream_profile(color_w, color_h, OBFormat.RGB, fps)
        except Exception: color_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(color_profile)

    depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    try: depth_profile = depth_profile_list.get_video_stream_profile(color_w, color_h, OBFormat.Y16, fps)
    except Exception: depth_profile = depth_profile_list.get_default_video_stream_profile()
    config.enable_stream(depth_profile)

    config.set_align_mode(OBAlignMode.HW_MODE)
    
    try:
        pipeline.start(config)
    except Exception as e:
        print(f"\n[致命错误] 无法启动 Orbbec Gemini 相机！")
        print(f"可能原因：1. 数据线未连接；2. 带宽不足；3. 设备被僵尸进程占用。")
        print(f"底层报错: {e}")
        sys.exit(1)

    camera_param = pipeline.get_camera_param()
    fx, fy = camera_param.rgb_intrinsic.fx, camera_param.rgb_intrinsic.fy
    cx, cy = camera_param.rgb_intrinsic.cx, camera_param.rgb_intrinsic.cy
    
    window_name = "Gemini 2 Object Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)

    try:
        while True:
            frames = pipeline.wait_for_frames(500)
            if frames is None: continue
            color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
            if not color_frame or not depth_frame: continue
            
            w_ob, h_ob, fmt_ob = color_frame.get_width(), color_frame.get_height(), color_frame.get_format()
            raw_color = np.ascontiguousarray(np.asanyarray(color_frame.get_data()), dtype=np.uint8)
            try:
                if fmt_ob == OBFormat.MJPG: color_image = cv2.imdecode(raw_color, cv2.IMREAD_COLOR)
                elif fmt_ob == OBFormat.RGB: color_image = cv2.cvtColor(raw_color.reshape((h_ob, w_ob, 3)), cv2.COLOR_RGB2BGR)
                else: color_image = raw_color.reshape((h_ob, w_ob, 3))
            except: continue
            
            if color_image is None: continue
            depth_map = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((h_ob, w_ob))

            results = model.predict(color_image, verbose=False, device=device)
            
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cls_id, conf = int(box.cls[0]), float(box.conf[0])
                    class_name = model.names[cls_id]
                    
                    u, v = (x1 + x2) // 2, (y1 + y2) // 2
                    z_mm = get_center_depth(depth_map, u, v, 5)
                    
                    if z_mm > 0:
                        z_m = z_mm / 1000.0
                        x_m, y_m = (u - cx) * z_m / fx, (v - cy) * z_m / fy
                        
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(color_image, (u, v), 5, (0, 0, 255), -1)
                        cv2.putText(color_image, f"{class_name} {conf:.2f}", (x1+5, y1-22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(color_image, f"X:{x_m:.2f} Y:{y_m:.2f} Z:{z_m:.2f} (m)", (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 165, 255), 2)

            cu, cv = clicked_point["u"], clicked_point["v"]
            if cu != -1 and cv != -1 and 0 <= cu < w_ob and 0 <= cv < h_ob:
                cz_mm = get_center_depth(depth_map, cu, cv)
                if cz_mm > 0:
                    cz_m = cz_mm / 1000.0
                    cx_m, cy_m = (cu - cx) * cz_m / fx, (cv - cy) * cz_m / fy
                    cv2.drawMarker(color_image, (cu, cv), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
                    cv2.putText(color_image, f"TEST-> X:{cx_m:.3f} Y:{cy_m:.3f} Z:{cz_m:.3f}m", (cu+10, cv-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            cv2.putText(color_image, f"Gemini 2 | {model_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow(window_name, color_image)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27] or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()