"""
直接读取深度相机硬件出厂内参并保存 (支持 RealSense & Orbbec)

用法：
    cd /home/chlorine/seeed/cameraws
    python scripts/get_camera_intrinsics.py

功能：
    读取 config/default.yaml 中的相机类型和分辨率配置，
    动态调用对应的 SDK (pyrealsense2 / pyorbbecsdk) 提取出厂内参，
    并格式化为 OpenCV 标准格式后保存到对应的 npz 文件中。
"""

import os
import sys
import yaml
import numpy as np
from pathlib import Path

def load_config(yaml_path):
    """加载 YAML 配置文件"""
    if not os.path.exists(yaml_path):
        print(f"[错误] 找不到配置文件: {yaml_path}")
        sys.exit(1)
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_realsense_intrinsics(color_w, color_h, fps):
    """使用 pyrealsense2 获取 Intel 相机内参 (适用 D435i, D405 等)"""
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("[错误] 未安装 pyrealsense2，请执行: pip install pyrealsense2")
        sys.exit(1)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, fps)
    
    try:
        profile = pipeline.start(config)
        color_stream = profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        
        # 相机矩阵 (Camera Matrix)
        camera_matrix = np.array([
            [intrinsics.fx, 0,             intrinsics.ppx],
            [0,             intrinsics.fy, intrinsics.ppy],
            [0,             0,             1.0           ]
        ], dtype=np.float64)
        
        # 畸变系数 (通常提供 5 个参数: k1, k2, p1, p2, k3)
        dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float64)
        resolution = np.array([intrinsics.width, intrinsics.height], dtype=np.int32)
        
        return camera_matrix, dist_coeffs, resolution
    finally:
        pipeline.stop()

def get_orbbec_intrinsics(color_w, color_h, fps):
    """使用 pyorbbecsdk 获取奥比中光相机内参 (适用 Gemini 2 等)"""
    try:
        from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBError
    except ImportError:
        print("[错误] 未安装 pyorbbecsdk，请根据官方 README 编译安装。")
        sys.exit(1)

    pipeline = Pipeline()
    config = Config()
    
    try:
        # 获取彩色传感器的流配置列表
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        
        try:
            # 尝试寻找完全匹配的分辨率和格式
            color_profile = profile_list.get_video_stream_profile(color_w, color_h, OBFormat.RGB, fps)
        except Exception:
            print(f"[警告] 找不到 {color_w}x{color_h}@{fps} 的RGB流配置，将使用默认配置！")
            color_profile = profile_list.get_default_video_stream_profile()

        config.enable_stream(color_profile)
        pipeline.start(config)
        
        # 启动后，获取相机的硬件参数 (包含内参和畸变)
        camera_param = pipeline.get_camera_param()
        rgb_int = camera_param.rgb_intrinsic
        rgb_dist = camera_param.rgb_distortion
        
        # 相机矩阵 (Camera Matrix)
        camera_matrix = np.array([
            [rgb_int.fx, 0,          rgb_int.cx],
            [0,          rgb_int.fy, rgb_int.cy],
            [0,          0,          1.0       ]
        ], dtype=np.float64)
        
        # 提取畸变系数 (映射为 OpenCV 需要的 [k1, k2, p1, p2, k3])
        dist_coeffs = np.array([
            rgb_dist.k1, rgb_dist.k2, rgb_dist.p1, rgb_dist.p2, rgb_dist.k3
        ], dtype=np.float64)
        
        resolution = np.array([rgb_int.width, rgb_int.height], dtype=np.int32)
        
        return camera_matrix, dist_coeffs, resolution
    finally:
        pipeline.stop()


def main():
    # 1. 加载配置
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config" / "default.yaml"
    
    print(f"正在读取配置: {config_path}")
    cfg = load_config(config_path)
    
    cam_cfg = cfg.get("camera", {})
    cam_type = cam_cfg.get("type", "realsense_d435i")
    color_w = cam_cfg.get("color_width", 1280)
    color_h = cam_cfg.get("color_height", 720)
    fps = cam_cfg.get("fps", 30)

    print(f"\n=== 开始获取 {cam_type} 出厂内参 ===")
    print(f"目标分辨率: {color_w}x{color_h} @ {fps}FPS")

    # 2. 根据相机类型分发任务
    try:
        if cam_type in ["realsense_d435i", "realsense_d405"]:
            camera_matrix, dist_coeffs, resolution = get_realsense_intrinsics(color_w, color_h, fps)
        elif cam_type == "orbbec_gemini2":
            camera_matrix, dist_coeffs, resolution = get_orbbec_intrinsics(color_w, color_h, fps)
        else:
            print(f"[错误] 暂不支持的相机型号: {cam_type}")
            sys.exit(1)
    except Exception as e:
        print(f"\n[错误] 获取内参失败，请检查相机连接或分辨率支持状态。")
        print(f"详情: {e}")
        sys.exit(1)

    # 3. 打印结果
    print("\n--- 成功读取硬件内参 ---")
    print(f"实际获取分辨率: {resolution[0]}x{resolution[1]}")
    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)

    # 4. 存储到对应的目录
    save_dir = project_root / "config" / "calibration" / cam_type
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "intrinsics.npz"
    
    np.savez(
        save_path, 
        camera_matrix=camera_matrix, 
        dist_coeffs=dist_coeffs,
        resolution=resolution
    )
    print(f"\n[成功] 内参已保存至: {save_path}")

if __name__ == "__main__":
    main()