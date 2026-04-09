"""
Gemini 2 畸变参数检验脚本 (已加入动态格式解码)

用法：
    cd /home/chlorine/seeed/cameraws
    python scripts/test_gemini2_distortion.py
"""

import sys
import cv2
import numpy as np

try:
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBError
except ImportError:
    print("[错误] 未安装 pyorbbecsdk，请确保在包含奥比中光 SDK 的环境下运行。")
    sys.exit(1)

def draw_grid(img, grid_size=50):
    """在图像上绘制辅助网格，方便观察直线是否弯曲"""
    h, w = img.shape[:2]
    for y in range(0, h, grid_size):
        cv2.line(img, (0, y), (w, y), (0, 255, 0), 1, cv2.LINE_AA)
    for x in range(0, w, grid_size):
        cv2.line(img, (x, 0), (x, h), (0, 255, 0), 1, cv2.LINE_AA)
    return img

def main():
    print("正在启动 Orbbec Gemini 2 ...")
    pipeline = Pipeline()
    config = Config()
    
    profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    try:
        color_profile = profile_list.get_video_stream_profile(1280, 720, OBFormat.RGB, 30)
    except Exception:
        print("[警告] 找不到 1280x720 的 RGB 配置，将使用相机默认配置。")
        color_profile = profile_list.get_default_video_stream_profile()

    config.enable_stream(color_profile)
    pipeline.start(config)

    try:
        camera_param = pipeline.get_camera_param()
        rgb_int = camera_param.rgb_intrinsic
        rgb_dist = camera_param.rgb_distortion
        
        camera_matrix = np.array([
            [rgb_int.fx, 0,          rgb_int.cx],
            [0,          rgb_int.fy, rgb_int.cy],
            [0,          0,          1.0       ]
        ], dtype=np.float64)
        
        raw_dist_coeffs = np.array([
            rgb_dist.k1, rgb_dist.k2, rgb_dist.p1, rgb_dist.p2, rgb_dist.k3
        ], dtype=np.float64)

        print("\n--- 硬件参数已读取 ---")
        print("Camera Matrix:\n", camera_matrix)
        print("Raw Distortion Coefficients:\n", raw_dist_coeffs)
        print("\n按 [Q] 键退出...")

        while True:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue
                
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
                
            width = color_frame.get_width()
            height = color_frame.get_height()
            color_format = color_frame.get_format()
            
            # 【终极修复点】：提取数据并强制转换为 C-contiguous 的 uint8 连续数组
            raw_data = np.asanyarray(color_frame.get_data())
            data_buffer = np.ascontiguousarray(raw_data, dtype=np.uint8)
            
            # 动态判断相机真实的输出格式并进行正确解码
            if color_format == OBFormat.RGB:
                raw_img = data_buffer.reshape((height, width, 3))
                raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
            elif color_format == OBFormat.BGR:
                # 已经是 BGR，直接 reshape 即可
                raw_img = data_buffer.reshape((height, width, 3))
            elif color_format == OBFormat.YUYV:
                # YUYV 格式每个像素占 2 字节
                raw_img = data_buffer.reshape((height, width, 2))
                raw_img = cv2.cvtColor(raw_img, cv2.COLOR_YUV2BGR_YUYV)
            elif color_format == OBFormat.MJPG:
                # MJPG 是压缩格式，必须使用 imdecode 解码
                raw_img = cv2.imdecode(data_buffer, cv2.IMREAD_COLOR)
            else:
                print(f"[警告] 未知的图像格式 {color_format}，无法渲染。")
                continue

            try:
                # 这一步必然会崩溃触发 except
                undistorted_img = cv2.undistort(raw_img, camera_matrix, raw_dist_coeffs)
            except Exception as e:
                # 捕获异常，生成黑屏
                undistorted_img = np.zeros_like(raw_img)
                cv2.putText(undistorted_img, "MATH CRASHED!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)

            raw_img_grid = raw_img.copy()
            undistorted_img_grid = undistorted_img.copy()
            
            draw_grid(raw_img_grid)
            draw_grid(undistorted_img_grid)

            cv2.putText(raw_img_grid, "Left: Real Camera Output (ISP Rectified)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(undistorted_img_grid, "Right: After cv2.undistort (Crash)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            resize_w, resize_h = 640, 360
            raw_resized = cv2.resize(raw_img_grid, (resize_w, resize_h))
            undist_resized = cv2.resize(undistorted_img_grid, (resize_w, resize_h))
            
            combined_view = np.hstack((raw_resized, undist_resized))
            cv2.imshow("Gemini 2 Distortion Test", combined_view)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()