"""
内参标定 - 步骤1：采集棋盘格图像

用法：
    cd /home/chlorine/seeed/cameraws
    python scripts/collect_intrinsic_frames.py

操作说明：
    - 在【终端】中操作，不需要点击图像窗口
    - 将棋盘格放在相机前，从不同角度和距离拍摄
    - 按 Enter  保存当前帧（检测到棋盘格才保存）
    - 输入 q + Enter  退出
    - 建议采集 ≥ 20 帧
"""

import os
# 告知 Qt 去哪找系统字体（解决 Qt 字体警告）
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

import sys
import threading
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cameraws.config import load_config, resolve_calib_paths
from cameraws.camera.factory import create_camera
from cameraws.calibration.intrinsic import CheckerboardCalibrator

WARMUP_FRAMES = 30   # 丢弃前 N 帧等待曝光稳定


def main():
    cfg = load_config("config/default.yaml")
    cfg = resolve_calib_paths(cfg)
    cb_cfg = cfg["calibration"]["checkerboard"]
    cam_type = cfg["camera"]["type"]

    save_dir = Path("data/intrinsic_frames") / cam_type
    save_dir.mkdir(parents=True, exist_ok=True)

    cal = CheckerboardCalibrator(
        board_cols=cb_cfg["cols"],
        board_rows=cb_cfg["rows"],
        square_size_m=cb_cfg["square_size_m"],
    )

    # 续接已有图像编号
    frame_idx = len(list(save_dir.glob("frame_*.png")))

    print(f"\n=== 内参标定 - 采集棋盘格图像 ===")
    print(f"相机: {cam_type}")
    print(f"棋盘格: {cb_cfg['cols']}x{cb_cfg['rows']} 内角点, "
          f"格大小 {cb_cfg['square_size_m']*100:.1f} cm")
    print(f"保存目录: {save_dir}/")
    print()
    print("【终端操作】")
    print("  按 Enter      → 保存当前帧")
    print("  输入 q Enter  → 退出")
    print()

    # 用线程读取终端输入（不阻塞相机循环）
    cmd_queue = []
    stop_flag = [False]

    def input_thread():
        while not stop_flag[0]:
            try:
                line = input()
                cmd_queue.append(line.strip().lower())
            except EOFError:
                cmd_queue.append("q")
                break

    t = threading.Thread(target=input_thread, daemon=True)
    t.start()

    # 保存最新帧供采集使用
    latest_frame = [None]
    latest_detected = [False]

    with create_camera(cfg) as camera:
        # 相机预热
        print(f"相机预热中（{WARMUP_FRAMES} 帧）...", end="", flush=True)
        for _ in range(WARMUP_FRAMES):
            camera.get_frame()
        print(" 就绪\n")

        cv2.namedWindow("预览 - 操作请看终端", cv2.WINDOW_AUTOSIZE)

        while True:
            frame = camera.get_frame()
            latest_frame[0] = frame

            # 棋盘格检测（用于预览高亮，不加入标定缓冲）
            vis = cal.draw_corners(frame.color)
            latest_detected[0] = (vis is not None)

            # 状态文字叠加
            n = cal.n_frames
            info = f"已采集: {n}/20  | 操作在终端中进行"
            cv2.putText(vis, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
            cv2.imshow("预览 - 操作请看终端", vis)
            cv2.waitKey(30)   # 仅用于刷新窗口，不用于捕获按键

            # 处理终端命令
            while cmd_queue:
                cmd = cmd_queue.pop(0)
                if cmd == "q":
                    stop_flag[0] = True
                    break
                else:  # Enter（空字符串）
                    f = latest_frame[0]
                    if f is None:
                        print("  等待相机帧，请重试")
                        continue
                    found = cal.add_frame(f.color)
                    if found:
                        fname = save_dir / f"frame_{frame_idx:04d}.png"
                        cv2.imwrite(str(fname), f.color)
                        frame_idx += 1
                        print(f"  [✓ {cal.n_frames:3d}] 已保存: {fname.name}")
                        if cal.n_frames >= 20:
                            print("  已达到 20 帧，可以继续或输入 q 退出")
                    else:
                        print("  [✗] 未检测到棋盘格，请调整角度后重试")

            if stop_flag[0]:
                break

    cv2.destroyAllWindows()
    print(f"\n共采集 {cal.n_frames} 帧，保存至 {save_dir}/")
    if cal.n_frames >= 15:
        print("帧数足够，运行以下命令计算内参：")
        print(f"  python scripts/run_intrinsic_calib.py")
    else:
        print(f"帧数不足（{cal.n_frames} < 15），建议继续采集")


if __name__ == "__main__":
    main()
