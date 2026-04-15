"""
内参标定 - 步骤2：从已保存图像计算内参

用法：
    cd /home/chlorine/seeed/cameraws
    python scripts/run_intrinsic_calib.py [图像目录]

默认从 data/intrinsic_frames/ 读取图像，结果保存至 config/calibration/intrinsics.npz
"""

import sys
import glob
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cameraws.config import load_config, resolve_calib_paths
from cameraws.calibration.intrinsic import CheckerboardCalibrator


def main():
    cfg = load_config("config/default.yaml")
    cfg = resolve_calib_paths(cfg)       # 按相机类型自动设置路径
    cb_cfg = cfg["calibration"]["checkerboard"]
    cam_type = cfg["camera"]["type"]

    default_img_dir = Path("data/intrinsic_frames") / cam_type
    img_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default_img_dir
    out_path = cfg["calibration"]["intrinsics_path"]

    image_paths = sorted(
        list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
    )
    if not image_paths:
        print(f"错误：{img_dir} 中未找到图像，请先运行 collect_intrinsic_frames.py")
        sys.exit(1)

    cal = CheckerboardCalibrator(
        board_cols=cb_cfg["cols"],
        board_rows=cb_cfg["rows"],
        square_size_m=cb_cfg["square_size_m"],
    )

    print(f"处理 {len(image_paths)} 张图像...")
    for p in image_paths:
        img = cv2.imread(str(p))
        found = cal.add_frame(img)
        print(f"  {'✓' if found else '✗'} {p.name}")

    print(f"\n成功检测: {cal.n_frames}/{len(image_paths)} 张")

    if cal.n_frames < 10:
        print("有效帧数太少（<10），请补充采集更多图像")
        sys.exit(1)

    result = cal.calibrate()
    CheckerboardCalibrator.save(result, out_path)

    print(f"\n标定结果：")
    print(f"  RMS 重投影误差: {result.rms_error:.4f} px  ({'良好' if result.rms_error < 0.5 else '偏高，建议重新采集'})")
    print(f"  fx={result.fx():.2f}  fy={result.fy():.2f}")
    print(f"  cx={result.cx():.2f}  cy={result.cy():.2f}")
    print(f"  图像尺寸: {result.image_size[0]}x{result.image_size[1]}")


if __name__ == "__main__":
    main()
