"""
生成 ArUco 标记图片并保存为 PNG，用于手眼标定。

用法：
    cd /home/chlorine/seeed/cameraws
    python scripts/gen_aruco.py

输出：aruco_marker_0.png（默认 DICT_4X4_50，ID=0）
打印建议：
  - 实际打印尺寸 = config 中的 marker_length_m（默认 5cm）
  - 打印后用尺子量实际边长，与配置对齐
  - 贴在平整硬纸板或薄金属板上，避免卷曲反光
"""

import sys
import cv2
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    root = Path(__file__).resolve().parent.parent
    cfg_path = root / "config" / "default.yaml"

    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    aruco_cfg = cfg["calibration"]["aruco"]
    dict_id = aruco_cfg.get("dict_id", 0)
    marker_id = aruco_cfg.get("target_marker_id") or 0
    marker_length_m = aruco_cfg.get("marker_length_m", 0.05)

    # 生成图像（600px，无外侧白边，便于打印对齐）
    px = 600
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, px)

    out_path = root / f"aruco_marker_{marker_id}.png"
    cv2.imwrite(str(out_path), marker_img)

    print(f"已生成: {out_path}")
    print(f"  字典: DICT_4X4_50 (dict_id={dict_id})")
    print(f"  标记 ID: {marker_id}")
    print(f"  配置边长: {marker_length_m*100:.1f} cm")
    print()
    print("打印提示：")
    print(f"  将图片打印后，实际边长（黑色边框外沿）应为 {marker_length_m*100:.1f} cm")
    print("  打印后用尺子量实际尺寸，如有偏差请修改 config/default.yaml 中的 marker_length_m")


if __name__ == "__main__":
    main()
