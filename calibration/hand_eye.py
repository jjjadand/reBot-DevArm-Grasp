"""手眼标定 — 基于 OpenCV calibrateHandEye。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np


class CalibMode(Enum):
    EYE_IN_HAND = "eye_in_hand"   # 相机在末端，随末端运动
    EYE_TO_HAND = "eye_to_hand"   # 相机固定，观察末端


_METHOD_MAP = {
    "TSAI":       cv2.CALIB_HAND_EYE_TSAI,
    "PARK":       cv2.CALIB_HAND_EYE_PARK,
    "HORAUD":     cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF":    cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


@dataclass
class CalibResult:
    T_result: np.ndarray    # (4, 4) 手眼变换矩阵
    mode: str               # CalibMode.value
    n_samples: int
    method: str


@dataclass
class _Sample:
    T_gripper2base: np.ndarray   # (4, 4)
    T_marker2cam:   np.ndarray   # (4, 4)


class HandEyeCalibrator:
    """
    手眼标定器。

    Eye-in-Hand 模式：
        求解 T_cam2gripper，使得
            T_marker2base = T_gripper2base @ T_cam2gripper @ T_marker2cam
        在所有姿态下恒成立。

    Eye-to-Hand 模式：
        求解 T_cam2base，使得
            T_marker2base = T_cam2base @ T_marker2cam
        在所有姿态下恒成立。

    使用方法：
        calib = HandEyeCalibrator(CalibMode.EYE_IN_HAND)
        calib.add_sample(T_gripper2base, T_marker2cam)
        ...
        result = calib.calibrate()
        HandEyeCalibrator.save(result, "hand_eye.npz")
    """

    def __init__(
        self,
        mode: CalibMode = CalibMode.EYE_IN_HAND,
        method: str = "TSAI",
    ) -> None:
        self._mode = mode
        self._method = method.upper()
        self._samples: List[_Sample] = []

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    def add_sample(
        self,
        T_gripper2base: np.ndarray,
        T_marker2cam: np.ndarray,
    ) -> None:
        """
        添加一个标定样本。

        Args:
            T_gripper2base: (4,4) 末端到基座的变换（正运动学 FK 输出）
            T_marker2cam:   (4,4) 标记到相机的变换（ArUco 检测输出）
        """
        self._samples.append(_Sample(
            T_gripper2base=np.asarray(T_gripper2base, dtype=np.float64),
            T_marker2cam=np.asarray(T_marker2cam, dtype=np.float64),
        ))

    def calibrate(self, min_samples: int = 5) -> CalibResult:
        """
        计算手眼变换。

        Args:
            min_samples: 最少样本数（< 此值会抛出异常）

        Returns:
            CalibResult，T_result 即手眼变换矩阵
        """
        if self.n_samples < min_samples:
            raise ValueError(
                f"样本不足：{self.n_samples} < {min_samples}，请继续采集"
            )

        cv_method = _METHOD_MAP.get(self._method, cv2.CALIB_HAND_EYE_TSAI)

        if self._mode == CalibMode.EYE_IN_HAND:
            # OpenCV 接口：R_gripper2base, t_gripper2base, R_target2cam, t_target2cam
            R_g2b = [s.T_gripper2base[:3, :3] for s in self._samples]
            t_g2b = [s.T_gripper2base[:3,  3].reshape(3, 1) for s in self._samples]
            R_t2c = [s.T_marker2cam[:3, :3] for s in self._samples]
            t_t2c = [s.T_marker2cam[:3,  3].reshape(3, 1) for s in self._samples]

            R_c2g, t_c2g = cv2.calibrateHandEye(
                R_g2b, t_g2b, R_t2c, t_t2c, method=cv_method
            )
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R_c2g
            T[:3,  3] = t_c2g.flatten()

        else:  # EYE_TO_HAND
            # OpenCV eye-to-hand 约定：将两组输入互换
            #   第一位（名义上的 R_gripper2base）← 传 R_target2cam（ArUco）
            #   第二位（名义上的 R_target2cam）  ← 传 R_gripper2base（FK，直接用，不倒置）
            # 输出 R_cam2gripper / t_cam2gripper 在此场景下即 R_cam2base / t_cam2base
            R_t2c = [s.T_marker2cam[:3, :3] for s in self._samples]
            t_t2c = [s.T_marker2cam[:3,  3].reshape(3, 1) for s in self._samples]
            R_g2b = [s.T_gripper2base[:3, :3] for s in self._samples]
            t_g2b = [s.T_gripper2base[:3,  3].reshape(3, 1) for s in self._samples]

            R_c2b, t_c2b = cv2.calibrateHandEye(
                R_t2c, t_t2c, R_g2b, t_g2b, method=cv_method
            )
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R_c2b
            T[:3,  3] = t_c2b.flatten()

        return CalibResult(
            T_result=T,
            mode=self._mode.value,
            n_samples=self.n_samples,
            method=self._method,
        )

    def calibrate_point_cloud(self, min_samples: int = 4) -> CalibResult:
        """
        用点云配准（Kabsch/Procrustes）求 T_cam2base。

        不依赖末端姿态旋转多样性，只用 ArUco 检测到的 3D 位置
        （相机系）与 FK 末端位置（基座系）做点集对齐。

        假设：ArUco 标记近似在末端执行器原点处（T_marker2gripper ≈ 纯平移）。
        若标记有固定偏移，结果平移量会有对应误差，但旋转仍准确。

        Args:
            min_samples: 最少样本数

        Returns:
            CalibResult，T_result 即 T_cam2base
        """
        if self.n_samples < min_samples:
            raise ValueError(
                f"样本不足：{self.n_samples} < {min_samples}"
            )

        # 取出两组 3D 点
        P_cam  = np.array([s.T_marker2cam[:3, 3]   for s in self._samples])   # (N,3) 相机系
        P_base = np.array([s.T_gripper2base[:3, 3]  for s in self._samples])  # (N,3) 基座系

        # 去质心
        mu_cam  = P_cam.mean(axis=0)
        mu_base = P_base.mean(axis=0)
        Q_cam  = P_cam  - mu_cam
        Q_base = P_base - mu_base

        # 交叉协方差矩阵 H = Q_cam.T @ Q_base
        H = Q_cam.T @ Q_base   # (3,3)

        # SVD
        U, _, Vt = np.linalg.svd(H)

        # 修正反射（保证行列式为 +1）
        d = np.linalg.det(Vt.T @ U.T)
        D = np.diag([1.0, 1.0, d])
        R = Vt.T @ D @ U.T    # R_cam2base

        t = mu_base - R @ mu_cam   # t_cam2base

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3,  3] = t

        # 计算残差（均方根误差，单位 m）
        P_est = (R @ P_cam.T).T + t
        rmse = float(np.sqrt(((P_est - P_base) ** 2).sum(axis=1).mean()))
        print(f"  [点云配准] RMSE = {rmse*1000:.1f} mm")

        return CalibResult(
            T_result=T,
            mode=self._mode.value,
            n_samples=self.n_samples,
            method="POINT_CLOUD",
        )

    @staticmethod
    def save(result: CalibResult, path: Union[str, Path]) -> None:
        """保存标定结果为 .npz 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(path),
            T_result=result.T_result,
            mode=np.array([result.mode]),
            n_samples=np.array([result.n_samples]),
            method=np.array([result.method]),
        )

    @staticmethod
    def load(path: Union[str, Path]) -> CalibResult:
        """从 .npz 文件加载标定结果。"""
        data = np.load(str(path), allow_pickle=False)
        return CalibResult(
            T_result=data["T_result"],
            mode=str(data["mode"][0]),
            n_samples=int(data["n_samples"][0]),
            method=str(data["method"][0]) if "method" in data else "TSAI",
        )
