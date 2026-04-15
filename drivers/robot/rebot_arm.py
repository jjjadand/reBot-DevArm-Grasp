"""
cameraws.drivers.robot.rebot_arm
=================================
轻量包装层，将 reBotArm_control_py 的低层 API 封装为
相机感知系统需要的简洁接口：

    connect()         — 使能电机
    disconnect()      — 失能并关闭
    get_tcp_pose()    — 通过 FK 读取末端位姿 (4×4 T_gripper2base)
    move_to(x,y,z)    — 通过 IK + 轨迹控制器移动末端
    safe_home()       — 回零位

依赖：~/seeed/reBotArm_control_py（pip install -e .）
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


def _find_repo_root(hint: Optional[str] = None) -> Path:
    """查找 reBotArm_control_py 仓库根目录。"""
    candidates = []
    if hint:
        candidates.append(Path(hint).expanduser().resolve())
    candidates += [
        Path.home() / "seeed" / "reBotArm_control_py",
        Path("/home/chlorine/seeed/reBotArm_control_py"),
    ]
    for p in candidates:
        if (p / "reBotArm_control_py").is_dir():
            return p
    raise FileNotFoundError(
        "找不到 reBotArm_control_py 仓库，请在 config/default.yaml 中设置 "
        "robot.repo_root 或执行 pip install -e ~/seeed/reBotArm_control_py"
    )


class RebotArm:
    """
    相机感知系统 ↔ 机械臂接口。

    Args:
        config_path: robot.yaml 路径；None = 使用仓库默认
        urdf_path:   URDF 路径；None = 使用仓库默认
        repo_root:   reBotArm_control_py 仓库根目录；None = 自动搜索
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        urdf_path:   Optional[str] = None,
        repo_root:   Optional[str] = None,
    ) -> None:
        repo = _find_repo_root(repo_root)
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        from reBotArm_control_py.actuator import RobotArm
        from reBotArm_control_py.kinematics import (
            load_robot_model,
            compute_fk,
            get_end_effector_frame_id,
        )
        from reBotArm_control_py.controllers import ArmEndPos

        cfg = str(config_path) if config_path else None
        self._arm = RobotArm(cfg_path=cfg)

        if urdf_path:
            self._model = load_robot_model(urdf_path=str(urdf_path))
        else:
            self._model = load_robot_model()

        self._data = self._model.createData()
        self._ee_frame_id = get_end_effector_frame_id(self._model)
        self._compute_fk = compute_fk

        self._endpos_ctrl: Optional[ArmEndPos] = None
        self._ArmEndPos = ArmEndPos

        self._connected = False

    # ── 生命周期 ─────────────────────────────────────────────────────────────

    def connect(self, enable: bool = True) -> None:
        """连接机械臂。

        Args:
            enable: True = 使能电机（运动控制用）
                    False = 仅读取编码器，电机保持失能（只读模式）
        """
        self._arm.connect()
        if enable:
            self._arm.enable()
            time.sleep(0.5)
            self._endpos_ctrl = self._ArmEndPos(self._arm)
            self._endpos_ctrl.start()
            print("[RebotArm] 连接成功，电机已使能")
        else:
            self._arm._request_and_poll()
            print("[RebotArm] 连接成功，电机保持失能（只读模式）")
        self._connected = True

    def disconnect(self) -> None:
        """停止控制器，失能电机，关闭连接。"""
        if self._endpos_ctrl is not None:
            try:
                self._endpos_ctrl.end()
            except Exception:
                pass
            self._endpos_ctrl = None
        try:
            self._arm.disconnect()
        except Exception:
            pass
        self._connected = False
        print("[RebotArm] 已断开连接")

    # ── 状态读取 ─────────────────────────────────────────────────────────────

    def get_tcp_pose(self) -> np.ndarray:
        """通过正运动学读取当前末端位姿。

        Returns:
            T_gripper2base: (4, 4) 齐次变换矩阵
        """
        self._arm._request_and_poll()
        q, _, _ = self._arm.get_state()
        position, rotation, _ = self._compute_fk(self._model, q)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rotation
        T[:3,  3] = position
        return T

    # ── 运动控制 ─────────────────────────────────────────────────────────────

    def move_to(
        self,
        x: float,
        y: float,
        z: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        duration: float = 2.0,
    ) -> bool:
        """通过 IK + 轨迹规划将末端移动到目标位置。"""
        if self._endpos_ctrl is None:
            raise RuntimeError("未连接机械臂，请先调用 connect()")
        return bool(self._endpos_ctrl.move_to_traj(
            x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, duration=duration,
        ))

    def safe_home(self, duration: float = 3.0) -> None:
        """回零位（关节全部归零）。"""
        if self._endpos_ctrl is None:
            raise RuntimeError("未连接机械臂，请先调用 connect()")
        q_zero = np.zeros(self._arm.num_joints, dtype=np.float64)
        pos_zero, _, _ = self._compute_fk(self._model, q_zero)
        ok = self._endpos_ctrl.move_to_traj(
            x=float(pos_zero[0]), y=float(pos_zero[1]), z=float(pos_zero[2]),
            duration=duration,
        )
        if not ok:
            self._arm.mode_pos_vel()
            self._arm.pos_vel(q_zero)
            time.sleep(duration)

    # ── 上下文管理器 ─────────────────────────────────────────────────────────

    def __enter__(self) -> "RebotArm":
        self.connect()
        return self

    def __exit__(self, *args) -> None:
        self.disconnect()
