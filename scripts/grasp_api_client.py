#!/usr/bin/env python3
"""
grasp_api_client.py
===================
HTTP / curl 封装层，为 rebot_grasp 和 rebot_arm_service 提供简洁的 Python
调用接口，同时生成等效的 curl 命令行示例。

依赖
----
    pip install requests

使用 rebot_arm_service (FastAPI, 默认端口 8000)
------------------------------------------------
    from grasp_api_client import ArmServiceClient
    arm = ArmServiceClient(base_url="http://localhost:8000")
    arm.connect()
    arm.home()
    state = arm.get_state()
    arm.move_pose(x=0.3, y=0.0, z=0.3, roll=0.0, pitch=1.2, yaw=0.0, duration=3.0)
    arm.open_gripper()
    arm.close_gripper()
    arm.set_gripper_position(position=-2.0, vlim=3.0)

使用 rebot_grasp 的 grasp_web.py (自定义 HTTP, 默认端口 8000)
-------------------------------------------------------------
    from grasp_api_client import GraspWebClient
    web = GraspWebClient(base_url="http://localhost:8000")
    web.start_pipeline()          # 启动 graspnet 抓取
    web.set_target("bottle")      # 发送目标类别名称
    web.reset()                  # 复位请求
    web.run_grasp()              # 执行规划运动（预览）
    web.execute_grasp()           # 执行真实抓取
    web.base_jog(delta_deg=30)   # 控制单个底座电机
    web.get_stream()             # 获取 MJPEG 视频流
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import requests

# ── 默认端口 ──────────────────────────────────────────────────────────────────
DEFAULT_ARM_SERVICE_PORT = 8000
DEFAULT_GRASP_WEB_PORT = 8000


# ══════════════════════════════════════════════════════════════════════════════
#  rebot_arm_service (FastAPI) 客户端
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ArmServiceClient:
    """
    rebot_arm_service 的 Python 封装。

    参数
    ----
    base_url : str
        服务地址，例如 "http://localhost:8000"
    timeout : float
        HTTP 请求超时时间（秒），默认 30s
    """

    base_url: str = f"http://localhost:{DEFAULT_ARM_SERVICE_PORT}"
    timeout: float = 30.0
    _session: requests.Session = field(default_factory=requests.Session, repr=False)

    # ── 基础请求 ──────────────────────────────────────────────────────────────

    def _post(self, path: str, **kwargs) -> dict[str, Any]:
        resp = self._session.post(
            f"{self.base_url}{path}",
            timeout=kwargs.pop("timeout", self.timeout),
            **kwargs,
        )
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str, **kwargs) -> dict[str, Any]:
        resp = self._session.get(
            f"{self.base_url}{path}",
            timeout=kwargs.pop("timeout", self.timeout),
            **kwargs,
        )
        resp.raise_for_status()
        return resp.json()

    # ── 连接 / 断开 ────────────────────────────────────────────────────────────

    def connect(self, cfg_path: str | None = None) -> dict[str, Any]:
        """连接机械臂并使能电机。"""
        payload = {"cfg_path": cfg_path} if cfg_path else {}
        return self._post("/connect", json=payload)

    def disconnect(self, safe_home: bool = True) -> dict[str, Any]:
        """断开机械臂连接。"""
        return self._post("/disconnect")

    def home(self, vlim: float | None = None) -> dict[str, Any]:
        """回零位（关节归零）。"""
        payload = {"vlim": vlim} if vlim is not None else {}
        return self._post("/home", json=payload)

    def get_state(self) -> dict[str, Any]:
        """读取当前状态：关节角、末端位姿、电机使能状态等。"""
        return self._get("/state")

    def health(self) -> dict[str, Any]:
        """健康检查。"""
        return self._get("/healthz")

    # ── 运动控制 ──────────────────────────────────────────────────────────────

    def move_pose(
        self,
        *,
        x: float,
        y: float,
        z: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        mode: str = "ik",
        duration: float = 2.0,
    ) -> dict[str, Any]:
        """移动末端到指定位姿（IK 求解或轨迹规划）。

        参数
        ----
        x, y, z      : 末端目标位置（米）
        roll, pitch, yaw : 末端目标姿态（弧度）
        mode          : "ik" 或 "traj"（轨迹插值）
        duration      : 运动时长（秒），仅 traj 模式生效
        """
        return self._post(
            "/move/pose",
            json={
                "x": x,
                "y": y,
                "z": z,
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw,
                "mode": mode,
                "duration": duration,
            },
        )

    def move_joints(self, joints_rad: list[float]) -> dict[str, Any]:
        """直接控制 6 个关节角（弧度）。

        参数
        ----
        joints_rad : 长度为 6 的列表，分别对应 joint1 ~ joint6
        """
        return self._post("/move/joints", json={"joints_rad": joints_rad})

    # ── 夹爪控制 ──────────────────────────────────────────────────────────────

    def open_gripper(self, cfg_path: str | None = None) -> dict[str, Any]:
        """张开夹爪。"""
        payload: dict[str, Any] = {"action": "open"}
        if cfg_path:
            payload["cfg_path"] = cfg_path
        return self._post("/gripper", json=payload)

    def close_gripper(self, cfg_path: str | None = None) -> dict[str, Any]:
        """闭合夹爪。"""
        payload: dict[str, Any] = {"action": "close"}
        if cfg_path:
            payload["cfg_path"] = cfg_path
        return self._post("/gripper", json=payload)

    def set_gripper_position(
        self,
        position: float,
        vlim: float = 3.0,
        cfg_path: str | None = None,
    ) -> dict[str, Any]:
        """精确设置夹爪目标位置（弧度）。"""
        payload: dict[str, Any] = {
            "action": "set",
            "position": position,
            "vlim": vlim,
        }
        if cfg_path:
            payload["cfg_path"] = cfg_path
        return self._post("/gripper", json=payload)

    # ── IK / FK 仿真（不驱动真实机械臂）────────────────────────────────────────

    def solve_sim_ik(
        self,
        *,
        x: float,
        y: float,
        z: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        seed_joints_rad: list[float] | None = None,
    ) -> dict[str, Any]:
        """求解逆运动学，返回 6 个关节角（不实际移动机械臂）。"""
        payload: dict[str, Any] = {
            "x": x,
            "y": y,
            "z": z,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
        }
        if seed_joints_rad:
            payload["seed_joints_rad"] = seed_joints_rad
        return self._post("/sim/ik", json=payload)

    def compute_sim_fk(self, joints_rad: list[float]) -> dict[str, Any]:
        """正运动学：由关节角计算末端位姿（不实际移动机械臂）。"""
        return self._post("/sim/fk", json={"joints_rad": joints_rad})

    # ── 视觉抓取任务 ──────────────────────────────────────────────────────────

    def compute_grasp_pose_on_z(
        self,
        *,
        target_z: float = 0.03,
        line_z_shift_m: float = 0.05,
        offset_x_m: float = 0.0,
        offset_y_m: float = 0.0,
        offset_z_m: float = 0.0,
    ) -> dict[str, Any]:
        """计算抓取位姿：沿当前工具坐标系前向射线与 z=target_z 平面的交点。

        返回 grasp_pose (position_m, rpy_rad)，可直接传给 move_pose。
        """
        return self._get(
            "/mission/teddy/grasp-pose",
            params={
                "target_z": target_z,
                "line_z_shift_m": line_z_shift_m,
                "offset_x_m": offset_x_m,
                "offset_y_m": offset_y_m,
                "offset_z_m": offset_z_m,
            },
        )

    def get_stream(self) -> bytes:
        """获取摄像头 MJPEG 视频流（单帧 JPEG）。"""
        resp = self._session.get(
            f"{self.base_url}/camera/frame.jpg",
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.content

    # ── 便捷高级封装 ──────────────────────────────────────────────────────────

    def get_end_effector_pose(self) -> dict[str, float]:
        """快速获取末端位姿 dict，方便直接传给 move_pose。"""
        state = self.get_state()
        ee = state["end_effector"]
        return {
            "x": ee["position_m"][0],
            "y": ee["position_m"][1],
            "z": ee["position_m"][2],
            "roll": ee["rpy_rad"][0],
            "pitch": ee["rpy_rad"][1],
            "yaw": ee["rpy_rad"][2],
        }

    def move_to_ready(self, duration: float = 3.0) -> dict[str, Any]:
        """移动到默认就绪位（夹爪朝前、高度适中）。"""
        return self.move_pose(
            x=0.25,
            y=0.0,
            z=0.35,
            roll=0.0,
            pitch=1.2,
            yaw=0.0,
            mode="traj",
            duration=duration,
        )

    def execute_planned_motion(
        self,
        *,
        target_z: float = 0.03,
        line_z_shift_m: float = 0.05,
        forward_offset_m: float = 0.03,
        duration: float = 0.8,
    ) -> dict[str, Any]:
        """
        一句话执行规划运动：
        1. 读取当前末端位姿
        2. 计算抓取点
        3. 移动到抓取点并闭合夹爪
        """
        pose = self.get_end_effector_pose()
        grasp = self.compute_grasp_pose_on_z(
            target_z=target_z,
            line_z_shift_m=line_z_shift_m,
        )
        gp = grasp["grasp_pose"]
        move_result = self.move_pose(
            x=gp["position_m"][0],
            y=gp["position_m"][1],
            z=gp["position_m"][2],
            roll=gp["rpy_rad"][0],
            pitch=gp["rpy_rad"][1],
            yaw=gp["rpy_rad"][2],
            mode="traj",
            duration=duration,
        )
        self.close_gripper()
        return {
            "grasp_pose": grasp,
            "move_result": move_result,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  grasp_web.py (自定义 HTTP 服务器) 客户端
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class GraspWebClient:
    """
    grasp_web.py (rebot_grasp) 的 Python 封装。

    grasp_web.py 内置 HTTP 服务器，支持以下端点：
        GET  /              — Web UI
        GET  /state         — 当前状态
        GET  /robot/state   — 机械臂关节和末端位姿
        GET  /stream.mjpg   — MJPEG 视频流
        POST /start         — 触发一次 GraspNet 预览推理
        POST /target        — 设置目标类别
        POST /reset         — 复位请求
        POST /infer         — 刷新抓取点（不执行）
        POST /grasp         — 执行真实抓取
        POST /compensation  — 设置外参补偿
        POST /base_jog      — 控制底座电机（兼容旧接口）
        POST /joint/jog     — 控制单个关节相对运动
        POST /move/pose     — 末端位姿轨迹/IK 运动
        POST /move/joints   — 6 关节绝对运动
    """

    base_url: str = f"http://localhost:{DEFAULT_GRASP_WEB_PORT}"
    timeout: float = 60.0
    _session: requests.Session = field(default_factory=requests.Session, repr=False)

    # ── 基础请求 ──────────────────────────────────────────────────────────────

    def _post(self, path: str, **kwargs) -> dict[str, Any]:
        resp = self._session.post(
            f"{self.base_url}{path}",
            timeout=kwargs.pop("timeout", self.timeout),
            **kwargs,
        )
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str, **kwargs) -> dict[str, Any]:
        resp = self._session.get(
            f"{self.base_url}{path}",
            timeout=kwargs.pop("timeout", self.timeout),
            **kwargs,
        )
        resp.raise_for_status()
        return resp.json()

    # ── 核心 API ──────────────────────────────────────────────────────────────

    def start_pipeline(self) -> dict[str, Any]:
        """启动 graspnet 抓取流水线（启动检测、姿态估计线程）。"""
        return self._post("/start")

    def set_target(self, class_name: str) -> dict[str, Any]:
        """发送目标类别名称，设置后触发 GraspNet 更新。

        示例::

            client.set_target("bottle")
            client.set_target("cup")
            client.set_target("apple")
        """
        return self._post("/target", json={"class_name": class_name})

    def reset(self) -> dict[str, Any]:
        """复位请求：停止当前执行、重置状态。"""
        return self._post("/reset")

    def get_state(self) -> dict[str, Any]:
        """读取当前状态：目标类别、检测结果、抓取点信息、补偿值等。"""
        return self._get("/state")

    def get_robot_state(self) -> dict[str, Any]:
        """读取机械臂状态：关节角、末端位置/RPY、夹爪状态。"""
        return self._get("/robot/state")

    def run_grasp(self, execute: bool = False) -> dict[str, Any]:
        """刷新抓取点预览（execute=False）或执行真实抓取（execute=True）。

        等价于 POST /infer（预览）或 POST /grasp（执行）。
        """
        path = "/grasp" if execute else "/infer"
        return self._post(path)

    def execute_grasp(self) -> dict[str, Any]:
        """执行真实机器人抓取（需要 --enable-robot 启动 grasp_web.py）。"""
        return self._post("/grasp")

    def set_compensation(
        self,
        *,
        forward_m: float = 0.0,
        lateral_m: float = 0.0,
        vertical_m: float = 0.0,
        roll_deg: float = 0.0,
        pitch_deg: float = 0.0,
        yaw_deg: float = 0.0,
        camera_x_m: float = 0.0,
        camera_y_m: float = 0.0,
        camera_z_m: float = 0.0,
        camera_roll_deg: float = 0.0,
        camera_pitch_deg: float = 0.0,
        camera_yaw_deg: float = 0.0,
        base_x_m: float = 0.0,
        base_y_m: float = 0.0,
        base_z_m: float = 0.0,
        base_roll_deg: float = 0.0,
        base_pitch_deg: float = 0.0,
        base_yaw_deg: float = 0.0,
    ) -> dict[str, Any]:
        """设置外参补偿值（夹爪偏移 / 相机偏移 / 基座偏移）。"""
        return self._post(
            "/compensation",
            json={
                "forward_m": forward_m,
                "lateral_m": lateral_m,
                "vertical_m": vertical_m,
                "roll_deg": roll_deg,
                "pitch_deg": pitch_deg,
                "yaw_deg": yaw_deg,
                "camera_x_m": camera_x_m,
                "camera_y_m": camera_y_m,
                "camera_z_m": camera_z_m,
                "camera_roll_deg": camera_roll_deg,
                "camera_pitch_deg": camera_pitch_deg,
                "camera_yaw_deg": camera_yaw_deg,
                "base_x_m": base_x_m,
                "base_y_m": base_y_m,
                "base_z_m": base_z_m,
                "base_roll_deg": base_roll_deg,
                "base_pitch_deg": base_pitch_deg,
                "base_yaw_deg": base_yaw_deg,
            },
        )

    def base_jog(
        self,
        *,
        delta_deg: float = -30.0,
        duration_s: float = 2.5,
        safety_margin_deg: float = 5.0,
    ) -> dict[str, Any]:
        """控制单个底座电机（joint1）相对转动（兼容旧接口）。

        参数
        ----
        delta_deg         : 相对转动角度（度），负数走负方向
        duration_s        : 运动时长（秒）
        safety_margin_deg : 限位安全边距（度）
        """
        return self._post(
            "/base_jog",
            json={
                "delta_deg": delta_deg,
                "duration_s": duration_s,
                "safety_margin_deg": safety_margin_deg,
            },
        )

    def get_joint_limits(self) -> dict[str, Any]:
        """读取所有 6 个关节的当前角度和限位（度）。"""
        return self._get("/joint/limits")

    def jog_joint(
        self,
        *,
        joint: str = "joint1",
        delta_deg: float = -10.0,
        duration_s: float = 2.5,
        safety_margin_deg: float = 5.0,
    ) -> dict[str, Any]:
        """相对转动任意关节（推荐用此接口代替 base_jog）。

        参数
        ----
        joint             : 关节名称，"joint1" ~ "joint6"
        delta_deg         : 相对转动角度（度），负数走负方向
        duration_s        : 运动时长（秒）
        safety_margin_deg : 限位安全边距（度）

        示例::

            client.jog_joint(joint="joint1", delta_deg=-30)  # 底座转负30度
            client.jog_joint(joint="joint2", delta_deg=15)   # 肩关节转正15度
            client.jog_joint(joint="joint3", delta_deg=-5)   # 大臂转负5度
            client.jog_joint(joint="joint4", delta_deg=10)   # 小臂转正10度
            client.jog_joint(joint="joint5", delta_deg=20)   # 腕关节转正20度
            client.jog_joint(joint="joint6", delta_deg=-10)  # 腕转转负10度
        """
        return self._post(
            "/joint/jog",
            json={
                "joint": joint,
                "delta_deg": delta_deg,
                "duration_s": duration_s,
                "safety_margin_deg": safety_margin_deg,
            },
        )

    def move_joints(
        self,
        joints_rad: list[float],
        duration_s: float = 3.0,
    ) -> dict[str, Any]:
        """一次性设置所有 6 个关节的绝对位置（弧度）。

        参数
        ----
        joints_rad : 长度为 6 的列表，分别对应 joint1 ~ joint6
        duration_s : 运动时长（秒）

        示例::

            # 回零位
            client.move_joints([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], duration_s=3.0)
            # 移动到就绪位
            client.move_joints([0.0, -1.0, -1.5, 0.5, 0.0, 0.0], duration_s=3.0)
        """
        return self._post(
            "/move/joints",
            json={
                "joints_rad": joints_rad,
                "duration_s": duration_s,
            },
        )

    def move_pose(
        self,
        *,
        x: float,
        y: float,
        z: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        duration: float = 3.0,
        mode: str = "traj",
    ) -> dict[str, Any]:
        """Move TCP to a target pose.

        参数
        ----
        x, y, z    : 末端目标位置（米）
        roll, pitch, yaw : 末端目标姿态（弧度）
        duration   : 运动时长（秒）
        mode       : "traj" 轨迹规划，或 "ik" 即时 IK

        示例::

            # 移动到就绪位
            client.move_pose(x=0.25, y=0.0, z=0.35, roll=0.0, pitch=1.2, yaw=0.0, duration=3.0)
            # 移动到目标抓取位
            client.move_pose(x=0.3, y=0.1, z=0.2, roll=0.0, pitch=0.0, yaw=0.0, duration=2.0)
        """
        return self._post(
            "/move/pose",
            json={
                "x": x,
                "y": y,
                "z": z,
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw,
                "duration_s": duration,
                "mode": mode,
            },
        )

    # ── 视频流 ───────────────────────────────────────────────────────────────

    def get_stream_frame(self) -> bytes:
        """获取单帧 JPEG 图像（MJPEG 流中的最新帧）。"""
        resp = self._session.get(
            f"{self.base_url}/stream.mjpg",
            timeout=self.timeout,
            headers={"Accept": "image/jpeg"},
        )
        resp.raise_for_status()
        return resp.content

    def stream_mjpeg(self) -> Iterator[bytes]:
        """MJPEG 流迭代器，逐帧产出 JPEG bytes。

        示例用法::

            for jpeg_bytes in client.stream_mjpeg():
                with open("frame.jpg", "wb") as f:
                    f.write(jpeg_bytes)
        """
        with self._session.get(
            f"{self.base_url}/stream.mjpg",
            stream=True,
            timeout=self.timeout,
            headers={"Accept": "multipart/x-mixed-replace"},
        ) as resp:
            resp.raise_for_status()
            # MJPEG boundary = "--frame"
            import re

            boundary = b"--frame"
            payload_pattern = re.compile(
                rb"Content-Length: (\\d+)\\r\\n\\r\\n", re.DOTALL
            )
            buffer = b""
            for chunk in resp.iter_content(chunk_size=4096):
                buffer += chunk
                while boundary in buffer:
                    parts = buffer.split(boundary)
                    for part in parts[:-1]:
                        m = payload_pattern.search(part)
                        if m:
                            length = int(m.group(1))
                            start = m.end()
                            end = start + length
                            if len(part) >= end:
                                yield part[start:end]
                    buffer = parts[-1]

    # ── 便捷封装 ──────────────────────────────────────────────────────────────

    def poll_until_target_detected(
        self,
        target_class: str,
        timeout: float = 30.0,
        poll_interval: float = 1.0,
    ) -> dict[str, Any]:
        """等待目标被检测到（轮询 /state）。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.get_state()
            detections = state.get("detections", [])
            for det in detections:
                if det.get("class_name", "").lower() == target_class.lower():
                    return {"ok": True, "detection": det, "state": state}
            time.sleep(poll_interval)
        return {"ok": False, "error": f"timeout: {target_class} not detected"}
