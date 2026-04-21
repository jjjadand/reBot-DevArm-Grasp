"""坐标变换工具函数。"""
import numpy as np


_ROT_X_PI = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)


def _nearest_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """Project a near-rotation matrix onto SO(3)."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"rotation matrix must be (3, 3), got {R.shape}")

    if not np.all(np.isfinite(R)):
        raise ValueError("rotation matrix contains non-finite values")

    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0.0:
        U[:, -1] *= -1.0
        R_ortho = U @ Vt
    return R_ortho.astype(np.float64)


def pose6d_to_mat4(x, y, z, rx, ry, rz, degrees=False) -> np.ndarray:
    """
    将 6D 位姿 (平移 + ZYX 内旋欧拉角) 转换为 4×4 齐次变换矩阵。

    Args:
        x, y, z: 平移 (米)
        rx, ry, rz: 欧拉角，ZYX 内旋约定 (roll=rx around X, pitch=ry around Y, yaw=rz around Z)
        degrees: True 时输入为度，False 时为弧度

    Returns:
        T: (4, 4) numpy array
    """
    if degrees:
        rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)

    # 绕 X 轴
    Rx = np.array([
        [1,          0,           0],
        [0,  np.cos(rx), -np.sin(rx)],
        [0,  np.sin(rx),  np.cos(rx)],
    ])
    # 绕 Y 轴
    Ry = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [          0, 1,          0],
        [-np.sin(ry), 0, np.cos(ry)],
    ])
    # 绕 Z 轴
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [         0,           0, 1],
    ])

    # ZYX 内旋 = R = Rz @ Ry @ Rx
    R = Rz @ Ry @ Rx

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = [x, y, z]
    return T


def quat_to_mat4(x, y, z, qx, qy, qz, qw) -> np.ndarray:
    """
    将平移 + 四元数转换为 4×4 齐次变换矩阵。

    Args:
        x, y, z: 平移 (米)
        qx, qy, qz, qw: 四元数 (Hamilton 约定)

    Returns:
        T: (4, 4) numpy array
    """
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    R = np.array([
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [  2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),   2*(qy*qz - qx*qw)],
        [  2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = [x, y, z]
    return T


def mat4_to_pose6d(T: np.ndarray) -> tuple:
    """
    将 4×4 齐次变换矩阵转换为 (x, y, z, rx, ry, rz)，ZYX 内旋约定，弧度。
    """
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    rpy = rotation_matrix_to_euler_zyx(T[:3, :3])
    return float(x), float(y), float(z), float(rpy[0]), float(rpy[1]), float(rpy[2])


def rotation_matrix_to_euler_zyx(R: np.ndarray) -> np.ndarray:
    """将旋转矩阵转换为 ZYX 内旋欧拉角 (roll, pitch, yaw)。"""
    R = _nearest_rotation_matrix(R)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0.0
    return np.array([rx, ry, rz], dtype=np.float64)


def canonicalize_parallel_gripper_tcp_rotation(R: np.ndarray) -> np.ndarray:
    """规范并联夹爪 TCP 姿态的等效 180 度扭转。

    对于对称的并联夹爪，沿工具 X 轴旋转 180 度通常是抓取等效的。
    这里在 ``R`` 和 ``R @ Rx(pi)`` 之间选择 roll 绝对值更小的那一支，
    让输出的 RPY 更贴近人的直觉，也更方便调试。
    """
    R = _nearest_rotation_matrix(R)
    alt = R @ _ROT_X_PI

    roll = float(rotation_matrix_to_euler_zyx(R)[0])
    alt_roll = float(rotation_matrix_to_euler_zyx(alt)[0])
    return alt if abs(alt_roll) < abs(roll) else R


def grasp_axes_to_rebot_tcp_rotation(
    grip_axis: np.ndarray,
    open_axis: np.ndarray,
    approach_axis: np.ndarray,
) -> np.ndarray:
    """将抓取坐标系映射到 reBotArm 的 TCP 坐标系。

    视觉抓取结果约定：
      - X = grip_axis
      - Y = open_axis
      - Z = approach_axis

    reBotArm 末端期望：
      - X = 工具前向 / 接近方向
      - Y = 夹爪开合方向
      - Z = 由右手系补齐
    """
    grip = np.asarray(grip_axis, dtype=np.float64)
    open_vec = np.asarray(open_axis, dtype=np.float64)
    approach = np.asarray(approach_axis, dtype=np.float64)

    grip /= max(np.linalg.norm(grip), 1e-8)
    open_vec /= max(np.linalg.norm(open_vec), 1e-8)
    approach /= max(np.linalg.norm(approach), 1e-8)

    # tcp_x = tool-forward = approach direction pointing INTO the object (downward in base).
    # plane.normal points toward the camera (upward), so negate it here.
    tcp_x = -approach
    tcp_y = open_vec - float(np.dot(open_vec, tcp_x)) * tcp_x
    tcp_y /= max(np.linalg.norm(tcp_y), 1e-8)
    tcp_z = np.cross(tcp_x, tcp_y)
    tcp_z /= max(np.linalg.norm(tcp_z), 1e-8)

    # 期望 tcp_z 与 grip 同向（取反后方向一致）。
    if float(np.dot(tcp_z, grip)) < 0.0:
        tcp_y = -tcp_y
        tcp_z = -tcp_z

    R = np.column_stack([tcp_x, tcp_y, tcp_z]).astype(np.float64)
    if np.linalg.det(R) < 0.0:
        R[:, 2] *= -1.0
    return R


def grasp_rotation_to_rebot_tcp_rotation(grasp_rotation: np.ndarray) -> np.ndarray:
    """将 [grip, open, approach] 旋转矩阵转换为 reBotArm TCP 旋转矩阵。"""
    R = np.asarray(grasp_rotation, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"grasp_rotation 必须为 (3, 3)，实际为 {R.shape}")
    return grasp_axes_to_rebot_tcp_rotation(R[:, 0], R[:, 1], R[:, 2])
