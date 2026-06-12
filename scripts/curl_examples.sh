#!/usr/bin/env bash
# =============================================================================
# curl_examples.sh — grasp_web.py curl 命令速查表
# =============================================================================
#
# 服务启动:
#   cd ~/Downloads/rebot_grasp-jetson
#   conda activate graspnet
#   python scripts/grasp_web.py --host 0.0.0.0 --port 8000 --enable-robot --num-point 12000 --cloud-crop-nsample 32
#
# Web UI: http://localhost:8000
# MJPEG 流: http://localhost:8000/stream.mjpg
#
# =============================================================================

set -euo pipefail

BASE="${GRASP_BASE:-http://localhost:8000}"

BOLD="\033[1m"
CYAN="\033[36m"
GREEN="\033[32m"
YELLOW="\033[33m"
NC="\033[0m"

section() {
    echo ""
    echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo ""
}

subsection() {
    echo ""
    echo -e "${BOLD}── $1 ──${NC}"
    echo ""
}

cmd() {
    echo -e "  ${GREEN}$1${NC}"
}

note() {
    echo -e "  ${YELLOW}# $1${NC}"
}

# ══════════════════════════════════════════════════════════════════════════════
section "grasp_web.py — curl 速查表"
# ══════════════════════════════════════════════════════════════════════════════

note "前提条件: python scripts/grasp_web.py --host 0.0.0.0 --port 8000 --enable-robot --num-point 12000 --cloud-crop-nsample 32"
note "BASE = $BASE (可通过 export GRASP_BASE=http://host:port 自定义)"

# ── 状态查询 ─────────────────────────────────────────────────────────────────

subsection "状态查询"
cmd "curl -s \"$BASE/state\""
note "读取当前状态：目标类别、检测结果、抓取点信息。"
cmd ""
cmd "curl -s \"$BASE/robot/state\""
note "读取机器人状态：关节角度、末端位姿、夹爪状态（位置/速度/力矩）。"
cmd ""
cmd "curl -s \"$BASE/joint/limits\""
note "读取所有 6 个关节的当前角度和限位（度）。"

# ── 基础操作 ──────────────────────────────────────────────────────────────────

subsection "基础操作"
cmd "curl -s -X POST \"$BASE/infer\" -H \"Content-Type: application/json\" -d \"{}\""
note "启动推理：触发一次 GraspNet 抓取点推理（不会执行机械臂运动）。"
cmd ""
cmd "curl -s -X POST \"$BASE/reset\" -H \"Content-Type: application/json\" -d \"{}\""
note "复位：停止当前执行、松开夹爪、机械臂回原点。"
cmd ""
cmd "curl -s -X POST \"$BASE/grasp\" -H \"Content-Type: application/json\" -d \"{}\""
note "执行真实机器人抓取（需要 --enable-robot 启动）。"
note "抓取流程: 就绪位 -> 计算抓取点 -> 张开夹爪 -> 移动到抓取位 -> 闭合夹爪 -> 抬起"

# ── 目标类别 ──────────────────────────────────────────────────────────────────

subsection "目标类别设置"
cmd "curl -s -X POST \"$BASE/target\" -H \"Content-Type: application/json\" -d '{\"class_name\":\"bottle\"}'"
note "设置目标类别为 bottle（设置后自动触发推理更新）。"
cmd ""
cmd "curl -s -X POST \"$BASE/target\" -H \"Content-Type: application/json\" -d '{\"class_name\":\"\"}'"
note "传空字符串取消类别过滤，扫描全场所有检测到的物体。"

# ── 就绪位 ───────────────────────────────────────────────────────────────────

subsection "就绪位"
cmd "curl -s -X POST \"$BASE/ready\" -H \"Content-Type: application/json\" -d \"{}\""
note "移动机械臂到就绪位（张开夹爪 + 移动到预定义就绪姿态）。"

# ── 外参补偿 ──────────────────────────────────────────────────────────────────

subsection "外参补偿设置"
cmd "curl -s -X POST \"$BASE/compensation\" -H \"Content-Type: application/json\" -d '{\"forward_m\":0.01,\"lateral_m\":0.0,\"vertical_m\":0.0,\"roll_deg\":0.0,\"pitch_deg\":0.0,\"yaw_deg\":0.0}'"
note "夹爪偏移补偿（米/度）：forward=前后，lateral=左右，vertical=上下，roll/pitch/yaw=姿态。"
cmd ""
cmd "curl -s -X POST \"$BASE/compensation\" -H \"Content-Type: application/json\" -d '{\"camera_x_m\":0.005,\"camera_y_m\":0.0,\"camera_z_m\":0.0}'"
note "相机外参补偿（米）：修正相机安装误差。"
cmd ""
cmd "curl -s -X POST \"$BASE/compensation\" -H \"Content-Type: application/json\" -d '{\"base_x_m\":0.0,\"base_y_m\":0.0,\"base_z_m\":0.0}'"
note "基座外参补偿（米）：修正机器人基座标定误差。"
cmd ""
cmd "curl -s -X POST \"$BASE/offset\" -H \"Content-Type: application/json\" -d '{\"offset_m\":0.05}'"
note "前向偏移（米）：沿工具坐标系前向方向额外移动的距离。"

# ── 夹爪控制 ─────────────────────────────────────────────────────────────────

subsection "夹爪控制"
cmd "curl -s -X POST \"$BASE/gripper\" -H \"Content-Type: application/json\" -d '{\"action\":\"state\"}'"
note "读取夹爪状态：位置（弧度/度）、速度、力矩、是否夹住。"
cmd ""
cmd "curl -s -X POST \"$BASE/gripper\" -H \"Content-Type: application/json\" -d '{\"action\":\"open\",\"distance_m\":0.09}'"
note "张开夹爪到指定距离（米），默认 0.09m。"
cmd ""
cmd "curl -s -X POST \"$BASE/gripper\" -H \"Content-Type: application/json\" -d '{\"action\":\"close\"}'"
note "闭合夹爪（非阻塞力矩模式）。"
cmd ""
cmd "curl -s -X POST \"$BASE/gripper\" -H \"Content-Type: application/json\" -d '{\"action\":\"release\"}'"
note "释放夹爪（松开，解除 HOLDING 状态）。"

# ── 关节控制 ─────────────────────────────────────────────────────────────────

subsection "关节控制 — 单关节点动"
cmd "curl -s -X POST \"$BASE/joint/jog\" -H \"Content-Type: application/json\" -d '{\"joint\":\"joint1\",\"delta_deg\":-30,\"duration_s\":2.5,\"safety_margin_deg\":5.0}'"
note "joint1 (底座) 负方向旋转 30 度。"
cmd ""
cmd "curl -s -X POST \"$BASE/joint/jog\" -H \"Content-Type: application/json\" -d '{\"joint\":\"joint2\",\"delta_deg\":-15,\"duration_s\":2.0,\"safety_margin_deg\":5.0}'"
note "joint2 (肩) 旋转 -15 度。"
cmd ""
cmd "curl -s -X POST \"$BASE/joint/jog\" -H \"Content-Type: application/json\" -d '{\"joint\":\"joint3\",\"delta_deg\":-10,\"duration_s\":2.0,\"safety_margin_deg\":5.0}'"
note "joint3 (大臂) 旋转 -10 度。"
cmd ""
cmd "curl -s -X POST \"$BASE/joint/jog\" -H \"Content-Type: application/json\" -d '{\"joint\":\"joint4\",\"delta_deg\":10,\"duration_s\":2.0,\"safety_margin_deg\":5.0}'"
note "joint4 (小臂) 旋转 +10 度。"
cmd ""
cmd "curl -s -X POST \"$BASE/joint/jog\" -H \"Content-Type: application/json\" -d '{\"joint\":\"joint5\",\"delta_deg\":20,\"duration_s\":2.0,\"safety_margin_deg\":5.0}'"
note "joint5 (腕) 旋转 +20 度。"
cmd ""
cmd "curl -s -X POST \"$BASE/joint/jog\" -H \"Content-Type: application/json\" -d '{\"joint\":\"joint6\",\"delta_deg\":-10,\"duration_s\":2.0,\"safety_margin_deg\":5.0}'"
note "joint6 (腕转) 旋转 -10 度。"
cmd ""
note "参数: joint (joint1~joint6), delta_deg (度), duration_s (秒), safety_margin_deg (限位边距)"
cmd ""
subsection "关节控制 — 全部关节一次性移动"
cmd "curl -s -X POST \"$BASE/joint/move\" -H \"Content-Type: application/json\" -d '{\"joints_rad\":[0.0,0.0,0.0,0.0,0.0,0.0],\"duration_s\":3.0}'"
note "一次性设置所有关节绝对位置（弧度）。关节顺序: [j1,j2,j3,j4,j5,j6]。回零位。"
cmd ""
cmd "curl -s -X POST \"$BASE/joint/move\" -H \"Content-Type: application/json\" -d '{\"joints_rad\":[0.0,-1.0,-1.5,0.5,0.0,0.0],\"duration_s\":3.0}'"
note "一次性设置所有关节绝对位置（弧度）。就绪位示例。"

# ── 末端位姿控制 ─────────────────────────────────────────────────────────────

subsection "末端位姿控制（IK 规划）"
note "先用 /ready 或 /joint/move 将机械臂移到可达位置，再使用 move/pose。"
note "当前机械臂实际末端位置（从 /robot/state 获取）: x≈0.260, y≈0, z≈0.192"
cmd ""
cmd "curl -s -X POST \"$BASE/move/pose\" -H \"Content-Type: application/json\" -d '{\"x\":0.25,\"y\":0.0,\"z\":0.35,\"roll\":0.0,\"pitch\":1.2,\"yaw\":0.0,\"duration\":3.0}'"
note "移动到就绪位。参数: x,y,z (米) roll,pitch,yaw (弧度) duration (秒)。需先处于可达状态。"
cmd ""
cmd "curl -s -X POST \"$BASE/move/pose\" -H \"Content-Type: application/json\" -d '{\"x\":0.260,\"y\":0.0,\"z\":0.20,\"roll\":0.0,\"pitch\":0.0,\"yaw\":0.0,\"duration\":2.0}'"
note "在当前位置附近微调末端位姿。从零位出发可达的保守目标位。"

# ── 底座点动 ─────────────────────────────────────────────────────────────────

subsection "底座点动"
cmd "curl -s -X POST \"$BASE/base_jog\" -H \"Content-Type: application/json\" -d '{\"delta_deg\":30,\"duration_s\":2.0}'"
note "底座旋转 +30 度。"
cmd ""
cmd "curl -s -X POST \"$BASE/base_jog\" -H \"Content-Type: application/json\" -d '{\"delta_deg\":-30,\"duration_s\":2.0}'"
note "底座旋转 -30 度。"

# ── 完整流程示例 ─────────────────────────────────────────────────────────────

subsection "完整 GraspNet 抓取流程示例"
note "1. 先将机械臂移到可达位置（从零位出发需先做这一步）"
cmd "curl -s -X POST \"$BASE/ready\" -H \"Content-Type: application/json\" -d \"{}\""
note "2. 查看当前状态"
cmd "curl -s \"$BASE/state\""
note "3. 设置目标类别"
cmd "curl -s -X POST \"$BASE/target\" -H \"Content-Type: application/json\" -d '{\"class_name\":\"bottle\"}'"
note "4. 刷新抓取点（预览）"
cmd "curl -s -X POST \"$BASE/infer\" -H \"Content-Type: application/json\" -d \"{}\""
note "5. 执行真实抓取"
cmd "curl -s -X POST \"$BASE/grasp\" -H \"Content-Type: application/json\" -d \"{}\""
note "6. 复位"
cmd "curl -s -X POST \"$BASE/reset\" -H \"Content-Type: application/json\" -d \"{}\""

# ── 视频流 ────────────────────────────────────────────────────────────────────

subsection "视频流"
cmd "curl -s \"$BASE/stream.mjpg\" --max-time 1 -o stream_part.mjpg"
note "获取 MJPEG 流中的一小段（可用于测试视频流是否正常）。"

echo ""
