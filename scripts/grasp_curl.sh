#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${GRASP_BASE:-http://127.0.0.1:8000}"

usage() {
  cat <<'EOF'
Usage:
  scripts/grasp_curl.sh serve [extra grasp_web.py args...]
  scripts/grasp_curl.sh state
  scripts/grasp_curl.sh robot-state
  scripts/grasp_curl.sh target CLASS_NAME
  scripts/grasp_curl.sh start
  scripts/grasp_curl.sh infer
  scripts/grasp_curl.sh grasp
  scripts/grasp_curl.sh ready
  scripts/grasp_curl.sh reset
  scripts/grasp_curl.sh joint-limits
  scripts/grasp_curl.sh joint-jog JOINT DELTA_DEG [DURATION_S] [SAFETY_MARGIN_DEG]
  scripts/grasp_curl.sh joint-move-rad J1,J2,J3,J4,J5,J6 [DURATION_S]
  scripts/grasp_curl.sh joint-move-deg J1,J2,J3,J4,J5,J6 [DURATION_S]
  scripts/grasp_curl.sh move-pose X Y Z ROLL PITCH YAW [DURATION_S] [traj|ik]

Environment:
  GRASP_BASE=http://host:port      Default: http://127.0.0.1:8000
  GRASP_NUM_POINT=12000           Low-memory GraspNet point count for serve
  GRASP_CLOUD_CROP_NSAMPLE=32     Low-memory GraspNet CloudCrop samples for serve

Examples:
  scripts/grasp_curl.sh serve --enable-robot --no-auto-graspnet
  scripts/grasp_curl.sh target cup
  scripts/grasp_curl.sh start
  scripts/grasp_curl.sh grasp
  scripts/grasp_curl.sh reset
  scripts/grasp_curl.sh joint-jog joint1 -30 2.5 5
  scripts/grasp_curl.sh robot-state
  scripts/grasp_curl.sh move-pose 0.25 0.0 0.35 0.0 1.2 0.0 3.0 traj
EOF
}

post_json() {
  local path="$1"
  local body="${2:-{}}"
  curl -sS -X POST "${BASE_URL}${path}" \
    -H "Content-Type: application/json" \
    -d "${body}"
}

get_json() {
  local path="$1"
  curl -sS "${BASE_URL}${path}"
}

json_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

csv_to_json_array() {
  local csv="$1"
  local old_ifs="${IFS}"
  IFS=','
  read -r -a vals <<< "${csv}"
  IFS="${old_ifs}"
  if [[ "${#vals[@]}" -ne 6 ]]; then
    echo "expected 6 comma-separated joint values, got ${#vals[@]}" >&2
    exit 2
  fi
  local out="["
  local i
  for i in "${!vals[@]}"; do
    [[ "${i}" -gt 0 ]] && out+=","
    out+="${vals[$i]}"
  done
  out+="]"
  printf '%s' "${out}"
}

cmd="${1:-}"
if [[ -z "${cmd}" || "${cmd}" == "-h" || "${cmd}" == "--help" ]]; then
  usage
  exit 0
fi
shift || true

case "${cmd}" in
  serve)
    cd "$(dirname "$0")/.."
    exec python scripts/grasp_web.py \
      --host 0.0.0.0 \
      --port 8000 \
      --num-point "${GRASP_NUM_POINT:-12000}" \
      --cloud-crop-nsample "${GRASP_CLOUD_CROP_NSAMPLE:-32}" \
      "$@"
    ;;
  state)
    get_json "/state"
    ;;
  robot-state)
    get_json "/robot/state"
    ;;
  target)
    [[ "$#" -ge 1 ]] || { usage >&2; exit 2; }
    class_name="$(json_escape "$1")"
    post_json "/target" "{\"class_name\":\"${class_name}\"}"
    ;;
  start|infer)
    post_json "/infer" "{}"
    ;;
  grasp)
    post_json "/grasp" "{}"
    ;;
  ready)
    post_json "/ready" "{}"
    ;;
  reset)
    post_json "/reset" "{}"
    ;;
  joint-limits)
    get_json "/joint/limits"
    ;;
  joint-jog)
    [[ "$#" -ge 2 ]] || { usage >&2; exit 2; }
    joint="$1"
    delta_deg="$2"
    duration_s="${3:-2.5}"
    safety_margin_deg="${4:-5.0}"
    post_json "/joint/jog" \
      "{\"joint\":\"$(json_escape "${joint}")\",\"delta_deg\":${delta_deg},\"duration_s\":${duration_s},\"safety_margin_deg\":${safety_margin_deg}}"
    ;;
  joint-move-rad)
    [[ "$#" -ge 1 ]] || { usage >&2; exit 2; }
    joints="$(csv_to_json_array "$1")"
    duration_s="${2:-3.0}"
    post_json "/move/joints" "{\"joints_rad\":${joints},\"duration_s\":${duration_s}}"
    ;;
  joint-move-deg)
    [[ "$#" -ge 1 ]] || { usage >&2; exit 2; }
    joints="$(csv_to_json_array "$1")"
    duration_s="${2:-3.0}"
    post_json "/move/joints" "{\"joints_deg\":${joints},\"duration_s\":${duration_s}}"
    ;;
  move-pose)
    [[ "$#" -ge 6 ]] || { usage >&2; exit 2; }
    x="$1"; y="$2"; z="$3"; roll="$4"; pitch="$5"; yaw="$6"
    duration_s="${7:-3.0}"
    mode="${8:-traj}"
    post_json "/move/pose" \
      "{\"x\":${x},\"y\":${y},\"z\":${z},\"roll\":${roll},\"pitch\":${pitch},\"yaw\":${yaw},\"duration_s\":${duration_s},\"mode\":\"$(json_escape "${mode}")\"}"
    ;;
  *)
    echo "unknown command: ${cmd}" >&2
    usage >&2
    exit 2
    ;;
esac
