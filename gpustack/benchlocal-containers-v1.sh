#!/usr/bin/env bash
# Host-side lifecycle for the three BenchLocal verifier containers.
# Usage: benchlocal-containers-v1.sh start|stop|status|sync [--replace]
#
# start  creates/uses an internal Docker network with automatic IPAM, starts
#        the three verifier containers without GPU access, discovers the
#        network gateway and container IPs, and writes $DATA_ROOT/benchlocal.env.
# stop   stops (does not remove) the three containers; their IPs persist.
# sync   follows the panel: panel down or benchlocal.enabled=0 -> stop;
#        panel up and enabled -> start. Intended for a systemd timer.
#
# The panel reads benchlocal.env from SM75_BENCH_VERIFIER_CONFIG_FILE. The
# panel writes benchlocal.enabled into SM75_BENCH_VERIFIER_STATE_DIR.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
VERSION=""
if [ -f "$REPO_ROOT/docker/VERSION" ]; then
  VERSION=$(tr -d ' \r\n' < "$REPO_ROOT/docker/VERSION")
fi
IMAGE=${IMAGE:-}
if [ -z "$IMAGE" ]; then
  [ -n "$VERSION" ] || { echo "红：未设置 IMAGE，且找不到 docker/VERSION" >&2; exit 2; }
  IMAGE="vllm-sm75-gpustack:v${VERSION}-v7"
fi
NETWORK=${BENCHLOCAL_NETWORK:-sm75-bench-verifiers}
GPUSTACK_DATA_ROOT=${GPUSTACK_DATA_ROOT:-}
BENCHLOCAL_DATA_ROOT=${BENCHLOCAL_DATA_ROOT:-}
if [ -z "$BENCHLOCAL_DATA_ROOT" ]; then
  [ -n "$GPUSTACK_DATA_ROOT" ] || {
    echo "红：请设置 GPUSTACK_DATA_ROOT（宿主挂到 /var/lib/gpustack 的目录）或 BENCHLOCAL_DATA_ROOT" >&2
    exit 2
  }
  BENCHLOCAL_DATA_ROOT="$GPUSTACK_DATA_ROOT/benchlocal"
fi
DATA_ROOT=$BENCHLOCAL_DATA_ROOT
CONFIG=${BENCHLOCAL_CONFIG:-$DATA_ROOT/benchlocal.env}
FLAG=${BENCHLOCAL_FLAG:-$DATA_ROOT/benchlocal.enabled}
PANEL_PORT=${SM75_CONSOLE_PORT:-1615}
PANEL_URL=${SM75_PANEL_URL:-http://127.0.0.1:$PANEL_PORT/}
ROLES=(bugfind-15 cli-40 hermesagent-20)
NAMES=(sm75-bench-bugfind sm75-bench-cli sm75-bench-hermes)
ACTION=${1:-status}
REPLACE=0
[ "${2:-}" = "--replace" ] && REPLACE=1
QUIET=0
START_ENV=""
trap 'if [ -n "${START_ENV:-}" ]; then rm -f "$START_ENV"; fi' EXIT

case "$ACTION" in
  start|stop|status|sync) ;;
  *) echo "Usage: $0 start|stop|status|sync [--replace]" >&2; exit 2 ;;
esac
case "$DATA_ROOT" in
  /*) ;;
  *) echo "红：BENCHLOCAL_DATA_ROOT 必须是绝对路径" >&2; exit 2 ;;
esac
[ "$DATA_ROOT" != "/" ] || { echo "红：BENCHLOCAL_DATA_ROOT 不能是 /" >&2; exit 2; }

if [ ! -d "$DATA_ROOT" ]; then
  mkdir -p "$DATA_ROOT"
  chmod 700 "$DATA_ROOT"
fi

need_docker() {
  command -v docker >/dev/null 2>&1 || { echo "红：PATH 上没有 docker" >&2; exit 2; }
}

need_image() {
  docker image inspect "$IMAGE" >/dev/null 2>&1 || {
    echo "红：镜像不存在：$IMAGE（先构建 v7）" >&2
    exit 2
  }
  [ "$(docker image inspect --format '{{index .Config.Labels "io.sm75.benchlocal.unified"}}' "$IMAGE")" = "1" ] || {
    echo "红：镜像缺少 io.sm75.benchlocal.unified=1：$IMAGE" >&2
    exit 2
  }
}

ensure_network() {
  if ! docker network inspect "$NETWORK" >/dev/null 2>&1; then
    docker network create --internal --driver bridge \
      --label io.sm75.benchlocal.role=network "$NETWORK" >/dev/null
  fi
}

network_gateway() {
  docker network inspect "$NETWORK" \
    --format '{{range .IPAM.Config}}{{.Gateway}}{{"\n"}}{{end}}' | sed -n '1p'
}

container_exists() { docker container inspect "$1" >/dev/null 2>&1; }
container_running() {
  [ "$(docker inspect --format '{{.State.Running}}' "$1" 2>/dev/null || true)" = "true" ]
}
container_ip() {
  docker inspect "$1" --format "{{(index .NetworkSettings.Networks \"$NETWORK\").IPAddress}}"
}
container_env() {
  docker inspect "$1" --format '{{range .Config.Env}}{{println .}}{{end}}' \
    | sed -n "s/^$2=//p" | sed -n '1p'
}
health_ok() {
  curl -fsS --max-time 3 "http://$1:4200/health" 2>/dev/null | grep -q '"ok":true'
}

read_config_key() {
  [ -f "$CONFIG" ] || return 0
  sed -n 's/^SM75_BENCH_VERIFIER_KEY=//p' "$CONFIG" | sed -n '1p'
}

valid_key() {
  case "$1" in
    ""|*[!0-9a-fA-F]*) return 1 ;;
  esac
  [ "${#1}" -eq 64 ]
}

write_config() {
  local key=$1 gateway=$2 verifiers=$3 tmp
  tmp=$(mktemp "$DATA_ROOT/.benchlocal.env.XXXXXX")
  chmod 600 "$tmp"
  {
    printf 'SM75_BENCH_VERIFIER_KEY=%s\n' "$key"
    printf 'SM75_BENCH_VERIFIERS=%s\n' "$verifiers"
    printf 'SM75_BENCH_PROXY_BASE=http://%s:%s/benchlocal-inference/v1\n' "$gateway" "$PANEL_PORT"
  } > "$tmp"
  mv "$tmp" "$CONFIG"
  chmod 600 "$CONFIG"
}

log() { [ "$QUIET" = 1 ] || printf '%s\n' "$*"; }

do_start() {
  need_docker
  need_image
  ensure_network

  local gateway key name role origin ip have_key have_origin verifiers=""
  gateway=$(network_gateway)
  [ -n "$gateway" ] || {
    echo "红：网络 $NETWORK 没有网关，无法让 verifier 回连面板" >&2
    exit 2
  }

  key=$(read_config_key || true)
  if [ -z "$key" ]; then
    for name in "${NAMES[@]}"; do
      container_exists "$name" || continue
      key=$(container_env "$name" SM75_BENCH_VERIFIER_KEY || true)
      [ -n "$key" ] && break
    done
  fi
  if [ -z "$key" ]; then
    key=$(od -An -N32 -tx1 /dev/urandom | tr -d ' \n')
  fi
  valid_key "$key" || {
    echo "红：验签 key 不是 64 位十六进制；请检查 $CONFIG" >&2
    exit 2
  }

  START_ENV=$(mktemp "$DATA_ROOT/.benchlocal.start.XXXXXX")
  chmod 600 "$START_ENV"
  printf 'SM75_BENCH_VERIFIER_KEY=%s\n' "$key" > "$START_ENV"
  origin="http://$gateway:$PANEL_PORT"

  for i in "${!ROLES[@]}"; do
    role=${ROLES[$i]}
    name=${NAMES[$i]}
    if container_exists "$name"; then
      [ "$(docker inspect --format '{{index .Config.Labels "io.sm75.benchlocal.role"}}' "$name")" = "verifier" ] || {
        echo "红：$name 不是受管 verifier" >&2
        exit 2
      }
      have_key=$(container_env "$name" SM75_BENCH_VERIFIER_KEY || true)
      have_origin=$(container_env "$name" SM75_BENCH_PROXY_ORIGIN || true)
      if [ "$have_key" != "$key" ] || [ "$have_origin" != "$origin" ] \
        || [ "$(docker inspect --format '{{.Image}}' "$name")" != "$(docker image inspect --format '{{.Id}}' "$IMAGE")" ]; then
        [ "$REPLACE" = 1 ] || {
          echo "红：$name 的 key/origin/镜像与当前配置不一致；用 --replace 重建" >&2
          exit 2
        }
        docker rm -f "$name" >/dev/null
      elif ! container_running "$name"; then
        docker start "$name" >/dev/null
      fi
    fi

    if ! container_exists "$name"; then
      docker run -d --name "$name" --network "$NETWORK" --runtime runc --restart unless-stopped \
        --init --read-only --pids-limit 512 --memory 4g --cpus 2 \
        --cap-drop ALL --cap-add SETUID --cap-add SETGID --cap-add KILL --security-opt no-new-privileges \
        --tmpfs /tmp:rw,exec,nosuid,nodev,size=2g,mode=1777 \
        --tmpfs /home/verifier:rw,nosuid,nodev,size=256m,uid=10001,gid=10001,mode=700 \
        --tmpfs /workspace:rw,exec,nosuid,nodev,size=512m,uid=10001,gid=10001,mode=700 \
        --env NVIDIA_VISIBLE_DEVICES=void --env CUDA_VISIBLE_DEVICES= \
        --env SM75_BENCHLOCAL_ROLE="$role" --env-file "$START_ENV" \
        --env SM75_BENCH_PROXY_ORIGIN="$origin" \
        --label io.sm75.benchlocal.role=verifier --label io.sm75.benchlocal.unified=1 \
        --entrypoint python3 "$IMAGE" /opt/vllm-sm75/benchlocal-entrypoint.py >/dev/null
    fi

    ip=$(container_ip "$name" || true)
    [ -n "$ip" ] || {
      echo "红：$name 没有拿到 $NETWORK 的 IP" >&2
      exit 2
    }
    for _ in $(seq 1 30); do
      health_ok "$ip" && break
      sleep 1
    done
    health_ok "$ip" || {
      echo "红：$name 健康检查失败（$ip:4200）" >&2
      docker logs --tail 40 "$name" 2>&1 || true
      exit 2
    }

    case "$role" in
      bugfind-15)
        verifiers=$(printf '{"bugfind-15":"http://%s:4200"' "$ip")
        ;;
      cli-40)
        verifiers="$verifiers,\"cli-40\":\"http://$ip:4200\""
        ;;
      hermesagent-20)
        verifiers="$verifiers,\"hermesagent-20\":\"http://$ip:4200\"}"
        ;;
    esac
  done

  write_config "$key" "$gateway" "$verifiers"
  log "CONFIG $CONFIG"
  log "GATEWAY $origin"
  log "VERIFIERS $verifiers"
}

do_stop() {
  need_docker
  local name
  for name in "${NAMES[@]}"; do
    container_exists "$name" || continue
    if container_running "$name"; then
      docker stop "$name" >/dev/null
      log "STOP $name"
    else
      log "STOPPED $name"
    fi
  done
}

panel_up() {
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "$PANEL_URL" 2>/dev/null || true)
  case "$code" in
    2??|3??|401) return 0 ;;
    *) return 1 ;;
  esac
}

do_status() {
  need_docker
  local name ip state health
  echo "IMAGE $IMAGE"
  echo "NETWORK $NETWORK"
  echo "DATA_ROOT $DATA_ROOT"
  echo "CONFIG $([ -f "$CONFIG" ] && echo present || echo missing)"
  echo "FLAG $([ -f "$FLAG" ] && tr -d ' \r\n' < "$FLAG" || echo missing)"
  echo "PANEL $(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "$PANEL_URL" 2>/dev/null || echo fail)"
  for name in "${NAMES[@]}"; do
    if ! container_exists "$name"; then
      echo "$name missing"
      continue
    fi
    state=$(docker inspect --format '{{.State.Status}}' "$name")
    ip=$(container_ip "$name" 2>/dev/null || true)
    health=down
    [ -n "$ip" ] && health_ok "$ip" && health=ok
    echo "$name $state ip=${ip:-none} health=$health"
  done
}

do_sync() {
  QUIET=1
  need_docker
  if ! panel_up; then
    do_stop
    echo "SYNC stop panel=down"
    return
  fi
  local enabled=1
  if [ -f "$FLAG" ]; then
    enabled=$(tr -d ' \r\n' < "$FLAG")
  fi
  if [ "$enabled" = "0" ]; then
    do_stop
    echo "SYNC stop benchlocal=off"
    return
  fi
  do_start
  echo "SYNC start panel=up benchlocal=on"
}

case "$ACTION" in
  start) do_start ;;
  stop) do_stop ;;
  status) do_status ;;
  sync) do_sync ;;
esac
