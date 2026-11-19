#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

output="${1:?Usage: bash run.sh OUTPUT bench|load [RPM ...] -- [vLLM flags]}"
mode="${2:?Choose bench or load}"
shift 2

rates=()
while [[ $# -gt 0 && "$1" != -- ]]; do
  [[ "$1" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid RPM: $1" >&2; exit 2; }
  rates+=("$1")
  shift
done
if [[ $# -gt 0 ]]; then shift; fi

case "$mode" in
  bench) [[ ${#rates[@]} -eq 0 ]] || { echo "bench uses 120 RPM" >&2; exit 2; } ;;
  load) [[ ${#rates[@]} -gt 0 ]] || { echo "load needs at least one RPM" >&2; exit 2; } ;;
  *) echo "Choose bench or load" >&2; exit 2 ;;
esac

port="${PORT:-8766}"
export URL="http://127.0.0.1:$port"
if curl -s --max-time 2 "$URL/health" >/dev/null 2>&1; then
  echo "Port $port is already in use" >&2
  exit 1
fi

mkdir -p "$(dirname "$output")"
mkdir "$output"
server=(bash serve.sh --host 127.0.0.1 --port "$port"
        --revision b968826d9c46dd6066d109eabc6255188de91218 "$@")
printf '%q ' "${server[@]}" > "$output/server_command.txt"
printf '\n' >> "$output/server_command.txt"
"${server[@]}" > "$output/server.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ready=0
for ((i = 0; i < 180; i++)); do
  if curl -fsS --max-time 2 "$URL/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  kill -0 "$server_pid" 2>/dev/null || break
  sleep 2
done
if [[ "$ready" -ne 1 ]]; then
  echo "vLLM did not become ready. See $output/server.log" >&2
  exit 1
fi

if [[ "$mode" == bench ]]; then
  bash benchmark.sh "$output/120rpm"
else
  for rpm in "${rates[@]}"; do
    bash load_test.sh "$rpm" "$output/${rpm}rpm"
  done
fi
