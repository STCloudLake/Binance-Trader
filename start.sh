#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Binance Trader - quick start (bash / Linux / macOS / MSYS2)
#    ./start.sh              start in sim mode on 127.0.0.1:8899
#    ./start.sh live         start in live mode (testnet in this env)
#    ./start.sh status       is it running?
#    ./start.sh stop         stop the instance
#    PORT=8900 ./start.sh    use another port
#  Logs: run_logs/start.log
# ─────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")"

# On Windows (Git Bash / MSYS2 / Cygwin) use the batch launcher, which is the
# tested path there; this script's own logic targets Linux and macOS.
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) exec cmd //c start.cmd "$@" ;;
esac

PORT="${PORT:-8899}"
MODE="${MODE:-sim}"

find_pid() {
  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
      # Windows netstat: TCP  <local>  <foreign>  LISTENING  <pid>
      netstat -ano 2>/dev/null | awk -v p=":$PORT" '$2 ~ p"$" && $4 == "LISTENING" {print $5; exit}'
      ;;
    *)
      if command -v lsof >/dev/null 2>&1; then
        lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | head -1
      elif command -v ss >/dev/null 2>&1; then
        ss -tlnp 2>/dev/null | awk -v p=":$PORT" '$4 ~ p"$" {split($6, a, "="); split(a[2], b, ","); print b[1]; exit}'
      fi
      ;;
  esac
}

stop_pid() {
  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) taskkill //PID "$1" //T //F >/dev/null 2>&1 ;;
    *) kill "$1" 2>/dev/null || kill -9 "$1" 2>/dev/null ;;
  esac
}

NO_BROWSER=0
[ "${1:-}" = "--no-browser" ] && NO_BROWSER=1 && shift

pid="$(find_pid)"
case "${1:-}" in
  status)
    if [ -n "$pid" ]; then echo "[ok] running - PID $pid - http://127.0.0.1:$PORT"; exit 0; fi
    echo "[i] not running"; exit 1 ;;
  stop)
    if [ -n "$pid" ]; then stop_pid "$pid"; echo "[ok] stopped PID $pid"; exit 0; fi
    echo "[i] not running"; exit 0 ;;
esac

if [ -n "$pid" ]; then
  echo "[i] Already running on port $PORT (PID $pid) - opening browser."
  if [ "$NO_BROWSER" = "0" ]; then
    command -v xdg-open >/dev/null 2>&1 && xdg-open "http://127.0.0.1:$PORT" >/dev/null 2>&1 &
    command -v open >/dev/null 2>&1 && open "http://127.0.0.1:$PORT" >/dev/null 2>&1 &
  fi
  exit 0
fi

PY="python3"
command -v python3 >/dev/null 2>&1 || PY="python"
[ -x ".venv/bin/python" ] && PY=".venv/bin/python"
[ -x ".venv/Scripts/python.exe" ] && PY=".venv/Scripts/python.exe"
mkdir -p run_logs

echo "[i] Starting $MODE mode on http://127.0.0.1:$PORT ..."
nohup "$PY" -m app.main --mode "$MODE" --port "$PORT" >> run_logs/start.log 2>&1 &
echo "[i] PID $! - waiting for /health"

for _ in $(seq 1 45); do
  if curl -s -m 2 -o /dev/null "http://127.0.0.1:$PORT/health"; then
    echo "[ok] Up: http://127.0.0.1:$PORT"
    exit 0
  fi
  sleep 1
done

echo "[x] No healthy response after ~45s. See run_logs/start.log"
exit 1
