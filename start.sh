#!/usr/bin/env bash
set -euo pipefail

# Miniforge Start Script
# Launches: Miniforge API server + OpenWebUI + Grafana dashboard
# Default Grafana login: admin / admin

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_PORT="${MINIFORGE_PORT:-8000}"
API_HOST="${MINIFORGE_HOST:-0.0.0.0}"
MODEL="${MINIFORGE_MODEL:-MiniMaxAI/MiniMax-M2.7}"
BACKEND="${MINIFORGE_BACKEND:-llama_cpp}"
QUANTIZATION="${MINIFORGE_QUANTIZATION:-}"

cd "$SCRIPT_DIR"

echo "================================================"
echo "  Miniforge Orchestrator"
echo "================================================"
echo ""

# Auto-detect hardware info
echo "[1/5] Detecting hardware..."
python3 - <<'PY'
import sys
sys.path.insert(0, "src")
try:
    from miniforge.utils.hardware import detect_hardware
    hw = detect_hardware()
    print(f"  CPU: {hw.cpu.brand} ({hw.cpu.physical_cores}c/{hw.cpu.logical_cores}t)")
    print(f"  RAM: {hw.total_ram_gb:.1f} GB")
    print(f"  GPUs: {len(hw.gpus)}")
    for g in hw.gpus:
        print(f"    - {g.vendor} {g.name} ({g.vram_gb:.1f}GB)")
    print(f"  OS: {hw.os_name} (WSL={hw.is_wsl})")
except Exception as e:
    print(f"  Hardware detection skipped: {e}")
PY

echo ""
echo "[2/5] Checking dependencies..."
if ! command -v docker &>/dev/null; then
  echo "WARNING: Docker not found. OpenWebUI and Grafana will not start."
  echo "Install Docker: https://docs.docker.com/get-docker/"
  USE_DOCKER=false
else
  echo "  Docker found."
  USE_DOCKER=true
fi

if ! python3 -c "import fastapi" 2>/dev/null; then
  echo "  Installing server dependencies..."
  pip install -e ".[server]" || pip install fastapi uvicorn prometheus-client
fi

# Export env vars for the API server
export MINIFORGE_MODEL="$MODEL"
export MINIFORGE_BACKEND="$BACKEND"
[ -n "$QUANTIZATION" ] && export MINIFORGE_QUANTIZATION="$QUANTIZATION"

echo ""
echo "[3/5] Starting Miniforge API server..."
echo "  Endpoint: http://$API_HOST:$API_PORT"
echo "  Model:    $MODEL"
echo "  Backend:  $BACKEND"

python3 - <<'PY' &
import sys, os
sys.path.insert(0, "src")
from miniforge.webui.server import run_server
run_server(
    host=os.environ.get("MINIFORGE_HOST", "0.0.0.0"),
    port=int(os.environ.get("MINIFORGE_PORT", "8000")),
)
PY
API_PID=$!

# Wait for API to be ready
echo "  Waiting for API to be ready..."
for i in {1..60}; do
  if curl -sf http://localhost:$API_PORT/health &>/dev/null; then
    echo "  API is ready."
    break
  fi
  if ! kill -0 $API_PID 2>/dev/null; then
    echo "ERROR: API server exited unexpectedly."
    exit 1
  fi
  sleep 1
done

# Start Docker services
if [ "$USE_DOCKER" = true ]; then
  echo ""
  echo "[4/5] Starting OpenWebUI + Grafana via Docker Compose..."
  docker compose up -d

  echo ""
  echo "  Waiting for services..."
  for i in {1..60}; do
    if curl -sf http://localhost:8080 &>/dev/null && curl -sf http://localhost:3000/api/health &>/dev/null; then
      echo "  All services are ready."
      break
    fi
    sleep 2
  done
else
  echo ""
  echo "[4/5] Skipping Docker services (docker not available)."
fi

echo ""
echo "================================================"
echo "  Miniforge is running!"
echo "================================================"
echo ""
echo "  API Server:     http://localhost:$API_PORT"
echo "  Health Check:   http://localhost:$API_PORT/health"
echo "  Metrics:        http://localhost:$API_PORT/metrics"
echo ""
if [ "$USE_DOCKER" = true ]; then
  echo "  OpenWebUI:      http://localhost:8080"
  echo "  Grafana:        http://localhost:3000  (admin / admin)"
  echo "  Prometheus:     http://localhost:9090"
  echo ""
  echo "  OpenWebUI should auto-connect to the Miniforge API."
  echo "  If not, go to Admin Settings → Connections → OpenAI API"
  echo "  and set the URL to: http://host.docker.internal:$API_PORT/v1"
fi
echo ""
echo "  Press Ctrl+C to stop everything."
echo ""

# Trap to clean up
cleanup() {
  echo ""
  echo "[5/5] Shutting down..."
  kill $API_PID 2>/dev/null || true
  if [ "$USE_DOCKER" = true ]; then
    docker compose down
  fi
  echo "  Goodbye!"
  exit 0
}
trap cleanup INT TERM

# Wait for API process
wait $API_PID
