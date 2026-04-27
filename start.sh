#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Load .env for PORT
set -a
source .env 2>/dev/null || true
set +a
PORT="${PORT:-9002}"

cleanup() {
    echo "[*] Shutting down..."
    kill $AGENT_PID 2>/dev/null
    kill $NGROK_PID 2>/dev/null
    wait 2>/dev/null
    echo "[*] Stopped"
}
trap cleanup EXIT

# Start ngrok in the background
echo "[*] Starting ngrok tunnel on port $PORT..."
ngrok http "$PORT" --log=stdout --log-level=warn &
NGROK_PID=$!

# Wait for ngrok to be ready and get the URL
for i in $(seq 1 15); do
    NGROK_URL=$(curl -sf http://localhost:4040/api/tunnels 2>/dev/null \
        | python3 -c "import sys,json; ts=json.load(sys.stdin)['tunnels']; print(next(t['public_url'] for t in ts if t['proto']=='https'))" 2>/dev/null) && break
    sleep 1
done

if [ -z "${NGROK_URL:-}" ]; then
    echo "[!] Failed to get ngrok URL"
    exit 1
fi

echo "[*] ngrok URL: $NGROK_URL"
echo "[!] Set your Twilio webhook to: $NGROK_URL/call/inbound"

# Start the agent (it auto-detects ngrok URL via localhost:4040)
echo "[*] Starting telephony agent on port $PORT..."
.venv/bin/python server.py &
AGENT_PID=$!

# Wait for agent to be ready
for i in $(seq 1 15); do
    curl -sf "http://localhost:$PORT/call/inbound" -X POST >/dev/null 2>&1 && break
    sleep 1
done

echo "[*] Ready — waiting for calls"
echo ""

# Wait for either process to exit
wait -n $AGENT_PID $NGROK_PID
