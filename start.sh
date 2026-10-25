#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Model argument: ./start.sh [model]   (default: kokoro)
TTS_MODEL="${1:-kokoro}"
TTS_SERVER_DIR="${TTS_SERVER_DIR:-$(cd "$(dirname "$0")/.." && pwd)/tts-server}"

# Load .env for PORT
set -a
source .env 2>/dev/null || true
set +a
PORT="${PORT:-9002}"
TTS_URL="${TTS_URL:-http://localhost:9100}"
TTS_PORT="${TTS_PORT:-9100}"

# Log full console output to timestamped file
LOGS_DIR="$(dirname "$0")/logs"
mkdir -p "$LOGS_DIR"
LOG_FILE="$LOGS_DIR/$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[*] Logging to $LOG_FILE"

cleanup() {
    echo "[*] Shutting down..."
    kill "${AGENT_PID:-}" 2>/dev/null || true
    kill "${NGROK_PID:-}" 2>/dev/null || true
    kill "${TTS_PID:-}" 2>/dev/null || true
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

# Start the TTS server (kokoro by default; override with ./start.sh <model>)
echo "[*] Starting TTS server (model: $TTS_MODEL) on port $TTS_PORT..."
if [[ ! -x "$TTS_SERVER_DIR/run.sh" ]]; then
    echo "[!] TTS server not found at $TTS_SERVER_DIR (set TTS_SERVER_DIR to override)"
    exit 1
fi
( cd "$TTS_SERVER_DIR" && exec ./run.sh "$TTS_MODEL" --port "$TTS_PORT" ) &
TTS_PID=$!

# Wait for TTS to come up (models load slowly on GPU)
for i in $(seq 1 60); do
    curl -sf "$TTS_URL/health" >/dev/null 2>&1 && break
    sleep 1
done
if curl -sf "$TTS_URL/health" >/dev/null 2>&1; then
    echo "[*] TTS server ready"
else
    echo "[!] TTS server not responding on $TTS_URL — agent voice replies will fail"
fi

# Start the agent (it auto-detects ngrok URL via localhost:4040)
echo "[*] Starting telephony agent on port $PORT..."
.venv/bin/python server.py &
AGENT_PID=$!

# Wait for agent to be ready
for i in $(seq 1 15); do
    curl -sf "http://localhost:$PORT/docs" >/dev/null 2>&1 && break
    sleep 1
done

# --- Setup Checks ---
echo ""
echo "============================================"
echo "  SETUP CHECKS"
echo "============================================"
CHECKS_PASSED=0
CHECKS_FAILED=0

pass() { echo "  ✓ $1"; CHECKS_PASSED=$((CHECKS_PASSED + 1)); }
fail() { echo "  ✗ $1"; CHECKS_FAILED=$((CHECKS_FAILED + 1)); }

# Environment variables
echo ""
echo "  Environment"
echo "  -----------"
for var in ANTHROPIC_API_KEY TWILIO_ACCOUNT_SID TWILIO_AUTH_TOKEN TWILIO_API_KEY TWILIO_API_SECRET TWILIO_PHONE; do
    if eval "[ -n \"\${$var:-}\" ]"; then
        pass "$var is set"
    else
        fail "$var is NOT set"
    fi
done

# Services
echo ""
echo "  Services"
echo "  --------"

# TTS server
TTS_URL="${TTS_URL:-http://localhost:9100}"
if curl -sf "$TTS_URL" >/dev/null 2>&1 || curl -sf "$TTS_URL/health" >/dev/null 2>&1; then
    pass "TTS server reachable at $TTS_URL"
else
    fail "TTS server NOT reachable at $TTS_URL"
fi

# ngrok
pass "ngrok tunnel active: $NGROK_URL"

# Agent server
if curl -sf "http://localhost:$PORT/docs" >/dev/null 2>&1; then
    pass "Agent server running on port $PORT"
else
    fail "Agent server NOT responding on port $PORT"
fi

# Twilio webhook configuration
echo ""
echo "  Twilio Webhooks"
echo "  ---------------"
EXPECTED_VOICE_URL="$NGROK_URL/call/inbound"
EXPECTED_SMS_URL="$NGROK_URL/sms/inbound"

if [ -n "${TWILIO_ACCOUNT_SID:-}" ] && [ -n "${TWILIO_AUTH_TOKEN:-}" ] && [ -n "${TWILIO_PHONE:-}" ]; then
    # Fetch the current webhook config from Twilio
    PHONE_SID=$(.venv/bin/python -c "
from twilio.rest import Client
c = Client('$TWILIO_ACCOUNT_SID', '$TWILIO_AUTH_TOKEN')
nums = c.incoming_phone_numbers.list(phone_number='$TWILIO_PHONE')
print(nums[0].sid if nums else '')
" 2>/dev/null) || PHONE_SID=""

    if [ -n "$PHONE_SID" ]; then
        WEBHOOK_INFO=$(.venv/bin/python -c "
from twilio.rest import Client
c = Client('$TWILIO_ACCOUNT_SID', '$TWILIO_AUTH_TOKEN')
n = c.incoming_phone_numbers('$PHONE_SID').fetch()
print(f'{n.voice_url}|{n.sms_url}')
" 2>/dev/null) || WEBHOOK_INFO=""

        if [ -n "$WEBHOOK_INFO" ]; then
            CURRENT_VOICE_URL="${WEBHOOK_INFO%%|*}"
            CURRENT_SMS_URL="${WEBHOOK_INFO##*|}"

            if [ "$CURRENT_VOICE_URL" = "$EXPECTED_VOICE_URL" ]; then
                pass "Voice webhook matches: $CURRENT_VOICE_URL"
            else
                fail "Voice webhook MISMATCH"
                echo "        Current:  $CURRENT_VOICE_URL"
                echo "        Expected: $EXPECTED_VOICE_URL"
            fi

            if [ "$CURRENT_SMS_URL" = "$EXPECTED_SMS_URL" ]; then
                pass "SMS webhook matches: $CURRENT_SMS_URL"
            else
                fail "SMS webhook MISMATCH"
                echo "        Current:  $CURRENT_SMS_URL"
                echo "        Expected: $EXPECTED_SMS_URL"
            fi
        else
            fail "Could not fetch webhook config from Twilio API"
        fi
    else
        fail "Phone number $TWILIO_PHONE not found in Twilio account"
    fi
else
    fail "Cannot check webhooks — missing TWILIO credentials"
fi

# Summary
echo ""
echo "  --------"
echo "  $CHECKS_PASSED passed, $CHECKS_FAILED failed"
if [ $CHECKS_FAILED -gt 0 ]; then
    echo ""
    echo "  Fix the issues above before expecting calls to work."
    echo "  To update Twilio webhooks, run:"
    echo "    .venv/bin/python -c \""
    echo "from twilio.rest import Client; import os"
    echo "c = Client(os.environ['TWILIO_ACCOUNT_SID'], os.environ['TWILIO_AUTH_TOKEN'])"
    echo "nums = c.incoming_phone_numbers.list(phone_number=os.environ['TWILIO_PHONE'])"
    echo "nums[0].update(voice_url='$EXPECTED_VOICE_URL', sms_url='$EXPECTED_SMS_URL')"
    echo "print('Updated!')\""
fi
echo "============================================"
echo ""

echo "[*] Ready — waiting for calls"
echo ""

# Wait for any of the three processes to exit
wait -n "$AGENT_PID" "$NGROK_PID" "$TTS_PID"
