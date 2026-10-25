#!/usr/bin/env bash
set -euo pipefail

# Provisions the telephony agent + Kokoro TTS server on a fresh GPU host.
#
# Usage:
#   ./deploy.sh [DEST_DIR]              # default: ~/telephony
#   GPU=1 ./deploy.sh                   # pin a specific GPU (default: freest card)
#   TTS_MODEL=chatterbox ./deploy.sh    # different TTS backend (default: kokoro)
#
# It installs everything and verifies it, but does NOT start the server —
# the final block prints the exact start command, so the GPU is your call.

DEST_DIR="${1:-$HOME/telephony}"
AGENT_REPO="git@github.com:michaelv2/telephony-agent.git"
TTS_REPO="git@github.com:michaelv2/tts-server.git"
TTS_MODEL="${TTS_MODEL:-kokoro}"

log()  { printf '\033[1;34m[deploy]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[deploy]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[deploy]\033[0m %s\n' "$*" >&2; exit 1; }

# --- Prerequisites ---
command -v git >/dev/null 2>&1 || fail "git is required"

ver_ok() { "$1" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 11) else 1)' 2>/dev/null; }
AGENT_PYTHON=""
for cand in python3.11 python3; do
    if command -v "$cand" >/dev/null 2>&1 && ver_ok "$cand"; then
        AGENT_PYTHON="$cand"
        break
    fi
done
[[ -n "$AGENT_PYTHON" ]] || fail "Python 3.11+ not found (tried python3.11, python3)"
ver_ok python3 2>/dev/null || warn "python3 is not 3.11+ — tts-server/setup-backend.sh calls python3 literally; point it at 3.11 first"

command -v nvidia-smi >/dev/null 2>&1 \
    && log "nvidia-smi OK: $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd', ' -)" \
    || warn "nvidia-smi not found — Whisper + Kokoro need a CUDA GPU (or change WHISPER_MODEL/device to CPU, expect slow STT)"
command -v espeak-ng >/dev/null 2>&1 || warn "espeak-ng missing — Kokoro needs it: sudo apt-get install espeak-ng"
command -v ngrok >/dev/null 2>&1 || warn "ngrok missing — needed for the Twilio tunnel unless PUBLIC_URL is set (sudo apt install via snap or brew/ngrok release)"

# --- GPU selection ---
if [[ -n "${GPU:-}" ]]; then
    GPU_CHOICE="$GPU"
elif command -v nvidia-smi >/dev/null 2>&1; then
    GPU_CHOICE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
else
    GPU_CHOICE="0"
fi
log "Target GPU: index ${GPU_CHOICE} (override: GPU=<n> $0)"
GPU_MEM_INFO=""
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_MEM_INFO=$(nvidia-smi --query-gpu=index,memory.used,memory.total \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk -F', ' -v i="$GPU_CHOICE" '$1==i {out="free " ($3-$2) "/" $3" MB"} END{print out}')
fi

# --- Clone or update repos (siblings, as start.sh expects) ---
clone_or_update() {
    local dir="$1" url="$2"
    if [[ -d "$dir/.git" ]]; then
        log "Updating $dir"
        git -C "$dir" fetch origin
        git -C "$dir" checkout main
        git -C "$dir" pull --ff-only origin main
    else
        log "Cloning $url -> $dir"
        git clone --branch main "$url" "$dir"
    fi
}
mkdir -p "$DEST_DIR"
clone_or_update "$DEST_DIR/telephony-agent" "$AGENT_REPO"
clone_or_update "$DEST_DIR/tts-server" "$TTS_REPO"

AGENT_DIR="$DEST_DIR/telephony-agent"
TTS_DIR="$DEST_DIR/tts-server"

# --- Environment (agent venv) ---
log "Building agent venv ($AGENT_PYTHON) + pip install -r requirements.txt"
"$AGENT_PYTHON" -m venv "$AGENT_DIR/.venv"
"$AGENT_DIR/.venv/bin/pip" install -q --upgrade pip
# first run pulls torch + faster-whisper wheels (~5GB); weights download on first call
"$AGENT_DIR/.venv/bin/pip" install -q -r "$AGENT_DIR/requirements.txt"

# --- TTS venv ---
log "Building TTS venv for: $TTS_MODEL"
( cd "$TTS_DIR" && ./setup-backend.sh "$TTS_MODEL" )

# --- .env scaffolding ---
ENV_FILE="$AGENT_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
    log ".env found, leaving it alone"
else
    log "Scaffolding $ENV_FILE with placeholders — fill in or copy your existing .env"
    cat > "$ENV_FILE" <<'EOF'
# Required
ANTHROPIC_API_KEY=
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_API_KEY=
TWILIO_API_SECRET=
TWILIO_PHONE=

# TTS (started by start.sh)
TTS_URL=http://localhost:9100
TTS_VOICE=af_heart

# STT / server
WHISPER_MODEL=base.en
PORT=9002
# PUBLIC_URL=  # leave unset; start.sh auto-detects ngrok
EOF
fi

# --- Smoke checks (no server started) ---
log "Verifying installs"
"$TTS_DIR/.venv-$TTS_MODEL/bin/python" -c "import torch; print('  tts venv torch', torch.__version__, '| cuda available:', torch.cuda.is_available())" || warn "torch import check failed"
if [[ "$TTS_MODEL" == "kokoro" ]]; then
    "$TTS_DIR/.venv-kokoro/bin/python" -c "import kokoro; print('  tts venv kokoro OK')" || warn "kokoro import check failed"
else
    # qwen3/chatterbox/orpheus each live in their own venv
    venv_for="$TTS_MODEL"; case "$TTS_MODEL" in qwen3-*) venv_for=qwen3 ;; esac
    "[ -d \"$TTS_DIR/.venv-$venv_for/bin/python\" ]" && log "  tts venv present: .venv-$venv_for" || warn "tts venv .venv-$venv_for missing"
fi
"$AGENT_DIR/.venv/bin/python" -c "import torch, faster_whisper, fastapi, twilio, anthropic; print(f'  agent venv imports OK | torch {torch.__version__}, cuda available: {torch.cuda.is_available()}')"

# --- Handoff ---
cat <<EOF

============================================
  DEPLOY READY
============================================
  Agent : $AGENT_DIR
  TTS   : $TTS_DIR   (model: $TTS_MODEL)
  GPU   : CUDA_VISIBLE_DEVICES=$GPU_CHOICE${GPU_MEM_INFO:+  ($GPU_MEM_INFO)}

  1. Check credentials:      $ENV_FILE
     (twilio:  TTS voice af_heart; webhook set after first start)
  2. Start:
       cd $AGENT_DIR
       CUDA_VISIBLE_DEVICES=$GPU_CHOICE ./start.sh $TTS_MODEL
  3. Twilio webhook →  https://<ngrok-url>/call/inbound  (shown in start.sh output)

  Notes:
  - First run downloads Whisper/Silero/Kokoro weights from Hugging Face (one time).
  - Ollama: the LLM leg is the Anthropic API, not local — only Whisper (~0.5GB)
    and Kokoro (~0.5GB) sit on the GPU, plus two CUDA contexts.
  - start.sh's setup check auto-verifies TTS, ngrok, and Twilio webhooks.
============================================
EOF
