#!/usr/bin/env bash
# Run the collector, then keep the container alive so logs/state can be inspected.
set -uo pipefail

export PYTHONUNBUFFERED=1

# Log file on the shared volume so it can be tailed from outside the container
mkdir -p /app/data/logs
LOG_FILE="/app/data/logs/$(hostname).log"

# Stagger startup: extract collector number from -o argument (e.g. data/collector-5 -> 5)
CONTAINER_NUM=$(echo "$*" | grep -oP '(?<=collector-)\d+' | tail -1)
CONTAINER_NUM=${CONTAINER_NUM:-1}
STARTUP_DELAY=$(( (CONTAINER_NUM - 1) * 5 ))
echo "[entrypoint] Container ${CONTAINER_NUM}: waiting ${STARTUP_DELAY}s before starting..." | tee -a "$LOG_FILE"
sleep "$STARTUP_DELAY"

echo "[entrypoint] Starting collector with args: $*" | tee -a "$LOG_FILE"
python whisper_leak_collect_conversations.py "$@" 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo "[entrypoint] Collector finished successfully (exit 0). Container kept alive for inspection." | tee -a "$LOG_FILE"
else
    echo "[entrypoint] Collector exited with code $EXIT_CODE. Container kept alive for inspection." | tee -a "$LOG_FILE"
fi

# Sleep forever — use 'docker stop <container>' or Ctrl-C to kill.
tail -f /dev/null
