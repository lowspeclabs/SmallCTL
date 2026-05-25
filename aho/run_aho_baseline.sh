#!/usr/bin/env bash
# Helper script to run AHO baseline from ./aho
# Proxies to ./run_baseline.py with sensible defaults

set -euo pipefail

cd "$(dirname "$0")"

python3 run_baseline.py \
  --endpoint "http://192.168.1.9:8080" \
  --model "qwen3.5-4b" \
  --provider-profile llamacpp \
  --max-prompt-tokens 32768 \
  --context-limit 32768 \
  --tool-profiles "core,data,mutate,network" \
  --run-mode loop \
  --phase execute \
  --hide-thinking \
  --timeout 600 \
  --report "temp/4-b-baseline-2.md" \
  "$@"
