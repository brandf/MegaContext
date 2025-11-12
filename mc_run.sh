#!/usr/bin/env bash
set -euo pipefail

git pull

if [ ! -f ".mc_env" ]; then
    echo "[mc_run] No .mc_env found; running mc_setup..."
    bash mc_setup
fi

bash run10.sh --mc \
    --mc_train_report 1 \
    --mc_log_timers 1 \
    --mc_log_lod_ascii_train 1
