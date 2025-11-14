#!/usr/bin/env bash
set -euo pipefail

source_shell_profile() {
    if [ -n "${MC_SHELL_PROFILE:-}" ] && [ -f "$MC_SHELL_PROFILE" ]; then
        # shellcheck disable=SC1090
        source "$MC_SHELL_PROFILE"
        echo "[mc_run] Sourced MC_SHELL_PROFILE=$MC_SHELL_PROFILE"
        return
    fi
    for candidate in "$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.profile" "$HOME/.zshrc"; do
        if [ -f "$candidate" ]; then
            # shellcheck disable=SC1090
            source "$candidate"
            echo "[mc_run] Sourced $candidate"
            return
        fi
    done
}

ensure_uv_in_path() {
    if command -v uv >/dev/null 2>&1; then
        return
    fi
    if [ -f "$HOME/.cargo/env" ]; then
        # shellcheck disable=SC1090
        source "$HOME/.cargo/env"
    fi
    if command -v uv >/dev/null 2>&1; then
        return
    fi
    for dir in "$HOME/.cargo/bin" "$HOME/.local/bin"; do
        if [ -d "$dir" ] && [[ ":$PATH:" != *":$dir:"* ]]; then
            export PATH="$dir:$PATH"
        fi
        if command -v uv >/dev/null 2>&1; then
            return
        fi
    done
    if [ -f "$HOME/.cargo/env" ]; then
        echo "[mc_run] uv not found on PATH. Run the following in your shell, then rerun mc_run:"
        echo "    source \"$HOME/.cargo/env\""
    else
        echo "[mc_run] uv not found. Make sure it is installed or add it via:"
        echo "    export PATH=\"\$HOME/.cargo/bin:\$PATH\""
    fi
    exit 1
}

source_shell_profile
ensure_uv_in_path

git pull

if [ ! -f ".mc_env" ]; then
    echo "[mc_run] No .mc_env found; running mc_setup..."
    python3 mc_setup
fi

bash run10.sh --mc \
    --gpu 5090 \
    --mc_train_report 1 \
    --mc_log_timers 1 \
    --mc_log_lod_ascii_train 1 \
    --mc_log_lens_debug 1 \
    --mc_disable_val 1
