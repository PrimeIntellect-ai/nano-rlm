#!/usr/bin/env bash
set -eo pipefail

has_command() {
    command -v "$1" >/dev/null 2>&1
}

install_system_packages() {
    if has_command apt-get; then
        apt-get -o Acquire::Retries=3 update -qq
        DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=3 install -y -qq --no-install-recommends "$@"
    elif has_command apk; then
        apk add --no-cache "$@"
    elif has_command dnf; then
        dnf install -y "$@"
        dnf clean all
    elif has_command yum; then
        yum install -y "$@"
        yum clean all
    elif has_command microdnf; then
        microdnf install -y "$@"
        microdnf clean all
    elif has_command zypper; then
        zypper --non-interactive refresh
        zypper --non-interactive install --no-recommends "$@"
        zypper clean --all
    else
        echo "No supported package manager found to install: $*" >&2
        return 127
    fi
    hash -r
}

ensure_command() {
    local name="$1"
    shift
    if has_command "$name"; then
        return 0
    fi
    install_system_packages "$@"
    has_command "$name"
}

install_uv() {
    local installer="/tmp/rlm-uv-install.sh"
    if ! curl -LsSf https://astral.sh/uv/install.sh -o "$installer"; then
        install_system_packages ca-certificates
        curl -LsSf https://astral.sh/uv/install.sh -o "$installer"
    fi
    sh "$installer"
    rm -f "$installer"
}

python_supports_rlm() {
    "$1" - <<'PY'
import sys

raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
}

select_tool_python() {
    if [ -n "${RLM_TOOL_PYTHON:-}" ]; then
        echo "$RLM_TOOL_PYTHON"
        return
    fi
    if has_command python3 && python_supports_rlm "$(command -v python3)"; then
        command -v python3
        return
    fi
    if has_command python && python_supports_rlm "$(command -v python)"; then
        command -v python
        return
    fi
    echo "3.10"
}

ensure_command curl ca-certificates curl

# Always install latest uv (sandbox images may have stale versions). Pin
# UV_INSTALL_DIR (uv installer target) and UV_TOOL_BIN_DIR (`uv tool install`
# shim target) to the same directory, and make the PATH export agree — on
# images that set XDG_DATA_HOME independent of HOME (e.g. SWE-rebench-V2
# python images with HOME=/root, XDG_DATA_HOME=/workspace/.local/share),
# both uv and its tool shims otherwise resolve via $XDG_DATA_HOME/../bin
# while a $HOME-based PATH export points at an empty dir — leaving `uv`
# and any `uv tool install`-ed executables (e.g. `rlm`) off PATH.
export UV_INSTALL_DIR="${UV_INSTALL_DIR:-$HOME/.local/bin}"
export UV_TOOL_BIN_DIR="${UV_TOOL_BIN_DIR:-$UV_INSTALL_DIR}"
install_uv
export PATH="$UV_INSTALL_DIR:$PATH"

# Clone the repo
RLM_REPO_URL="${RLM_REPO_URL:-github.com/mannuch/nano-rlm.git}"
RLM_REPO_BRANCH="${RLM_REPO_BRANCH:-main}"
RLM_CHECKOUT="${RLM_CHECKOUT_PATH:-/tmp/rlm-checkout}"
if [ ! -f "$RLM_CHECKOUT/install.sh" ] || [ ! -f "$RLM_CHECKOUT/pyproject.toml" ]; then
    ensure_command git git
    case "$RLM_REPO_URL" in
        https://*|http://*)
            CLONE_URL="$RLM_REPO_URL"
            ;;
        github.com/*)
            CLONE_URL="https://${GH_TOKEN:+${GH_TOKEN}@}${RLM_REPO_URL}"
            ;;
        *)
            CLONE_URL="$RLM_REPO_URL"
            ;;
    esac
    rm -rf "$RLM_CHECKOUT"
    git clone --depth 1 --branch "$RLM_REPO_BRANCH" "$CLONE_URL" "$RLM_CHECKOUT"
fi

if ! has_command rg; then
    if ! install_system_packages ripgrep || ! has_command rg; then
        echo "Warning: ripgrep is not available; continuing without rg" >&2
    fi
fi

# Install rlm as an isolated CLI tool (separate venv, on PATH).
# Skills are owned by the environment (e.g. ComposableEnv uploads them to
# /task/rlm-skills before this script runs).  Discover and install any
# that are present so they're both importable and on PATH.
SKILL_ARGS=""
SKILL_TOOL_NAMES=""
if [ -d /task/rlm-skills ]; then
    for skill_dir in /task/rlm-skills/*/; do
        [ -f "$skill_dir/pyproject.toml" ] || continue
        skill_name=$(grep '^name' "$skill_dir/pyproject.toml" | head -1 | sed 's/.*"\(.*\)".*/\1/')
        SKILL_ARGS="$SKILL_ARGS --with-editable $skill_dir --with-executables-from $skill_name"
        for candidate in "$skill_dir"/src/*; do
            [ -d "$candidate" ] || continue
            [ "$(basename "$candidate")" = "__pycache__" ] && continue
            SKILL_TOOL_NAMES="$SKILL_TOOL_NAMES $(basename "$candidate")"
            break
        done
    done
fi

# Forward caller-supplied `uv tool install` extras (e.g. "--with numpy
# --with sympy") when set, so environments can inject extra pip deps into
# the rlm tool venv without patching this script. Tokens are word-split by
# the shell, so avoid shell metacharacters (e.g. `>=`) inside package specs.
EXTRA_UV_ARGS=""
if [ -n "${RLM_EXTRA_UV_ARGS:-}" ]; then
    EXTRA_UV_ARGS="$RLM_EXTRA_UV_ARGS"
fi

uv tool install --python "$(select_tool_python)" --editable "$RLM_CHECKOUT" $SKILL_ARGS $EXTRA_UV_ARGS

REAL_TOOL_DIR="$UV_TOOL_BIN_DIR/.rlm-real-tools"
mkdir -p "$REAL_TOOL_DIR"

wrap_skill_cli() {
    local tool_name="$1"
    local wrapper_path="$UV_TOOL_BIN_DIR/$tool_name"
    local real_path="$REAL_TOOL_DIR/$tool_name"

    if [ -x "$wrapper_path" ] && [ ! -x "$real_path" ]; then
        mv "$wrapper_path" "$real_path"
    fi
    [ -x "$real_path" ] || return 0

    cat > "$wrapper_path" <<EOF
#!/usr/bin/env bash
set -eo pipefail
REAL_TOOL="$real_path"
TOOL_NAME="$tool_name"
SOURCE="\${RLM_TOOL_CALL_SOURCE:-bash}"
if [ -n "\${RLM_SESSION_DIR:-}" ]; then
  printf '{"tool":"%s","source":"%s","timestamp":%s}\n' \\
      "\$TOOL_NAME" "\$SOURCE" "\$(date +%s.%N)" \\
      >> "\${RLM_SESSION_DIR}/programmatic_tool_calls.jsonl" 2>/dev/null || true
fi
exec "\$REAL_TOOL" "\$@"
EOF
    chmod +x "$wrapper_path"
}

for tool_name in $SKILL_TOOL_NAMES; do
    wrap_skill_cli "$tool_name"
done
