#!/usr/bin/env bash
# 检查 docker build 前置条件（面向已完成 venv + curobo build 的环境）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LEROOT_REV="a445d9c9da6bea99a8972daa4fe1fdd053d711d2"
LEROOT_DIR="policy/pi0/vendor/lerobot"
CUROBO_SO_DIR="envs/curobo/src/curobo/curobolib"
VENV="policy/pi0/.venv"

echo "==> [1/4] lerobot vendor（build 前 clone，不提交 git）"
mkdir -p "$(dirname "$LEROOT_DIR")"
if [[ ! -d "$LEROOT_DIR/.git" ]]; then
    git clone https://github.com/huggingface/lerobot "$LEROOT_DIR"
fi
git -C "$LEROOT_DIR" checkout "$LEROOT_REV"

actual_rev="$(git -C "$LEROOT_DIR" rev-parse HEAD)"
if [[ "$actual_rev" != "$LEROOT_REV" ]]; then
    echo "ERROR: lerobot 版本不匹配: 期望 $LEROOT_REV, 实际 $actual_rev"
    exit 1
fi
echo "    lerobot OK @ $actual_rev"

echo "==> [2/4] pi0 venv"
if [[ ! -x "$VENV/bin/python" ]]; then
    cat <<EOF
ERROR: 未找到 $VENV
      请先在 policy/pi0 完成 uv sync：

        cd policy/pi0 && uv sync --frozen --python python3.11
EOF
    exit 1
fi
echo "    venv OK ($("$VENV/bin/python" --version))"

echo "==> [3/4] curobo 预编译 .so"
if ! compgen -G "$CUROBO_SO_DIR/*.so" > /dev/null; then
    cat <<EOF
ERROR: $CUROBO_SO_DIR 下没有 .so 文件。
      请先在宿主机编译 curobo：

        cd policy/pi0 && source .venv/bin/activate
        cd ../../envs/curobo && pip install -e . --no-build-isolation

      Docker build 会直接从 envs/curobo 拷贝 .so，无需 docker-vendor/。
EOF
    exit 1
fi
echo "    curobo .so OK ($(ls -1 "$CUROBO_SO_DIR"/*.so | wc -l) 个)"

echo "==> [4/4] Dockerfile 相关文件"
required=(
    Dockerfile
    .dockerignore
    "$LEROOT_DIR/pyproject.toml"
    policy/pi0/pyproject.toml
    policy/pi0/uv.lock
    policy/pi0/LICENSE
    envs/curobo/setup.py
)

missing=0
for f in "${required[@]}"; do
    if [[ ! -e "$f" ]]; then
        echo "ERROR: 缺少 $f"
        missing=1
    fi
done
[[ "$missing" -eq 0 ]] || exit 1

echo ""
echo "检查通过。接下来："
echo "  docker build -t robotwin-pi0:cu124 ."
echo ""
echo "说明："
echo "  - 无需 docker-vendor/，curobo .so 来自 envs/curobo/"
echo "  - assets 运行时挂载，不打进镜像"
echo "  - eval_data/beat_block_hammer/0 可选，见 docs/docker-build.md"
