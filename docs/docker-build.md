# RoboTwin pi0 Docker 构建指南

镜像标签：`robotwin-pi0:cu124`  
预期体积：约 30GB（不含 `assets`）

## 适用环境

本仓库 **不提交** curobo `.so` 等大文件。目标使用者应已在本机完成：

1. `policy/pi0` 的 `uv sync`（venv 已就绪）
2. `envs/curobo` 的编译安装（`curobolib/*.so` 已生成）

## 前置条件

- Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- 磁盘空间 ≥ 60GB
- 可访问 `ghcr.io`（拉取 uv 基础层）

## 快速开始

```bash
cd /path/to/RoboTwin

# clone lerobot + 检查 venv / curobo .so
bash scripts/prepare-docker-build.sh

docker build -t robotwin-pi0:cu124 .
```

构建后验证：

```bash
docker run --rm --gpus all robotwin-pi0:cu124 bash -lc '
source /workspace/RoboTwin/policy/pi0/.venv/bin/activate
python -c "import torch; print(\"cuda\", torch.cuda.is_available())"
python -c "from curobo.wrap.reacher.motion_gen import MotionGen; print(\"curobo ok\")"
python /workspace/RoboTwin/script/test_render.py
'
```

预期：`cuda True`、`curobo ok`、`Render Well`。

## Git 提交策略（本分支）

| 内容 | 是否提交 git | 来源 |
|------|-------------|------|
| `Dockerfile`、`.dockerignore`、文档、脚本 | ✅ 提交 | 仓库 |
| `policy/pi0/vendor/lerobot` | ❌ 不提交 | `prepare-docker-build.sh` build 前 clone |
| `envs/curobo/`（含 `.so`） | ❌ 不提交 | 本地 curobo build |
| `docker-vendor/` | ❌ 不提交 | 已废弃，无需维护 |
| `assets/`、`eval_data/` | ❌ 不提交 | 挂载或本地准备 |

## 本地准备 curobo .so

若尚未编译 curobo：

```bash
cd policy/pi0 && source .venv/bin/activate
cd ../../envs/curobo && pip install -e . --no-build-isolation
```

完成后应存在：

```
envs/curobo/src/curobo/curobolib/*.so   # 5 个文件
```

Docker build 时 `COPY envs/curobo` 会带上这些 `.so`，**无需** `docker-vendor/`。

`.so` 须与容器内 Python 3.11 / CUDA 12.4 / torch 版本匹配。

## lerobot（build 前自动 clone）

```bash
bash scripts/prepare-docker-build.sh
# 或手动：
git clone https://github.com/huggingface/lerobot policy/pi0/vendor/lerobot
cd policy/pi0/vendor/lerobot && git checkout a445d9c9da6bea99a8972daa4fe1fdd053d711d2
```

## 运行时挂载

| 路径 | 说明 |
|------|------|
| `assets/` | 仿真资源（约 16GB），必须 `-v` 挂载 |
| `eval_result/` | 评测输出，建议挂载 |
| PyTorch 模型 | 按需挂载到 `policy/pi0/torch_model/...` |

## 导出与分发

```bash
docker save robotwin-pi0:cu124 -o robotwin-pi0-cu124.tar
docker load -i robotwin-pi0-cu124.tar
```

## 常见问题

**Q: 缺少 curobo .so 导致 build 失败？**  
先在本机 `pip install -e .` 编译 curobo，再运行 `bash scripts/prepare-docker-build.sh`。

**Q: 还需要 docker-vendor 吗？**  
不需要。旧方案已移除，可直接删除本地 `docker-vendor/` 目录。

**Q: build 从 `COPY . .` 开始重编？**  
改业务代码会 bust 该层，但 `uv sync` / curobo 等前置层仍 CACHED。
